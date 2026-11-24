"""報道のまとめを作る：記事 → 束 → 台本 → 音声 → json と mp3。

  python -m briefing --feed URL_or_FILE --out DIR [--server http://127.0.0.1:8080] [--vv-root DIR] [--no-audio]

出力（DIR）：latest.json（形は 司令部が固定した形、項目の追加のみ）、latest.mp3、credits.txt、timing.json。
作れなかったとき（出来事が MIN_EVENTS 未満など）は hold: true の json だけを書き、音声は作らない。
"""
import argparse
import collections
import datetime
import json
import os
import sys
import time
import urllib.request

from . import audio as A
from . import cluster as C
from . import places as P
from . import script as T
from . import sources as S

MIN_EVENTS = 3
CAT_JA = {'military': '軍事', 'conflict': '紛争', 'diplomacy': '外交', 'alert': '警報'}
JST = datetime.timezone(datetime.timedelta(hours=9))
MODEL_NAME = 'Qwen3-8B'


def load_feed(src):
    if src.startswith(('http://', 'https://')):
        req = urllib.request.Request(src, headers={'Accept': 'application/json', 'User-Agent': 'observatory-data briefing'})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode('utf-8'))
    with open(src, encoding='utf-8') as f:
        return json.load(f)


def load_terms():
    with open(os.path.join(os.path.dirname(__file__), 'glossary.json'), encoding='utf-8') as f:
        return json.load(f)['terms']


def safe_url(u):
    return u if isinstance(u, str) and u.startswith(('http://', 'https://')) and len(u) <= 600 else None


def event_meta(cand, reps):
    """分類・国・地名・出典の欄（モデルを使わない部分）。"""
    recs = cand['records']
    st = cand['stats']
    cat = collections.Counter(r.get('category') for r in recs).most_common(1)[0][0]
    hits = P.find([f"{r.get('title', '')} {r.get('titleJa', '')} {r.get('summary', '')}" for r in reps])
    codes = P.country_codes(hits)
    if not codes:
        sc = collections.Counter(r.get('scope') for r in recs if r.get('scope') and r.get('scope') != '-')
        codes = [c for c, _ in sc.most_common(1)]
    main = codes[0] if codes else None
    label = CAT_JA.get(cat, '報道') + (' · ' + S.COUNTRY_JA.get(main, main) if main else '')
    ordered = sorted(reps, key=lambda r: S.is_primary_or_state(r.get('source')))
    items = [{'source': r.get('source'), 'title': T.clean(r.get('title') or r.get('titleJa'), 200), 'url': safe_url(r.get('url'))}
             for r in ordered if safe_url(r.get('url'))][:6]
    return {
        'category': cat, 'label': label, 'country': main, 'countries': codes, 'place': P.main_place(hits),
        'sources': {'count': len(st['families']), 'countries': len(st['countries']),
                    'names': [T.display_name(r.get('source')) for r in ordered][:3], 'items': items},
    }


def source_line(meta):
    n, m = meta['sources']['count'], meta['sources']['countries']
    if m:
        return f'この件は、独立した{n}の報道機関、{m}か国が伝えています。'
    return f'この件は、独立した{n}の報道機関が伝えています。'


def make_scripts(records, args, terms, tok, log, deadline):
    """選んだ束の台本。(events, dropped, 統計)。events は音声の前の形。"""
    tc = time.time()
    clusters = C.cluster(records, tok[0], args.threshold)
    cands = C.select(clusters)
    stats = {'clusterSec': round(time.time() - tc, 2), 'clusters': len(clusters), 'multiSource': sum(1 for c in clusters if len(C.describe(c)['families']) >= 2),
             'candidates': len(cands)}
    known = [w[0] for w in A.load_readings()]
    events, dropped, picked_heads = [], [], []
    for rank, cand in enumerate(cands[:args.candidates], 1):
        if len(events) >= args.max_events or time.time() > deadline:
            break
        heads = [C.head(r) for r in cand['records']]
        if any(C.same_story(heads, h, tok[0]) for h in picked_heads):
            dropped.append({'reason': '選んだ束と同じ話（見出しの語が重なる）', 'headline': heads[0][:40]})
            continue
        reps = C.representatives(cand['records'])
        conflicts = T.number_conflicts(reps)
        use_terms = T.glossary_for(reps, terms)
        script, why, notes, tries = T.write(args.server, reps, use_terms, conflicts, known)
        meta = event_meta(cand, reps)
        if script is None:
            dropped.append({'reason': '、'.join(why), 'headline': (heads[0])[:40]})
            log(f'#{rank} hold: {"、".join(why)}')
            continue
        picked_heads.append(heads)
        log(f'#{rank} ok: {script["headline"]}（系列 {meta["sources"]["count"]}・国 {meta["sources"]["countries"]}）')
        events.append({'rank': rank, 'script': script, 'meta': meta, 'notes': notes, 'tries': tries,
                       'conflicts': conflicts, 'families': cand['stats']['families']})
    return events, dropped, stats


def make_audio(events, now_ms, voice_root, no_audio):
    """音声と、各文・各出来事の開始秒。events を書き換えずに (out_events, intro, outro, mp3, pcm秒) を返す。"""
    d = datetime.datetime.fromtimestamp(now_ms / 1000, JST)
    intro = f'世界情勢観測、{d.month}月{d.day}日のまとめです。過去24時間の報道から、{len(events)}件の出来事をお伝えします。'
    outro = '以上です。この音声は、公開されている報道の見出しと要約を、機械で要約して読み上げたものです。'
    seq = [('intro', intro, A.INTRO_GAP)]
    for i, e in enumerate(events):
        lines = [ln['text'] for ln in e['script']['lines']] + [source_line(e['meta'])]
        e['spoken'] = lines
        for j, t in enumerate(lines):
            last = j == len(lines) - 1
            seq.append((f'{i}.{j}', t, A.SAMPLE_PAUSE['event'] if last else A.SAMPLE_PAUSE['line']))
    seq.append(('outro', outro, 0.5))
    if no_audio:
        return None, intro, outro, None, 0.0, []
    voice = A.Voice(voice_root)
    kana = {'intro': voice.kana(intro)}
    starts, pcm = A.assemble(seq, voice.wav)
    for key, text, _ in seq:
        if key != 'intro':
            kana[key] = voice.kana(text)
    return starts, intro, outro, A.to_mp3(pcm), pcm.seconds, kana


def assemble_json(events, dropped, stats, starts, intro, outro, mp3, seconds, now_ms, counts, build, voice, hold, hold_reason):
    out_events = []
    for i, e in enumerate(events):
        lines = []
        for j, t in enumerate(e['spoken']):
            sub = e['script']['lines'][j]['sub'] if j < len(e['script']['lines']) else '出典'
            lines.append({'start': (starts or {}).get(f'{i}.{j}', 0.0), 'sub': sub, 'text': t})
        nxt = (starts or {}).get(f'{i + 1}.0', (starts or {}).get('outro', seconds))
        end = round(max(lines[-1]['start'], nxt - A.SAMPLE_PAUSE['event']), 2) if starts else 0.0
        ev = {'start': lines[0]['start'], 'end': end}
        ev.update({k: e['meta'][k] for k in ('category', 'label', 'country', 'countries', 'place')})
        ev.update({'headline': e['script']['headline'], 'lines': lines, 'sources': e['meta']['sources'],
                   'debug': {'rank': e['rank'], 'families': e['families'], 'notes': e['notes'], 'tries': e['tries'],
                             'numberConflicts': e['conflicts']}})
        out_events.append(ev)
    return {
        'version': 1, 'createdAt': now_ms, 'windowHours': 24, 'hold': hold, 'holdReason': hold_reason,
        'generated': 'machine', 'model': MODEL_NAME,
        'voice': {'credit': voice['credit'], 'styleId': voice['style_id'], 'speaker': voice['speaker'], 'style': voice['style']},
        'audio': {'file': 'latest.mp3', 'seconds': round(seconds, 2), 'bytes': len(mp3 or b'')} if mp3 else None,
        'counts': counts, 'intro': {'start': 0, 'text': intro}, 'outro': {'start': (starts or {}).get('outro', 0.0), 'text': outro},
        'events': out_events, 'dropped': dropped, 'build': build,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--feed', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--server', default='http://127.0.0.1:8080')
    ap.add_argument('--vv-root', default=os.environ.get('VV_ROOT', ''))
    ap.add_argument('--no-audio', action='store_true')
    ap.add_argument('--now-ms', type=int, default=0)
    ap.add_argument('--max-events', type=int, default=6)
    ap.add_argument('--candidates', type=int, default=14)
    ap.add_argument('--threshold', type=float, default=C.THRESHOLD)
    ap.add_argument('--minutes', type=float, default=30, help='台本に使う時間の上限')
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    log = lambda s: print(s, flush=True)
    t0 = time.time()
    now_ms = a.now_ms or int(time.time() * 1000)
    feed = load_feed(a.feed)
    allrecs = [r for r in (feed.get('records') or []) if isinstance(r, dict)]
    recs = C.in_window(allrecs, now_ms)
    if not recs:
        print('::error title=briefing::24 時間以内の記事が無い')
        return 2
    tok = C.default_tokenizer()
    log(f'記事 {len(allrecs)}、24 時間以内 {len(recs)}、形態素解析 {tok[1]}')
    t1 = time.time()
    terms = load_terms()
    events, dropped, stats = make_scripts(recs, a, terms, tok, log, t1 + a.minutes * 60)
    t2 = time.time()
    voice = A.load_voice()
    hold, reason = False, ''
    if len(events) < MIN_EVENTS:
        hold, reason = True, f'読める出来事が {len(events)} 件（{MIN_EVENTS} 件未満）'
        dropped += [{'reason': 'まとめ全体を hold（' + reason + '）', 'headline': e['script']['headline']} for e in events]
        events = []
    starts = mp3 = None
    seconds, kana = 0.0, []
    intro = outro = ''
    if not hold:
        starts, intro, outro, mp3, seconds, kana = make_audio(events, now_ms, a.vv_root, a.no_audio)
    t3 = time.time()
    counts = {'records': len(recs), 'sources': len({r.get('source') for r in recs}), 'clusters': stats['clusters'],
              'multiSource': stats['multiSource']}
    timing = {'clusterSec': stats['clusterSec'], 'scriptSec': round(t2 - t1 - stats['clusterSec'], 1), 'audioSec': round(t3 - t2, 1), 'totalSec': round(t3 - t0, 1)}
    build = {'schema': 'briefing-1', 'tokenizer': tok[1], 'threshold': a.threshold, 'timing': timing,
             'candidates': stats['candidates'], 'kana': kana}
    doc = assemble_json(events, dropped, stats, starts, intro, outro, mp3, seconds, now_ms, counts, build, voice, hold, reason)
    with open(os.path.join(a.out, 'latest.json'), 'w', encoding='utf-8') as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
        f.write('\n')
    if mp3:
        with open(os.path.join(a.out, 'latest.mp3'), 'wb') as f:
            f.write(mp3)
    with open(os.path.join(a.out, 'credits.txt'), 'w', encoding='utf-8') as f:
        f.write(voice['credit'] + '\n台本: Qwen3-8B（Apache-2.0）で機械作成。元記事の出典は latest.json の sources。\n')
    with open(os.path.join(a.out, 'timing.json'), 'w', encoding='utf-8') as f:
        json.dump(timing, f)
    log(f'hold={hold} 出来事 {len(events)} 落ち {len(dropped)} 音声 {seconds:.1f} 秒 {timing}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
