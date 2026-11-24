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
import re
import sys
import time
import urllib.request

from . import audio as A
from . import cluster as C
from . import places as P
from . import script as T
from . import sources as S

MIN_EVENTS = 3
TARGET_SECONDS = 120      # 音声の長さの上限の目安。超えるぶんは順位の低い出来事から外す
CHARS_PER_SEC = 6.0       # もち子さん（ノーマル）の実測：1 秒に約 6 字（2026-11-24 の試作）
INTRO_OUTRO_SEC = 14.0    # 冒頭と結びの定型文（実測：冒頭 約 7.5 秒、結び 約 6 秒）
CAT_JA = {'military': '軍事', 'conflict': '紛争', 'diplomacy': '外交', 'alert': '警報'}
# 分類：配信元の既定の分類（多くは外交）に寄るので、代表の記事の半数以上に出る語で上書きする。先に当たった方。
CAT_HINT = (
    ('conflict', re.compile(r'air ?strike|shelling|killed|\bdead\b|wounded|casualt|\battack|bombing|bombard|\bsunk\b|\bsank\b|\bsinks?\b|'
                            r'空爆|空袭|空襲|砲撃|炮击|炮擊|死亡|遇难|遇難|襲撃|袭击|襲擊|击沉|擊沉|撃沈', re.I)),
    ('military', re.compile(r'warship|\bnavy\b|naval|frigate|destroyer|submarine|aircraft carrier|missile|\bdrones?\b|fighter jet|warplane|'
                            r'\btroops\b|military|\bdrills?\b|live-fire|blockade|\bmines?\b|coast guard|'
                            r'军舰|軍艦|海军|海軍|导弹|飛彈|演习|演習|潜艇|潛艦|军事|軍事|封锁|封鎖|水雷|機雷|海警|军机|軍機', re.I)),
)
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
    cat = category_of(recs, reps)
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


def category_of(recs, reps):
    """束の分類：代表の記事の半数以上に紛争・軍事の語が出ればそれ、無ければ記事の分類の多数。"""
    texts = [f"{r.get('title', '')} {r.get('summary', '')}" for r in reps]
    for cat, rx in CAT_HINT:
        if texts and sum(1 for t in texts if rx.search(t)) * 2 >= len(texts):
            return cat
    return collections.Counter(r.get('category') for r in recs).most_common(1)[0][0]


def same_event(a, b, tok):
    """分類・主な国の組（先頭 2 つ）・地点が同じで、見出しの語も重なる束は、同じ出来事として 1 つだけ読む
    （束ねが分かれた続報・別の配信元の組。a・b は {'meta', 'heads'}）。"""
    key = lambda m: (m['category'], frozenset(m['countries'][:2]), (m['place'] or {}).get('name'))
    if not a['meta']['countries'] or key(a['meta']) != key(b['meta']):
        return False
    names = set(S.COUNTRY_JA.values())   # 国名は国の組で見たので、語の重なりには数えない
    wa = (set().union(*(tok(h) for h in a['heads'])) if a['heads'] else set()) - names
    wb = (set().union(*(tok(h) for h in b['heads'])) if b['heads'] else set()) - names
    return bool(wa and wb) and len(wa & wb) / min(len(wa), len(wb)) >= 0.4


def place_terms():
    """地名の表を定訳として使う（Zaporizhzhia・札波羅熱 → ザポリージャ）。3 字以下の英字の別名は誤って当たるので除く。"""
    out = []
    for p in P.table():
        src = [a.strip() for a in p['aliases'] if a.strip() and (len(a.strip()) > 3 or not a.isascii())]
        if src:
            out.append({'src': src, 'ja': p['name']})
    return out


def estimate_seconds(events):
    """台本の字数からの音声の長さの見積もり（冒頭・結び・間を含む）。"""
    s = INTRO_OUTRO_SEC + A.INTRO_GAP
    for e in events:
        lines = e['script']['lines']
        s += sum(len(ln['text']) for ln in lines) / CHARS_PER_SEC + A.SAMPLE_PAUSE['line'] * (len(lines) - 1) + A.SAMPLE_PAUSE['event']
    return s


def trim_to_length(events):
    """見積もりが TARGET_SECONDS を超えるぶんを、順位の低い出来事から外す（MIN_EVENTS は残す）。(残す, 外す)。"""
    keep, cut = list(events), []
    while len(keep) > MIN_EVENTS and estimate_seconds(keep) > TARGET_SECONDS:
        cut.append(keep.pop())
    return keep, cut


def make_scripts(records, args, terms, tok, log, deadline):
    """選んだ束の台本。(events, dropped, 統計)。events は音声の前の形。"""
    tc = time.time()
    clusters = C.cluster(records, tok[0], args.threshold)
    cands = C.select(clusters)
    stats = {'clusterSec': round(time.time() - tc, 2), 'clusters': len(clusters), 'multiSource': sum(1 for c in clusters if len(C.describe(c)['families']) >= 2),
             'candidates': len(cands)}
    known = [w[0] for w in A.load_readings()]
    places = place_terms()
    events, dropped, picked_heads = [], [], []
    for rank, cand in enumerate(cands[:args.candidates], 1):
        if len(events) >= args.max_events or time.time() > deadline:
            break
        heads = [C.head(r) for r in cand['records']]
        if any(C.same_story(heads, h, tok[0]) for h in picked_heads):
            dropped.append({'reason': '選んだ束と同じ話（見出しの語が重なる）', 'headline': heads[0][:40]})
            continue
        reps = C.representatives(cand['records'])
        meta = event_meta(cand, reps)
        twin = next((e for e in events if same_event({'meta': e['meta'], 'heads': e['heads']}, {'meta': meta, 'heads': heads}, tok[0])), None)
        if twin:
            dropped.append({'reason': '選んだ出来事と同じ（分類・国・地点が同じで見出しの語が重なる：' + twin['script']['headline'] + '）',
                            'headline': heads[0][:40]})
            log(f'#{rank} same: {twin["script"]["headline"]}')
            continue
        conflicts = T.number_conflicts(reps)
        use_terms = T.glossary_for(reps, terms + places)
        script, why, notes, tries = T.write(args.server, reps, use_terms, conflicts, known,
                                            need_attribution=meta['category'] == 'conflict')
        if script is None:
            dropped.append({'reason': '、'.join(why), 'headline': (heads[0])[:40]})
            log(f'#{rank} hold: {"、".join(why)}')
            continue
        picked_heads.append(heads)
        log(f'#{rank} ok: {script["headline"]}（系列 {meta["sources"]["count"]}・国 {meta["sources"]["countries"]}）')
        events.append({'rank': rank, 'script': script, 'meta': meta, 'heads': heads, 'notes': notes, 'tries': tries,
                       'conflicts': conflicts, 'families': cand['stats']['families']})
    return events, dropped, stats


def make_audio(events, now_ms, voice_root, no_audio):
    """音声と、各文・各出来事の開始秒。events を書き換えずに (out_events, intro, outro, mp3, pcm秒) を返す。"""
    d = datetime.datetime.fromtimestamp(now_ms / 1000, JST)
    intro = f'世界情勢観測、{d.month}月{d.day}日。過去24時間の主な出来事、{len(events)}件です。'
    outro = '以上です。報道の見出しと要約を、機械でまとめて読み上げました。'
    seq = [('intro', intro, A.INTRO_GAP)]
    for i, e in enumerate(events):
        lines = [ln['text'] for ln in e['script']['lines']]   # 出典の数は読まない（画面が json の sources から出す）
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
            sub = e['script']['lines'][j].get('sub', '') if j < len(e['script']['lines']) else ''
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
    ap.add_argument('--max-events', type=int, default=8)
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
    events, cut = trim_to_length(events)
    for e in cut:
        dropped.append({'reason': f'長さの上限（{TARGET_SECONDS} 秒）で外した', 'headline': e['script']['headline']})
        log(f'#{e["rank"]} cut: 長さの上限（{e["script"]["headline"]}）')
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
