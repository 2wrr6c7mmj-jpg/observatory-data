"""報道のまとめを作る：記事 → 束 → 台本 → 音声 → json と mp3。

  python -m briefing --feed URL_or_FILE --out DIR [--server http://127.0.0.1:8080] [--vv-root DIR] [--no-audio]
  python -m briefing --feed URL_or_FILE --out DIR --script script.json [--vv-root DIR]     # 人が書いた台本（briefing/EDITOR.md）

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
INTRO_OUTRO_SEC = 10.0    # 冒頭と結びの定型文（冒頭 約 3 秒〈2026-11-28 に短縮〉、結び 約 6 秒）
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


def same_script(a, b, tok):
    """台本どうしの比べ：分類と主な国の組が同じで、台本の語（国名を除く）が重なれば同じ出来事。
    見出しの語では分かれたままの束を拾う（a・b は {'script', 'meta'}）。"""
    if a['meta']['category'] != b['meta']['category'] or frozenset(a['meta']['countries'][:2]) != frozenset(b['meta']['countries'][:2]):
        return False
    names = set(S.COUNTRY_JA.values())
    words = lambda e: set(tok(e['script']['headline'] + ' ' + ' '.join(ln['text'] for ln in e['script']['lines']))) - names
    wa, wb = words(a), words(b)
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


KATAKANA = re.compile(r'[ァ-ヶー]+')
CODE = re.compile(r'[A-Za-z]{2}')


def parse_tune(text):
    """--tune の「speed=1.08,intonation=0.9」→ {'speed': 1.08, 'intonation': 0.9}。読めない項目は捨てる。"""
    out = {}
    for part in (text or '').split(','):
        k, _, v = part.partition('=')
        try:
            if k.strip() in A.TUNE:
                out[k.strip()] = float(v)
        except ValueError:
            pass
    return out


def extra_readings(doc):
    """台本に付けたその日の読み（調声）：{"readings": [[表記, カタカナの読み, アクセント核の位置], …]}。
    形の合うものだけ、最大 40 語。briefing/readings.json の後ろに足して辞書に入れる。"""
    out = []
    for w in (doc.get('readings') or []) if isinstance(doc, dict) else []:
        if (isinstance(w, (list, tuple)) and len(w) == 3 and all(isinstance(x, str) for x in w[:2]) and isinstance(w[2], int)
                and not isinstance(w[2], bool) and 1 <= len(w[0].strip()) <= 20 and not T.BAD_CHARS.search(w[0])
                and KATAKANA.fullmatch(w[1]) and len(w[1]) <= 30 and 0 <= w[2] <= len(w[1])):
            out.append((w[0].strip(), w[1], w[2]))
    return out[:40]


# 再生画面の動き（map.anim の kind）：strike 攻撃・爆発、aircraft 航空機の移動、ship 艦船の移動、missile 発射と飛翔、
# launch ロケット・衛星の打ち上げ、alert 警報・緊張、none 動かさない。from・to は [経度, 緯度]（航空機・艦船・ミサイルの経路）。
ANIM_KINDS = ('strike', 'aircraft', 'ship', 'missile', 'launch', 'alert', 'none')


def map_hint(ev):
    """台本に付けた地図の行き先（再生画面のアニメーション）。形の合う項目だけを返す。
    {"map": {"countries": ["LY", "UA", "RU"], "name": "リビア", "place": {"name": "ベンガジ", "lon": 20.07, "lat": 32.12}}}
    countries：輪郭を出す国（ISO の 2 文字、最大 3。先頭が地図の寄る先）／name：札に出す国・地域の名前／place：点を打つ地点。"""
    m = ev.get('map') if isinstance(ev, dict) else None
    if not isinstance(m, dict):
        return {}
    out = {}
    cs = list(dict.fromkeys(c.upper() for c in (m.get('countries') or []) if isinstance(c, str) and CODE.fullmatch(c)))[:3] \
        if isinstance(m.get('countries'), list) else []
    if cs:
        out['countries'] = cs
    ok_name = lambda x, n: isinstance(x, str) and 1 <= len(x.strip()) <= n and T.JAPANESE.search(x) and not T.BAD_CHARS.search(x)
    if ok_name(m.get('name'), 12):
        out['name'] = m['name'].strip()
    p = m.get('place')
    num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)
    if isinstance(p, dict) and ok_name(p.get('name'), 16) and num(p.get('lon')) and num(p.get('lat')) and -180 <= p['lon'] <= 180 and -85 <= p['lat'] <= 85:
        out['place'] = {'name': p['name'].strip(), 'lon': round(float(p['lon']), 3), 'lat': round(float(p['lat']), 3)}
    a = m.get('anim')
    pt = lambda q: isinstance(q, list) and len(q) == 2 and all(num(v) for v in q) and -180 <= q[0] <= 180 and -85 <= q[1] <= 85
    if isinstance(a, dict) and a.get('kind') in ANIM_KINDS:
        anim = {'kind': a['kind']}
        for k in ('from', 'to'):
            if pt(a.get(k)):
                anim[k] = [round(float(a[k][0]), 3), round(float(a[k][1]), 3)]
        out['anim'] = anim
    return out


def script_line(x):
    """台本の 1 文：文字列、または {"text": 文, "kana": 調声のカナ}。形が違えば None。"""
    if isinstance(x, str):
        return {'text': x.strip(), 'sub': ''}
    if isinstance(x, dict) and isinstance(x.get('text'), str):
        row = {'text': x['text'].strip(), 'sub': ''}
        k = x.get('kana')
        if isinstance(k, str) and 0 < len(k) <= 400 and A.KANA_SCRIPT.fullmatch(k):
            row['kana'] = k
        return row
    return None


def apply_map_hint(meta, hint):
    """地図の行き先の指定を、記事から機械で決めた値に上書きする（札の国名も合わせる）。"""
    if hint.get('countries'):
        meta['countries'] = hint['countries']
        meta['country'] = hint['countries'][0]
    if hint.get('countries') or hint.get('name'):
        name = hint.get('name') or S.COUNTRY_JA.get(meta['country'], meta['country'])
        meta['label'] = CAT_JA.get(meta['category'], '報道') + (' · ' + name if name else '')
    if 'place' in hint:
        meta['place'] = hint['place']
    if 'anim' in hint:
        meta['anim'] = hint['anim']
    return meta


INTRO = '本日の主要トピックを紹介します。'
# 発表の順で少し先にする国（所有者の指示 2026-11-28「中国、北朝鮮、ロシアのものをやや優先」。ほかの国の出来事も出す）
FOCUS = ('CN', 'KP', 'RU')
FOCUS_EXTRA = 4           # 上位 limit 件に入らなかった FOCUS の国の候補を、後ろにこの数まで足す


def focus_of(meta):
    """候補が FOCUS の国にかかわるか（主体の国・関係する国のどれか）。"""
    codes = [meta.get('country')] + list(meta.get('countries') or [])
    return any(c in FOCUS for c in codes if c)


def pick_candidates(records, tok, threshold, limit, focus_extra=FOCUS_EXTRA):
    """台本を書く前の候補：束ね → 2 源以上 → 順位 → 同じ出来事（分類・国・地点・見出しの語）を除く、上から limit 件。
    ([{'rank', 'cand', 'reps', 'heads', 'meta'}], 落とした一覧, 統計)。モデルは使わない。"""
    tc = time.time()
    clusters = C.cluster(records, tok[0], threshold)
    cands = C.select(clusters)
    out, dropped = [], []
    extra = 0
    for rank, cand in enumerate(cands, 1):
        if len(out) >= limit + focus_extra:
            break
        heads = [C.head(r) for r in cand['records']]
        reps = C.representatives(cand['records'])
        meta = event_meta(cand, reps)
        if len(out) >= limit:
            # 上位 limit 件の後ろは、中国・北朝鮮・ロシアの候補だけを focus_extra 件まで足す
            if not focus_of(meta) or extra >= focus_extra:
                continue
            extra += 1
        twin = next((p for p in out if same_event({'meta': p['meta'], 'heads': p['heads']}, {'meta': meta, 'heads': heads}, tok[0])), None)
        if twin:
            dropped.append({'reason': f'候補 {twin["rank"]} と同じ出来事（分類・国・地点が同じで見出しの語が重なる）', 'headline': heads[0][:40]})
            continue
        out.append({'rank': rank, 'cand': cand, 'reps': reps, 'heads': heads, 'meta': meta})
    stats = {'clusterSec': round(time.time() - tc, 2), 'clusters': len(clusters), 'candidates': len(cands),
             'multiSource': sum(1 for c in clusters if len(C.describe(c)['families']) >= 2)}
    return out, dropped, stats


def scripts_from_file(records, doc, args, terms, tok, log):
    """人（毎朝の Claude）が書いた台本を、モデルの台本と同じ確かめに通す。(events, dropped, 統計)。
    台本の出来事は記事の id の頭（key、8 字以上）で引き、その記事を含む束から、出典の数・分類・国を取り直す
    （台本に書かれた数は使わない）。確かめに使う記事は、束の代表の記事と、key の記事。"""
    tc = time.time()
    clusters = C.cluster(records, tok[0], args.threshold)
    where = {id(r): ci for ci, recs in enumerate(clusters) for r in recs}
    stats = {'clusterSec': round(time.time() - tc, 2), 'clusters': len(clusters), 'candidates': 0,
             'multiSource': sum(1 for c in clusters if len(C.describe(c)['families']) >= 2)}
    known = [w[0] for w in A.load_readings()]
    places = place_terms()
    events, dropped = [], []
    listed = doc.get('events') if isinstance(doc, dict) else None
    for rank, ev in enumerate((listed if isinstance(listed, list) else [])[:args.max_events + 6], 1):
        if not isinstance(ev, dict):
            continue
        head = T.clean(ev.get('headline'), 60) or '（見出しなし）'
        keys = [str(k).lower()[:64] for k in (ev.get('key') or []) if isinstance(k, str) and len(k) >= 8][:12]
        named = [r for r in records if any(str(r.get('id', '')).lower().startswith(k) for k in keys)]
        hit = collections.Counter(where[id(r)] for r in named if id(r) in where)
        if not hit:
            dropped.append({'reason': '台本の記事（key）が、いまの 24 時間ぶんの記事に無い', 'headline': head})
            log(f'#{rank} hold: key の記事が無い（{head}）')
            continue
        recs = clusters[hit.most_common(1)[0][0]]
        stat = C.describe(recs)
        if len(stat['families']) < C.MIN_FAMILIES:
            dropped.append({'reason': f'独立した出典が {C.MIN_FAMILIES} つに満たない', 'headline': head})
            log(f'#{rank} hold: 出典が足りない（{head}）')
            continue
        reps = C.representatives(recs)
        arts = reps + [r for r in named if all(r is not x for x in reps)]
        meta = event_meta({'records': recs, 'stats': stat}, reps)
        need_attribution = meta['category'] == 'conflict'   # 機械の分類で決める（下の指定では緩めない）
        if ev.get('category') in CAT_JA:
            meta['category'] = ev['category']
            meta['label'] = CAT_JA[meta['category']] + (' · ' + S.COUNTRY_JA.get(meta['country'], meta['country']) if meta['country'] else '')
        meta = apply_map_hint(meta, map_hint(ev))
        need_attribution = need_attribution or meta['category'] == 'conflict'
        conflicts = T.number_conflicts(arts)
        use_terms = T.glossary_for(arts, terms + places)
        lines = ev.get('lines')
        script = None
        rows = [script_line(t) for t in lines] if isinstance(lines, list) else [None]
        if isinstance(ev.get('headline'), str) and rows and all(rows):
            script = T.apply_fixed_terms({'headline': ev['headline'].strip(), 'lines': rows}, use_terms)
            for before, after in zip(rows, script['lines']):
                if after['text'] != before['text']:
                    after.pop('kana', None)   # 定訳の置き換えで文が変わったら、調声のカナは使わない
        why, notes = T.verify(script, arts, conflicts, use_terms, known + [w[0] for w in extra_readings(doc)], need_attribution=need_attribution)
        dup = None if why else next((e for e in events if same_script(e, {'script': script, 'meta': meta}, tok[0])), None)
        if dup:
            why = ['選んだ出来事と同じ（台本の語が重なる：' + dup['script']['headline'] + '）']
        if why:
            dropped.append({'reason': '、'.join(why), 'headline': head})
            log(f'#{rank} hold: {"、".join(why)}（{head}）')
            continue
        log(f'#{rank} ok: {script["headline"]}（系列 {meta["sources"]["count"]}・国 {meta["sources"]["countries"]}・{meta["label"]}）')
        events.append({'rank': rank, 'script': script, 'meta': meta, 'heads': [C.head(r) for r in recs], 'notes': notes, 'tries': [],
                       'conflicts': conflicts, 'families': stat['families']})
        if len(events) >= args.max_events:
            break
    return events, dropped, stats


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
        dup = next((e for e in events if same_script(e, {'script': script, 'meta': meta}, tok[0])), None)
        if dup:
            dropped.append({'reason': '選んだ出来事と同じ（台本の語が重なる：' + dup['script']['headline'] + '）', 'headline': script['headline']})
            log(f'#{rank} same(script): {dup["script"]["headline"]}')
            continue
        picked_heads.append(heads)
        log(f'#{rank} ok: {script["headline"]}（系列 {meta["sources"]["count"]}・国 {meta["sources"]["countries"]}）')
        events.append({'rank': rank, 'script': script, 'meta': meta, 'heads': heads, 'notes': notes, 'tries': tries,
                       'conflicts': conflicts, 'families': cand['stats']['families']})
    return events, dropped, stats


# サイト名の読み：合成器は 1 句につなげて読む（ひがしあじあじょーせーか＼んそく）。「東アジア」「情勢観測」の 2 句にする
NAME_KANA = ("ヒガシアジアジョオセエカ'ンソク", "ヒガシア'ジア/ジョオセエカ'ンソク")


def spread(key, text, gap):
    """冒頭・結びの文言を 1 文ずつに分ける（文末の落ち着かせは 1 文ごとにかかる）。2 文目からのキーは key+1、key+2…"""
    parts = [s for s in re.findall(r'[^。]+。?', text) if s.strip()] or [text]
    return [(key if k == 0 else f'{key}+{k}', s, gap if k == len(parts) - 1 else A.SAMPLE_PAUSE['line']) for k, s in enumerate(parts)]


def make_audio(events, now_ms, voice_root, no_audio, extra=(), tune=None):
    """音声と、各文・各出来事の開始秒。events を書き換えずに (out_events, intro, outro, mp3, pcm秒) を返す。"""
    # 冒頭は短く（所有者の指示 2026-11-28「冒頭が長いので省略。『本日の主要トピックを紹介します』程度でいい」）
    intro = INTRO
    outro = '以上です。報道の見出しと要約を、機械でまとめて読み上げました。'
    seq = spread('intro', intro, A.INTRO_GAP)
    for i, e in enumerate(events):
        lines = [ln['text'] for ln in e['script']['lines']]   # 出典の数は読まない（画面が json の sources から出す）
        e['spoken'] = lines
        for j, t in enumerate(lines):
            last = j == len(lines) - 1
            seq.append((f'{i}.{j}', t, A.SAMPLE_PAUSE['event'] if last else A.SAMPLE_PAUSE['line']))
    seq += spread('outro', outro, 0.5)
    if no_audio:
        return None, intro, outro, None, 0.0, []
    voice = A.Voice(voice_root, readings=A.load_readings() + list(extra)) if extra else A.Voice(voice_root)
    voice.tune.update(tune or {})
    given = {ln['text']: ln['kana'] for e in events for ln in e['script']['lines'] if ln.get('kana')}   # 調声のカナ（文 → カナ）
    given.setdefault(seq[0][1], voice.kana(seq[0][1]).replace(*NAME_KANA))   # 名乗りは 2 句で読む（読みは同じなので tuned を通る）
    voice.calibrate([(t, given.get(t)) for _, t, _ in seq])
    kana = {'intro': voice.tuned(seq[0][1], given.get(seq[0][1])) or voice.kana(seq[0][1])}
    starts, pcm = A.assemble(seq, lambda t: voice.wav(t, given.get(t)))
    for key, text, _ in seq:
        if key != 'intro':
            kana[key] = voice.tuned(text, given.get(text)) or voice.kana(text)
    if voice.untuned:
        kana['untuned'] = voice.untuned   # 調声のカナが使えず、文から読んだ文（検品用）
    return starts, intro, outro, A.to_mp3(pcm), pcm.seconds, kana


def assemble_json(events, dropped, stats, starts, intro, outro, mp3, seconds, now_ms, counts, build, voice, hold, hold_reason, model=MODEL_NAME):
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
        if e['meta'].get('anim'):
            ev['anim'] = e['meta']['anim']
        ev.update({'headline': e['script']['headline'], 'lines': lines, 'sources': e['meta']['sources'],
                   'debug': {'rank': e['rank'], 'families': e['families'], 'notes': e['notes'], 'tries': e['tries'],
                             'numberConflicts': e['conflicts']}})
        out_events.append(ev)
    return {
        'version': 1, 'createdAt': now_ms, 'windowHours': 24, 'hold': hold, 'holdReason': hold_reason,
        'generated': 'machine', 'model': model,
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
    ap.add_argument('--tune', default='', help='声の設定の上書き（試聴用）。例 speed=1.08,intonation=0.9')
    ap.add_argument('--script', default='', help='人が書いた台本（briefing/EDITOR.md の形）。指定すると、モデルを使わずにこの台本を確かめて読む')
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
    writer, written = MODEL_NAME, None
    if a.script:
        with open(a.script, encoding='utf-8') as f:
            written = json.load(f)
        writer = 'Claude'
        events, dropped, stats = scripts_from_file(recs, written, a, terms, tok, log)
    else:
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
        starts, intro, outro, mp3, seconds, kana = make_audio(events, now_ms, a.vv_root, a.no_audio,
                                                              extra_readings(written) if a.script else (), parse_tune(a.tune))
    t3 = time.time()
    counts = {'records': len(recs), 'sources': len({r.get('source') for r in recs}), 'clusters': stats['clusters'],
              'multiSource': stats['multiSource']}
    timing = {'clusterSec': stats['clusterSec'], 'scriptSec': round(t2 - t1 - stats['clusterSec'], 1), 'audioSec': round(t3 - t2, 1), 'totalSec': round(t3 - t0, 1)}
    build = {'schema': 'briefing-1', 'tokenizer': tok[1], 'threshold': a.threshold, 'timing': timing,
             'candidates': stats['candidates'], 'kana': kana}
    doc = assemble_json(events, dropped, stats, starts, intro, outro, mp3, seconds, now_ms, counts, build, voice, hold, reason, model=writer)
    with open(os.path.join(a.out, 'latest.json'), 'w', encoding='utf-8') as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
        f.write('\n')
    if mp3:
        with open(os.path.join(a.out, 'latest.mp3'), 'wb') as f:
            f.write(mp3)
    with open(os.path.join(a.out, 'credits.txt'), 'w', encoding='utf-8') as f:
        who = 'Qwen3-8B（Apache-2.0）' if writer == MODEL_NAME else writer
        f.write(voice['credit'] + f'\n台本: {who} で機械作成。元記事の出典は latest.json の sources。\n')
    with open(os.path.join(a.out, 'timing.json'), 'w', encoding='utf-8') as f:
        json.dump(timing, f)
    log(f'hold={hold} 出来事 {len(events)} 落ち {len(dropped)} 音声 {seconds:.1f} 秒 {timing}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
