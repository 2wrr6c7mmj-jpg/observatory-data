"""台本を人（毎朝の定期タスクの Claude）が書く経路。モデルサーバーは使わない。手順は briefing/EDITOR.md。

  python -m briefing.editor candidates --feed URL_or_FILE --out cands.json
  python -m briefing.editor check --feed URL_or_FILE --script script.json

candidates：過去 24 時間の記事を束ね、選び、同じ出来事を除いた候補を json に出す（代表の記事・定訳・数字の食い違い・分類と国）。
check：書いた台本を、公開の処理（python -m briefing --script）と同じ確かめに通し、通らない理由を出す。--vv-root を付けると、
読み上げのカナ（調声用）も出す。地図の行き先（国・地点）も出す。
台本の形（EDITOR.md）：{"version": 1, "readings": [[表記, 読み, 下がり目], …], "events": [{"key": [候補の key], "headline": "…",
"lines": ["…", "…"], "map": {"countries": ["LY"], "name": "リビア", "place": {"name": "ベンガジ", "lon": 20.07, "lat": 32.12}}}]}
"""
import argparse
import json
import sys
import time
from types import SimpleNamespace

from . import build as B
from . import cluster as C
from . import script as T

KEY_LEN = 12   # 候補の key：記事の id の頭（24 時間ぶんの記事の中で 1 本に決まる長さ）


def window(a):
    feed = B.load_feed(a.feed)
    now = a.now_ms or int(time.time() * 1000)
    return C.in_window([r for r in (feed.get('records') or []) if isinstance(r, dict)], now), now


def cmd_candidates(a):
    recs, now = window(a)
    tok = C.default_tokenizer()
    terms = B.load_terms() + B.place_terms()
    picked, dropped, stats = B.pick_candidates(recs, tok, a.threshold, a.limit)
    cands = []
    for p in picked:
        reps, meta = p['reps'], p['meta']
        cands.append({
            'rank': p['rank'],
            'key': [str(r.get('id'))[:KEY_LEN] for r in reps[:3]],
            'label': meta['label'], 'place': (meta['place'] or {}).get('name'), 'countries': meta['countries'],
            'sources': {'count': meta['sources']['count'], 'countries': meta['sources']['countries']},
            'needAttribution': meta['category'] == 'conflict',
            'focus': B.focus_of(meta),
            'articles': [dict(T.article_payload(r), hoursAgo=round((now - (r.get('publishedAt') or now)) / 3600e3, 1)) for r in reps],
            'terms': [dict({'原文': t['src'][:3], '訳': t['ja']}, **({'書かない': t['bad']} if t.get('bad') else {}))
                      for t in T.glossary_for(reps, terms)],
            'numberConflicts': T.number_conflicts(reps),
        })
    out = {
        'version': 1, 'createdAt': now,
        'counts': {'records': len(recs), 'sources': len({r.get('source') for r in recs}), 'clusters': stats['clusters'],
                   'multiSource': stats['multiSource']},
        'limits': {'events': a.max_events, 'seconds': B.TARGET_SECONDS, 'headline': T.HEADLINE_MAX, 'lines': [T.LINES_MIN, T.LINES_MAX],
                   'lineChars': [T.LINE_MIN, T.LINE_MAX], 'bodyChars': T.BODY_MAX},
        'candidates': cands, 'dropped': dropped,
    }
    with open(a.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
        f.write('\n')
    print(f'記事 {len(recs)}、2 源以上の束 {stats["multiSource"]}、候補 {len(cands)}（同じ出来事として除いた束 {len(dropped)}）、形態素解析 {tok[1]}')
    for c in cands:
        print(f'#{c["rank"]}{" ★" if c["focus"] else ""} [{c["label"]}] 出典 {c["sources"]["count"]}・{c["sources"]["countries"]} か国 | {c["articles"][0]["title"][:60]}')
    return 0


def cmd_check(a):
    recs, _ = window(a)
    tok = C.default_tokenizer()
    with open(a.script, encoding='utf-8') as f:
        doc = json.load(f)
    args = SimpleNamespace(threshold=a.threshold, max_events=a.max_events)
    events, dropped, _ = B.scripts_from_file(recs, doc, args, B.load_terms(), tok, print)
    events, cut = B.trim_to_length(events)
    for e in cut:
        print(f'#{e["rank"]} cut: 長さの上限（{B.TARGET_SECONDS} 秒）で外れる（{e["script"]["headline"]}）')
    for e in events:
        m = e['meta']
        print(f'  地図 #{e["rank"]} [{m["label"]}] 国 {m["countries"]} 地点 {m["place"]}')
    if a.vv_root:   # 調声：読み上げのカナ（/ はアクセント句の区切り、' はアクセント核、_ は無声化）
        from . import audio as A
        extra = B.extra_readings(doc)
        voice = A.Voice(a.vv_root, readings=A.load_readings() + list(extra))
        for e in events:
            print(f'  読み #{e["rank"]}')
            for ln in e['script']['lines']:
                auto = voice.kana(ln['text'])
                if ln.get('kana'):
                    ok = voice.tuned(ln['text'], ln['kana'])
                    print('    ' + (ok or auto) + ('   ← 調声' if ok else '   ← 調声のカナは使えない（記法の誤りか、読みが文と違う）。自動の読み'))
                else:
                    print('    ' + auto)
        if voice.skipped:
            print('  辞書に入らなかった読み: ' + '、'.join(voice.skipped))
    print(f'読める出来事 {len(events)} 件、確かめに落ちた出来事 {len(dropped)} 件、長さの見積もり {B.estimate_seconds(events):.0f} 秒')
    return 0 if len(events) >= B.MIN_EVENTS and not dropped else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('candidates', 'check'):
        p = sub.add_parser(name)
        p.add_argument('--feed', required=True)
        p.add_argument('--now-ms', type=int, default=0)
        p.add_argument('--threshold', type=float, default=C.THRESHOLD)
        p.add_argument('--max-events', type=int, default=8)
        if name == 'candidates':
            p.add_argument('--out', required=True)
            p.add_argument('--limit', type=int, default=12, help='出す候補の数')
        else:
            p.add_argument('--script', required=True)
            p.add_argument('--vv-root', default='', help='tools/voicevox_setup.sh が作ったディレクトリ。指定すると読み上げのカナも出す（調声）')
    a = ap.parse_args(argv)
    return cmd_candidates(a) if a.cmd == 'candidates' else cmd_check(a)


if __name__ == '__main__':
    sys.exit(main())
