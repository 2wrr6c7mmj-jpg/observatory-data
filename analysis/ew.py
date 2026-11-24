"""早期警戒（ew-1）：既存の公開データから、層別の指標と総合判定を作る。

標準ライブラリだけで動く。書き込むのは docs/data/ew/ だけ（ew.yml）。

考え方（設計の要約。詳しくは docs/data/ew/meta.json の method）
- 指標を「何日前に動くか」で 4 層（戦略・動員・展開・行動）に分ける。
- 件数の指標は、直近 7 日の合計を、過去 365 日の 7 日合計（重ならない週ごと）の
  中央値と MAD から作るロバストな z で比べ、0〜3 点にする。
- 総合判定：兵站系のカテゴリー（お金・医療・人員・輸送）が 2 つ以上 2 点以上なら「警戒」、
  展開層・行動層のどれかが 2 点以上なら「演習」、それ以外は「平常」。
  兵站系の指標はまだ収集していないため、今の判定の上限は「演習」（meta.json と画面に明記）。
- null は不明（データ不足・未収集）。0 点は「平常値の範囲内と確かめた」という意味。
"""
import csv
import json
import os
import sys
from datetime import date, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'docs', 'data')
OUT = os.path.join(DATA, 'ew')
VERSION = 'ew-1'

WINDOW = 7            # 直近の合計をとる日数
BASE_DAYS = 365       # 平常値に使う期間
GAP_DAYS = 14         # 直近を平常値から外す日数（演習の前ぶれを平常値に混ぜない）
MIN_WEEKS = 20        # 平常値に要る週の数
MIN_DAYS_IN_WINDOW = 5
Z_CUTS = (2.0, 3.0, 5.0)   # 1 点・2 点・3 点の境
SERIES_WEEKS = 26     # 画面のグラフに出す週の数

LAYERS = [
    ('strategic', '戦略層', '数か月前'),
    ('mobilization', '動員層', '数週間前'),
    ('deployment', '展開層', '数日前'),
    ('action', '行動層', '数時間前'),
]
LOGISTICS = {'money', 'medical', 'personnel', 'transport'}
CATEGORY_JA = {
    'money': 'お金', 'medical': '医療', 'personnel': '人員', 'transport': '輸送',
    'military': '軍の活動', 'law_enforcement': '海上法執行', 'messaging': '宣伝',
    'electronic': '電子', 'infrastructure': '通信基盤',
}

# 未収集の指標（予定）。値は常に null。画面で「未収集」と出す
PLANNED = [
    ('strategic', 'oil-stock', '石油備蓄（浮き屋根タンクの影）', 'money', 'Sentinel-2'),
    ('strategic', 'gold', '金準備・金の輸入', 'money', '人民銀行・税関統計'),
    ('strategic', 'amphib-build', '揚陸艦・特殊はしけの建造', 'transport', '造船所の衛星画像'),
    ('strategic', 'rhetoric', '宣伝の論調', 'messaging', '人民日報・解放軍報'),
    ('mobilization', 'mobilization-drills', '国防動員訓練の件数', 'personnel', '福建・浙江の地方政府サイト'),
    ('mobilization', 'blood', '血液・病院の準備', 'medical', '血液センター・調達公告'),
    ('mobilization', 'conscription', '退役延期・徴兵の変更', 'personnel', '徴兵公告'),
    ('mobilization', 'war-risk', '戦争保険の料率', 'money', '海上保険市場'),
    ('deployment', 'ferries', 'フェリーの定期航路からの離脱', 'transport', 'AIS'),
    ('deployment', 'gnss', 'GPS 妨害', 'electronic', 'ADS-B の精度値'),
    ('deployment', 'cables', '海底ケーブル障害', 'infrastructure', '通信事業者の発表'),
]


def d(s):
    return date.fromisoformat(s[:10])


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if not n:
        return None
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def score_from_z(z):
    if z is None:
        return None
    s = 0
    for cut in Z_CUTS:
        if z >= cut:
            s += 1
    return s


def window_sum(daily, end, days=WINDOW):
    """end を含む days 日の合計と、値のある日数。"""
    total, n = 0, 0
    for i in range(days):
        v = daily.get(end - timedelta(days=i))
        if v is not None:
            total += v
            n += 1
    return total, n


def assess_count(daily, as_of, since=None, zero_fill=False):
    """件数の系列（{date: 数}）を as_of 時点で採点する。

    since：これより前の日は平常値に使わない（定義の変更）。
    zero_fill：記録の無い日を 0 とみなす（統幕の公表のように「無い＝0」の系列）。
    """
    if zero_fill:
        first = min(daily) if daily else as_of
        daily = {first + timedelta(days=i): daily.get(first + timedelta(days=i), 0)
                 for i in range((as_of - first).days + 1)}
    value, n = window_sum(daily, as_of)
    if n < MIN_DAYS_IN_WINDOW:
        return {'value': None, 'z': None, 'score': None, 'reason': '直近 7 日のデータが足りない'}
    base = []
    end = as_of - timedelta(days=GAP_DAYS)
    start = as_of - timedelta(days=BASE_DAYS)
    while end - timedelta(days=WINDOW - 1) >= start:
        if since and end - timedelta(days=WINDOW - 1) < since:
            break
        s, k = window_sum(daily, end)
        if k >= MIN_DAYS_IN_WINDOW:
            base.append(s * WINDOW / k)
        end -= timedelta(days=WINDOW)
    if len(base) < MIN_WEEKS:
        return {'value': value * WINDOW / n, 'z': None, 'score': None,
                'reason': f'平常値を作る週が足りない（{len(base)} 週、要 {MIN_WEEKS} 週）'}
    med = median(base)
    mad = median([abs(b - med) for b in base])
    scale = max(1.4826 * mad, 0.15 * med, 1.0)
    v = value * WINDOW / n
    z = (v - med) / scale
    q = sorted(base)
    return {'value': round(v, 1), 'baseline_median': round(med, 1),
            'band': [round(q[int(0.1 * (len(q) - 1))], 1), round(q[int(0.9 * (len(q) - 1))], 1)],
            'scale': round(scale, 2), 'z': round(z, 2), 'score': score_from_z(z), 'weeks': len(base)}


def weekly_series(daily, as_of, weeks=SERIES_WEEKS):
    out = []
    for i in range(weeks - 1, -1, -1):
        end = as_of - timedelta(days=WINDOW * i)
        s, n = window_sum(daily, end)
        out.append([end.isoformat(), round(s * WINDOW / n, 1) if n >= MIN_DAYS_IN_WINDOW else None])
    return out


# ---- 入力 ----

def read_csv(rel):
    path = os.path.join(DATA, rel)
    if not os.path.exists(path):
        return []
    with open(path, encoding='utf-8') as f:
        return list(csv.DictReader(f))


def load_json(rel, default=None):
    path = os.path.join(DATA, rel) if not os.path.isabs(rel) else rel
    if not os.path.exists(path):
        return default
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def int_or_none(s):
    try:
        return int(float(s)) if s not in (None, '') else None
    except ValueError:
        return None


def load_inputs():
    pla = read_csv('pla-tracker/daily.csv')
    jcg = read_csv('jcg-senkaku/daily.csv')
    jso = read_csv('jso/passages.csv')
    series = {'aircraft': {}, 'crossings': {}, 'ships': {}, 'senkaku': {}, 'jso': {}}
    for r in pla:
        day = d(r['date'])
        for key, col in (('aircraft', 'aircraft_total'), ('crossings', 'median_line_cross'), ('ships', 'ships_total')):
            v = int_or_none(r.get(col))
            if v is not None:
                series[key][day] = v
    for r in jcg:
        v = int_or_none(r.get('contiguous_zone'))
        if v is not None:
            series['senkaku'][d(r['date'])] = v
    for r in jso:
        if r.get('nation') == 'CN' and r.get('date'):
            day = d(r['date'])
            series['jso'][day] = series['jso'].get(day, 0) + 1
    with open(os.path.join(ROOT, 'analysis', 'exercises.json'), encoding='utf-8') as f:
        named = json.load(f)['exercises']
    events = []
    idx = os.path.join(DATA, 'exercises', 'events')
    if os.path.isdir(idx):
        for name in sorted(os.listdir(idx)):
            if name[:4].isdigit() and name.endswith('.json'):
                events += load_json(os.path.join(idx, name), {}).get('events', [])
    return series, named, events


# ---- 指標 ----

SHIPS_DEFINITION_CHANGE = date(2026, 3, 23)   # pla-tracker の ships_total が公務船を含むようになった日（meta.json）
NEAR_REGIONS = {'taiwan', 'taiwan-strait', 'east-china-sea', 'philippine-sea', 'china', 'south-china-sea', 'okinawa'}

COUNT_INDICATORS = [
    # (層, id, 名前, カテゴリー, 出典, 系列, 無い日を 0 とみなすか)
    ('action', 'pla-aircraft', '台湾周辺の中国軍機（延べ機数）', 'military', 'PLA Activity Tracker（台湾国防部の日報）', 'aircraft', False),
    ('action', 'pla-crossings', '中間線越えの機数', 'military', 'PLA Activity Tracker（台湾国防部の日報）', 'crossings', False),
    ('action', 'pla-ships', '台湾周辺の艦艇（海軍＋公務船）', 'military', 'PLA Activity Tracker（台湾国防部の日報）', 'ships', False),
    ('action', 'senkaku', '尖閣 接続水域の中国海警船（隻・日の合計）', 'law_enforcement', '海上保安庁', 'senkaku', False),
    ('action', 'jso-transits', '統幕公表の中国艦艇の動向（件数）', 'military', '統合幕僚監部', 'jso', True),
]


def exercise_indicator(as_of, named, events):
    """展開層：演習の告知・報道。名前つきの大規模演習の期間なら 3、確度「高」の中国の演習なら 2、「中」なら 1。"""
    for e in named:
        if d(e['start']) <= as_of <= d(e['end']) + timedelta(days=1):
            return {'score': 3, 'value': None, 'note': f"名前つきの大規模演習：{e['name']}"}
    best, note = 0, '該当なし'
    for e in events:
        if '中国' not in (e.get('countries') or []) or not e.get('start'):
            continue
        if e.get('region') not in NEAR_REGIONS:
            continue
        start, end = d(e['start']), d(e.get('end') or e['start'])
        if not (start <= as_of <= end + timedelta(days=1)):
            continue
        s = {'高': 2, '中': 1}.get(e.get('confidence'), 0)
        if s > best:
            best, note = s, f"{e.get('titleJa') or e.get('name')}（確度 {e.get('confidence')}）"
    return {'score': best, 'value': None, 'note': note}


def build_indicators(series, named, events, as_of, with_series=True):
    out = []
    for layer, iid, name, cat, src, key, zero in COUNT_INDICATORS:
        since = SHIPS_DEFINITION_CHANGE if key == 'ships' and as_of >= SHIPS_DEFINITION_CHANGE else None
        daily = series[key]
        if key == 'ships' and as_of < SHIPS_DEFINITION_CHANGE:
            daily = {k: v for k, v in daily.items() if k < SHIPS_DEFINITION_CHANGE}
        last = max((k for k in daily if k <= as_of), default=None)
        if last is None:
            a = {'value': None, 'z': None, 'score': None, 'reason': 'データが無い'}
        else:
            # 日報の遅れの分だけ、評価日を最新の日に寄せる（3 日まで）
            at = min(as_of, last) if (as_of - last).days <= 3 else as_of
            a = assess_count(daily, at, since=since, zero_fill=zero)
            a['as_of'] = at.isoformat()
        item = {'layer': layer, 'id': iid, 'name': name, 'category': cat, 'source': src,
                'kind': 'count-7d', 'status': 'collected', **a}
        if since:
            item['note'] = f'艦艇は {since.isoformat()} から公務船を含む定義のため、平常値はそれ以降だけで作る'
        if with_series and last is not None:
            item['series'] = weekly_series(daily, min(as_of, last))
        out.append(item)
    ex = exercise_indicator(as_of, named, events)
    out.append({'layer': 'deployment', 'id': 'exercises', 'name': '中国の演習の告知・報道', 'category': 'military',
                'source': 'exercises（統幕・中国国防部・報道の照合）と analysis/exercises.json', 'kind': 'event',
                'status': 'collected', **ex})
    for layer, iid, name, cat, src in PLANNED:
        out.append({'layer': layer, 'id': iid, 'name': name, 'category': cat, 'source': src,
                    'kind': 'planned', 'status': 'planned', 'value': None, 'score': None, 'reason': '未収集'})
    return out


def judge(indicators):
    layers = []
    for lid, name, lead in LAYERS:
        scores = [i['score'] for i in indicators if i['layer'] == lid and i['score'] is not None]
        layers.append({'id': lid, 'name': name, 'lead': lead,
                       'mean': round(sum(scores) / len(scores), 2) if scores else None,
                       'max': max(scores) if scores else None,
                       'collected': len(scores),
                       'total': sum(1 for i in indicators if i['layer'] == lid)})
    hot_logistics = sorted({i['category'] for i in indicators
                            if i['category'] in LOGISTICS and (i['score'] or 0) >= 2})
    hot_ops = [i['id'] for i in indicators if i['layer'] in ('deployment', 'action') and (i['score'] or 0) >= 2]
    logistics_collected = any(i['category'] in LOGISTICS and i['status'] == 'collected' for i in indicators)
    if len(hot_logistics) >= 2:
        verdict = 'alert'
    elif hot_ops:
        verdict = 'exercise'
    else:
        verdict = 'normal'
    return {'verdict': verdict, 'layers': layers, 'hot_logistics': hot_logistics, 'hot_ops': hot_ops,
            'ceiling': None if logistics_collected else 'exercise',
            'ceiling_note': None if logistics_collected else
            '兵站系（お金・医療・人員・輸送）の指標をまだ収集していないため、「警戒」は出せない（判定の上限は「演習」）'}


def backtest(series, named, events):
    rows = []
    for e in named:
        start, end = d(e['start']), d(e['end'])
        row = {'id': e['id'], 'name': e['name'], 'start': e['start'], 'end': e['end'], 'points': []}
        # 日報は前日 06:00〜当日 06:00 の集計なので、演習の最終日の翌日付までを「期間中」とみなす
        for label, at in (('7 日前', start - timedelta(days=7)), ('前日', start - timedelta(days=1)),
                          ('期間中', end + timedelta(days=1))):
            inds = build_indicators(series, [], events, at, with_series=False)
            j = judge(inds)
            row['points'].append({'label': label, 'as_of': at.isoformat(), 'verdict': j['verdict'],
                                  'layers': {l['id']: l['max'] for l in j['layers']},
                                  'scores': {i['id']: i['score'] for i in inds if i['status'] == 'collected'}})
        rows.append(row)
    return rows


def false_alarms(series, named, events, as_of, days=730, margin=7):
    """名前つきの演習の前後 margin 日を除いた日のうち、「演習」以上と判定した日の割合（誤報の目安）。"""
    quiet, hits, months = 0, 0, {}
    for i in range(days):
        at = as_of - timedelta(days=i)
        if any(d(e['start']) - timedelta(days=margin) <= at <= d(e['end']) + timedelta(days=margin) for e in named):
            continue
        v = judge(build_indicators(series, [], events, at, with_series=False))['verdict']
        quiet += 1
        if v != 'normal':
            hits += 1
            months[at.isoformat()[:7]] = months.get(at.isoformat()[:7], 0) + 1
    return {'days': quiet, 'flagged': hits, 'rate': round(hits / quiet, 3) if quiet else None,
            'by_month': dict(sorted(months.items())),
            'note': f'過去 {days} 日のうち、名前つきの大規模演習の前後 {margin} 日を除いた日で「演習」以上と出た割合。'
                    '名前の無い演習・示威行動も「演習」に入るので、すべてが誤報とは限らない'}


def write_if_changed(name, payload):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    body = json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=False) + '\n'
    old = None
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            old = f.read()
    def strip(s):
        try:
            j = json.loads(s)
            j.pop('generated_utc', None)
            return j
        except (TypeError, ValueError):
            return None
    if old is not None and strip(old) == strip(body):
        return False
    with open(path, 'w', encoding='utf-8') as f:
        f.write(body)
    return True


def main(argv=None):
    from datetime import datetime, timezone
    argv = argv if argv is not None else sys.argv[1:]
    series, named, events = load_inputs()
    as_of = d(argv[0]) if argv else datetime.now(timezone.utc).date()
    inds = build_indicators(series, named, events, as_of)
    now = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    current = {'version': VERSION, 'generated_utc': now, 'as_of': as_of.isoformat(),
               **judge(inds), 'indicators': inds, 'category_ja': CATEGORY_JA}
    changed = [n for n, p in (('indicators.json', current),
                              ('backtest.json', {'version': VERSION, 'generated_utc': now,
                                                 'note': '各時点で、その日までのデータだけを使って採点した。名前つきの演習の一覧（analysis/exercises.json）は答えなので採点に使わない',
                                                 'rows': backtest(series, named, events),
                                                 'false_alarm': false_alarms(series, named, events, as_of)}))
               if write_if_changed(n, p)]
    print('changed:', ', '.join(changed) or 'none')


if __name__ == '__main__':
    main()
