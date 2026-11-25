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
import re
import sys
from datetime import date, datetime, timedelta, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'docs', 'data')
OUT = os.path.join(DATA, 'ew')
VERSION = 'ew-2'

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
LOGISTICS = {'money', 'personnel', 'transport'}   # 医療（血液・病院）は自動で集められず、欄ごと外した（司令部判断 2026-11-25）
CATEGORY_JA = {
    'money': 'お金', 'personnel': '人員', 'transport': '輸送',
    'military': '軍の活動', 'law_enforcement': '海上法執行', 'messaging': '宣伝',
    'electronic': '電子', 'infrastructure': '通信基盤',
}

# 集められない指標。値は常に null。画面で理由を出す
BLOCKED = [
    ('deployment', 'gnss', 'GPS 妨害', 'electronic', 'ADS-B の精度値（airplanes.live）',
     'airplanes.live の API は事前の連絡（メール）が要り、規約の頁はボット対策で読めない（probe/ew2-robots、2026-11-25）。許可を得るまで集めない'),
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


def assess_count(daily, as_of, since=None, zero_fill=False, down=False):
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
        return {'value': round(value * WINDOW / n, 1), 'z': None, 'score': None,
                'reason': f'平常値を作る週が足りない（{len(base)} 週、要 {MIN_WEEKS} 週）'}
    med = median(base)
    mad = median([abs(b - med) for b in base])
    scale = max(1.4826 * mad, 0.15 * med, 1.0)
    v = value * WINDOW / n
    z = (med - v) / scale if down else (v - med) / scale
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
    """入力をまとめる。台湾周辺の数は、国防部の日報（taiwan-mnd）を先に使い、値が決まらない日だけ PLA Tracker で補う。"""
    mnd = [r for r in read_csv('taiwan-mnd/daily.csv') if r.get('format') == 'around-taiwan']
    pla = read_csv('pla-tracker/daily.csv')
    src = {'aircraft': {}, 'crossings': {}, 'navy': {}, 'ships': {}, 'senkaku': {}}
    for r in mnd:
        day = d(r['report_date'])
        v = int_or_none(r.get('aircraft'))
        if v is not None:
            src['aircraft'][day] = v
        pv, os_ = int_or_none(r.get('plan_vessels')), int_or_none(r.get('official_ships'))
        if pv is not None:
            src['navy'][day] = pv
        if pv is not None and os_ is not None:
            src['ships'][day] = pv + os_
    for r in pla:
        day = d(r['date'])
        v = int_or_none(r.get('aircraft_total'))
        if v is not None and day not in src['aircraft']:
            src['aircraft'][day] = v
        # 中間線越えは国防部の median_line_or_adiz と別の量なので、PLA Tracker の値だけでつなぐ
        v = int_or_none(r.get('median_line_cross'))
        if v is not None:
            src['crossings'][day] = v
        # ships_total は SHIPS_DEFINITION_CHANGE 以降だけ「海軍＋公務船」と同じ量
        v = int_or_none(r.get('ships_total'))
        if v is not None and day >= SHIPS_DEFINITION_CHANGE and day not in src['ships']:
            src['ships'][day] = v
        # それより前の ships_total は海軍艦艇だけ（国防部の plan_vessels と同じ量）
        if v is not None and day < SHIPS_DEFINITION_CHANGE and day not in src['navy']:
            src['navy'][day] = v
    for r in read_csv('jcg-senkaku/daily.csv'):
        v = int_or_none(r.get('contiguous_zone'))
        if v is not None:
            src['senkaku'][d(r['date'])] = v
    src['jso_rows'] = [(d(r['date']), d(r['pub_date'])) for r in read_csv('jso/passages.csv')
                       if r.get('nation') == 'CN' and r.get('date') and r.get('pub_date')]
    last = (load_json('jso/status.json', {}) or {}).get('last_release')
    src['jso_last_release'] = d(last) if last else None
    src.update(load_ew2())
    src['ew2'] = {k: src.pop(k) for k in ('gold', 'oil', 'jwc', 'rmrb', 'feed', 'ioda', 'ferries')}
    with open(os.path.join(ROOT, 'analysis', 'exercises.json'), encoding='utf-8') as f:
        named = json.load(f)['exercises']
    events = []
    idx = os.path.join(DATA, 'exercises', 'events')
    if os.path.isdir(idx):
        for name in sorted(os.listdir(idx)):
            if name[:4].isdigit() and name.endswith('.json'):
                events += load_json(os.path.join(idx, name), {}).get('events', [])
    return src, named, events


def jso_daily(src, at):
    """at の時点で公表済みだった統幕の中国艦艇の目撃を、航過日ごとに数える。

    公表の無い日は 0（統幕は目撃のたびに公表する）。0 で埋めるのは、収集で確かめた最後の公表日（status.json）と
    at の早い方まで。"""
    rows = [day for day, pub in src['jso_rows'] if pub <= at]
    if not rows:
        return {}
    end = min(at, src['jso_last_release']) if src.get('jso_last_release') else at
    first = min(rows)
    daily = {first + timedelta(days=i): 0 for i in range((end - first).days + 1)}
    for day in rows:
        if day <= end:
            daily[day] = daily.get(day, 0) + 1
    return daily


# ---- 指標 ----

SHIPS_DEFINITION_CHANGE = date(2026, 3, 23)   # pla-tracker の ships_total が公務船を含むようになった日（meta.json）
EXERCISES_SINCE = date(2026, 11, 14)          # exercises の公開の始まり（docs/data/exercises/current.json の最初のコミット）
NEAR_REGIONS = {'taiwan', 'taiwan-strait', 'east-china-sea', 'philippine-sea', 'china', 'south-china-sea', 'okinawa'}
MIN_ACTION = 2        # 行動層で点の付いた指標がこれより少なければ「不明」

COUNT_INDICATORS = [
    # (層, id, 名前, カテゴリー, 出典, 系列)
    ('action', 'pla-aircraft', '台湾周辺の中国軍機（延べ機数）', 'military', '台湾国防部の日報（欠けた日は PLA Activity Tracker）', 'aircraft'),
    ('action', 'pla-crossings', '中間線越えの機数（PLA Tracker の定義）', 'military', 'PLA Activity Tracker（台湾国防部の日報）', 'crossings'),
    ('action', 'pla-navy', '台湾周辺の中国海軍艦艇', 'military', '台湾国防部の日報（欠けた日は PLA Activity Tracker）', 'navy'),
    ('action', 'pla-ships', '台湾周辺の艦艇（海軍＋公務船）', 'military', '台湾国防部の日報（欠けた日は PLA Activity Tracker）', 'ships'),
    ('action', 'senkaku', '尖閣 接続水域の中国海警船（隻・日の合計）', 'law_enforcement', '海上保安庁', 'senkaku'),
    ('action', 'jso-transits', '統幕公表の中国艦艇の目撃（件数、航過日ごと）', 'military', '統合幕僚監部', 'jso'),
]


def reported_by(e, at):
    """事象が at の日の終わり（UTC）までに報じられていたか。firstReported が無ければ使わない。"""
    fr = e.get('firstReported')
    if not isinstance(fr, (int, float)):
        return False
    return datetime.fromtimestamp(fr / 1000, tz=timezone.utc).date() <= at


def exercise_indicator(as_of, named, events):
    """展開層：演習の告知・報道。名前つきの大規模演習の期間なら 3、確度「高」の中国の演習なら 2、「中」なら 1。

    exercises の公開が始まる前（EXERCISES_SINCE）は確かめていないので null。報道が at より後の事象は使わない。"""
    for e in named:
        if d(e['start']) <= as_of <= d(e['end']) + timedelta(days=1):
            return {'score': 3, 'value': None, 'note': f"名前つきの大規模演習：{e['name']}"}
    if as_of < EXERCISES_SINCE:
        return {'score': None, 'value': None, 'reason': f'演習の収集は {EXERCISES_SINCE.isoformat()} から'}
    best, note = 0, '該当なし'
    for e in events:
        if '中国' not in (e.get('countries') or []) or not e.get('start'):
            continue
        if e.get('region') not in NEAR_REGIONS or not reported_by(e, as_of):
            continue
        start, end = d(e['start']), d(e.get('end') or e['start'])
        if not (start <= as_of <= end + timedelta(days=1)):
            continue
        s = {'高': 2, '中': 1}.get(e.get('confidence'), 0)
        if s > best:
            best, note = s, f"{e.get('titleJa') or e.get('name')}（確度 {e.get('confidence')}）"
    return {'score': best, 'value': None, 'note': note}


# ---- ew-2：戦略層・動員層・展開層の追加の源（collectors/ew2.py、docs/data/ew/src/）----

MONTHS_BASE = 24      # 月次の平常値に使う月数
MIN_MONTHS = 12
SAMPLE_DAYS = 30      # 衛星の数（まばらな標本）を平均する日数
MIN_SAMPLE_WINDOWS = 6
HOSTILE = re.compile(r'台独|分裂|外部势力|惩戒|反制|亮剑|备战|打仗|斗争|统一|武力')
RMRB_SINCE_NOTE = '人民日報 1〜4 版の見出しのうち、台湾・闘争・備戦の語（台独・分裂・外部势力・惩戒・反制・亮剑・备战・打仗・斗争・统一・武力）を含む数'


def month_end(ym):
    y, m = int(ym[:4]), int(ym[5:7])
    return (date(y + (m == 12), m % 12 + 1, 1) - timedelta(days=1))


def assess_monthly(months, as_of, lag_days, diff=False, max_age_days=45):
    """月次の系列 {YYYY-MM: 値}。月末から lag_days たって公表されたとみなし、as_of で見えていた最新の月を、
    その前の MONTHS_BASE か月の中央値と MAD でロバスト z にする。diff=True は前月からの増え。"""
    seen = sorted(k for k in months if month_end(k) + timedelta(days=lag_days) <= as_of)
    vals = {k: months[k] for k in seen}
    if diff:
        vals = {k: vals[k] - vals[p] for p, k in zip(seen, seen[1:]) if (month_end(k) - month_end(p)).days < 40}
    keys = sorted(vals)
    if not keys:
        return {'value': None, 'z': None, 'score': None, 'reason': 'データが無い'}
    last = keys[-1]
    if (as_of - (month_end(last) + timedelta(days=lag_days))).days > max_age_days:
        return {'value': vals[last], 'z': None, 'score': None, 'month': last, 'reason': f'最新の月（{last}）が古すぎる（収集が止まっている疑い）'}
    base = [vals[k] for k in keys[:-1][-MONTHS_BASE:]]
    if len(base) < MIN_MONTHS:
        return {'value': vals[last], 'z': None, 'score': None, 'month': last,
                'reason': f'平常値を作る月が足りない（{len(base)} か月、要 {MIN_MONTHS} か月）'}
    med = median(base)
    mad = median([abs(b - med) for b in base])
    scale = max(1.4826 * mad, 0.15 * abs(med), 1.0)
    z = (vals[last] - med) / scale
    q = sorted(base)
    return {'value': vals[last], 'month': last, 'baseline_median': round(med, 1),
            'band': [round(q[int(0.1 * (len(q) - 1))], 1), round(q[int(0.9 * (len(q) - 1))], 1)],
            'z': round(z, 2), 'score': score_from_z(z), 'months': len(base)}


def assess_samples(samples, as_of):
    """まばらな標本（衛星の撮影ごとの数）。直近 SAMPLE_DAYS 日の平均を、それより前の 1 年の SAMPLE_DAYS 日ごとの平均と比べる。"""
    def mean_in(end):
        xs = [v for k, v in samples.items() if end - timedelta(days=SAMPLE_DAYS) < k <= end]
        return sum(xs) / len(xs) if xs else None
    v = mean_in(as_of)
    if v is None:
        return {'value': None, 'z': None, 'score': None, 'reason': f'直近 {SAMPLE_DAYS} 日の撮影が無い'}
    base = []
    end = as_of - timedelta(days=SAMPLE_DAYS)
    while end > as_of - timedelta(days=BASE_DAYS):
        m = mean_in(end)
        if m is not None:
            base.append(m)
        end -= timedelta(days=SAMPLE_DAYS)
    if len(base) < MIN_SAMPLE_WINDOWS:
        return {'value': round(v, 1), 'z': None, 'score': None, 'reason': f'平常値を作る期間が足りない（{len(base)}、要 {MIN_SAMPLE_WINDOWS}）'}
    med = median(base)
    mad = median([abs(b - med) for b in base])
    scale = max(1.4826 * mad, 0.15 * med, 1.0)
    z = (v - med) / scale
    q = sorted(base)
    return {'value': round(v, 1), 'baseline_median': round(med, 1),
            'band': [round(q[int(0.1 * (len(q) - 1))], 1), round(q[int(0.9 * (len(q) - 1))], 1)],
            'z': round(z, 2), 'score': score_from_z(z)}


def covered(spans, first_day, last_day):
    """Is every moment of the UTC days first_day..last_day inside the collected spans (ISO times)?
    A span written with dates only counts as whole days."""
    start = datetime(first_day.year, first_day.month, first_day.day, tzinfo=timezone.utc)
    end = datetime(last_day.year, last_day.month, last_day.day, tzinfo=timezone.utc) + timedelta(days=1)
    def t(x, upper):
        if len(x) == 10:
            v = datetime.fromisoformat(x).replace(tzinfo=timezone.utc)
            return v + timedelta(days=1) if upper else v
        return datetime.fromisoformat(x.replace('Z', '+00:00'))
    for a, b in sorted(spans or []):
        a, b = t(a, False), t(b, True)
        if a <= start < b:
            start = b
        if start >= end:
            return True
    return start >= end


def load_ew2():
    """docs/data/ew/src/ の各源と、衛星の数（sat/ の ship_counts.csv の馬尾造船所）。"""
    src = {}
    for name in ('gold', 'oil', 'jwc', 'rmrb', 'feed', 'ioda', 'ferries'):
        src[name] = load_json(os.path.join(OUT, 'src', name + '.json'), None)
    maw = {}
    for r in read_csv('sat/ship_counts.csv'):
        if r.get('site_id') == 'MAW' and r.get('acquired_utc') and (r.get('platform') or '').startswith('sentinel-1') \
                and (float(r.get('coverage') or 0) >= 0.95):
            a, b = int_or_none(r.get('len_100_200')), int_or_none(r.get('len_ge200'))
            if a is not None and b is not None:
                maw[d(r['acquired_utc'])] = a + b
    src['maw'] = maw
    rm = {}
    for day, heads in ((src.get('rmrb') or {}).get('days') or {}).items():
        rm[d(day)] = sum(1 for h in heads if HOSTILE.search(h.get('title') or ''))
    src['rmrb_daily'] = rm
    for z in ('fujian', 'bohai'):
        src['ferry_' + z] = {d(k): v for k, v in (((src.get('ferries') or {}).get('daily') or {}).get(z) or {}).items()}
    return src


# ew-2 の指標は、誤報の目安と過去事例で確かめるまで「表示だけ」（判定に入れない）。確かめたものから外す
TRIAL = {'gold', 'oil-imports', 'amphib-build', 'rhetoric', 'mobilization-drills', 'conscription', 'war-risk',
         'ferries-fujian', 'ferries-bohai', 'cables'}


def item(layer, iid, name, cat, source, kind, a, **extra):
    out = {'layer': layer, 'id': iid, 'name': name, 'category': cat, 'source': source, 'kind': kind, 'status': 'collected', **a, **extra}
    if iid in TRIAL:
        out['judged'] = False
    return out


def lagged(daily, as_of, max_lag):
    last = max((k for k in daily if k <= as_of), default=None)
    if last is None:
        return None, as_of
    return last, (last if (as_of - last).days <= max_lag else as_of)


def last_covered_day(spans, as_of, back=2):
    """The latest day (as_of or up to ``back`` days before) whose 24 hours are inside the spans; None when none is.
    The run of the day has not reached midnight yet, so 'today' is usually not complete."""
    for i in range(back + 1):
        day = as_of - timedelta(days=i)
        if covered(spans, day, day):
            return day
    return None


def feed_event(doc, key, as_of, days=30):
    """報道の語の事象：窓（as_of までの days 日）の全体が収集の期間（covered）に入っていなければ null。1 件で 1 点、3 件で 2 点。"""
    end = last_covered_day((doc or {}).get('covered'), as_of)
    if not doc or end is None or not covered(doc.get('covered'), end - timedelta(days=days - 1), end):
        return {'value': None, 'score': None, 'reason': f'報道の語の収集が {days} 日の窓を覆っていない（2026-11-25 から集め始めた。途中の穴も不明）'}
    n = [i for i in (doc.get('items') or {}).get(key, []) if end - timedelta(days=days) < d(i['date']) <= end]
    return {'value': len(n), 'score': 2 if len(n) >= 3 else 1 if n else 0, 'as_of': end.isoformat(),
            'note': '；'.join(i['title'] for i in n[-3:]) or '該当なし'}


def ew2_indicators(src, as_of, with_series=True):
    e2 = src.get('ew2') or {}
    out = []
    # 戦略層
    g = e2.get('gold')
    a = assess_monthly({k: v['value'] for k, v in (g or {}).get('months', {}).items()}, as_of, 7, diff=True) if g else \
        {'value': None, 'score': None, 'reason': 'データが無い'}
    if a.get('score') is not None:
        a['score'] = min(a['score'], 1)   # 金だけでは 1 点まで（スイスからの金の輸出などで裏付けが取れるまで）
    out.append(item('strategic', 'gold', '金準備の月の増え（万盎司）', 'money', '国家外汇管理局「官方储备资产」', 'monthly', a,
                    note='公表は翌月 7 日ごろとみなす。前月からの増えを直近 24 か月と比べる。金だけでは 1 点まで'))
    o = e2.get('oil')
    a = assess_monthly({k: v['value'] for k, v in (o or {}).get('months', {}).items()}, as_of, 90) if o else \
        {'value': None, 'score': None, 'reason': 'データが無い'}
    out.append(item('strategic', 'oil-imports', '原油の輸入（千バレル/日）', 'money', 'JODI-Oil World Database', 'monthly', a,
                    note='中国は原油の在庫を出していないため、備蓄の積み増しの代わりに輸入を見る。公表は約 3 か月後とみなす'))
    last, at = lagged(src.get('maw', {}), as_of, 30)
    a = assess_samples(src.get('maw', {}), at) if last else {'value': None, 'score': None, 'reason': 'データが無い'}
    out.append(item('strategic', 'amphib-build', '馬尾造船所の 100 m 以上の船（撮影ごとの数の 30 日平均）', 'transport',
                    'Sentinel-1/2（sat/ の ship_counts、MAW）', 'samples', a, as_of=at.isoformat(),
                    note='揚陸艦・特殊はしけの造船所（広州・龍穴島、上海・滬東中華、長興島）は sat/ の地点の準備（陸マスク・基準・岸壁線）がまだで、今は馬尾だけ。Contains modified Copernicus Sentinel data'))
    last, at = lagged(src.get('rmrb_daily', {}), as_of, 2)
    a = assess_count(src.get('rmrb_daily', {}), at) if last else {'value': None, 'score': None, 'reason': 'データが無い'}
    out.append(item('strategic', 'rhetoric', '人民日報の要聞の見出しの台湾・闘争の語（7 日合計）', 'messaging', '人民日報 電子版（1〜4 版）',
                    'count-7d', a, as_of=at.isoformat(), note=RMRB_SINCE_NOTE,
                    **({'series': weekly_series(src.get('rmrb_daily', {}), at)} if with_series and last else {})))
    # 動員層
    out.append(item('mobilization', 'mobilization-drills', '国防動員・民兵・人防の報道（30 日）', 'personnel',
                    '東アジア情勢観測の報道の一覧（語の一致）', 'event', feed_event(e2.get('feed'), 'mobilization', as_of)))
    out.append(item('mobilization', 'conscription', '徴兵・退役延期の報道（30 日）', 'personnel',
                    '東アジア情勢観測の報道の一覧（語の一致）', 'event', feed_event(e2.get('feed'), 'conscription', as_of)))
    j = e2.get('jwc')
    known = sorted(((k, v) for k, v in ((j or {}).get('circulars') or {}).items() if d(v['first_seen']) <= as_of), key=lambda x: x[0])
    if not known:
        a = {'value': None, 'score': None, 'reason': 'JWC の指定海域の記録の期間外（2026-11-25 から）'}
    else:
        k, v = known[-1]
        a = {'value': None, 'score': 3 if v.get('east_asia_terms') else 0,
             'note': f"{k}：" + ('、'.join(v['east_asia_terms']) if v.get('east_asia_terms') else '東アジアの語なし')}
    a['note'] = ((a.get('note') or '') + '（料率は公開されないため、指定海域の有無で見る）').strip()
    out.append(item('mobilization', 'war-risk', '戦争保険の指定海域（JWC Listed Areas）に東アジア', 'money',
                    "Lloyd's Market Association, Joint War Committee", 'event', a))
    # 展開層
    for z, name, down in (('fujian', '台湾海峡の西岸（福建沿岸）の旅客船の時間（7 日合計）', False),
                          ('bohai', '渤海の定期航路の旅客船の時間の減り（7 日合計）', True)):
        series = src.get('ferry_' + z, {})
        last, at = lagged(series, as_of, 7)
        a = assess_count(series, at, down=down) if last else {'value': None, 'score': None, 'reason': 'データが無い'}
        out.append(item('deployment', 'ferries-' + z, name, 'transport', 'Global Fishing Watch（Powered by Global Fishing Watch、CC BY-NC 4.0）',
                        'count-7d', a, as_of=at.isoformat(), note='GFW の在りかは約 4 日遅れ。格子の合計だけを使い、船の識別子は保存しない',
                        **({'series': weekly_series(series, at)} if with_series and last else {})))
    io_ = e2.get('ioda')
    end = last_covered_day((io_ or {}).get('covered'), as_of)
    if not io_ or not io_.get('verified') or end is None or not covered(io_.get('covered'), end - timedelta(days=6), end):
        a = {'value': None, 'score': None, 'reason': 'IODA の取得が 7 日の窓を覆っていない（または地域のコードが未確認）'}
    else:
        t1 = int(datetime(end.year, end.month, end.day, tzinfo=timezone.utc).timestamp()) + 86400
        ev = [x for x in io_['events'] if t1 - 7 * 86400 <= x['start'] < t1]
        longest = max((x['duration'] for x in ev), default=0)
        a = {'value': len(ev), 'score': 3 if longest >= 6 * 3600 else 2 if longest >= 3600 else 1 if ev else 0,
             'note': f'7 日の障害 {len(ev)} 件、最長 {round(longest / 3600, 1)} 時間（連江・金門・澎湖）'}
    fc = feed_event(e2.get('feed'), 'cables', as_of, days=7)
    if fc.get('score') and (a.get('score') is None or fc['score'] > a['score']):
        a = {**a, 'score': fc['score'], 'note': (a.get('note') or '') + ' 報道：' + fc['note']}
    out.append(item('deployment', 'cables', '海底ケーブル・離島の通信障害', 'infrastructure', 'IODA（Georgia Tech）と報道の語', 'event', a))
    return out


def build_indicators(src, named, events, as_of, with_series=True):
    out = []
    for layer, iid, name, cat, source, key in COUNT_INDICATORS:
        daily = jso_daily(src, as_of) if key == 'jso' else src[key]
        last = max((k for k in daily if k <= as_of), default=None)
        if last is None:
            a = {'value': None, 'z': None, 'score': None, 'reason': 'データが無い'}
        else:
            # 日報の遅れの分だけ、評価日を最新の日に寄せる（3 日まで）
            at = min(as_of, last) if (as_of - last).days <= 3 else as_of
            a = assess_count(daily, at)
            a['as_of'] = at.isoformat()
        item = {'layer': layer, 'id': iid, 'name': name, 'category': cat, 'source': source,
                'kind': 'count-7d', 'status': 'collected', **a}
        if iid.startswith('pla-'):
            item['group'] = 'taiwan'
        if key == 'navy':
            item['note'] = '国防部の日報の自動取得を 2026-10-31 に止め、PLA Tracker は 2026-03 下旬から公務船と合わせた数しか無いため、それ以降は値が無い（過去事例の検証用）'
        if key == 'ships':
            item['note'] = f'公務船の数が分かる日だけ（PLA Tracker は {SHIPS_DEFINITION_CHANGE.isoformat()} 以降の定義が同じ量）'
        if with_series and last is not None:
            item['series'] = weekly_series(daily, min(as_of, last))
        out.append(item)
    ex = exercise_indicator(as_of, named, events)
    out.append({'layer': 'deployment', 'id': 'exercises', 'name': '中国の演習の告知・報道', 'category': 'military',
                'source': 'exercises（統幕・中国国防部・報道の照合）と analysis/exercises.json', 'kind': 'event',
                'status': 'collected', **ex})
    out += ew2_indicators(src, as_of, with_series)
    for layer, iid, name, cat, source, reason in BLOCKED:
        out.append({'layer': layer, 'id': iid, 'name': name, 'category': cat, 'source': source,
                    'kind': 'blocked', 'status': 'blocked', 'value': None, 'score': None, 'reason': reason})
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
    used = [i for i in indicators if i.get('judged', True)]
    hot_logistics = sorted({i['category'] for i in used if i['category'] in LOGISTICS and (i['score'] or 0) >= 2})
    hot_ops = [i['id'] for i in used if i['layer'] in ('deployment', 'action') and (i['score'] or 0) >= 2]
    logistics_collected = any(i['category'] in LOGISTICS and i['status'] == 'collected' for i in used)
    trial_hot = [i['id'] for i in indicators if not i.get('judged', True) and (i['score'] or 0) >= 2]
    action_scored = sum(1 for i in indicators if i['layer'] == 'action' and i['score'] is not None)
    taiwan = [i for i in indicators if i.get('group') == 'taiwan']
    if len(hot_logistics) >= 2:
        verdict = 'alert'
    elif action_scored < MIN_ACTION or (taiwan and all(i['score'] is None for i in taiwan)):
        verdict = 'unknown'
    elif hot_ops:
        verdict = 'exercise'
    else:
        verdict = 'normal'
    return {'verdict': verdict, 'layers': layers, 'hot_logistics': hot_logistics, 'hot_ops': hot_ops, 'trial_hot': trial_hot,
            'ceiling': None if logistics_collected else 'exercise',
            'ceiling_note': None if logistics_collected else
            '兵站系（お金・人員・輸送）の指標は集め始めたが、確かめるまで判定に入れない（表示だけ）。「警戒」はまだ出ない（判定の上限は「演習」）'}


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
    quiet, hits, unknown, months = 0, 0, 0, {}
    for i in range(days):
        at = as_of - timedelta(days=i)
        if any(d(e['start']) - timedelta(days=margin) <= at <= d(e['end']) + timedelta(days=margin) for e in named):
            continue
        v = judge(build_indicators(series, [], events, at, with_series=False))['verdict']
        if v == 'unknown':
            unknown += 1
            continue
        quiet += 1
        if v != 'normal':
            hits += 1
            months[at.isoformat()[:7]] = months.get(at.isoformat()[:7], 0) + 1
    return {'days': quiet, 'flagged': hits, 'unknown_days': unknown, 'rate': round(hits / quiet, 3) if quiet else None,
            'by_month': dict(sorted(months.items())),
            'note': f'過去 {days} 日のうち、名前つきの大規模演習の前後 {margin} 日を除いた日で「演習」以上と出た割合。'
                    '名前の無い演習・示威行動も「演習」に入るので、すべてが誤報とは限らない。「不明」の日は分母に入れない'}


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
    argv = argv if argv is not None else sys.argv[1:]
    series, named, events = load_inputs()
    as_of = d(argv[0]) if argv else datetime.now(timezone.utc).date()
    inds = build_indicators(series, named, events, as_of)
    now = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    current = {'version': VERSION, 'generated_utc': now, 'as_of': as_of.isoformat(),
               **judge(inds), 'indicators': inds, 'category_ja': CATEGORY_JA}
    changed = [n for n, p in (('indicators.json', current),
                              ('backtest.json', {'version': VERSION, 'generated_utc': now,
                                                 'note': '各時点で、その日までに公表・報道されていたものだけを使って採点した（統幕は公表日、演習の報道は初報の日で絞る）。名前つきの演習の一覧（analysis/exercises.json）は答えなので、ここと誤報の目安では採点に使わない。台湾周辺の数と海保は後から取得した版で、公表の時点は再現していない',
                                                 'rows': backtest(series, named, events),
                                                 'false_alarm': false_alarms(series, named, events, as_of)}))
               if write_if_changed(n, p)]
    print('changed:', ', '.join(changed) or 'none')


if __name__ == '__main__':
    main()
