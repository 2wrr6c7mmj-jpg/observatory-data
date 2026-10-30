"""分析の試作（世界情勢観測）: 台湾国防部の日報から静的 JSON を作る。

標準ライブラリだけで動く（CI と Actions で追加の導入が要らないように）。
書き出し先は docs/proto/data/ だけ。

    python -m analysis.proto            # docs/proto/data/*.json を作り直す
"""
import csv
import json
import math
import os
import random
import sys
from datetime import date, datetime, timedelta, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'docs', 'data', 'taiwan-mnd')
OUT = os.path.join(ROOT, 'docs', 'proto', 'data')
EXERCISES = os.path.join(ROOT, 'analysis', 'exercises.json')

METRICS = ['aircraft', 'plan_vessels', 'official_ships', 'median_line_or_adiz']
LABELS = {
    'aircraft': '軍機（延べ機数）',
    'plan_vessels': '艦艇',
    'official_ships': '公務船',
    'median_line_or_adiz': '中間線越え等（延べ機数）',
    'balloons': '気球',
}
ALGO_VERSION = 1


# ---------------------------------------------------------------- 基本の道具

def d(s):
    return date.fromisoformat(s)


def days_between(a, b):
    """a から b まで（両端を含む）の日付。"""
    out, x = [], a
    while x <= b:
        out.append(x)
        x += timedelta(days=1)
    return out


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return None
    m = n // 2
    return float(xs[m]) if n % 2 else (xs[m - 1] + xs[m]) / 2.0


def quantile(xs, q):
    """線形補間の分位点（numpy の既定と同じ）。"""
    xs = sorted(xs)
    if not xs:
        return None
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def mad_scale(xs):
    """正規分布の標準偏差に換算した MAD（1.4826 倍）。"""
    m = median(xs)
    if m is None:
        return None
    return 1.4826 * median([abs(x - m) for x in xs])


def r(x, nd=3):
    return None if x is None else round(x, nd)


# ---------------------------------------------------------------- 読み込み

def load_daily(path=None, fmt='around-taiwan'):
    """daily.csv を読む。既定では around-taiwan 書式（2022-11 以降）の日だけ。
    sw-adiz 書式（それ以前）は機数の定義が違うため、試作では使わない。"""
    path = path or os.path.join(SRC, 'daily.csv')
    rows = {}
    with open(path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if fmt and row.get('format') != fmt:
                continue
            rec = {}
            for k in METRICS + ['balloons']:
                v = row.get(k, '')
                rec[k] = int(v) if v not in ('', None) else None
            rec['source_url'] = row.get('source_url')
            rec['format'] = row.get('format')
            rows[d(row['report_date'])] = rec
    return rows


def load_json(name):
    with open(os.path.join(SRC, name), encoding='utf-8') as f:
        return json.load(f)


def series(rows, metric):
    """暦日ごとの値（日報なし・null は None）。"""
    first, last = min(rows), max(rows)
    return [(x, rows[x][metric] if x in rows else None) for x in days_between(first, last)]


# ---------------------------------------------------------------- 1. 収集の穴

def build_gaps(rows, reports, definition_changes):
    first, last = min(rows), max(rows)
    by_date = {}
    for rid, rep in reports.items():
        rd = rep.get('report_date')
        if rd and first <= d(rd) <= last:
            by_date.setdefault(rd, []).append((rid, rep))
    days = []
    totals = {'no_report': 0, 'unparsed_only': 0, 'parsed': 0, 'multiple': 0}
    per_metric = {m: {'value': 0, 'zero': 0, 'null': 0} for m in METRICS + ['balloons']}
    for x in days_between(first, last):
        iso = x.isoformat()
        reps = by_date.get(iso, [])
        if x in rows:
            st = 'parsed'
        elif reps:
            st = 'unparsed_only'
        else:
            st = 'no_report'
        totals[st] += 1
        if len(reps) > 1:
            totals['multiple'] += 1
        cell = {'date': iso, 'status': st, 'reports': len(reps),
                'unparsed': [rid for rid, rep in reps if not rep.get('parsed')]}
        if x in rows:
            for m in METRICS + ['balloons']:
                v = rows[x][m]
                per_metric[m]['null' if v is None else ('zero' if v == 0 else 'value')] += 1
                cell[m] = v
        days.append(cell)
    # 指標ごとに「記載が始まった日」（それより前の null は書式上の未記載）
    first_seen = {}
    for m in METRICS + ['balloons']:
        seen = [x for x in sorted(rows) if rows[x][m] is not None]
        first_seen[m] = seen[0].isoformat() if seen else None
    # 月ごとの集計
    months = {}
    for c in days:
        mo = months.setdefault(c['date'][:7], {'days': 0, 'no_report': 0, 'unparsed_only': 0,
                                               **{'null_' + m: 0 for m in METRICS}})
        mo['days'] += 1
        if c['status'] != 'parsed':
            mo[c['status']] += 1
        else:
            for m in METRICS:
                if c[m] is None:
                    mo['null_' + m] += 1
    return {
        'span': [first.isoformat(), last.isoformat()],
        'calendar_days': len(days),
        'totals': totals,
        'per_metric': per_metric,
        'first_reported': first_seen,
        'days': days,
        'months': [{'month': k, **v} for k, v in sorted(months.items())],
        'definition_changes': definition_changes,
    }


# ---------------------------------------------------------------- 2. 平常帯

def nb_upper_tail(y, mu, var):
    """P(Y >= y)。var <= mu のときはポアソン、それ以外は負の二項（平均 mu・分散 var）。"""
    if y <= 0:
        return 1.0
    mu = max(mu, 1e-6)
    if var <= mu * 1.0001:
        logp = -mu + y * math.log(mu) - math.lgamma(y + 1)
        ratio = lambda k: mu / (k + 1)
    else:
        rr = mu * mu / (var - mu)
        q = mu / (rr + mu)
        logp = (math.lgamma(y + rr) - math.lgamma(rr) - math.lgamma(y + 1)
                + rr * math.log(rr / (rr + mu)) + y * math.log(q))
        ratio = lambda k: (k + rr) / (k + 1) * q
    term = math.exp(logp)
    total, k = 0.0, y
    while term > 0 and k < y + 100000:
        total += term
        if term < total * 1e-14 and ratio(k) < 1:
            break
        term *= ratio(k)
        k += 1
    return min(1.0, total)


def nb_logpmf(y, mu, size):
    return (math.lgamma(y + size) - math.lgamma(size) - math.lgamma(y + 1)
            + size * math.log(size / (size + mu)) + y * math.log(mu / (size + mu)))


NB_GRID = [0.3 * 1.25 ** k for k in range(36)]  # 0.3〜約740


def fit_nb_size(pairs):
    """平均を固定した負の二項の size を格子上の最尤で選ぶ（大きいほどポアソンに近い）。"""
    best, arg = -math.inf, None
    for size in NB_GRID:
        ll = sum(nb_logpmf(y, mu, size) for mu, y in pairs)
        if ll > best:
            best, arg = ll, size
    return arg


def build_baseline(rows, metric, trend_window=56, min_obs=20, k=3.0, min_scale=0.15):
    """log(1+件数) の上で、移動中央値（その日を含まない過去 trend_window 日）＋曜日・月の効果（中央値）で
    平常値を作り、過去 trend_window 日の残差の MAD で幅を引く（件数に戻すと上下非対称の幅になる）。
    起こりにくさは負の二項（平均＝平常値、size は過去 365 日で最尤）の上側確率。"""
    s = series(rows, metric)
    vals = [v for _, v in s]
    lv = [math.log1p(v) if v is not None else None for v in vals]
    n = len(s)
    trend = [None] * n
    for i in range(n):
        past = [v for v in lv[max(0, i - trend_window):i] if v is not None]
        if len(past) >= min_obs:
            trend[i] = median(past)
    # 曜日効果 → 月効果（それぞれ残差の中央値、中央値で中心化）
    r1 = {}
    for i, (x, v) in enumerate(s):
        if lv[i] is not None and trend[i] is not None:
            r1.setdefault(x.weekday(), []).append(lv[i] - trend[i])
    wd = {w: median(xs) for w, xs in r1.items()}
    c = median(list(wd.values())) if wd else 0.0
    wd = {w: e - c for w, e in wd.items()}
    r2 = {}
    for i, (x, v) in enumerate(s):
        if lv[i] is not None and trend[i] is not None:
            r2.setdefault(x.month, []).append(lv[i] - trend[i] - wd.get(x.weekday(), 0.0))
    mo = {m: median(xs) for m, xs in r2.items() if len(xs) >= 30}
    c = median(list(mo.values())) if mo else 0.0
    mo = {m: e - c for m, e in mo.items()}
    elog = [None] * n
    for i, (x, v) in enumerate(s):
        if trend[i] is not None:
            elog[i] = max(0.0, trend[i] + wd.get(x.weekday(), 0.0) + mo.get(x.month, 0.0))
    expected = [math.expm1(e) if e is not None else None for e in elog]
    resid = [(lv[i] - elog[i]) if (lv[i] is not None and elog[i] is not None) else None for i in range(n)]
    # 負の二項の形（size）を、過去 365 日の（平常値, 実際）から最尤で求める。30 日ごとに当て直す。
    r_at = [None] * n
    cur = None
    for i in range(n):
        if i % 30 == 0 or cur is None:
            pairs = [(max(expected[j], 0.5), vals[j]) for j in range(max(0, i - 365), i)
                     if expected[j] is not None and vals[j] is not None]
            cur = fit_nb_size(pairs) if len(pairs) >= 60 else None
        r_at[i] = cur
    out = []
    for i, (x, v) in enumerate(s):
        e = expected[i]
        rec = {'date': x.isoformat(), 'y': v}
        if e is not None and r_at[i] is not None:
            past = [z for z in resid[max(0, i - trend_window):i] if z is not None]
            if len(past) >= min_obs:
                sc = max(min_scale, mad_scale(past))
                hi = math.expm1(elog[i] + k * sc)
                rec.update(exp=r(e, 2), lo=r(max(0.0, math.expm1(elog[i] - k * sc)), 2), hi=r(hi, 2), sc=r(sc, 3))
                if v is not None:
                    mu = max(e, 0.5)
                    p = nb_upper_tail(v, mu, mu + mu * mu / r_at[i])
                    rec['nb_r'] = r(r_at[i], 2)
                    rec['p'] = float('%.3g' % p)
                    rec['out'] = v > hi
        out.append(rec)
    flagged = [o for o in out if o.get('out')]
    return {
        'metric': metric, 'label': LABELS[metric],
        'params': {'trend_window_days': trend_window, 'min_obs': min_obs, 'k_mad': k, 'scale': 'log1p', 'min_scale': min_scale},
        # 効果は倍率（log(1+件数) の差を exp したもの）
        'weekday_effect': {str(w): r(math.exp(e), 3) for w, e in sorted(wd.items())},
        'month_effect': {str(m): r(math.exp(e), 3) for m, e in sorted(mo.items())},
        'n_evaluated': sum(1 for o in out if 'p' in o),
        'n_out': len(flagged),
        'days': out,
    }


# ---------------------------------------------------------------- 3. 変化点

def week_start(x):
    return x - timedelta(days=x.weekday())


def weekly(rows, metric, min_days=4):
    """月曜始まりの週ごとの日次値の中央値（値のある日が min_days 未満なら None）と、記載のない日の割合。"""
    first, last = min(rows), max(rows)
    weeks = {}
    for x in days_between(week_start(first), last):
        w = weeks.setdefault(week_start(x), {'vals': [], 'days': 0, 'missing': 0})
        if x < first:
            continue
        w['days'] += 1
        v = rows[x][metric] if x in rows else None
        if v is None:
            w['missing'] += 1
        else:
            w['vals'].append(v)
    out = []
    for ws in sorted(weeks):
        w = weeks[ws]
        out.append({'week': ws.isoformat(),
                    'value': median(w['vals']) if len(w['vals']) >= min_days else None,
                    'missing_share': r(w['missing'] / w['days'], 3) if w['days'] else None,
                    'n': len(w['vals'])})
    return out


def segment_cost_fn(x):
    s1, s2 = [0.0], [0.0]
    for v in x:
        s1.append(s1[-1] + v)
        s2.append(s2[-1] + v * v)

    def cost(a, b):  # x[a:b] の二乗誤差
        n = b - a
        return s2[b] - s2[a] - (s1[b] - s1[a]) ** 2 / n
    return cost


def optimal_partition(x, penalty, min_size=4):
    """平均の変化点（二乗誤差＋変化点ごとの罰則）を厳密に最小化する（PELT と同じ解。n が小さいので枝刈りなし）。
    戻り値は各区間の開始位置（0 を除く）。"""
    n = len(x)
    if n < 2 * min_size:
        return []
    cost = segment_cost_fn(x)
    F = [math.inf] * (n + 1)
    F[0] = -penalty
    prev = [0] * (n + 1)
    for t in range(min_size, n + 1):
        best, arg = math.inf, 0
        for s in range(0, t - min_size + 1):
            if F[s] == math.inf:
                continue
            c = F[s] + cost(s, t) + penalty
            if c < best:
                best, arg = c, s
        F[t], prev[t] = best, arg
    cps, t = [], n
    while t > 0:
        s = prev[t]
        if s > 0:
            cps.append(s)
        t = s
    return sorted(cps)


def noise_scale(x):
    """差分の MAD から雑音の標準偏差を見積もる（段差に強い）。"""
    if len(x) < 3:
        return 1.0
    dif = [x[i + 1] - x[i] for i in range(len(x) - 1)]
    return max(mad_scale(dif) / math.sqrt(2), 1e-3)


def changepoints(x, min_size=4, n_boot=100, seed=1, beta=None, sigma_floor=0.0):
    """変化点、区間の平均、ブートストラップによる位置の幅（5〜95%）と再検出率。"""
    sigma = max(noise_scale(x), sigma_floor)
    z = [v / sigma for v in x]
    n = len(z)
    beta = beta if beta is not None else 3.0 * math.log(max(n, 2))
    cps = optimal_partition(z, beta, min_size)
    bounds = [0] + cps + [n]
    means, resid = [], []
    for a, b in zip(bounds, bounds[1:]):
        m = sum(x[a:b]) / (b - a)
        means.append(m)
        resid.append([v - m for v in x[a:b]])
    rng = random.Random(seed)
    hits = {c: [] for c in cps}
    for _ in range(n_boot):
        xb = []
        for (a, b), m, rs in zip(zip(bounds, bounds[1:]), means, resid):
            xb.extend(m + rs[rng.randrange(len(rs))] for _ in range(b - a))
        cb = optimal_partition([v / sigma for v in xb], beta, min_size)
        for c in cps:
            near = [e for e in cb if abs(e - c) <= 8]
            if near:
                hits[c].append(min(near, key=lambda e: abs(e - c)))
    res = []
    for i, c in enumerate(cps):
        h = sorted(hits[c])
        res.append({'index': c, 'lo': quantile(h, 0.05) if h else None, 'hi': quantile(h, 0.95) if h else None,
                    'redetect': r(len(h) / n_boot, 2), 'before': r(means[i], 2), 'after': r(means[i + 1], 2)})
    return {'sigma': r(sigma, 3), 'penalty': r(beta, 2), 'segments': [
        {'start': a, 'end': b - 1, 'mean': r(m, 2)} for (a, b), m in zip(zip(bounds, bounds[1:]), means)],
        'changepoints': res}


def build_changepoints(rows, definition_changes, n_boot=100):
    out = {}
    for m in METRICS:
        w = weekly(rows, m)
        obs = [i for i, e in enumerate(w) if e['value'] is not None]
        x = [w[i]['value'] for i in obs]
        cp = changepoints(x, n_boot=n_boot, sigma_floor=0.5) if len(x) >= 8 else {'segments': [], 'changepoints': []}

        def wk(j):  # 観測のある週の番号 → 週の日付
            return w[obs[int(round(j))]]['week'] if j is not None else None
        for c in cp['changepoints']:
            c['week'] = wk(c['index'])
            c['week_lo'], c['week_hi'] = wk(c['lo']), wk(c['hi'])
        for s in cp['segments']:
            s['start'], s['end'] = wk(s['start']), wk(s['end'])
        # 記載のない日の割合（報告の仕方の指標）にも同じ手法をかける
        ms = [e['missing_share'] for e in w if e['missing_share'] is not None]
        mw = [e['week'] for e in w if e['missing_share'] is not None]
        mcp = changepoints(ms, n_boot=n_boot, sigma_floor=0.1) if len(ms) >= 8 else {'segments': [], 'changepoints': []}
        for c in mcp['changepoints']:
            c['week'] = mw[c['index']]
            c['week_lo'] = mw[int(round(c['lo']))] if c['lo'] is not None else None
            c['week_hi'] = mw[int(round(c['hi']))] if c['hi'] is not None else None
        for s in mcp['segments']:
            s['start'], s['end'] = mw[s['start']], mw[s['end']]
        out[m] = {'label': LABELS[m], 'weeks': w, 'value': cp, 'missing': mcp}
    return {'metrics': out, 'definition_changes': definition_changes, 'n_boot': n_boot}


# ---------------------------------------------------------------- 4. 持続度と割合

def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None, None)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (p, max(0.0, centre - half), min(1.0, centre + half))


def runs_above(rows, metric='plan_vessels', window=365, q=0.9, min_obs=180):
    """その日より前 window 日の q 分位点を超えた日の連続。欠測（null・日報なし）で連続は切れる。"""
    s = series(rows, metric)
    flags = []
    for i, (x, v) in enumerate(s):
        past = [z for _, z in s[max(0, i - window):i] if z is not None]
        if v is None or len(past) < min_obs:
            flags.append((x, v, None, None))
            continue
        th = quantile(past, q)
        flags.append((x, v, th, v > th))
    runs, cur, start = [], 0, None
    for x, v, th, f in flags:
        if f:
            if cur == 0:
                start = x
            cur += 1
        else:
            if cur:
                runs.append({'start': start.isoformat(), 'length': cur, 'ended_by': 'missing' if f is None else 'below'})
            cur = 0
    current = cur
    if cur:
        runs.append({'start': start.isoformat(), 'length': cur, 'ended_by': 'ongoing'})
    evaluated = [f for _, _, _, f in flags if f is not None]
    p = sum(evaluated) / len(evaluated) if evaluated else 0
    # 隣り合う評価日どうしの「超え」の続きやすさ（独立なら p と同じになる）
    pairs = [(a[3], b[3]) for a, b in zip(flags, flags[1:]) if a[3] is not None and b[3] is not None]
    after_hit = [b for a, b in pairs if a]
    hist = {}
    for rn in runs:
        hist[rn['length']] = hist.get(rn['length'], 0) + 1
    n_runs = len(runs)
    geo = {L: r(n_runs * (1 - p) * p ** (L - 1), 2) for L in range(1, max(hist or {1: 0}) + 1)}
    return {
        'metric': metric, 'label': LABELS[metric], 'window_days': window, 'quantile': q,
        'share_above': r(p, 3), 'p_next_given_above': r(sum(after_hit) / len(after_hit), 3) if after_hit else None,
        'runs': runs, 'histogram': {str(k): v for k, v in sorted(hist.items())},
        'independent_expectation': {str(k): v for k, v in geo.items()},
        'current_run': current, 'last_day': flags[-1][0].isoformat(),
        'last_threshold': r(flags[-1][2], 2), 'last_value': flags[-1][1],
        'days': [{'date': x.isoformat(), 'y': v, 'th': r(th, 2), 'above': f} for x, v, th, f in flags],
    }


def weekly_share(rows, num='median_line_or_adiz', den='aircraft'):
    first, last = min(rows), max(rows)
    weeks = {}
    for x in days_between(first, last):
        if x not in rows:
            continue
        a, b = rows[x][num], rows[x][den]
        w = weeks.setdefault(week_start(x), [0, 0, 0])
        if a is not None and b is not None:
            w[0] += a
            w[1] += b
            w[2] += 1
    out = []
    for ws in sorted(weeks):
        k, n, nd = weeks[ws]
        p, lo, hi = wilson(k, n)
        out.append({'week': ws.isoformat(), 'k': k, 'n': n, 'days': nd, 'p': r(p), 'lo': r(lo), 'hi': r(hi)})
    return {'num': num, 'den': den, 'weeks': out}


# ---------------------------------------------------------------- 5. イベントスタディ

MIN_CASES = 3  # 使える事例がこれ未満の要約値（中央値・偽の起点との比較）は出さない（1部 審査）


def event_study(rows, exercises, metrics=METRICS, pre=30, post=10):
    rel = list(range(-pre, post + 1))
    evs = []
    for e in exercises:
        day0 = d(e['start']) + timedelta(days=1)
        evs.append({**e, 'day0_report': day0.isoformat()})
    ev_days = [d(e['day0_report']) for e in evs]
    res = {}
    for m in metrics:
        mat = []
        for e in evs:
            day0 = d(e['day0_report'])
            mat.append([rows[day0 + timedelta(days=k)][m] if (day0 + timedelta(days=k)) in rows else None for k in rel])

        def summarize(idx):
            med, lo, hi, n = [], [], [], []
            for j in range(len(rel)):
                col = [mat[i][j] for i in idx if mat[i][j] is not None]
                med.append(median(col) if len(col) >= MIN_CASES else None)
                lo.append(min(col) if col else None)
                hi.append(max(col) if col else None)
                n.append(len(col))
            return med, lo, hi, n

        allidx = list(range(len(evs)))
        med, lo, hi, n = summarize(allidx)

        def pre_ratio(vals):
            a = [v for k, v in zip(rel, vals) if -7 <= k <= -1 and v is not None]
            b = [v for k, v in zip(rel, vals) if -30 <= k <= -8 and v is not None]
            if len(a) < 4 or len(b) < 12:
                return None
            ma, mb = median(a), median(b)
            return (ma + 0.5) / (mb + 0.5)

        ratios = [pre_ratio(row) for row in mat]
        loo = []
        for i in allidx:
            mm, _, _, _ = summarize([j for j in allidx if j != i])
            diffs = [abs(a - b) for k, a, b in zip(rel, mm, med) if k < 0 and a is not None and b is not None]
            rs = [x for j, x in enumerate(ratios) if j != i and x is not None]
            loo.append({'excluded': evs[i]['id'], 'median': mm,
                        'max_abs_diff_pre': r(max(diffs) if diffs else None, 2),
                        'median_pre_ratio': r(median(rs), 3) if len(rs) >= MIN_CASES else None})
        # 偽の起点（演習の前後 45 日を除くすべての日）で同じ比を取ったときの分布
        placebo = []
        for x in sorted(rows):
            if any(abs((x - e0).days) <= 45 for e0 in ev_days):
                continue
            vals = [rows.get(x + timedelta(days=k), {}).get(m) for k in rel]
            pr = pre_ratio(vals)
            if pr is not None:
                placebo.append(pr)
        rs = [x for x in ratios if x is not None]
        obs = median(rs) if len(rs) >= MIN_CASES else None
        pct = (sum(1 for p in placebo if p < obs) / len(placebo)) if (placebo and obs is not None) else None
        res[m] = {'label': LABELS[m], 'rel': rel, 'matrix': mat, 'median': med, 'min': lo, 'max': hi, 'n': n,
                  'pre_ratio': [r(x, 3) for x in ratios], 'n_cases': len(rs), 'median_pre_ratio': r(obs, 3),
                  'placebo': {'n': len(placebo), 'q05': r(quantile(placebo, 0.05), 3), 'q50': r(quantile(placebo, 0.5), 3),
                              'q95': r(quantile(placebo, 0.95), 3), 'percentile_of_observed': r(pct, 3)},
                  'loo': loo}
    return {'events': evs, 'pre': pre, 'post': post, 'min_cases': MIN_CASES, 'metrics': res}


# ---------------------------------------------------------------- 書き出し

def data_version():
    dj = load_json('daily.json')
    st = load_json('status.json')
    return {'daily_generated_utc': dj.get('generated_utc'), 'status_generated_utc': st.get('generated_utc'),
            'parser_version': st.get('parser_version'), 'days': st.get('days'),
            'first_day': st.get('first_day'), 'last_day': st.get('last_day'), 'algo_version': ALGO_VERSION}


def write_if_changed(name, payload):
    """computed_utc 以外が同じなら書き換えない（毎日の無駄なコミットを避ける）。"""
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            try:
                old = json.load(f)
            except ValueError:
                old = None
        if old is not None:
            old.pop('computed_utc', None)
            if old == json.loads(json.dumps(payload)):
                return False
    body = {'computed_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), **payload}
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(body, f, ensure_ascii=False, separators=(',', ':'))
        f.write('\n')
    os.replace(tmp, path)
    return True


def main(argv=None):
    rows = load_daily()
    meta = load_json('meta.json')
    reports = load_json('reports.json')['reports']
    with open(EXERCISES, encoding='utf-8') as f:
        exercises = json.load(f)
    ver = data_version()
    defs = meta.get('definition_changes', [])
    src = {'name': meta['source']['name'], 'urls': meta['source']['urls'], 'rating': meta['source'].get('rating_note')}
    common = {'data_version': ver, 'source': src, 'definition_changes': defs}
    changed = []
    payloads = {
        'gaps.json': build_gaps(rows, reports, defs),
        'baseline.json': {'metrics': {m: build_baseline(rows, m) for m in METRICS}},
        'changepoints.json': build_changepoints(rows, defs),
        'persistence.json': {'runs': runs_above(rows), 'share': weekly_share(rows)},
        'events.json': {**event_study(rows, exercises['exercises']), 'exercise_note': exercises['note'],
                        'checked_utc': exercises.get('checked_utc')},
    }
    for name, p in payloads.items():
        if write_if_changed(name, {**common, **p}):
            changed.append(name)
    print('changed:', ', '.join(changed) or 'none')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
