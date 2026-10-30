"""分析の試作 第2弾（6〜10）: 統幕・海保・台湾国防部の公開データから静的 JSON を作る。

第1弾（analysis/proto.py）と同じく標準ライブラリだけで動き、書き出し先は docs/proto/data/ だけ。

    python -m analysis.proto2           # docs/proto/data/{heatmap,stay,joint,grayzone,iw}.json を作り直す

6. heatmap  海峡×月の件数（中国・ロシア別、推定の通過を区別）と年ごとの率の比
7. stay     進出→帰投の往復の滞在日数の年ごとの分布（Kaplan-Meier、往路の行先：太平洋／その他）
8. joint    中露の共同行動の候補（明記・同じ記述・時期の重なり）
9. grayzone 尖閣の中国海警（海保）と台湾周辺の公務船（台湾国防部）を日ごとに並べる
10. iw      シナリオごとの、複数の指標の同時の上振れ（各指標の過去365日でのパーセンタイル）
"""
import csv
import json
import math
import os
import sys
from datetime import date, timedelta

from analysis import proto
from analysis.proto import d, days_between, median, quantile, r, write_if_changed

ROOT = proto.ROOT
DATA = os.path.join(ROOT, 'docs', 'data')
JSO = os.path.join(DATA, 'jso')
JCG = os.path.join(DATA, 'jcg-senkaku')
MND = os.path.join(DATA, 'taiwan-mnd')
LAUNCHES = os.path.join(DATA, 'launches')
CLASSES = os.path.join(ROOT, 'jso', 'ref', 'vessel_classes.csv')
ALGO_VERSION = 1

HOME = {'CN': '東シナ海', 'RU': '日本海'}  # meta.json の direction の定義と同じ
# 往路の行先（waters_to）での絞り込み。自国側の海の外＝太平洋とは限らない
# （ロシア艦の宗谷経由のオホーツク海行きは自国の海域どうしの移動、中国艦の対馬経由は日本海行き）
DESTS = ('all', 'pacific', 'other')


def dest_of(to):
    return 'pacific' if to == '太平洋' else 'other'
NATION_JP = {'CN': '中国', 'RU': 'ロシア'}


# ---------------------------------------------------------------- 道具

def read_csv(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as f:
        return list(csv.DictReader(f))


def read_json(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def to_int(v):
    return int(v) if v not in ('', None) else None


def log_binom_pmf(k, n, p):
    if p <= 0:
        return 0.0 if k == 0 else -math.inf
    if p >= 1:
        return 0.0 if k == n else -math.inf
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
            + k * math.log(p) + (n - k) * math.log(1 - p))


def binom_cdf(k, n, p):
    """P(X <= k), X ~ Bin(n, p)。"""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return min(1.0, sum(math.exp(log_binom_pmf(i, n, p)) for i in range(k + 1)))


def clopper_pearson(k, n, alpha=0.05):
    """二項割合の正確な（Clopper-Pearson）信頼区間。二分法で求める。"""
    if n == 0:
        return (0.0, 1.0)

    def solve(f):  # f は p について単調減少、f(p)=alpha/2 の根
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if f(mid) > alpha / 2:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2
    # 下限：P(X>=k) は p について増加
    lower = 0.0 if k == 0 else _solve_increasing(lambda p: 1 - binom_cdf(k - 1, n, p), alpha / 2)
    upper = 1.0 if k == n else solve(lambda p: binom_cdf(k, n, p))
    return (lower, upper)


def _solve_increasing(f, target):
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if f(mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def rate_ratio(a, ta, b, tb, alpha=0.05):
    """率の比 (a/ta)/(b/tb) と、条件付き二項（a | a+b）による正確な信頼区間。
    a か b が 0 でも区間は出る（点推定は 0 か None＝無限大）。"""
    n = a + b
    if n == 0 or ta <= 0 or tb <= 0:
        return {'rr': None, 'lo': None, 'hi': None}
    lo_p, hi_p = clopper_pearson(a, n, alpha)
    conv = lambda p: None if p >= 1 else p / (1 - p) * tb / ta
    return {'rr': None if b == 0 else (a / ta) / (b / tb), 'lo': conv(lo_p), 'hi': conv(hi_p)}


def midrank_percentile(value, ref):
    """ref の中での value の中央順位パーセンタイル（同じ値は半分として数える、0〜100）。"""
    if value is None or not ref:
        return None
    less = sum(1 for x in ref if x < value)
    eq = sum(1 for x in ref if x == value)
    return 100.0 * (less + 0.5 * eq) / len(ref)


# ---------------------------------------------------------------- 読み込み

def load_classes(path=None):
    path = path or CLASSES
    """艦級名 → 国（CN/RU）。units の文字列から国を判定するのに使う。"""
    out = []
    for row in read_csv(path) or []:
        names = [row['class_jp'], row['class_key']] + [a for a in (row.get('aliases') or '').split(';') if a]
        for nm in names:
            if nm:
                out.append((nm, row['nation']))
    out.sort(key=lambda x: -len(x[0]))  # 長い名前から当てる
    return out


def unit_nations(units, nation, classes):
    """1行の艦艇の国の集合。中露の資料（CNRU）は艦級名で判定する。

    判定できない行は「両国」とせず空の集合を返す（どちらの国にも数えず、「判定できない」として別に数える）。"""
    if nation in ('CN', 'RU'):
        return {nation}
    found = set()
    for u in (units or '').split(' / '):
        name = u.split('(')[0].strip()
        for nm, nat in classes:
            if name.startswith(nm) or (len(name) >= 3 and nm.startswith(name)):
                found.add(nat)
                break
    return found


def load_jso(src=None, classes=None):
    src = src or JSO
    classes = load_classes() if classes is None else classes
    passages = read_csv(os.path.join(src, 'passages.csv')) or []
    for p in passages:
        p['nations'] = sorted(unit_nations(p['units'], p['nation'], classes))
        p['inferred'] = p['inferred'] == '1'
    return {
        'passages': passages,
        'round_trips': read_csv(os.path.join(src, 'round_trips.csv')) or [],
        'meta': read_json(os.path.join(src, 'meta.json')) or {},
        'status': read_json(os.path.join(src, 'status.json')) or {},
    }


def jso_source(J):
    m, st = J['meta'], J['status']
    s = m.get('source', {})
    return {'key': 'jso', 'name': s.get('name'), 'urls': s.get('urls', []), 'rating': s.get('rating_note'),
            'version': f"status.json 作成 {st.get('built_utc')}／資料 {st.get('releases')} 件（{st.get('first_release')}〜{st.get('last_release')} 公表分）",
            'files': ['jso/passages.csv', 'jso/round_trips.csv', 'jso/meta.json', 'jso/status.json']}


def load_jcg(src=None):
    src = src or JCG
    rows = read_csv(os.path.join(src, 'daily.csv'))
    meta = read_json(os.path.join(src, 'meta.json')) or {}
    status = read_json(os.path.join(src, 'status.json')) or {}
    daily = {}
    for row in rows or []:
        daily[d(row['date'])] = {'cz': to_int(row.get('contiguous_zone')), 'ts': to_int(row.get('territorial_sea')),
                                 'check': row.get('check'), 'url': row.get('source_url')}
    return {'daily': daily, 'meta': meta, 'status': status, 'available': bool(daily)}


def jcg_source(G):
    m, st = G['meta'], G['status']
    s = m.get('source', {})
    days = sorted(G['daily'])
    ver = (f"status.json 作成 {st.get('generated_utc')}／{days[0]}〜{days[-1]}（{len(days)} 日分）" if days
           else 'まだ公開データに入っていない（jcg-senkaku.yml の収集後に入る）')
    return {'key': 'jcg', 'name': s.get('name'), 'urls': s.get('urls', []), 'rating': s.get('rating_note'),
            'credit': (m.get('attribution') or {}).get('credit'), 'license': (m.get('attribution') or {}).get('license'),
            'processing': (m.get('attribution') or {}).get('processing'),
            'version': ver, 'files': ['jcg-senkaku/daily.csv', 'jcg-senkaku/meta.json', 'jcg-senkaku/status.json']}


def load_mnd():
    rows = proto.load_daily(os.path.join(MND, 'daily.csv'))
    return {'daily': rows, 'meta': read_json(os.path.join(MND, 'meta.json')) or {},
            'status': read_json(os.path.join(MND, 'status.json')) or {},
            'daily_json': read_json(os.path.join(MND, 'daily.json')) or {}}


def mnd_source(M):
    s, st = M['meta'].get('source', {}), M['status']
    return {'key': 'mnd', 'name': s.get('name'), 'urls': s.get('urls', []), 'rating': s.get('rating_note'),
            'version': (f"daily.json 生成 {M['daily_json'].get('generated_utc')}／読み取り処理 第{st.get('parser_version')}版／"
                        f"{st.get('first_day')}〜{st.get('last_day')}（試作では 2022年11月以降の台湾周辺全体の書式の日だけ）"),
            'files': ['taiwan-mnd/daily.csv', 'taiwan-mnd/meta.json', 'taiwan-mnd/status.json']}


# ---------------------------------------------------------------- 6. 海峡×月

def year_exposure(first, last):
    """年ごとの観測日数（初回公表日〜最後の公表日）。"""
    out = {}
    for y in range(first.year, last.year + 1):
        a, b = max(first, date(y, 1, 1)), min(last, date(y, 12, 31))
        out[y] = (b - a).days + 1
    return out


def build_heatmap(J):
    st = J['status']
    first, last = d(st['first_release']), d(st['last_release'])
    straits = {}
    cells = {}  # (nation, strait, month, inferred) -> n
    for p in J['passages']:
        straits.setdefault(p['strait'], {'key': p['strait'], 'jp': p['strait_jp'], 'n': 0})['n'] += 1
        for nat in p['nations']:
            k = (nat, p['strait'], p['date'][:7], int(p['inferred']))
            cells[k] = cells.get(k, 0) + 1
    expo = year_exposure(first, last)
    years = sorted(expo)
    # 年ごとの件数と、前年との率の比（国 × 海峡〔all を含む〕× 推定を含むか）
    rr = []
    for nat in ('CN', 'RU'):
        for sk in ['all'] + sorted(straits):
            for incl in (0, 1):
                cnt = {y: 0 for y in years}
                for (n_, s_, mo, inf), n in cells.items():
                    if n_ == nat and (sk == 'all' or s_ == sk) and (incl or not inf):
                        cnt[int(mo[:4])] += n
                rows = []
                for y in years:
                    row = {'year': y, 'n': cnt[y], 'days': expo[y], 'per_year': r(cnt[y] / expo[y] * 365.25, 2)}
                    if y - 1 in cnt:
                        q = rate_ratio(cnt[y], expo[y], cnt[y - 1], expo[y - 1])
                        row.update({'rr': r(q['rr'], 3), 'lo': r(q['lo'], 3), 'hi': r(q['hi'], 3)})
                    rows.append(row)
                rr.append({'nation': nat, 'strait': sk, 'inferred': incl, 'years': rows})
    months = []
    y, m = first.year, first.month
    while (y, m) <= (last.year, last.month):
        months.append(f'{y:04d}-{m:02d}')
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return {
        'span': [first.isoformat(), last.isoformat()],
        'months': months,
        'partial_months': [first.isoformat()[:7], last.isoformat()[:7]],
        'straits': sorted(straits.values(), key=lambda s: -s['n']),
        'cells': [[n_, s_, mo, inf, n] for (n_, s_, mo, inf), n in sorted(cells.items())],
        'exposure': expo,
        'rate_ratios': rr,
        'n_passages': len(J['passages']),
        'n_inferred': sum(1 for p in J['passages'] if p['inferred']),
        'n_cnru_rows': sum(1 for p in J['passages'] if p['nation'] == 'CNRU'),
        'n_cnru_both': sum(1 for p in J['passages'] if p['nation'] == 'CNRU' and len(p['nations']) == 2),
        'n_cnru_unknown': sum(1 for p in J['passages'] if p['nation'] == 'CNRU' and not p['nations']),
    }


# ---------------------------------------------------------------- 7. 滞在日数

def kaplan_meier(times):
    """times: [(t, event)]（event=1 は帰りを確認、0 は打ち切り）。段の一覧と分位点。"""
    times = sorted(times)
    n = len(times)
    s, steps, i = 1.0, [[0, 1.0]], 0
    at_risk = n
    while i < n:
        t = times[i][0]
        ev = sum(1 for x in times if x[0] == t and x[1])
        tot = sum(1 for x in times if x[0] == t)
        if ev:
            s *= 1 - ev / at_risk
            steps.append([t, r(s, 4)])
        at_risk -= tot
        i += tot

    def q(p):  # 生存が 1-p 以下になる最初の t
        for t, sv in steps[1:]:
            if sv <= 1 - p + 1e-12:
                return t
        return None
    return {'n': n, 'events': sum(1 for x in times if x[1]), 'censored': sum(1 for x in times if not x[1]),
            'steps': steps, 'q25': q(0.25), 'median': q(0.5), 'q75': q(0.75)}


def build_stay(J, horizon=90):
    st = J['status']
    last = d(st['last_release'])
    by_sid = {p['sighting_id']: p for p in J['passages']}
    trips, used = [], set()
    for t in J['round_trips']:
        used.add(t['out_sighting'])
        used.add(t['in_sighting'])
        if t['direction'] != '進出→帰投' or not t['in_strait'] or t['days_away'] in ('', None):
            continue
        op = by_sid.get(t['out_sighting'])
        trips.append({'year': int(t['out_date'][:4]), 'nation': t['nation'], 'method': t['method'],
                      'days': int(t['days_away']), 'out_date': t['out_date'], 'in_date': t['in_date'],
                      'out_strait': t['out_strait_jp'], 'in_strait': t['in_strait_jp'],
                      'to': (op['waters_to'] or None) if op else None, 'units': t['units'],
                      'cv': t['has_cv'] == '1', 'aor': t['with_aor'] == '1'})
    # 帰りが確認されていない往路：自国側の海から外へ出た通過で、どの往復にも入っていないもの
    open_out = []
    for p in J['passages']:
        if len(p['nations']) != 1:
            continue
        nat = p['nations'][0]
        if p['waters_from'] != HOME[nat] or not p['waters_to'] or p['waters_to'] == HOME[nat]:
            continue
        if p['sighting_id'] in used:
            continue
        age = (last - d(p['date'])).days
        open_out.append({'year': int(p['date'][:4]), 'nation': nat, 'date': p['date'], 'strait': p['strait_jp'],
                         'to': p['waters_to'], 'units': p['units'], 'age': age, 'recent': age <= horizon,
                         'inferred': p['inferred']})
    curves = []
    years = sorted({t['year'] for t in trips} | {o['year'] for o in open_out})
    combos = [(nat, dst, meth, y) for nat in ('ALL', 'CN', 'RU') for dst in DESTS
              for meth in ('all', 'stated', 'hull-match') for y in ['all'] + years]
    for nat, dst, meth, y in combos:
        sel = [t for t in trips if (nat == 'ALL' or t['nation'] == nat) and (meth == 'all' or t['method'] == meth)
               and (y == 'all' or t['year'] == y) and (dst == 'all' or dest_of(t['to']) == dst)]
        cen = [o for o in open_out if o['recent'] and (nat == 'ALL' or o['nation'] == nat)
               and (y == 'all' or o['year'] == y) and (dst == 'all' or dest_of(o['to']) == dst)]
        # 打ち切り（最近の、帰りがまだ確認されていない往路）は、どちらの方法で組になるか分からないので
        # 明記・推定のどちらの曲線にも入れる（ページの手法に明記）
        tm = [(t['days'], 1) for t in sel] + [(o['age'], 0) for o in cen]
        if not tm:
            continue
        km = kaplan_meier(tm)
        curves.append({'nation': nat, 'dest': dst, 'method': meth, 'year': y, **km,
                       'empirical_median': median([t['days'] for t in sel])})
    dest_counts = {}  # 「国:行先」→ 組の数（行先の記載がない組は「不明」）
    for t in trips:
        k = f"{t['nation']}:{t['to'] or '不明'}"
        dest_counts[k] = dest_counts.get(k, 0) + 1
    return {'last_release': st['last_release'], 'horizon': horizon, 'trips': trips, 'open_out': open_out,
            'curves': curves, 'years': years,
            'dest_counts': dict(sorted(dest_counts.items())),
            'counts': {'trips': len(trips), 'stated': sum(1 for t in trips if t['method'] == 'stated'),
                       'hull_match': sum(1 for t in trips if t['method'] == 'hull-match'),
                       'open_out': len(open_out), 'open_recent': sum(1 for o in open_out if o['recent'])}}


# ---------------------------------------------------------------- 8. 中露の共同行動の候補

def same_direction(p, q):
    if p['waters_to'] and q['waters_to']:
        return p['waters_to'] == q['waters_to']
    if p['dir'] and q['dir']:
        return p['dir'] == q['dir']
    return None


def joint_pairs(passages, window=2, shift=0):
    """中国艦だけの行とロシア艦だけの行の組（同じ海峡・±window 日・同じ向き）。shift はロシア側の日付をずらす（偽の比較用）。"""
    cn = [p for p in passages if p['nations'] == ['CN']]
    ru = [p for p in passages if p['nations'] == ['RU']]
    by = {}
    for q in ru:
        by.setdefault(q['strait'], []).append((d(q['date']) + timedelta(days=shift), q))
    out = []
    for p in cn:
        dp = d(p['date'])
        for dq, q in by.get(p['strait'], []):
            gap = (dq - dp).days
            if abs(gap) <= window:
                out.append((p, q, gap, same_direction(p, q)))
    return out


def build_joint(J, window=2):
    P = J['passages']
    stated, together = [], []
    for p in P:
        if p['nation'] != 'CNRU' or len(p['nations']) != 2:
            continue
        rec = {'date': p['date'], 'strait': p['strait_jp'], 'dir': p['dir'], 'to': p['waters_to'], 'units': p['units'],
               'title': p['title'], 'url': p['url'], 'inferred': p['inferred']}
        (stated if '共同' in p['title'] else together).append(rec)
    pairs = joint_pairs(P, window)
    cand, unknown = [], []
    for p, q, gap, same in pairs:
        if same is False:
            continue
        rec = {'cn_date': p['date'], 'ru_date': q['date'], 'gap': gap, 'strait': p['strait_jp'],
               'dir': p['dir'] or q['dir'], 'to': p['waters_to'] or q['waters_to'],
               'cn_units': p['units'], 'ru_units': q['units'], 'cn_url': p['url'], 'ru_url': q['url'],
               'same_release': p['release_id'] == q['release_id'], 'inferred': p['inferred'] or q['inferred']}
        (cand if same else unknown).append(rec)
    cand.sort(key=lambda x: x['cn_date'])
    # 偶然の目安：ロシア側の日付を 30〜180 日（前後）ずらしたときの候補の数
    shifts = [s for s in range(-180, 181, 3) if abs(s) >= 30]
    placebo = [sum(1 for *_, same in joint_pairs(P, window, s) if same) for s in shifts]
    by_year = {}
    for c in cand:
        by_year[c['cn_date'][:4]] = by_year.get(c['cn_date'][:4], 0) + 1
    return {'window': window, 'stated': stated, 'together': together, 'candidates': cand, 'unknown_direction': unknown,
            'by_year': by_year,
            'placebo': {'shifts': [shifts[0], shifts[-1], 3], 'n': len(placebo), 'median': median(placebo),
                        'q05': quantile(placebo, 0.05), 'q95': quantile(placebo, 0.95), 'max': max(placebo),
                        'observed': len(cand)}}


# ---------------------------------------------------------------- 9. グレーゾーンの一覧

def build_grayzone(G, M):
    mnd = M['daily']
    days = sorted(set(G['daily']) | {x for x, v in mnd.items()})
    out = []
    for x in days:
        g = G['daily'].get(x, {})
        m = mnd.get(x, {})
        out.append([x.isoformat(), g.get('cz'), g.get('ts'), m.get('official_ships') if x in mnd else None,
                    1 if x in mnd else 0])
    launches_ready = any(os.path.exists(os.path.join(LAUNCHES, f)) for f in ('events.json', 'notices.json'))
    return {'jcg_available': G['available'], 'jcg_span': [min(G['daily']).isoformat(), max(G['daily']).isoformat()] if G['available'] else None,
            'mnd_span': [min(mnd).isoformat(), max(mnd).isoformat()] if mnd else None,
            'jcg_not_published': (G['status'] or {}).get('not_published'),
            # 月合計と突き合わせられなかった月（check=no_totals、例：2014-02 は PDF に合計行がない）。値はあるが未検算
            'jcg_unverified_months': sorted({x.strftime('%Y-%m') for x, g in G['daily'].items() if g.get('check') == 'no_totals'}),
            'fields': ['date', 'jcg_contiguous', 'jcg_territorial', 'mnd_official_ships', 'mnd_report'],
            'days': out, 'launches_ready': launches_ready}


# ---------------------------------------------------------------- 10. 同時の上振れ

def rolling_count(dates, x, k):
    return sum(1 for z in dates if x - timedelta(days=k - 1) <= z <= x)


def jso_daily_count(J, nation, straits=None, k=7):
    """通過の k 日の合計（日ごと）。最後の公表日の前日までを有効とし、それ以降は None。"""
    first, last = d(J['status']['first_release']), d(J['status']['last_release']) - timedelta(days=1)
    ds = sorted(d(p['date']) for p in J['passages'] if nation in p['nations']
                and (straits is None or p['strait'] in straits))
    out, cnt = {}, {}
    for z in ds:
        cnt[z] = cnt.get(z, 0) + 1
    run = 0
    for x in days_between(first, last):
        run += cnt.get(x, 0) - cnt.get(x - timedelta(days=k), 0)
        out[x] = run
    return out


def joint_daily(J, k=30):
    first, last = d(J['status']['first_release']), d(J['status']['last_release']) - timedelta(days=1)
    ds = {}
    for p in J['passages']:
        if p['nation'] == 'CNRU' and len(p['nations']) == 2:
            ds[d(p['date'])] = ds.get(d(p['date']), 0) + 1
    out, run = {}, 0
    for x in days_between(first, last):
        run += ds.get(x, 0) - ds.get(x - timedelta(days=k), 0)
        out[x] = run
    return out


def indicators(J, G, M):
    mnd = M['daily']
    mser = lambda m: {x: v[m] for x, v in mnd.items()}
    sw = ['miyako', 'yonaguni_taiwan', 'yonaguni_iriomote', 'osumi', 'amami_yokoate']
    ind = {
        'mnd_aircraft': ('台湾国防部：軍機（延べ機数）', '日', mser('aircraft')),
        'mnd_median': ('台湾国防部：中間線越え等（延べ機数）', '日', mser('median_line_or_adiz')),
        'mnd_vessels': ('台湾国防部：艦艇', '日', mser('plan_vessels')),
        'mnd_official': ('台湾国防部：公務船', '日', mser('official_ships')),
        'jso_cn_sw': ('統幕：中国艦の南西諸島の海峡の通過（7日計）', '7日計', jso_daily_count(J, 'CN', sw)),
        'jso_cn': ('統幕：中国艦の海峡の通過（7日計、全海峡）', '7日計', jso_daily_count(J, 'CN')),
        'jso_ru': ('統幕：ロシア艦の海峡の通過（7日計、全海峡）', '7日計', jso_daily_count(J, 'RU')),
        'jso_joint': ('統幕：中露が同じ記述で通過（30日計）', '30日計', joint_daily(J)),
        'jcg_cz': ('海保：尖閣の接続水域の中国海警等（隻数）', '日', {x: v['cz'] for x, v in G['daily'].items()}),
        'jcg_ts': ('海保：尖閣の領海侵入（隻数）', '日', {x: v['ts'] for x, v in G['daily'].items()}),
    }
    return ind


SCENARIOS = [
    {'key': 'taiwan', 'label': '台湾周辺の大規模演習',
     'note': '台湾国防部の日報の4指標と、統幕の南西諸島の海峡の中国艦の通過。演習の有無を判定するものではない。',
     'indicators': ['mnd_aircraft', 'mnd_median', 'mnd_vessels', 'mnd_official', 'jso_cn_sw']},
    {'key': 'japan', 'label': '日本周辺の中露の行動',
     'note': '統幕の中国艦・ロシア艦の通過と中露の同じ記述、海保の尖閣の数。',
     'indicators': ['jso_cn', 'jso_ru', 'jso_joint', 'jcg_cz', 'jcg_ts']},
]


def build_iw(J, G, M, start=date(2022, 11, 1), window=365, min_obs=180, q=90.0):
    ind = indicators(J, G, M)
    ends = [max(s) for _, _, s in ind.values() if s]
    end = max(ends)
    days = days_between(start, end)
    meta, pct = {}, {}
    for key, (label, unit, ser) in ind.items():
        vals, ps = [], []
        for x in days:
            v = ser.get(x)
            ref = [ser[z] for z in days_between(x - timedelta(days=window), x - timedelta(days=1))
                   if ser.get(z) is not None] if v is not None else []
            p = midrank_percentile(v, ref) if len(ref) >= min_obs else None
            vals.append(v)
            ps.append(r(p, 1))
        avail = [x for x, v in ser.items() if v is not None]
        meta[key] = {'label': label, 'unit': unit, 'first': min(avail).isoformat() if avail else None,
                     'last': max(avail).isoformat() if avail else None}
        pct[key] = {'v': vals, 'p': ps}
    scen = []
    for s in SCENARIOS:
        hit, avail = [], []
        for i in range(len(days)):
            ks = [k for k in s['indicators'] if pct[k]['p'][i] is not None]
            avail.append(len(ks))
            hit.append(sum(1 for k in ks if pct[k]['p'][i] > q))
        scen.append({**s, 'above': hit, 'available': avail})
    return {'start': days[0].isoformat(), 'end': days[-1].isoformat(), 'window': window, 'min_obs': min_obs,
            'threshold': q, 'indicators': meta, 'series': pct, 'scenarios': scen}


# ---------------------------------------------------------------- 書き出し

def main(argv=None):
    J = load_jso()
    G = load_jcg()
    M = load_mnd()
    srcs = {'jso': jso_source(J), 'jcg': jcg_source(G), 'mnd': mnd_source(M)}
    jmeta = {'limitations': J['meta'].get('limitations', []), 'method_notes': J['meta'].get('method_notes', [])}
    common = lambda keys: {'algo_version': ALGO_VERSION, 'sources': [srcs[k] for k in keys]}
    payloads = {
        'heatmap.json': {**common(['jso']), **build_heatmap(J), 'jso_meta': jmeta},
        'stay.json': {**common(['jso']), **build_stay(J)},
        'joint.json': {**common(['jso']), **build_joint(J)},
        'grayzone.json': {**common(['jcg', 'mnd']), **build_grayzone(G, M),
                          'jcg_meta': {'missing_values': G['meta'].get('missing_values'),
                                       'definition_changes': G['meta'].get('definition_changes', []),
                                       'unverified_months': G['meta'].get('unverified_months', []),
                                       'limitations': G['meta'].get('limitations', [])}},
        'iw.json': {**common(['mnd', 'jso', 'jcg']), **build_iw(J, G, M)},
    }
    changed = [n for n, p in payloads.items() if write_if_changed(n, p)]
    print('changed:', ', '.join(changed) or 'none')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
