"""confirm-1: the tables of the triage design (docs/data/sat/eval/confirm-design.md, section 2 and section 5 item 1) from the
published per-cell file docs/data/sat/eval/pres3_oos.csv alone. numpy only; no working directory, no network.

Which cells may be settled automatically ("auto") and which go to the owner ("routed"), for a threshold pair on the pres-3o
score p (the configuration without the S1 columns: p_opt_lopo / p_opt_fwd), and for the recommended rule v1:

    auto  <=>  (p <= 0.03 or p >= 0.97)  and  the section has >= 2 owner answers before the day
               and the last of them is <= 120 days old  and  the port is one of SLP, QGZ, QJG, QDH, LSN

Every number is on out-of-sample scores, but the rule itself was chosen on these same cells (the design, section 2.6).
Intervals: bootstrap over scenes (port x day), each interval from its own generator (seed 0), so they differ in the third
decimal from the intervals printed in the design, which drew all its tables from one generator in sequence. Counts, rates
and accuracies are exact.
    python -m sat.pres3.triage [--csv PATH] [--json] [--no-ci]
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import sys

import numpy as np

from . import common, evalcv

CSV = common.SAT / 'eval' / 'pres3_oos.csv'
THRESHOLDS = [0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.5]
GRID = [0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5]             # the nested selection walks this from the strictest
RULE_V1 = dict(lo=0.03, hi=0.97, min_answers=2, max_age_days=120, ports_on=('SLP', 'QGZ', 'QJG', 'QDH', 'LSN'))
COLUMNS = {'opt': ('p_opt_lopo', 'p_opt_fwd'), 'pres3': ('p_pres3_lopo', 'p_pres3_fwd'), 'hist': ('p_hist_lopo', 'p_hist_fwd')}
NO_PREV = 9999.0


def load(path=CSV):
    """The cells of pres3_oos.csv as arrays, plus for every cell the number of the section's answers on earlier days (n_prev)
    and the days since the last of them (age; NO_PREV when there is none). The file holds every answered cell, so these are
    the owner's answers 'before the day' exactly as the daily rule would count them."""
    with open(path, newline='', encoding='utf-8') as fh:
        rows = list(csv.DictReader(fh))
    f = lambda v: float(v) if v != '' else np.nan
    D = dict(port=np.array([r['site_id'] for r in rows]), date=np.array([r['date'] for r in rows]),
             section=np.array([r['section_id'] for r in rows]), ev=np.array([r['in_eval'] == '1' for r in rows]),
             y=np.array([int(r['y']) for r in rows]), ch=np.array([int(r['changed']) for r in rows]))
    for c in rows[0]:
        if c.startswith('p_'): D[c] = np.array([f(r[c]) for r in rows])
    n_prev = np.zeros(len(rows), int); age = np.full(len(rows), NO_PREV); g = {}
    for i, r in enumerate(rows): g.setdefault((r['site_id'], r['section_id']), []).append(i)
    for v in g.values():
        v.sort(key=lambda i: rows[i]['date'])
        for j, i in enumerate(v):
            earlier = [t for t in v[:j] if rows[t]['date'] < rows[i]['date']]
            n_prev[i] = len(earlier)
            if earlier: age[i] = (dt.date.fromisoformat(rows[i]['date']) - dt.date.fromisoformat(rows[earlier[-1]]['date'])).days
    D['n_prev'] = n_prev; D['age'] = age; D['scene'] = np.array([p + '|' + d for p, d in zip(D['port'], D['date'])])
    return D


def p_of(D, model, fold):
    return D[COLUMNS[model][0 if fold == 'lopo' else 1]]


def band(p, lo, hi=None):
    hi = 1 - lo if hi is None else hi
    with np.errstate(invalid='ignore'):
        return (p <= lo) | (p >= hi)


def auto_v1(D, p, rule=RULE_V1, answers=True, age=True, ports=True):
    """The cells rule v1 settles automatically (a boolean array). The three switches drop a condition (the 'one more
    condition at a time' table)."""
    a = band(p, rule['lo'], rule['hi'])
    if answers: a &= D['n_prev'] >= rule['min_answers']
    if age: a &= D['age'] <= rule['max_age_days']
    if ports: a &= np.isin(D['port'], rule['ports_on'])
    return a


def boot(unit, den, a, cor, B=1000, seed=0):
    """95% intervals of (accuracy of the auto cells, share routed) when the units (scenes, or ports) among the cells of den
    are resampled."""
    rng = np.random.default_rng(seed); ids = np.where(den)[0]; us = np.unique(unit[ids]); by = [ids[unit[ids] == u] for u in us]
    acc = []; routed = []
    for _ in range(B):
        ix = np.concatenate([by[t] for t in rng.integers(0, len(us), len(us))]); k = a[ix]
        if k.any(): acc.append(cor[ix][k].mean())
        routed.append(1 - k.mean())
    f = lambda v: [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)] if v else None
    return f(acc), f(routed)


def line(D, den, a, p, B=1000, seed=0, unit='scene'):
    """One row: the cells of den, split by a (auto) / not a (routed). 'combined_if_owner_right' counts every routed cell as
    right; it is not the model's accuracy."""
    a = a & den; cor = (p >= 0.5) == (D['y'] == 1); n = int(den.sum()); na = int(a.sum()); rest = den & ~a
    r = dict(n=n, routed=n - na, routed_frac=round(1 - na / n, 3) if n else None, auto=na,
             auto_acc=round(float(cor[a].mean()), 3) if na else None, auto_errors=int((~cor[a]).sum()),
             routed_model_acc=round(float(cor[rest].mean()), 3) if rest.any() else None,
             combined_if_owner_right=round(float((cor[a].sum() + rest.sum()) / n), 3) if n else None)
    if B and n: r['auto_acc_ci'], r['routed_ci'] = boot(D[unit], den, a, cor, B, seed)
    return r


def eval_cells(D, p):
    return D['ev'] & ~np.isnan(p)


def t_calibration(D, model='opt'):
    """2.2: per band of p, the mean p and the share of cells that were present."""
    bins = [0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95, 0.98, 1.0001]; out = {}
    for fold in ('lopo', 'fwd'):
        p = p_of(D, model, fold); den = eval_cells(D, p); rows = []
        for lo, hi in zip(bins, bins[1:]):
            k = den & (p >= lo) & (p < hi)
            if k.any(): rows.append(dict(lo=lo, hi=min(hi, 1.0), n=int(k.sum()), mean_p=round(float(p[k].mean()), 3), observed=round(float(D['y'][k].mean()), 3)))
        out[fold] = rows
    return out


def t_thresholds(D, model='opt', B=1000, cells='eval'):
    """2.3: symmetric thresholds, auto = p <= lo or p >= 1 - lo."""
    out = {}
    for fold in ('lopo', 'fwd'):
        p = p_of(D, model, fold); den = eval_cells(D, p) if cells == 'eval' else ~np.isnan(p)
        out[fold] = {f'{lo}': line(D, den, band(p, lo), p, B) for lo in THRESHOLDS}
    return out


def t_sides(D, lo=0.05, model='opt'):
    """2.3: the absent side and the present side of the confident band."""
    out = {}
    for fold in ('lopo', 'fwd'):
        p = p_of(D, model, fold); den = eval_cells(D, p); cor = (p >= 0.5) == (D['y'] == 1)
        out[fold] = {nm: dict(n=int(k.sum()), acc=round(float(cor[k].mean()), 3) if k.any() else None)
                     for nm, k in (('absent', den & (p <= lo)), ('present', den & (p >= 1 - lo)))}
    return out


def t_changed(D, lo=0.05, model='opt', auto=None):
    """2.4: changed since the previous answer / the same / no previous answer."""
    out = {}
    for fold in ('lopo', 'fwd'):
        p = p_of(D, model, fold); den = eval_cells(D, p); a = band(p, lo) if auto is None else auto(D, p)
        out[fold] = {nm: line(D, den & (D['ch'] == v), a, p, 0) for nm, v in (('changed', 1), ('same', 0), ('no_prev', -1))}
        cor = (p >= 0.5) == (D['y'] == 1); e = den & a & ~cor
        out[fold]['auto_errors_on_changed'] = [int((e & (D['ch'] == 1)).sum()), int(e.sum())]
    return out


def t_age(D, lo=0.05, model='opt'):
    """2.4: by the days since the section's previous answer (cells that have one)."""
    out = {}
    for fold in ('lopo', 'fwd'):
        p = p_of(D, model, fold); den = eval_cells(D, p) & (D['age'] < NO_PREV); rows = []
        for a0, a1 in ((0, 21), (21, 45), (45, 90), (90, NO_PREV)):
            k = den & (D['age'] >= a0) & (D['age'] < a1); r = line(D, k, band(p, lo), p, 0)
            rows.append(dict(days=[a0, None if a1 == NO_PREV else a1], n=r['n'], changed_rate=round(float((D['ch'][k] == 1).mean()), 3),
                             auto=r['auto'], auto_acc=r['auto_acc']))
        out[fold] = rows
    return out


def t_answers(D, lo=0.05, model='opt', fold='lopo'):
    """2.4: by the number of the section's earlier answers."""
    p = p_of(D, model, fold); out = {}
    for nm, den in (('eval', eval_cells(D, p)), ('all', ~np.isnan(p))):
        out[nm] = []
        for k0, k1 in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 6), (6, 10 ** 6)):
            r = line(D, den & (D['n_prev'] >= k0) & (D['n_prev'] < k1), band(p, lo), p, 0)
            out[nm].append(dict(answers=[k0, None if k1 > 99 else k1 - 1], n=r['n'], auto=r['auto'], auto_acc=r['auto_acc']))
    return out


def t_rule(D, rule=RULE_V1, model='opt', B=2000):
    """2.5: rule v1 on the evaluation cells; by port; with the conditions added one at a time."""
    out = {}
    for fold in ('lopo', 'fwd'):
        p = p_of(D, model, fold); den = eval_cells(D, p); a = auto_v1(D, p, rule); r = line(D, den, a, p, B)
        if B: r['auto_acc_ci_ports'], r['routed_ci_ports'] = boot(D['port'], den, a & den, (p >= 0.5) == (D['y'] == 1), B)
        r['changed'] = line(D, den & (D['ch'] == 1), a, p, 0); r['same'] = line(D, den & (D['ch'] == 0), a, p, 0)
        r['by_port'] = {P: line(D, den & (D['port'] == P), a, p, 0) for P in common.PORTS}
        r['steps'] = {nm: line(D, den, auto_v1(D, p, rule, **kw), p, B)
                      for nm, kw in (('band', dict(answers=False, age=False, ports=False)), ('+answers', dict(age=False, ports=False)),
                                     ('+age', dict(ports=False)), ('+ports (v1)', {}))}
        r['looser_0.05'] = line(D, den, auto_v1(D, p, dict(rule, lo=0.05, hi=0.95)), p, B)
        out[fold] = r
    return out


def pick(p, y, sel, target, grid=GRID):
    """The loosest threshold of the grid whose auto cells among sel reach the target accuracy (walked from the strictest;
    stops at the first that fails)."""
    cor = (p >= 0.5) == (y == 1); best = grid[0]
    for t in grid:
        a = sel & band(p, t)
        if a.any() and cor[a].mean() >= target: best = t
        else: break
    return best


def t_nested(D, model='opt', B=1000):
    """2.6: choose the threshold on some cells, apply it to cells not used for the choice."""
    out = []
    for target in (0.97, 0.98):
        p = p_of(D, model, 'lopo'); den = eval_cells(D, p); a = np.zeros(len(p), bool); ts = {}
        for P in common.PORTS:
            ts[P] = pick(p, D['y'], den & (D['port'] != P), target); a |= den & (D['port'] == P) & band(p, ts[P])
        out.append(dict(how='leave one port out', target=target, thresholds=ts, **line(D, den, a, p, B)))
        p = p_of(D, model, 'fwd'); den = eval_cells(D, p)
        for cut in ('2026-03-01', '2026-06-01', '2026-09-01'):
            late = den & (D['date'] >= cut); t = pick(p, D['y'], den & (D['date'] < cut), target)
            out.append(dict(how=f'chosen before {cut}', target=target, thresholds=t, **line(D, late, band(p, t), p, B)))
    return out


def t_compare(D):
    """2.7: the same cells with the other scores in the file (history only; the adopted configuration with S1)."""
    out = {}
    for fold in ('lopo', 'fwd'):
        out[fold] = {}
        for model, lo in (('opt', 0.03), ('pres3', 0.05), ('hist', 0.05)):
            p = p_of(D, model, fold); den = eval_cells(D, p); a = band(p, lo); cor = (p >= 0.5) == (D['y'] == 1); c = den & (D['ch'] == 1)
            m = evalcv.metrics(D['y'][den], p[den])
            out[fold][f'{model} {lo}'] = dict(all_acc=round(m['acc'], 3), routed_frac=round(float(1 - (a & den).sum() / den.sum()), 3),
                                              auto_acc=round(float(cor[a & den].mean()), 3), changed=int(c.sum()), changed_auto_wrong=int((c & a & ~cor).sum()))
    return out


def t_generic(D):
    """Section 5, item 1: the generic judgement (no answers of the port) is not good enough even in its confident band."""
    p = D['p_gen_lopo']; den = eval_cells(D, p); cor = (p >= 0.5) == (D['y'] == 1); out = {}
    for lo in (0.03, 0.05):
        k = den & (p <= lo); kp = den & (p >= 1 - lo)
        out[f'{lo}'] = dict(absent_n=int(k.sum()), absent_acc=round(float(cor[k].mean()), 3) if k.any() else None, present_n=int(kp.sum()))
    return out


def tables(D, ci=True):
    b1, b2 = (1000, 2000) if ci else (0, 0)
    return {'cells': dict(all=len(D['y']), eval=int(D['ev'].sum()), eval_fwd=int(eval_cells(D, D['p_opt_fwd']).sum())),
            'rule_v1': dict(RULE_V1, ports_on=list(RULE_V1['ports_on'])),
            '2.2 calibration (pres-3o)': t_calibration(D), '2.3 thresholds (pres-3o)': t_thresholds(D, 'opt', b1),
            '2.3 thresholds (adopted, with S1: an upper reference)': t_thresholds(D, 'pres3', b1),
            '2.3 all answered cells (pres-3o)': t_thresholds(D, 'opt', 0, 'all'), '2.3 sides at 0.05/0.95': t_sides(D),
            '2.4 changed cells at 0.05/0.95': t_changed(D), '2.4 by days since the previous answer': t_age(D),
            '2.4 by number of earlier answers (lopo)': t_answers(D), '2.5 rule v1': t_rule(D, B=b2), '2.6 nested selection': t_nested(D, B=b1),
            '2.7 comparison': t_compare(D), '5 generic judgement, confident band': t_generic(D)}


def _ci(v): return '' if not v else f' [{v[0]:.3f}, {v[1]:.3f}]'


def _row(name, r):
    f = lambda v: '–' if v is None else f'{v:.3f}'
    return (f"| {name} | {r['n']} | {r['routed']} | {f(r['routed_frac'])}{_ci(r.get('routed_ci'))} | {r['auto']} | {f(r['auto_acc'])}{_ci(r.get('auto_acc_ci'))} "
            f"| {r['auto_errors']} | {f(r['routed_model_acc'])} | {f(r['combined_if_owner_right'])} |")


HEAD = '| | 升 | 回す | 回す割合 [95% CI] | 自動確定 | 自動確定の正答率 [95% CI] | 誤り | 回した升の判定器の正答率 | 合計（人＝正答） |\n|---|---|---|---|---|---|---|---|---|'


def render(T):
    o = [f"confirm-1 の仕分けの表（{CSV.name} から。升 {T['cells']['all']}、評価票の升 {T['cells']['eval']}、前進 {T['cells']['eval_fwd']}）", '']
    for fold in ('lopo', 'fwd'):
        o += [f'## 2.2 較正（pres-3o、{fold}）', '| p の帯 | 升 | 平均 p | 実際 |', '|---|---|---|---|']
        o += [f"| {r['lo']}〜{r['hi']} | {r['n']} | {r['mean_p']:.3f} | {r['observed']:.3f} |" for r in T['2.2 calibration (pres-3o)'][fold]] + ['']
    for key in ('2.3 thresholds (pres-3o)', '2.3 thresholds (adopted, with S1: an upper reference)', '2.3 all answered cells (pres-3o)'):
        for fold in ('lopo', 'fwd'):
            o += [f'## {key}、{fold}', HEAD] + [_row(f'{lo}／{round(1 - float(lo), 2)}', r) for lo, r in T[key][fold].items()] + ['']
    o += ['## 2.3 在の側・不在の側（pres-3o、0.05／0.95）', json.dumps(T['2.3 sides at 0.05/0.95'], ensure_ascii=False), '']
    for fold in ('lopo', 'fwd'):
        t = T['2.4 changed cells at 0.05/0.95'][fold]
        o += [f'## 2.4 変わった升（pres-3o、0.05／0.95、{fold}）', HEAD] + [_row(nm, t[nm]) for nm in ('changed', 'same', 'no_prev')]
        o += [f"自動確定の誤りのうち変わった升：{t['auto_errors_on_changed'][0]}／{t['auto_errors_on_changed'][1]}", '']
        o += [f'## 2.4 直前の答えからの日数（{fold}）', '| 日数 | 升 | 変わった割合 | 自動確定 | 正答率 |', '|---|---|---|---|---|']
        o += [f"| {r['days'][0]}〜{r['days'][1] or ''} | {r['n']} | {r['changed_rate']:.3f} | {r['auto']} | {r['auto_acc']} |" for r in T['2.4 by days since the previous answer'][fold]] + ['']
    o += ['## 2.4 区間の過去の答えの数（pres-3o、LOPO、0.05／0.95）', json.dumps(T['2.4 by number of earlier answers (lopo)'], ensure_ascii=False), '']
    for fold in ('lopo', 'fwd'):
        r = T['2.5 rule v1'][fold]
        o += [f'## 2.5 規則 v1（{fold}）', HEAD, _row('v1', r)]
        if r.get('auto_acc_ci_ports'): o += [f"港を単位にした CI：自動確定の正答率{_ci(r['auto_acc_ci_ports'])}、回す割合{_ci(r['routed_ci_ports'])}"]
        o += [_row('変わった升', r['changed']), _row('同じ升', r['same'])] + [_row(P, v) for P, v in r['by_port'].items()]
        o += [_row(nm, v) for nm, v in r['steps'].items()] + [_row('（比較）0.05／0.95 の v1', r['looser_0.05']), '']
    o += ['## 2.6 入れ子の選び方（pres-3o）', '| 選び方 | 目標 | 閾値 | ' + HEAD.split('\n')[0][4:], '|---|---|---|' + HEAD.split('\n')[1][5:]]
    o += [f"| {r['how']} | {r['target']} | {r['thresholds']} " + _row('', r)[3:] for r in T['2.6 nested selection']] + ['']
    o += ['## 2.7 比較', json.dumps(T['2.7 comparison'], ensure_ascii=False), '', '## 5 汎用の判定の確信帯（p_gen_lopo）', json.dumps(T['5 generic judgement, confident band'], ensure_ascii=False)]
    return '\n'.join(o)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--csv', default=str(CSV)); ap.add_argument('--json', action='store_true'); ap.add_argument('--no-ci', action='store_true')
    a = ap.parse_args(argv); T = tables(load(a.csv), ci=not a.no_ci)
    print(json.dumps(T, ensure_ascii=False, indent=1) if a.json else render(T))
    return 0


if __name__ == '__main__':
    sys.exit(main())
