"""Step 6: the numbers of the adopted configuration: <work>/pres3_results.json and <work>/pres3_oos.csv.

With --publish also writes docs/data/sat/eval/pres3.json (the same numbers plus version, time, sources and limits; NaN -> null)
and docs/data/sat/eval/pres3_oos.csv. No other published file is touched.
    python -m sat.pres3.final --work DIR [--publish]
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import math
import pickle
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np

from . import VERSION, common, evalcv, fuse

KEY = ('opt+diff+S1', 1)
NOTE = 'gen＝港を抜いた汎用の判定（答えを使わない特徴 545、HGB）。pres-3＝gen のロジット＋区間の過去の答え（見本・履歴）＋像の差分＋現行 S1 の p（HGB）。'
NOTE_PUBLISHED = NOTE[:-len('現行 S1 の p（HGB）。')] + '現行 S1 の p・対の S1 撮影の帯の VH 平均 xh・時刻差 dt_h（HGB）。'
OOS_COLUMNS = ['site_id', 'date', 'section_id', 'in_eval', 'y', 'changed', 'p_current_s1', 'p_gen_lopo', 'p_hist_lopo', 'p_pres3_lopo', 'p_hist_fwd', 'p_pres3_fwd']


def acc_ci(C, y, idx, p, B=2000):
    idx = [i for i in idx if not np.isnan(p[i])]
    if not idx: return None
    yy = y[idx]; m = evalcv.metrics(yy, p[idx]); lo, hi = evalcv.boot_ci(C, idx, yy, p[idx], B=B)
    return dict(n=m['n'], acc=round(m['acc'], 3), ci=[round(lo, 3), round(hi, 3)], recall=round(m['rec'], 3), precision=round(m['pre'], 3), f1=round(m['f1'], 3),
                auc=round(m['auc'], 3), brier=round(m['brier'], 3), tp=m['tp'], fp=m['fp'], fn=m['fn'], tn=m['tn'])


def diff_ci(C, y, idx, p, q, B=2000, seed=0):
    """Paired difference of accuracy (p minus q) on the same cells, bootstrap over scenes (port x day)."""
    idx = [i for i in idx if not np.isnan(p[i]) and not np.isnan(q[i])]; rng = np.random.default_rng(seed)
    keys = [C[i]['port'] + C[i]['date'] for i in idx]; uk = sorted(set(keys)); pos = {k: [j for j, kk in enumerate(keys) if kk == k] for k in uk}
    a = ((p[idx] >= 0.5) == y[idx]).astype(float); b = ((q[idx] >= 0.5) == y[idx]).astype(float); v = []
    for _ in range(B):
        j = np.concatenate([pos[uk[t]] for t in rng.integers(0, len(uk), len(uk))]); v.append(a[j].mean() - b[j].mean())
    return dict(n=len(idx), diff=round(float(a.mean() - b.mean()), 3), ci=[round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)])


def results(C, y, lab, ch, ev, O, G):
    """O: fuse_out (fold kind -> configuration -> p), G: gen_cache. Returns (R, the per-cell columns)."""
    PORTS = common.PORTS
    pl = O['LOPO'][KEY]; pf = O['FWD'][KEY]; hl = O['LOPO'][('hist', 1)]; hf = O['FWD'][('hist', 1)]
    gl = np.full(len(C), np.nan)
    for (fn, name), s in G.items():
        if fn == 'LOPO':
            te = [i for i in lab if C[i]['port'] == name]; gl[te] = 1 / (1 + np.exp(-s[te]))
    pc = np.array([c['p_cur'] if c['p_cur'] is not None else np.nan for c in C])
    R = {'config': {'key': KEY, 'note': NOTE}}
    for fn, p, h in (('lopo', pl, hl), ('fwd', pf, hf)):
        te = [i for i in ev if not np.isnan(p[i])]; r = {}
        r['pres3'] = acc_ci(C, y, te, p); r['current_s1'] = acc_ci(C, y, te, pc); r['history_only'] = acc_ci(C, y, te, h)
        if fn == 'lopo': r['gen_no_answers'] = acc_ci(C, y, te, gl); r['gen_changed'] = acc_ci(C, y, [i for i in te if ch[i] == 1], gl)
        r['pres3_minus_current'] = diff_ci(C, y, te, p, pc); r['pres3_minus_history'] = diff_ci(C, y, te, p, h)
        for nm, v in (('changed', 1), ('same', 0), ('no_prev', -1)):
            s = [i for i in te if ch[i] == v]; r[nm] = {'pres3': acc_ci(C, y, s, p, 500), 'current_s1': acc_ci(C, y, s, pc, 500), 'history_only': acc_ci(C, y, s, h, 500)}
        r['by_port'] = {P: {'pres3': acc_ci(C, y, [i for i in te if C[i]['port'] == P], p, 500), 'current_s1': acc_ci(C, y, [i for i in te if C[i]['port'] == P], pc, 500)} for P in PORTS}
        r['bands'] = {}
        for lo, hi in ((0.05, 0.95), (0.1, 0.9), (0.2, 0.8)):
            m = [i for i in te if p[i] <= lo or p[i] >= hi]; rest = [i for i in te if not (p[i] <= lo or p[i] >= hi)]
            r['bands'][f'{lo}_{hi}'] = {'coverage': round(len(m) / len(te), 3), 'confident': acc_ci(C, y, m, p), 'rest': acc_ci(C, y, rest, p, 500),
                                        'changed_in_band': sum(ch[i] == 1 for i in m), 'changed_total': sum(ch[i] == 1 for i in te)}
        r['all_labelled_cells'] = acc_ci(C, y, [i for i in lab if not np.isnan(p[i])], p)
        R[fn] = r
    return R, dict(pc=pc, gl=gl, hl=hl, pl=pl, hf=hf, pf=pf)


def write_oos(path, C, y, lab, ch, cols):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n'); w.writerow(OOS_COLUMNS)
        fm = lambda v: '' if np.isnan(v) else f'{v:.4f}'
        for i in lab:
            w.writerow([C[i]['port'], C[i]['date'], C[i]['section'], int(C[i]['eval']), int(y[i]), ch[i]] + [fm(cols[k][i]) for k in ('pc', 'gl', 'hl', 'pl', 'hf', 'pf')])


def nan_to_null(o):
    if isinstance(o, float) and math.isnan(o): return None
    if isinstance(o, dict): return {k: nan_to_null(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [nan_to_null(v) for v in o]
    return o


def published(R, generated_utc):
    """The published sheet: the results plus version, time, sources and limits. NaN (a ratio with a zero denominator) is null."""
    head = {
        'version': VERSION, 'status': '試行・研究（毎日の収集の流れには接続していない。公開の数・画面・既存の評価票は変えない）',
        'generated_utc': generated_utc,
        'design': 'docs/data/sat/eval/pres3-design.md', 'code': 'sat/pres3/（README.md に再現の手順）', 'cells': 'docs/data/sat/eval/pres3_oos.csv',
        'target': '海軍の岸壁区間 × 日の在/不在。目標の正答率 0.95 には全升では届いていない（LOPO 0.871、前進 0.893）',
        'truth': '所有者が Sentinel-2 の表示用画像を見て付けた答え（labels/<港>-s2-answers.csv、湛江は labels/ZJG-s2-labels.csv の verified）。入力と同じ撮影',
        'sources': ['Contains modified Copernicus Sentinel data 2023–2026（Sentinel-2 L2A。Element84 Earth Search／AWS Open Data 経由）',
                    '岸壁の線 © OpenStreetMap contributors', '現行 S1 の判定 p：docs/data/sat/eval/presence.json（eval-presence-6）',
                    'S1 の帯の VH 平均 xh・時刻差 dt_h：sat/pres3/inputs/s1_cells.csv'],
        'evaluation': {
            'lopo': '港を 1 つ抜く。汎用の判定と融合の係数は他の 8 港の升で学ぶ。見本・履歴は抜いた港の、判定日より前の答えだけ',
            'fwd': '時間の前進検証。境界 2025-12-01・2026-03-01・2026-06-01・2026-09-01。係数は境界より前の全港の升、見本・履歴は判定日より前の答えだけ',
            'ci': '撮影（港×日）単位のブートストラップ 2,000 回（港別・変わった升などの内訳は 500 回）、95%',
            'cells': '評価の升は評価票 presence.json の s2 の升のうち答えのある 737 升（前進は 2025-12-01 以後の 516 升）。升は 1 つも外していない'},
        'limits': [
            '特徴・構成は同じ升の LOPO・前進の数字を見ながら選んだ。手法の選び方については標本外でない（試した構成の正答率の幅 0.85〜0.905）',
            '正解は所有者が S2 の表示用画像を見た読みで、入力と同じ撮影。独立した正解（高分解能など）ではない',
            'S2 が晴れの日だけ。雲の日・夜は判定できない',
            '答えの無い 14 港は汎用の判定（答えなし）しか使えず、その正答率は lopo.gen_no_answers',
            '確信帯の閾値（0.1/0.9 など）は結果を見てから置いた',
            'S1 は現行の判定 p と、対の S1 撮影の帯の VH 平均 xh・時刻差 dt_h の 3 列だけ。それ以外の S1 の特徴は使っていない',
            '直前の答えから変わった升の正答率は LOPO 0.50、前進 0.65 で、入港・出港の検知には足りない'],
        'null_rule': 'null は計算できない値（分母が 0 の適合率など）。0 は 0 と計算された値',
    }
    out = {**head, **nan_to_null(R)}
    out['config'] = dict(out['config'], note=NOTE_PUBLISHED)
    return out


def publish(R, oos_csv, out_dir, generated_utc):
    """Writes exactly two files in out_dir: pres3.json and pres3_oos.csv (temporary file + replace)."""
    out_dir = Path(out_dir); js = out_dir / 'pres3.json'; cs = out_dir / 'pres3_oos.csv'
    tmp = out_dir / 'pres3.json.tmp'
    tmp.write_text(json.dumps(published(R, generated_utc), ensure_ascii=False, indent=1, allow_nan=False) + '\n', encoding='utf-8'); tmp.replace(js)
    tmp = out_dir / 'pres3_oos.csv.tmp'; shutil.copyfile(oos_csv, tmp); tmp.replace(cs)
    return js, cs


def main(argv=None):
    warnings.filterwarnings('ignore')
    ap = common.parser(__doc__.split('\n')[0]); ap.add_argument('--publish', action='store_true', help='also write docs/data/sat/eval/pres3.json and pres3_oos.csv')
    a = ap.parse_args(argv); work = common.work_dir(a.work)
    C, X, names, lab, y, ch, ev = fuse.load(work)
    O = pickle.load(open(f'{work}/fuse_out.pkl', 'rb')); G = pickle.load(open(f'{work}/gen_cache.pkl', 'rb'))
    R, cols = results(C, y, lab, ch, ev, O, G)
    json.dump(R, open(f'{work}/pres3_results.json', 'w'), ensure_ascii=False, indent=1)
    write_oos(f'{work}/pres3_oos.csv', C, y, lab, ch, cols)
    for fn in ('lopo', 'fwd'):
        r = R[fn]; print(fn, 'pres3', r['pres3']['acc'], r['pres3']['ci'], 'cur', r['current_s1']['acc'], 'hist', r['history_only']['acc'], 'Δcur', r['pres3_minus_current'],
                         'changed', r['changed']['pres3']['acc'], r['changed']['pres3']['ci'], 'same', r['same']['pres3']['acc'])
        print('  ports', {P: (v['pres3']['acc'], v['pres3']['n']) for P, v in r['by_port'].items() if v['pres3']})
        print('  bands', {k: (v['coverage'], v['confident']['acc'], v['confident']['ci']) for k, v in r['bands'].items()})
        if fn == 'lopo': print('  gen no answers', r['gen_no_answers']['acc'], r['gen_no_answers']['ci'], 'changed', r['gen_changed']['acc'])
    if a.publish:
        now = dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        print('published', *publish(R, f'{work}/pres3_oos.csv', common.SAT / 'eval', now))

if __name__ == '__main__':
    sys.exit(main())
