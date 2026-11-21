"""confirm-1 (PR-2): pres-3o one day at a time. Fetch a new scene -> patches -> features -> the score p of fixed coefficients.

pres-3o is the pres-3 configuration without the S1 columns, ('opt', 1): a generic judgement (545 answer-free features) fused
with the exemplar / history features of the section's EARLIER owner answers. Nothing here is wired into a workflow yet (that
is PR-5) and nothing here writes a published file: every path is given on the command line and must be outside docs/.

What a score of day d may see (the rule the tests guard):
    images   the patches of days <= d (the section's reference is made from the clear days strictly before d)
    answers  the owner's answers of days strictly before d; an automatic decision is never an answer
    model    coefficients fitted on answers before the model's cutoff, stored as plain arrays

Three stores, all keyed by (port, section, date), never by a row number:
    patch store   <store>/<port>/<YYYY-MM>.npz    patch [n, 24, 41, 5] float16 + one JSON record per cell (no answers in it)
    model         <model>.npz                     the trees of the two classifiers as arrays, the out-of-fold generic scores of
                                                  the training cells, the feature names, and a JSON header (version, cutoff,
                                                  scikit-learn / numpy versions). numpy alone predicts from it.
    scores        a CSV, one row per cell         site_id, section_id, date, acquired_utc, p, model_version, cloud_shadow, nodata,
                                                  status, answers_through, n_answers: the input of sat/confirm.py (which applies
                                                  the rule). n_answers = the owner's answers the judge used for the cell (the
                                                  section's answers of earlier days that have a patch in the store)

    python -m sat.pres3.daily import  --work DIR --store STORE [--ports ..]       the research patches -> the patch store
    python -m sat.pres3.daily fit     --work DIR --cutoff 2026-11-01 --model M.npz   coefficients (needs scikit-learn)
    python -m sat.pres3.daily fetch   --store STORE --windows DIR [--ports ..] [--days 10]   new scenes (needs rasterio, network)
    python -m sat.pres3.daily score   --store STORE --model M.npz --out scores.csv [--since D] [--until D] [--ports ..]
    python -m sat.pres3.daily check   --work DIR --store STORE [--ports ..] [--bounds ..]   daily == the research forward validation

Contains modified Copernicus Sentinel data (via Element84 Earth Search / AWS Open Data). CDSE is not used.
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

from . import common, feats, labels, model

KIND = 'pres3o'
CELL_KEYS = ('port', 'date', 'section', 'kind', 'sector', 'length_m', 'cloud', 'shadow', 'cirrus', 'nodata', 'tile_cloud', 'scene', 'time')
MAX_CLOUD_SHADOW = 0.2          # a cell is judged when cloud + shadow <= 0.2 and no nodata (the design, section 4: otherwise no_image)
MIN_REFERENCE_DAYS = 6          # feats.features: a section's reference needs 6 earlier clear days
# the score table: the input of sat/confirm.py (its SCORE_FIELDS, same names, same units, then n_answers: the number of the
# owner's answers the judge used, an integer; confirm-1 counts min(the answer file's count, n_answers) for the rule's "2 answers").
# --diagnostics adds the last two.
SCORE_COLUMNS = ['site_id', 'section_id', 'date', 'acquired_utc', 'p', 'model_version', 'cloud_shadow', 'nodata', 'status', 'answers_through',
                 'n_answers']
DIAGNOSTIC_COLUMNS = ['n_reference_days', 'scene_id']
NODE_FIELDS = ('value', 'feature_idx', 'num_threshold', 'missing_go_to_left', 'left', 'right', 'is_leaf')


def outside_repo(path, what):
    p = Path(path).resolve()
    if common.REPO in (p, *p.parents): raise SystemExit(f'{what} must be outside the repository: {p}')
    return p


# ---------------------------------------------------------------- the patch store

class Store:
    """<root>/<port>/<YYYY-MM>.npz. One record per (section, date); a record put again replaces the old one."""

    def __init__(self, root):
        self.root = Path(root)

    def months(self, port):
        return sorted(p.stem for p in (self.root / port).glob('????-??.npz'))

    def _read(self, port, month):
        f = self.root / port / f'{month}.npz'
        if not f.exists(): return [], np.zeros((0, labels.NA, 41, 5), np.float16)
        z = np.load(f, allow_pickle=False)
        return json.loads(str(z['cells'])), z['patch']

    def put(self, port, cells, patches):
        """cells: records with at least CELL_KEYS (answers and evaluation fields are dropped: the store holds images only)."""
        by = collections.defaultdict(list)
        for c, p in zip(cells, patches):
            if c['port'] != port: raise ValueError(f"a cell of {c['port']} in the store of {port}")
            by[c['date'][:7]].append(({k: c[k] for k in CELL_KEYS}, np.asarray(p, np.float16)))
        for month, new in by.items():
            C, P = self._read(port, month); rec = {(c['section'], c['date']): (c, p) for c, p in zip(C, P)}
            for c, p in new: rec[(c['section'], c['date'])] = (c, p)
            keys = sorted(rec, key=lambda k: (k[1], k[0])); f = self.root / port / f'{month}.npz'; f.parent.mkdir(parents=True, exist_ok=True)
            tmp = f.with_suffix('.tmp.npz')
            np.savez_compressed(tmp, patch=np.stack([rec[k][1] for k in keys]), cells=json.dumps([rec[k][0] for k in keys], ensure_ascii=False),
                                offs=np.arange(-80, 125, 5.0))
            os.replace(tmp, f)

    def load(self, port, until=None):
        """(cells, patches float16, offs), sorted by (date, section); only the days <= until when given."""
        C = []; P = []
        for month in self.months(port):
            if until is not None and month > until[:7]: continue
            c, p = self._read(port, month); keep = [i for i, x in enumerate(c) if until is None or x['date'] <= until]
            C += [c[i] for i in keep]; P.append(p[keep])
        P = np.concatenate(P) if P else np.zeros((0, labels.NA, 41, 5), np.float16)
        order = sorted(range(len(C)), key=lambda i: (C[i]['date'], C[i]['section']))
        return [C[i] for i in order], P[order], np.arange(-80, 125, 5.0)

    def dates(self, port):
        return sorted({c['date'] for month in self.months(port) for c in self._read(port, month)[0]})


# ---------------------------------------------------------------- the model: trees as arrays, numpy predicts

def trees_of(est):
    """The fitted HistGradientBoostingClassifier (binary) as arrays: every node of every tree, and the baseline."""
    nodes = [p[0].nodes for p in est._predictors]
    if any(n['is_categorical'].any() for n in nodes): raise ValueError('categorical splits are not supported')
    out = {f: np.concatenate([n[f] for n in nodes]) for f in NODE_FIELDS}
    out['start'] = np.cumsum([0] + [len(n) for n in nodes]); out['baseline'] = np.asarray(est._baseline_prediction, np.float64).reshape(-1)[:1]
    return out


def raw_predict(T, X):
    """The raw score (logit) of the trees T for the rows of X (float; NaN = missing), as scikit-learn computes it: at a split
    a row goes left when x <= threshold, a missing value goes where the split says."""
    X = np.asarray(X, np.float64); n = len(X); out = np.full(n, float(T['baseline'][0])); rows = np.arange(n)
    for a, b in zip(T['start'][:-1], T['start'][1:]):
        val, f, thr, mgl, left, right, leaf = (T[k][a:b] for k in NODE_FIELDS)
        node = np.zeros(n, np.int64)
        while True:
            live = leaf[node] == 0
            if not live.any(): break
            x = X[rows, f[node]]
            with np.errstate(invalid='ignore'):
                go_left = np.where(np.isnan(x), mgl[node] == 1, x <= thr[node])
            node = np.where(live, np.where(go_left, left[node], right[node]), node)
        out += val[node]
    return out


def proba(T, X):
    return 1.0 / (1.0 + np.exp(-raw_predict(T, X)))


def key_of(c):
    return f"{c['port']}|{c['section']}|{c['date']}"


def save_model(path, B):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix('.tmp.npz'); arr = {}
    for part in ('gen', 'stack'):
        for k, v in B[part].items(): arr[f'{part}_{k}'] = v
    np.savez_compressed(tmp, meta=json.dumps(B['meta'], ensure_ascii=False), names=np.array(B['names']), phys=np.asarray(B['phys'], np.int64),
                        train_keys=np.array(sorted(B['train_scores'])), train_scores=np.array([B['train_scores'][k] for k in sorted(B['train_scores'])], np.float64), **arr)
    os.replace(tmp, path)


def load_model(path):
    z = np.load(path, allow_pickle=False); meta = json.loads(str(z['meta']))
    if meta.get('kind') != KIND: raise SystemExit(f'{path} is not a {KIND} model')
    B = dict(meta=meta, names=[str(n) for n in z['names']], phys=z['phys'].tolist(),
             train_scores=dict(zip((str(k) for k in z['train_keys']), z['train_scores'].tolist())))
    for part in ('gen', 'stack'):
        B[part] = {k: z[f'{part}_{k}'] for k in NODE_FIELDS + ('start', 'baseline')}
    return B


def fit(C, X, names, cutoff, version=None, log=print):
    """Coefficients from the answered cells of days before ``cutoff`` (all ports), exactly as a fold of the research forward
    validation is fitted (fuse.stack_fold, configuration ('opt', 1)):
      gen     the generic classifier on the 545 answer-free features
      scores  for the training cells, the generic score out-of-fold over the ports (the exemplars of later days use these)
      stack   the fusion on the exemplar / history features of the training cells
    C: cell records with 'y' (None = no usable answer); X: their features. Needs scikit-learn."""
    import sklearn
    lab, y = model.labelled(C); tr = [i for i in lab if C[i]['date'] < cutoff]; cols = list(range(X.shape[1])); phys = model.phys_cols(names)
    if len(set(y[tr])) < 2: raise SystemExit(f'the answers before {cutoff} do not hold both present and absent cells')
    t = time.time(); s = model.gen_scores(C, X, y, tr, [], cols); log(f'out-of-fold generic scores: {len(tr)} cells, {time.time() - t:.0f} s')
    t = time.time(); gen = model.gen_model().fit(X[np.array(tr)][:, cols], y[np.array(tr)]); log(f'generic classifier: {time.time() - t:.0f} s')
    A = model.adapt_feats(C, X, y, lab, phys, s, set(lab)); trv = np.array([i for i in tr if not np.isnan(s[i])])
    stack = model.stack_model().fit(A[trv], y[trv], sample_weight=np.ones(len(trv)))
    meta = dict(kind=KIND, model_version=version or f'{KIND}-{cutoff[:7].replace("-", "")}', cutoff=cutoff, config="('opt', 1)",
                n_train=len(tr), n_train_stack=len(trv), n_present=int((y[tr] == 1).sum()), ports=sorted({C[i]['port'] for i in tr}),
                last_answer_date=max(C[i]['date'] for i in tr), n_features=len(names), n_fusion_features=int(A.shape[1]),
                scikit_learn=sklearn.__version__, numpy=np.__version__, python=sys.version.split()[0],
                created_utc=dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
                note='the trees of two HistGradientBoostingClassifier as arrays; predicted with numpy (sat.pres3.daily.proba)')
    return dict(meta=meta, names=list(names), phys=phys, gen=trees_of(gen), stack=trees_of(stack),
                train_scores={key_of(C[i]): float(s[i]) for i in tr if not np.isnan(s[i])})


# ---------------------------------------------------------------- answers

def owner_answers(port, sat_dir=None):
    """(section, date) -> 1 / 0 from the owner's answer files of the repository: labels/<port>-s2-answers.csv (Zhanjiang:
    ZJG-s2-labels.csv, verified rows) and, when it exists, labels/<port>-confirm-answers.csv (the answers of the confirmation
    queue, the design section 6.3; rows of verified_by other than the owner are not answers). 'Cannot tell' is no answer."""
    sat_dir = Path(sat_dir) if sat_dir else common.SAT; out = {}
    first = sat_dir / 'labels' / ('ZJG-s2-labels.csv' if port == 'ZJG' else f'{port}-s2-answers.csv')
    for f, confirm in ((first, False), (sat_dir / 'labels' / f'{port}-confirm-answers.csv', True)):
        if not f.exists(): continue
        with open(f, newline='', encoding='utf-8') as fh:
            for r in csv.DictReader(fh):
                if confirm and r.get('verified_by', 'owner') != 'owner': continue
                if r.get('verified') == '1' and r.get('verified_occupied') in ('0', '1'): out[(r['section_id'], r['date'])] = int(r['verified_occupied'])
                elif confirm: out.pop((r['section_id'], r['date']), None)       # a later 'cannot tell' withdraws the answer
    return out


# ---------------------------------------------------------------- scoring

def usable(c):
    return c['cloud'] + c['shadow'] <= MAX_CLOUD_SHADOW and c['nodata'] == 0


def score_days(port, C, P, offs, answers, B, days, sections=None, all_cells=False):
    """Score rows (dicts with SCORE_COLUMNS) of the cells of ``days``.

    C, P: the port's patch store (every section of every day: the water colour of a scene and its clear-day flag are made
    from all the port's sections). For a day d only the rows of days <= d are looked at, and only the answers of days < d
    (and only answers of cells that have a patch). sections: score only these (the others still shape the water colour).
    A cell under cloud / shadow / nodata gets no score (status = no_image), a cell with fewer than 6 reference days neither, unless all_cells (the check against the research
    numbers, which scored every answered cell)."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')             # numpy's all-NaN slices (cells without a reference)
        return _score_days(port, C, P, offs, answers, B, days, sections, all_cells)


def _score_days(port, C, P, offs, answers, B, days, sections, all_cells):
    out = []; W = feats.water(P, offs, C); good = feats.good_day(C); phys = B['phys']
    date = np.array([c['date'] for c in C]); sec = np.array([c['section'] for c in C])
    for d in sorted(days):
        day = [int(i) for i in np.where(date == d)[0] if sections is None or sec[i] in sections]
        if not day: continue
        day_secs = {sec[i] for i in day}
        ans = {int(i): answers[(sec[i], date[i])] for i in np.where(date < d)[0] if sec[i] in day_secs and answers.get((sec[i], date[i])) in (0, 1)}
        past = collections.defaultdict(list)
        for i in sorted(ans, key=lambda i: date[i]): past[sec[i]].append((str(date[i]), ans[i]))
        img = [i for i in day if all_cells or usable(C[i])]; res = {}
        if img:
            want = {sec[i] for i in img}; need = sorted(set(img) | {i for i in ans if sec[i] in want})
            rows = [int(i) for i in np.where(date <= d)[0] if sec[i] in want]; pos = {i: k for k, i in enumerate(rows)}
            X, names, _ = feats.features(np.asarray(P[rows], np.float32), offs, [C[i] for i in rows], only={pos[i] for i in need}, W=W[rows], good=good[rows])
            if names != B['names']: raise SystemExit('the features of this code are not the features the model was fitted on')
            Xn = X[[pos[i] for i in need]]; s = model.logit(proba(B['gen'], Xn))
            for k, i in enumerate(need):          # a training cell keeps the out-of-fold score it had when the fusion was fitted
                if i in ans and key_of(C[i]) in B['train_scores']: s[k] = B['train_scores'][key_of(C[i])]
            y = np.array([ans.get(i, -1) for i in need])
            A = model.adapt_feats([C[i] for i in need], Xn, y, list(range(len(need))), phys, s, {k for k, i in enumerate(need) if i in ans})
            k_img = [need.index(i) for i in img]; p = proba(B['stack'], A[k_img]); nref = Xn[:, names.index('nref')]
            for k, i, pi in zip(k_img, img, p): res[i] = (float(pi), int(nref[k]))
        for i in day:
            c = C[i]; pa = past.get(sec[i], []); p, nref = res.get(i, (None, None)); ok = usable(c)
            # p is left empty when the cell has no image, and when the section has fewer than 6 earlier clear days (the
            # judge has not enough input: sat/confirm.py shows such a cell to the owner with the reason no_reference)
            show = p is not None and (all_cells or nref >= MIN_REFERENCE_DAYS)
            out.append(dict(site_id=port, section_id=c['section'], date=d, acquired_utc=c['time'], p=f'{p:.6f}' if show else '',
                            model_version=B['meta']['model_version'], cloud_shadow=f"{c['cloud'] + c['shadow']:.4f}", nodata=f"{c['nodata']:.4f}",
                            status='' if ok else 'no_image', answers_through=pa[-1][0] if pa else '',
                            n_answers=len(pa), n_reference_days='' if nref is None else nref, scene_id=c['scene']))
    return out


def write_scores(path, rows, diagnostics=False):
    path = outside_repo(path, 'the score table'); path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix('.tmp')
    with open(tmp, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, SCORE_COLUMNS + (DIAGNOSTIC_COLUMNS if diagnostics else []), lineterminator='\n', extrasaction='ignore')
        w.writeheader(); w.writerows(rows)
    os.replace(tmp, path)


# ---------------------------------------------------------------- fetching one new day

def port_tile(port):
    """The MGRS tile the port's recent history was fetched from: the most frequent tile among the scenes of
    inputs/scenes.csv of the list's last 365 days (the tile of two ports changed between the two fetch periods)."""
    from . import fetch
    ids = [i.split('_') for i in fetch.manifest(port)]
    if not ids: raise SystemExit(f'{port} has no scene in inputs/scenes.csv: its tile is not known')
    last = dt.datetime.strptime(max(i[2] for i in ids), '%Y%m%d'); since = f'{last - dt.timedelta(days=365):%Y%m%d}'
    t = collections.Counter(i[1] for i in ids if i[2] > since)
    return sorted(t, key=lambda k: (-t[k], k))[0]


def pick_new(feats_found, tile, have_dates, max_cloud=85):
    """From the features of a STAC search: the scenes of the port's tile, cloud cover <= max_cloud, of days the store does not
    hold yet; the last scene of each day (as sat.pres3.fetch chose the history)."""
    pick = {}
    for f in feats_found:
        d = f['properties']['datetime'][:10]
        if f['id'].split('_')[1] != tile or d in have_dates or (f['properties'].get('eo:cloud_cover') or 0) > max_cloud: continue
        if d not in pick or f['id'] > pick[d]['id']: pick[d] = f
    return [pick[d] for d in sorted(pick)]


def fetch_port(store, windows, port, days, sites, now=None, log=print):
    """Search the last ``days`` days, fetch the windows of the new scenes into <windows>/data/<port>/ and put their patches
    into the store. No retry around a refusal: a refusal (HTTP 4xx) or a failure that fetch.search's few retries of a transient
    failure (no connection, cut off, 5xx) did not cure stops the port (cmd_fetch reports it). The COG reads: fetch.gdal_env."""
    from . import build, fetch
    now = now or dt.datetime.now(dt.timezone.utc); bb = sites[port]['bbox']; tile = port_tile(port); have = set(store.dates(port))
    found = fetch.search({"bbox": bb, "datetime": f"{now - dt.timedelta(days=days):%Y-%m-%dT%H:%M:%SZ}/{now:%Y-%m-%dT%H:%M:%SZ}"})
    new = pick_new(found, tile, have); log(f'{port} tile {tile}: {len(found)} scenes in the catalogue, {len(new)} new days')
    secs = labels.frames(port); geo = build.sample_points(secs); done = []
    for f in new:
        fetch.one(str(windows), port, f, bb); time.sleep(1.0)
        d = f['properties']['datetime'][:10]; fn = f'{windows}/data/{port}/{f["id"]}.npz'
        got = build.day_cells(port, d, [(fn, json.loads(str(np.load(fn)['meta'])))], secs, geo, {})
        if got is None: log(f'{port} {d}: more than half nodata, not stored'); continue
        store.put(port, *got); done.append(d)
    return done


# ---------------------------------------------------------------- command line

def research_cells(work, ports):
    from . import dataset
    return dataset.load(work, ports)


def cmd_import(a):
    work = common.work_dir(a.work); store = Store(outside_repo(a.store, 'the patch store'))
    for port in common.ports_of(a.ports):
        C = json.load(open(f'{work}/cells_{port}.json')); P = np.load(f'{work}/patch_{port}.npz')['patch']
        store.put(port, C, P); print(port, 'cells', len(C), 'days', len({c['date'] for c in C}), 'months', len(store.months(port)), flush=True)
    return 0


def cmd_fit(a):
    warnings.filterwarnings('ignore'); work = common.work_dir(a.work)
    C, X, names = research_cells(work, common.ports_of(a.ports)); B = fit(C, X, names, a.cutoff, a.version)
    save_model(outside_repo(a.model, 'the model file'), B); print(json.dumps(B['meta'], ensure_ascii=False)); return 0


def cmd_fetch(a):
    store = Store(outside_repo(a.store, 'the patch store')); windows = outside_repo(a.windows, 'the window directory')
    sites = {x['id']: x for x in json.load(open(common.REPO / 'sat' / 'sites.json'))['sites']}; bad = 0
    for port in common.ports_of(a.ports):
        try: print(port, 'stored', fetch_port(store, windows, port, a.days, sites), flush=True)
        except Exception as e:          # reported, not worked around
            bad += 1; print(f'::error title=pres3 daily fetch::{port}: {e!r}'[:400], flush=True)
    return 1 if bad else 0


def cmd_score(a):
    store = Store(outside_repo(a.store, 'the patch store')); B = load_model(a.model); rows = []
    for port in common.ports_of(a.ports):
        C, P, offs = store.load(port, a.until); every = sorted({c['date'] for c in C})
        days = [d for d in every if d >= a.since] if a.since else every[-1:]
        if any(d < B['meta']['cutoff'] for d in days):
            print(f'{port}: days before the cutoff of the model ({B["meta"]["cutoff"]}) are scored: those scores are not out of sample', file=sys.stderr)
        r = score_days(port, C, P, offs, owner_answers(port, a.sat), B, days); rows += r
        print(port, 'days', len(days), 'cells', len(r), 'scored', sum(x['p'] != '' for x in r), 'no_image', sum(x['status'] == 'no_image' for x in r), flush=True)
    write_scores(a.out, rows, a.diagnostics); return 0


def cmd_check(a):
    """The research forward validation, redone one day at a time: for each boundary, fit on the answers before it, then walk
    the answered days of the fold in order, each from a store cut at that day. Compared with pres3_oos.csv (p_opt_fwd)."""
    from . import evalcv
    warnings.filterwarnings('ignore'); work = common.work_dir(a.work); store = Store(outside_repo(a.store, 'the patch store'))
    ports = common.ports_of(a.ports); C, X, names = research_cells(work, common.PORTS)
    with open(a.oos, newline='', encoding='utf-8') as fh:
        oos = {(r['site_id'], r['section_id'], r['date']): r['p_opt_fwd'] for r in csv.DictReader(fh)}
    bounds = evalcv.BOUNDS; worst = 0.0; n = 0; bad = 0
    for k, b in enumerate(bounds):
        if a.bounds and b not in a.bounds.split(','): continue
        e = bounds[k + 1] if k + 1 < len(bounds) else '9999'; mf = Path(a.models) / f'check_{b}.npz' if a.models else None
        if mf and mf.exists(): B = load_model(mf)
        else:
            B = fit(C, X, names, b, version=f'check-{b}')
            if mf: save_model(mf, B)
        for port in ports:
            Cp, Pp, offs = store.load(port); ans = {(c['section'], c['date']): c['y'] for c in C if c['port'] == port and c['y'] is not None}
            days = sorted({d for (s, d) in ans if b <= d < e}); t = time.time(); dmax = 0.0; m = 0
            for d in days:
                keep = [i for i, c in enumerate(Cp) if c['date'] <= d]          # the store as it was on day d
                for r in score_days(port, [Cp[i] for i in keep], Pp[keep], offs, ans, B, [d], sections={s for (s, dd) in ans if dd == d}, all_cells=True):
                    want = oos.get((port, r['section_id'], d))
                    if not want: continue
                    if r['p'] == '': bad += 1; print('  no score for an answered cell', port, r['section_id'], d); continue
                    dmax = max(dmax, abs(float(r['p']) - float(want))); m += 1
            worst = max(worst, dmax); n += m
            print(f'fold {b} {port}: days {len(days)} cells {m} max |daily - research| {dmax:.2e} ({time.time() - t:.0f} s)', flush=True)
    print(f'cells {n} max difference {worst:.2e} unscored {bad}')
    return 0 if n and worst <= a.tolerance and not bad else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0]); sub = ap.add_subparsers(dest='cmd', required=True)

    def add(name, fn, *opts):
        p = sub.add_parser(name); p.set_defaults(fn=fn); p.add_argument('--ports', default=','.join(common.PORTS))
        for o in opts:
            if o == 'work': p.add_argument('--work', help=f'the research working directory (default ${common.ENV_WORK})')
            if o == 'store': p.add_argument('--store', required=True, help='the patch store (outside the repository)')
        return p
    add('import', cmd_import, 'work', 'store')
    p = add('fit', cmd_fit, 'work'); p.add_argument('--cutoff', required=True); p.add_argument('--model', required=True); p.add_argument('--version')
    p = add('fetch', cmd_fetch, 'store'); p.add_argument('--windows', required=True); p.add_argument('--days', type=int, default=10)
    p = add('score', cmd_score, 'store'); p.add_argument('--model', required=True); p.add_argument('--out', required=True)
    p.add_argument('--since'); p.add_argument('--until'); p.add_argument('--diagnostics', action='store_true', help='also n_reference_days, scene_id'); p.add_argument('--sat', help='docs/data/sat of another checkout (default: this one)')
    p = add('check', cmd_check, 'work', 'store'); p.add_argument('--oos', default=str(common.SAT / 'eval' / 'pres3_oos.csv'))
    p.add_argument('--bounds'); p.add_argument('--models', help='a directory to keep the fitted models of the check'); p.add_argument('--tolerance', type=float, default=2e-6)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == '__main__':
    sys.exit(main())
