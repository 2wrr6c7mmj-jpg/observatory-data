"""confirm-1 PR-5: the daily run of .github/workflows/sat-confirm.yml (and its monthly re-fit), in one place.

    daily    for each port: search Earth Search for the last --days days, fetch the windows of the new days (AWS COGs), put
             their patches into the store, thin the store's old months; then score the last --score-days days with the latest
             model (sat.pres3.daily), judge them (sat.confirm run), draw the queue's pictures (sat.confirm_img) and write
             summary.json. A port that is refused (HTTP 4xx) is skipped and reported; nothing goes around a refusal.
             The requests and bytes are counted (the search through urllib, the COG reads from libcurl's own log) and the
             fetching stops before a scene would go over --max-requests or --max-bytes.
    refit    the coefficients of a new version from the store and the owner's answer files only: the answers of days before
             the cutoff (the 1st of the month), the answer-free features of those cells computed from the store.
    paths    the paths `git status --porcelain` reports, outside the two places this series may write: fails if there is one.
    verdict  exit 1 when the report of a daily run lists a refused / failed port, a budget stop or a missing model.

The store is the branch pres3-patches (a checkout given with --store, outside the repository); the models are in
<store>/models/. Writes in the repository only under docs/data/sat/confirm/ (and, through sat.confirm merge, which this
module does not call, docs/data/sat/labels/<site>-confirm-answers.csv). No CDSE (PU 0).
Contains modified Copernicus Sentinel data (Element84 Earth Search / AWS Open Data).
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import io
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import common

# Limits of one daily run. A scene is 6 COG files x (1 HEAD + 3 ranged GETs) = 24 requests and 7-25 MB (the PR-5 probe:
# 24 requests, 11.8 MB for Qingdao; Zhanjiang's window is about 3 times larger). A normal day brings 0-3 new days per port
# (9 ports: at most about 30 scenes, 720 requests, 0.5 GB with the searches); the limits leave room for a catch-up after a
# missed day or two, and stop a run that would fetch far more than that (a changed catalogue, a bug).
MAX_REQUESTS = 1000
MAX_BYTES = 1_500_000_000
SCENE_REQUESTS = 24                       # a scene's requests before any was seen in this run (then the most seen)
SCENE_BYTES = 30_000_000                  # a scene's bytes before any was seen in this run (then the largest seen)
SCORE_DAYS = 35                           # the days scored each run: the expiry of a queued cell (28 days) and a week more
ALLOWED = (re.compile(r'^docs/data/sat/confirm/'), re.compile(r'^docs/data/sat/labels/[A-Z0-9]+-confirm-answers\.csv$'))
MODEL_RE = re.compile(r'^pres3o-(\d{6})\.npz$')


def note(level, title, msg):
    print(f'::{level} title={title}::{msg}'[:3800], flush=True)


# ---------------------------------------------------------------- counting the requests

class Meter:
    """The requests and bytes of a run, and the limits."""

    def __init__(self, max_requests=MAX_REQUESTS, max_bytes=MAX_BYTES):
        self.max_requests, self.max_bytes = max_requests, max_bytes
        self.requests = 0; self.bytes = 0; self.status = collections.Counter(); self.search_requests = 0; self.scene_bytes = 0; self.scene_requests = 0
        self.stopped = None

    def add(self, requests, nbytes, status=None):
        self.requests += requests; self.bytes += nbytes; self.status.update(status or {})

    def urlopen(self, opener=None):
        """An urlopen for fetch.search that counts each request and the bytes of its answer."""
        opener = opener or urllib.request.urlopen

        def open_(req, timeout=None):
            self.search_requests += 1
            try:
                data = opener(req, timeout=timeout).read()
            except urllib.error.HTTPError as e:
                self.add(1, 0, {str(e.code): 1}); raise
            except Exception:
                self.add(1, 0, {'none': 1}); raise
            self.add(1, len(data), {'200': 1})
            return io.BytesIO(data)
        return open_

    def room_for_a_scene(self):
        est = max(self.scene_bytes, SCENE_BYTES); nreq = max(self.scene_requests, SCENE_REQUESTS)
        if self.requests + nreq > self.max_requests:
            self.stopped = f'requests: {self.requests} + {nreq} would pass the limit {self.max_requests}'
        elif self.bytes + est > self.max_bytes:
            self.stopped = f'bytes: {self.bytes} + {est} would pass the limit {self.max_bytes}'
        return self.stopped is None

    def as_dict(self):
        return dict(requests=self.requests, search_requests=self.search_requests, cog_requests=self.requests - self.search_requests,
                    bytes=self.bytes, status=dict(sorted(self.status.items())), max_requests=self.max_requests, max_bytes=self.max_bytes,
                    stopped=self.stopped)


def refusal(e):
    """A refusal (HTTP 4xx; also one a proxy gives, and GDAL's text of one): the port is skipped, never retried."""
    from . import fetch
    if isinstance(e, urllib.error.HTTPError): return 400 <= e.code < 500
    text = str(e) + ' ' + str(getattr(e, 'reason', ''))
    return not fetch.transient(e) and bool(fetch.STATUS_4XX.search(text))


# ---------------------------------------------------------------- the daily run

def fetch_port(store, windows, port, sites, meter, days, now, max_scenes=None, log=print, read=None, out=None):
    """Search, then fetch the new days of one port (sat.pres3.daily.fetch_port, with the requests counted and the limits).
    -> (stored days, the scenes left for the limits)."""
    from . import build, daily, fetch
    import numpy as np
    read = read or fetch.one
    bb = sites[port]['bbox']; tile = daily.port_tile(port); have = set(store.dates(port))
    secs, geo, trial, wide = daily.fetch_plan(port, bb)          # the window: widened to the port's trial sections, if any
    bad = redrawn_trial(out, port) if out is not None else []
    if bad:          # drawn again under the same id: no patch of the new line under the old id (its reference would mix them)
        secs = {k: v for k, v in secs.items() if k not in bad}; geo = {k: v for k, v in geo.items() if k not in bad}; trial = trial - set(bad)
    found = fetch.search({"bbox": bb, "datetime": f"{now - dt.timedelta(days=days):%Y-%m-%dT%H:%M:%SZ}/{now:%Y-%m-%dT%H:%M:%SZ}"},
                         urlopen=meter.urlopen())
    new = daily.pick_new(found, tile, have)
    if max_scenes is not None: new = new[-max_scenes:]
    log(f'{port} tile {tile}: {len(found)} scenes in the catalogue, {len(new)} new days')
    done = []
    for k, f in enumerate(new):
        if not meter.room_for_a_scene(): return done, len(new) - k
        err = None
        with fetch.captured_stderr() as cap:                  # libcurl's log of the reads: the requests GDAL really made
            try:
                read(str(windows), port, f, wide, env_extra={'CPL_CURL_VERBOSE': 'YES'}, **({'base': bb} if trial else {}))
            except Exception as e:
                err = e
        c = fetch.curl_counts(cap.text); meter.add(c['requests'], c['bytes'], c['status']); meter.scene_bytes = max(meter.scene_bytes, c['bytes'])
        meter.scene_requests = max(meter.scene_requests, c['requests'])
        if err is not None: raise err
        time.sleep(1.0)
        d = f['properties']['datetime'][:10]; fn = f'{windows}/data/{port}/{f["id"]}.npz'
        got = build.day_cells(port, d, [(fn, json.loads(str(np.load(fn)['meta'])))], secs, geo, {}, trial=trial)
        if got is None: log(f'{port} {d}: more than half nodata, not stored'); continue
        store.put(port, *got, trial=trial); done.append(d)
    return done, 0


def latest_model(models_dir, today):
    """The model file of the latest version whose cutoff is on or before today."""
    best = None
    for f in sorted(Path(models_dir).glob('pres3o-*.npz')):
        m = MODEL_RE.match(f.name)
        if m and f'{m.group(1)[:4]}-{m.group(1)[4:]}-01' <= today: best = f
    return best


def window_files(windows, port, day):
    return sorted(f for f in (Path(windows) / 'data' / port).glob(f'*_{day.replace("-", "")}_*.npz'))


def draw(out, windows, site, day, sites):
    """The queue's picture of one port and day from a window fetched in this run (natural colour, on a lon / lat grid over the
    port's bbox). False when there is no window of that day here."""
    import numpy as np
    import rasterio
    from rasterio.transform import from_bounds, from_origin
    from rasterio.warp import Resampling, reproject
    from .. import confirm_img
    fs = window_files(windows, site, day)
    if not fs: return False
    from .. import confirm_trial
    z = np.load(fs[-1]); meta = json.loads(str(z['meta'])); bb = confirm_trial.fetch_bbox(site, sites[site]['bbox'])   # the trial sections too
    lat = (bb[1] + bb[3]) / 2; dx = 10 / (111320 * np.cos(np.radians(lat))); dy = 10 / 110540
    w = max(1, int(round((bb[2] - bb[0]) / dx))); h = max(1, int(round((bb[3] - bb[1]) / dy)))
    src_t = from_origin(meta['x0'], meta['y1'], 10, 10); dst_t = from_bounds(bb[0], bb[1], bb[2], bb[3], w, h)
    refl = np.zeros((h, w, 3), np.float32)
    with rasterio.Env():
        for k, band in enumerate(('blue', 'green', 'red')):
            reproject(z[band].astype(np.float32), refl[..., k], src_transform=src_t, src_crs=meta['crs'], dst_transform=dst_t,
                      dst_crs='EPSG:4326', resampling=Resampling.bilinear, src_nodata=0, dst_nodata=0)
    confirm_img.write_image(out, site, day, refl, {"west": bb[0], "south": bb[1], "east": bb[2], "north": bb[3]})      # the label: the site id (the font has no CJK)
    return True


def daily_run(store_dir, windows, out, ports, report_path, now=None, days=10, score_days=SCORE_DAYS, meter=None, max_scenes=None,
              require_frozen=False, fetch_read=None, log=print):
    from .. import confirm, confirm_img
    from . import daily
    t0 = time.time(); now = now or dt.datetime.now(dt.timezone.utc); today = f'{now:%Y-%m-%d}'; nows = f'{now:%Y-%m-%dT%H:%M:%SZ}'
    out = Path(out); store = daily.Store(store_dir); meter = meter or Meter(); cfg = confirm.load_config()
    rep = dict(version='confirm-run-1', now=nows, ports={}, problems=[], seconds={})
    if require_frozen and not cfg.get('frozen_utc'):
        rep['problems'].append('sat/confirm.json frozen_utc is null: set it before the first run on main (the PR text)')
        _write(report_path, rep); return rep
    sites = {x['id']: x for x in json.load(open(common.REPO / 'sat' / 'sites.json'))['sites']}
    for port in ports:
        r = rep['ports'][port] = dict(stored=[], thinned=0)
        if meter.stopped:
            r['skipped'] = 'limits'; continue
        try:
            r['stored'], left = fetch_port(store, windows, port, sites, meter, days, now, max_scenes, log, read=fetch_read, out=out)
            if left: r['left_for_the_limits'] = left
        except Exception as e:                 # reported as it is: the port waits for the next run
            kind = 'refused' if refusal(e) else 'failed'; r[kind] = repr(e)[:300]
            note('error', f'sat-confirm {port} {kind}', f'{port}: {e!r}'[:600])
            rep['problems'].append(f'{port} {kind}')
        r['thinned'] = len(store.thin(port, today, answers=set(daily.owner_answers(port, out))))
    if meter.stopped:
        note('error', 'sat-confirm limits', meter.stopped); rep['problems'].append('limits: ' + meter.stopped)
    rep['seconds']['fetch'] = round(time.time() - t0, 1); rep['meter'] = meter.as_dict()
    mf = latest_model(Path(store_dir) / 'models', today)
    if mf is None:
        rep['problems'].append('no model in the store'); _write(report_path, rep); return rep
    t = time.time(); B = daily.load_model(mf); rep['model'] = B['meta']['model_version']; since = f'{now - dt.timedelta(days=score_days):%Y-%m-%d}'
    rows = []
    for port in ports:
        C, P, offs = store.load(port, today); ds = sorted({c['date'] for c in C if c['date'] >= since})
        rr = daily.score_days(port, C, P, offs, daily.owner_answers(port, out), B, ds); rows += rr
        rep['ports'][port].update(scored_days=len(ds), cells=len(rr), with_p=sum(x['p'] != '' for x in rr))
    rep['seconds']['score'] = round(time.time() - t, 1)
    t = time.time(); scores = [{k: str(v) for k, v in r.items()} for r in rows]
    skip = set()
    for port in ports:
        if confirm.load_sections(out, port) is None:
            confirm.sync_sections(out, port, confirm.review_sections(out, port), nows)
        bad = sync_trial(out, port, nows)             # the trial sections: a later sections_version of the ledger (§6.5)
        if bad:
            msg = (f'{port}: trial sections whose line differs from the ledger under the same id, left out of this run: {", ".join(bad)} '
                   '(a line drawn again needs a new id: sat/confirm_trial.json)')
            note('warning', 'sat-confirm trial', msg); rep.setdefault('warnings', []).append(msg)
            skip |= {(port, i) for i in bad}
    if skip: scores = [r for r in scores if (r['site_id'], r['section_id']) not in skip]
    res = confirm.run(scores, out, cfg, nows)
    queue = json.loads((out / confirm.DIR / 'queue.json').read_text(encoding='utf-8'))
    drawn = [f'{s}/{d}' for s, d in confirm_img.needed(out, queue) if draw(out, windows, s, d, sites)]
    if drawn: res = confirm.run(scores, out, cfg, nows); queue = json.loads((out / confirm.DIR / 'queue.json').read_text(encoding='utf-8'))
    gone = confirm_img.prune(out, queue, today, cfg.get('image_keep_days', 90))
    doc = confirm.summary(out, cfg, nows); confirm._write_json(out / confirm.DIR / 'summary.json', doc)
    rep['seconds']['confirm'] = round(time.time() - t, 1); rep['seconds']['all'] = round(time.time() - t0, 1)
    rep.update(queued=len(queue.get('cells', [])), images_drawn=drawn, images_removed=gone, skipped=len(res.get('skipped', [])))
    _write(report_path, rep)
    return rep


def redrawn_trial(out, port):
    """The ids of the port's trial sections whose line or frame differs from the ledger's under the same id (drawn again
    without a new id). Such a section is neither synced, judged nor stored until it gets a new id and the old one is retired."""
    from .. import confirm, confirm_trial
    known = {s['id']: s for s in (confirm.load_sections(out, port) or {}).get('sections', [])}
    return [s['id'] for s in confirm_trial.sections(port)
            if s['id'] in known and (known[s['id']].get('line') != s['line'] or known[s['id']].get('frame') != s['frame'])]


def sync_trial(out, port, nows):
    """Add the port's trial sections (sat/confirm_trial.json) to its ledger as a later sections_version. A trial section whose
    id is in the ledger with another line or frame (drawn again under the same id) is not synced and is returned, so that one
    redrawn line stops only itself, not the daily run of every port (sat.confirm.sync_sections would raise)."""
    from .. import confirm, confirm_trial
    trial = confirm_trial.sections(port)
    if not trial: return []
    bad = redrawn_trial(out, port)
    confirm.sync_sections(out, port, [s for s in trial if s['id'] not in bad], nows)
    return bad


def _write(path, doc):
    if path: Path(path).write_text(json.dumps(doc, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')


def annotate(rep):
    m = rep.get('meter', {})
    note('notice', 'sat-confirm requests', f"HTTP requests {m.get('requests')} (search {m.get('search_requests')}, COG {m.get('cog_requests')}), "
         f"bytes {m.get('bytes')} ({(m.get('bytes') or 0) / 1e6:.1f} MB), status {m.get('status')}, limits {m.get('max_requests')} requests / "
         f"{(m.get('max_bytes') or 0) / 1e9:.1f} GB, seconds {rep.get('seconds')}")
    note('notice', 'sat-confirm ports', json.dumps({p: {k: (len(v) if k == 'stored' else v) for k, v in r.items()} for p, r in rep['ports'].items()}, ensure_ascii=False))
    note('notice', 'sat-confirm queue', f"model {rep.get('model')}, queued {rep.get('queued')}, images drawn {len(rep.get('images_drawn', []))}, "
         f"removed {len(rep.get('images_removed', []))}, rows skipped by sat.confirm {rep.get('skipped')}")
    for p in rep.get('problems', []): note('error', 'sat-confirm problem', p)


# ---------------------------------------------------------------- re-fit from the store

def store_cells(store, ports, sat_dir=None, cutoff=None):
    """(cells, features) of the answered cells of days before cutoff, from the store: the training set of a re-fit."""
    import numpy as np
    from .. import confirm_trial
    from . import daily, feats
    C = []; X = []; names = None
    for port in ports:
        Cp, Pp, offs = store.load(port); ans = daily.owner_answers(port, sat_dir); trial = confirm_trial.trial_ids(port)
        # the trial sections (sat/confirm_trial.json) are not learnt from: the coefficients stay those of the counted sections
        lab = [i for i, c in enumerate(Cp) if (c['section'], c['date']) in ans and (cutoff is None or c['date'] < cutoff) and c['section'] not in trial]
        if not lab: continue
        W, good = daily.day_inputs(Pp, offs, Cp)
        Xp, names, _ = feats.features(np.asarray(Pp, np.float32), offs, Cp, only=set(lab), W=W, good=good)
        C += [dict(port=port, section=Cp[i]['section'], date=Cp[i]['date'], y=ans[(Cp[i]['section'], Cp[i]['date'])]) for i in lab]
        X.append(Xp[lab])
    return C, (np.concatenate(X) if X else None), names


def refit_allowed():
    from .. import confirm
    return bool(confirm.load_config().get('frozen_utc'))


def refit(store_dir, out, cutoff, ports, log=print):
    from . import daily
    store = daily.Store(store_dir); C, X, names = store_cells(store, ports, None, cutoff)      # the answer files of the repository
    B = daily.fit(C, X, names, cutoff, version=f'pres3o-{cutoff[:7].replace("-", "")}', log=log)
    mf = Path(store_dir) / 'models' / f"{B['meta']['model_version']}.npz"; daily.save_model(mf, B)
    from .. import confirm
    cfg = confirm.load_config()
    doc = dict(B['meta'], thresholds=dict(p_low=cfg['p_low'], p_high=cfg['p_high'], rule_version=cfg['rule_version']),
               file=f'pres3-patches:models/{mf.name}',
               note='判定器 pres-3o の係数の版。締め日（cutoff）より前の日付の所有者の答えだけで学んだ。点数はその版の締め日以後の日だけに使う（§6.3・§6.4）')
    p = Path(out) / confirm.DIR / 'model.json'; p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    return B, mf


# ---------------------------------------------------------------- the paths a run may write

def unexpected(porcelain):
    """The paths of `git status --porcelain` output that this series may not write (a rename counts with both names)."""
    bad = []
    for line in porcelain.splitlines():
        if not line.strip(): continue
        for p in line[3:].split(' -> '):
            p = p.strip().strip('"')
            if not any(r.match(p) for r in ALLOWED): bad.append(p)
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0]); sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('daily'); p.add_argument('--store', required=True); p.add_argument('--windows', required=True)
    p.add_argument('--out', default=str(common.SAT)); p.add_argument('--ports', default=','.join(common.PORTS)); p.add_argument('--report', required=True)
    p.add_argument('--days', type=int, default=10); p.add_argument('--score-days', type=int, default=SCORE_DAYS)
    p.add_argument('--max-requests', type=int, default=MAX_REQUESTS); p.add_argument('--max-bytes', type=int, default=MAX_BYTES)
    p.add_argument('--max-scenes', type=int, help='at most this many new days per port (the probe: 1)')
    p.add_argument('--require-frozen', action='store_true', help='stop when sat/confirm.json has no frozen_utc (the run on main)')
    p = sub.add_parser('refit'); p.add_argument('--store', required=True); p.add_argument('--out', default=str(common.SAT))
    p.add_argument('--cutoff', help='default: the 1st of this month (UTC)'); p.add_argument('--ports', default=','.join(common.PORTS))
    p.add_argument('--require-frozen', action='store_true', help='do nothing when sat/confirm.json has no frozen_utc (the run on main)')
    sub.add_parser('paths')
    p = sub.add_parser('verdict'); p.add_argument('report')
    a = ap.parse_args(argv)
    if a.cmd == 'daily':
        from . import daily
        daily.outside_repo(a.store, 'the patch store'); daily.outside_repo(a.windows, 'the window directory')
        rep = daily_run(a.store, a.windows, a.out, common.ports_of(a.ports), a.report, days=a.days, score_days=a.score_days,
                        meter=Meter(a.max_requests, a.max_bytes), max_scenes=a.max_scenes, require_frozen=a.require_frozen)
        annotate(rep)
        return 1 if 'meter' not in rep else 0          # stopped before fetching (no frozen_utc on main): nothing to publish
    if a.cmd == 'refit':
        from . import daily
        daily.outside_repo(a.store, 'the patch store')
        if a.require_frozen and not refit_allowed():
            note('error', 'sat-confirm refit', 'sat/confirm.json frozen_utc is null: no re-fit before the first run on main'); return 1
        cutoff = a.cutoff or f'{dt.datetime.now(dt.timezone.utc):%Y-%m}-01'
        B, mf = refit(a.store, a.out, cutoff, common.ports_of(a.ports))
        note('notice', 'sat-confirm refit', json.dumps(B['meta'], ensure_ascii=False)); return 0
    if a.cmd == 'paths':
        txt = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=all'], capture_output=True, text=True, check=True, cwd=common.REPO).stdout
        bad = unexpected(txt)
        for b in bad: note('error', 'sat-confirm paths', f'a path outside docs/data/sat/confirm/ and labels/<site>-confirm-answers.csv: {b}')
        return 1 if bad else 0
    rep = json.loads(Path(a.report).read_text(encoding='utf-8'))
    for p in rep.get('problems', []): print('problem:', p)
    return 1 if rep.get('problems') else 0


if __name__ == '__main__':
    sys.exit(main())
