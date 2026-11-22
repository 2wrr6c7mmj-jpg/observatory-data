"""The section frames and the owner's answers, read from the repository (read only).

frames(port)   section id -> the quay line (near) and its 120 m offshore normal (far), lon/lat
               (trial=True: with the port's trial sections of the confirmation queue, sat/confirm_trial.json, marked trial)
lookup(port)   (date, section id) -> the answer y, whether the cell is a row of the evaluation sheet (eval/presence.json),
               and that row's current S1 judgement p
"""
from __future__ import annotations

import csv
import json

import numpy as np

from sat.grid import m_per_deg

from . import common

NA = 24          # points along the quay (resampled, whatever the section's length)


def normals(pts, side):
    mlat, mlon = m_per_deg(float(pts[:, 1].mean()))
    xy = np.c_[pts[:, 0] * mlon, pts[:, 1] * mlat]
    t = np.gradient(xy, axis=0); t /= np.maximum(np.hypot(t[:, 0], t[:, 1]), 1e-9)[:, None]
    nrm = np.c_[t[:, 1], -t[:, 0]] * side      # the same sense as the frames of review.json (checked port by port)
    far = xy + 120.0 * nrm
    return np.c_[far[:, 0] / mlon, far[:, 1] / mlat]


def zjg_frames():
    """Zhanjiang: the normals are made from the lines of the geojson (side: L = left of the direction of the line)."""
    g = json.load(open(common.SAT / 'labels' / 'ZJG-s2-sections.geojson'))
    out = {}
    for f in g['features']:
        pr = f['properties']; pts = np.asarray(f['geometry']['coordinates'], float)
        out[pr['section_id']] = dict(id=pr['section_id'], kind=pr['kind'], sector=pr['sector'], length_m=pr['length_m'],
                                     near=pts, far=normals(pts, pr['side']))
    return out


def frames(port, trial=False):
    """The sections of the review set (Zhanjiang: of its geojson). trial: also the port's trial sections of the confirmation
    queue (sat/confirm_trial.json; the daily fetch of sat-confirm cuts their patches too), each with trial=True. The research
    code (build.py) calls it without: its cells are the review set's only."""
    secs = zjg_frames() if port == 'ZJG' else _review_frames(port)
    if trial:
        from sat import confirm_trial
        for s in confirm_trial.sections(port):
            if s['id'] in secs: raise ValueError(f"{port} {s['id']}: a trial section with the id of a section of the review set")
            k = len(s['line']); fr = np.asarray(s['frame'], float)
            secs[s['id']] = dict(id=s['id'], kind=s['kind'], sector=s['sector'], length_m=s['length_m'], near=fr[:k], far=fr[k:][::-1], trial=True)
    return secs


def _review_frames(port):
    rv = json.load(open(common.SAT / 'labels' / port / 'review.json')); secs = {}
    for s in rv['sections']:
        k = len(s['line']); fr = np.asarray(s['frame'], float)
        secs[s['id']] = dict(id=s['id'], kind=s['kind'], sector=s['sector'], length_m=s['length_m'], near=fr[:k], far=fr[k:][::-1])
    return secs


def answers(port):
    """(section id, date) -> y (1 present, 0 absent) for the cells the owner answered with a usable answer."""
    ans = {}
    if port == 'ZJG':
        rows = [r for r in csv.DictReader(open(common.SAT / 'labels' / 'ZJG-s2-labels.csv', newline='')) if r['verified'] != '']
    else:
        rows = list(csv.DictReader(open(common.SAT / 'labels' / f'{port}-s2-answers.csv', newline='')))
    for r in rows:
        ans[(r['section_id'], r['date'])] = int(r['verified_occupied']) if r['verified'] == '1' and r['verified_occupied'] != '' else None
    return ans


def lookup(port):
    pres = json.load(open(common.SAT / 'eval' / 'presence.json'))
    ev = {(r['section_id'], r['date']): r for r in pres['sites'][port]['rows']}
    ans = answers(port); out = {}
    for sid, date in set(ans) | set(ev):
        e = ev.get((sid, date))
        out[(date, sid)] = dict(y=ans.get((sid, date)), eval=bool(e), y_eval=(e['y'] if e else None), p_cur=(e['p'] if e else None),
                                p_pres1=(e.get('p_pres1') if e else None), dt_h=(e['dt_h'] if e else None))
    return out
