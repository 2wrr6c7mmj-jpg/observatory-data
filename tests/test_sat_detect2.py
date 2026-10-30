"""cfar-v2 PR2 (sat/detect2.py): hysteresis, shape, side lobes, azimuth ambiguities; v1 alongside.

The synthetic scene test (design 5.1 c/d) prints a table to stderr: detection rate by SCR and
length, false exclusions of real ships, exclusion rate of side-lobe pieces and ambiguities, length
error, and the same image through cfar-v1 ``detect()``.
"""
import math
import sys
import unittest
from pathlib import Path

try:
    import numpy as np
except ImportError:  # the collect workflows do not install numpy; Tests does
    np = None

if np is not None:
    from sat import cfar2 as c2
    from sat import detect2 as d2
    from sat.detect import detect

ROOT = Path(__file__).resolve().parents[1]
PX = 20.0
VV_MEAN = 0.01      # -20 dB
VH_MEAN = 0.002     # -27 dB


def _speckle(kind, shape, rng):
    x = rng.gamma(4.4, 1 / 4.4, shape)
    if kind.startswith("K"):
        nu = float(kind[1:])
        x = x * rng.gamma(nu, 1 / nu, shape)
    return x


def _footprint(shape, r0, c0, length, width, ang_deg, sub=4):
    """Fraction of each pixel covered by a rectangle (centre r0, c0 in px; heading ang from north)."""
    h = math.radians(ang_deg)
    ue, un = math.sin(h), math.cos(h)
    ext = int(math.ceil((length + width) / PX / 2)) + 2
    ri, ci = int(round(r0)), int(round(c0))
    rr0, rr1 = max(0, ri - ext), min(shape[0], ri + ext + 1)
    cc0, cc1 = max(0, ci - ext), min(shape[1], ci + ext + 1)
    off = (np.arange(sub) + 0.5) / sub - 0.5
    R, C = np.meshgrid(np.arange(rr0, rr1), np.arange(cc0, cc1), indexing="ij")
    frac = np.zeros(R.shape)
    for dy in off:
        for dx in off:
            e = (C + dx - c0) * PX
            n = -(R + dy - r0) * PX
            a = e * ue + n * un
            b = e * un - n * ue
            frac += (np.abs(a) <= length / 2) & (np.abs(b) <= width / 2)
    frac /= sub * sub
    return (slice(rr0, rr1), slice(cc0, cc1)), frac


class Scene:
    """Synthetic 20 m scene: ships, bright ships with side-lobe pieces and azimuth ambiguities."""

    def __init__(self, kind, heading, lia, seed, n=1024, n_ships=44, n_bright=8, crowd=True):
        rng = np.random.default_rng(seed)
        self.n, self.heading, self.lia = n, heading, lia
        shape = (n, n)
        vv = VV_MEAN * _speckle(kind, shape, rng)
        vh = VH_MEAN * _speckle(kind, shape, rng)
        self.ship_id = np.zeros(shape, np.int32)      # 1.. ships
        self.lobe_id = np.zeros(shape, np.int32)      # 1.. side-lobe pieces
        self.amb_id = np.zeros(shape, np.int32)       # 1.. ambiguity copies
        self.ships, self.lobes, self.ambs = [], [], []
        ue, un = d2._unit(heading)
        az = (-un / PX, ue / PX)                      # (d row, d col) per metre along azimuth
        rg = (ue / PX, un / PX)                       # range = azimuth rotated by +90 deg (east of north)
        sw = d2.subswaths(lia, 0.0)[0]
        delta = d2.ambiguity_offset_m(lia, sw)
        self.delta = delta
        keepout = []                                  # (row, col, radius px) where no other ship goes

        def free(r, c, rad):
            return all((r - a) ** 2 + (c - b) ** 2 >= (rad + q) ** 2 for a, b, q in keepout)

        def add_ship(r, c, L, W, ang, scr, kind_="plain", vh_scr=None):
            sl, f = _footprint(shape, r, c, L, W, ang)
            # SCR = mean target-to-clutter ratio over the pixels the ship touches (a ship narrower
            # than a pixel puts the same mean power into fewer, partly covered pixels)
            g = (f > 0).sum() / f.sum()
            sp = rng.gamma(4.4, 1 / 4.4, f.shape)
            vv[sl] = vv[sl] * (1 - f) + f * g * VV_MEAN * 10 ** (scr / 10) * sp
            vhs = scr if vh_scr is None else vh_scr
            vh[sl] = vh[sl] * (1 - f) + f * g * VH_MEAN * 10 ** (vhs / 10) * rng.gamma(4.4, 1 / 4.4, f.shape)
            self.ships.append({"r": r, "c": c, "L": L, "W": W, "ang": ang, "scr": scr, "kind": kind_})
            ids = self.ship_id[sl]
            ids[f > 0] = len(self.ships)
            return sl, f

        # bright ships (point-like superstructure +36..40 dB) with side lobes and ambiguities
        margin = int(delta / PX) + 40
        tries = 0
        while sum(s["kind"] == "bright" for s in self.ships) < n_bright and tries < 5000:
            tries += 1
            r = rng.uniform(80, n - 80)
            c = rng.uniform(80, n - 80)
            amb_pos = [(r + s * delta * az[0], c + s * delta * az[1]) for s in (1, -1)]
            if not all(45 < a < n - 45 and 45 < b < n - 45 for a, b in amb_pos):
                continue
            if not free(r, c, 45) or not all(free(a, b, 30) for a, b in amb_pos):
                continue
            # bright ship (1部: peak +33..38 dB, 140-180 m): body SCR 25-29 dB, 40 m hot spot +36..40 dB
            L, W = rng.uniform(140, 200), rng.uniform(20, 35)
            ang = rng.uniform(0, 180)
            sl, f = add_ship(r, c, L, W, ang, rng.uniform(25, 29), "bright")
            bid = len(self.ships)
            pi, pj = int(round(r)), int(round(c))
            hot = 10 ** (rng.uniform(36, 40) / 10)
            vv[pi:pi + 2, pj:pj + 2] = VV_MEAN * hot
            vh[pi:pi + 2, pj:pj + 2] = VH_MEAN * hot
            keepout.append((r, c, 12))
            # escorts in column behind the bright ship, along the azimuth axis and the range axis,
            # heading along that axis (1部 review: these were excluded as side lobes)
            for axis, hd_ax in ((az, heading), (rg, heading + 90.0)):
                Le = rng.uniform(60, 120)
                dist = min(L / 2 + rng.uniform(80, 300) + Le / 2, 600 - Le / 2)
                sgn = rng.choice((-1, 1))
                er, ec = r + sgn * dist * axis[0], c + sgn * dist * axis[1]
                add_ship(er, ec, Le, rng.uniform(10, 20), hd_ax % 180, rng.uniform(8, 13), "escort")
                keepout.append((er, ec, Le / PX / 2 + 3))
            # side-lobe pieces along azimuth and range, 3 px long, 14-28 dB below the hot spot
            for axis in (az, rg):
                for dist in (L / 2 + 80, L / 2 + 200, L / 2 + 330, 560):
                    if dist > 590 or dist < L / 2 + 60:
                        continue
                    pair_db = rng.uniform(14, 28)     # side lobes are symmetric: same level on both arms
                    pair = []
                    for sgn in (1, -1):
                        cells = []
                        for k in (0, 1, 2):          # 60 m piece (v1 keeps >= 3 px)
                            ai = int(round(r + sgn * (dist + k * PX) * axis[0]))
                            bi = int(round(c + sgn * (dist + k * PX) * axis[1]))
                            cells.append((ai, bi))
                        pair.append(cells)
                    # a pair touching an escort is left out (the escort's own test needs a clean mirror)
                    if any(self.ship_id[max(0, ai - 3):ai + 4, max(0, bi - 3):bi + 4].any()
                           for cells in pair for ai, bi in cells):
                        continue
                    for cells in pair:
                        drop_db = pair_db + rng.uniform(-1, 1)
                        self.lobes.append({"parent": bid, "drop_db": drop_db, "dist": dist})
                        for ai, bi in cells:
                            vv[ai, bi] = VV_MEAN * hot * 10 ** (-drop_db / 10)
                            vh[ai, bi] = VH_MEAN * hot * 10 ** (-drop_db / 10)
                            self.lobe_id[ai, bi] = len(self.lobes)
            # ambiguity copies: the ship (with its hot spot) 20 dB down at +-delta (+-4% error)
            for a, b in amb_pos:
                err = rng.uniform(-0.04, 0.04) * delta
                a2, b2 = a + err * az[0], b + err * az[1]
                dr, dc = int(round(a2 - r)), int(round(b2 - c))
                src = (slice(sl[0].start, sl[0].stop), slice(sl[1].start, sl[1].stop))
                dst = (slice(sl[0].start + dr, sl[0].stop + dr), slice(sl[1].start + dc, sl[1].stop + dc))
                fm = f > 0.25
                fm[pi - sl[0].start:pi - sl[0].start + 2, pj - sl[1].start:pj - sl[1].start + 2] = True
                vv[dst][fm] += 0.01 * vv[src][fm]
                vh[dst][fm] += 0.01 * vh[src][fm]
                self.ambs.append({"parent": bid})
                self.amb_id[dst][fm] = len(self.ambs)
                # crowded: another ship near the ambiguity -- outside the +-100 m across band, or
                # inside the ellipse but with a different heading (>= 40 deg)
                if crowd:
                    Lc = rng.uniform(50, 100)
                    if len(self.ambs) % 2:
                        off_rg = rng.choice((-1, 1)) * rng.uniform(150, 300)
                        off_az = rng.uniform(-150, 150)
                        hc = rng.uniform(0, 180)
                    else:
                        off_rg = 0.0
                        off_az = rng.choice((-1, 1)) * (L / 2 + Lc / 2 + 60)
                        hc = (ang + rng.uniform(40, 140)) % 180
                    cr = a2 + off_az * az[0] + off_rg * rg[0]
                    cc_ = b2 + off_az * az[1] + off_rg * rg[1]
                    if 40 < cr < n - 40 and 40 < cc_ < n - 40:
                        add_ship(cr, cc_, Lc, rng.uniform(10, 20), hc, rng.uniform(8, 13), "crowd_amb")
                        keepout.append((cr, cc_, Lc / PX / 2 + 3))
            keepout.append((r, c, 45))
            keepout.extend((a, b, 30) for a, b in amb_pos)

        # decoys: real ships next to a bright ship but off the arms, and near +-delta but off by 800 m
        brights = [s for s in self.ships if s["kind"] == "bright"]
        for i, s in enumerate(brights):
            if i % 2 == 0:
                hd = math.radians(heading + 45)
                r = s["r"] - 450 * math.cos(hd) / PX
                c = s["c"] + 450 * math.sin(hd) / PX
                kd = "decoy_lobe"
            else:
                r = s["r"] + delta * az[0] + 800 * rg[0]
                c = s["c"] + delta * az[1] + 800 * rg[1]
                kd = "decoy_amb"
            ri_, ci_ = int(round(r)), int(round(c))
            if 40 < r < n - 40 and 40 < c < n - 40 and not self.ship_id[ri_ - 10:ri_ + 11, ci_ - 10:ci_ + 11].any() \
                    and not self.amb_id[ri_ - 10:ri_ + 11, ci_ - 10:ci_ + 11].any():
                add_ship(r, c, rng.uniform(80, 150), rng.uniform(15, 25), rng.uniform(0, 180), rng.uniform(12, 15), kd)
                keepout.append((r, c, 12))

        # plain ships
        tries = 0
        while sum(s["kind"] == "plain" for s in self.ships) < n_ships and tries < 20000:
            tries += 1
            r, c = rng.uniform(40, n - 40), rng.uniform(40, n - 40)
            L = float(np.exp(rng.uniform(math.log(30), math.log(300))))
            W = rng.uniform(5, min(40, max(6, L / 3)))
            if not free(r, c, L / PX / 2 + 12):
                continue
            add_ship(r, c, L, W, rng.uniform(0, 180), rng.uniform(3, 15))
            keepout.append((r, c, L / PX / 2 + 12))

        q = lambda x: np.round(10 * np.log10(x) * 800) / 800
        self.vv, self.vh = q(vv), q(vh)
        self.land = np.zeros(shape, bool)

    @staticmethod
    def _hit(idmap, d):
        r, c = int(round(d["row"])), int(round(d["col"]))
        w = idmap[max(0, r - 2):r + 3, max(0, c - 2):c + 3]
        return int(w.max())

    def score(self, dets, v2=True):
        """Match detections to ships, side-lobe pieces, ambiguity copies."""
        det_ship = {}
        lobes_hit, lobes_excl, ambs_hit, ambs_excl, false_excl, other = set(), set(), set(), set(), [], 0
        lobes_flag, ambs_flag = set(), set()
        for d in dets:
            excluded = v2 and (d["sidelobe_of"] is not None or d["ambiguity_of"] is not None)
            s = self._hit(self.ship_id, d)
            lo = self._hit(self.lobe_id, d)
            am = self._hit(self.amb_id, d)
            if s:
                if excluded:
                    false_excl.append(s)
                else:
                    det_ship.setdefault(s, []).append(d)
            elif lo:
                lobes_hit.add(lo)
                if excluded:
                    lobes_excl.add(lo)
                elif v2 and (d["sidelobe_suspect"] is not None or d["ambiguity_suspect"] is not None):
                    lobes_flag.add(lo)
            elif am:
                ambs_hit.add(am)
                if excluded:
                    ambs_excl.add(am)
                elif v2 and (d["sidelobe_suspect"] is not None or d["ambiguity_suspect"] is not None):
                    ambs_flag.add(am)
            elif not excluded:
                other += 1
        len_err = []
        for sid, ds in det_ship.items():
            big = max(ds, key=lambda d: d["pixels"])
            len_err.append(big["length_m"] - self.ships[sid - 1]["L"])
        return {"detected": set(det_ship), "len_err": len_err, "lobes_hit": lobes_hit, "lobes_excl": lobes_excl,
                "ambs_hit": ambs_hit, "ambs_excl": ambs_excl, "false_excl": false_excl, "other": other,
                "lobes_flag": lobes_flag, "ambs_flag": ambs_flag,
                "suspect_ships": {self._hit(self.ship_id, d) for d in dets if v2 and self._hit(self.ship_id, d)
                                  and (d["sidelobe_suspect"] is not None or d["ambiguity_suspect"] is not None)},
                "fragments": sum(len(v) - 1 for v in det_ship.values())}


@unittest.skipUnless(np is not None, "numpy not installed")
class GeometryTest(unittest.TestCase):
    def test_platform_heading(self):
        self.assertAlmostEqual(d2.platform_heading("ascending", 22.0), 347.6, delta=0.3)
        self.assertAlmostEqual(d2.platform_heading("descending", 22.0), 192.4, delta=0.3)

    def test_ambiguity_offset_in_design_range(self):
        # design 3.5.7: about 4.3-6.4 km for S1 IW
        for inc in np.arange(29.5, 46.0, 0.5):
            for sw in d2.subswaths(float(inc)):
                self.assertTrue(4300 <= d2.ambiguity_offset_m(float(inc), sw) <= 6400, (inc, sw))
        self.assertAlmostEqual(d2.slant_range_m(0.0), d2.ORBIT_HEIGHT_M, delta=1)

    def test_minmax_filters_and_tophat(self):
        rng = np.random.default_rng(1)
        a = rng.random((37, 41))
        for w in (3, 7, 21):
            h = w // 2
            ref = np.array([[a[max(0, i - h):i + h + 1, max(0, j - h):j + h + 1].min() for j in range(41)]
                            for i in range(37)])
            np.testing.assert_array_equal(d2.min_filter(a, w), ref)
            ref = np.array([[a[max(0, i - h):i + h + 1, max(0, j - h):j + h + 1].max() for j in range(41)]
                            for i in range(37)])
            np.testing.assert_array_equal(d2.max_filter(a, w), ref)
        db = np.zeros((60, 60))
        db[30, 20:25] = 10.0
        th = d2.white_tophat(db, np.ones_like(db, bool), 21)
        self.assertEqual(th[30, 22], 10.0)
        self.assertEqual(th[10, 10], 0.0)

    def test_closing_and_holes(self):
        m = np.zeros((9, 9), bool)
        m[2:7, 2:7] = True
        m[4, 4] = False
        m[2, 5] = False
        self.assertTrue(d2.fill_holes(m)[4, 4])
        self.assertTrue(d2.closing3(m)[2, 5])
        self.assertFalse(d2.closing3(m)[0, 0])

    def test_shape_features(self):
        sl, f = _footprint((200, 200), 100.0, 100.0, 200.0, 40.0, 30.0)
        rr, cc = np.nonzero(f > 0.5)
        sf = d2.shape_features(rr + sl[0].start, cc + sl[1].start, (PX, PX))
        self.assertAlmostEqual(sf["heading_deg"], 30.0, delta=4)
        self.assertAlmostEqual(sf["length_m"], 200.0, delta=30)
        self.assertGreater(sf["fill"], 0.6)
        self.assertGreater(sf["lw_ratio"], 3)


@unittest.skipUnless(np is not None, "numpy not installed")
class DetectV2SmallTest(unittest.TestCase):
    def test_output_fields_and_edges(self):
        rng = np.random.default_rng(4)
        n = 384
        vv = 10 * np.log10(VV_MEAN * rng.gamma(4.4, 1 / 4.4, (n, n)))
        vh = 10 * np.log10(VH_MEAN * rng.gamma(4.4, 1 / 4.4, (n, n)))
        vv[100:104, 100:112] += 15
        vh[100:104, 100:112] += 12
        vv[250:252, 60:62] += 13          # small, VH only clutter -> dropped (not A, too short for B)
        vv[300:302, 200:208] += 14        # 160 m, no VH -> confidence B
        vv[:, :6] = np.nan                # no data on the left
        land = np.zeros((n, n), bool)
        land[:, 360:] = True
        static = np.zeros((n, n), bool)
        static[95:110, 95:120] = True
        dets, info = d2.detect_v2(vv, vh, land, (PX, PX), None, lia=35.0,
                                  heading_deg=d2.platform_heading("ascending", 22.0), static=static)
        v1_keys = ("row", "col", "rmin", "rmax", "cmin", "cmax", "pixels", "length_m", "width_m", "length_class",
                   "zone", "peak_vv_db", "contrast_vv_db", "vh_ratio_db")
        v2_keys = ("lon", "lat", "acquired_utc", "heading_deg", "area_m2", "lw_ratio", "fill", "z_peak", "z_t",
                   "model", "tail_rmse", "vh_z", "tophat_db", "incidence_deg", "sea_state", "near_coast", "static",
                   "edge", "sidelobe_of", "ambiguity_of", "wake_dark_db", "confidence", "large", "pixel_m",
                   "detector_version", "processing_version")
        def at(r, c):
            return [d for d in dets if d["rmin"] <= r <= d["rmax"] and d["cmin"] <= c <= d["cmax"]]
        a = at(101, 105)[0]
        for k in v1_keys + v2_keys:
            self.assertIn(k, a)
        self.assertFalse(any(k.startswith("_") for k in a))
        self.assertEqual(a["confidence"], "A")
        self.assertTrue(a["static"])
        self.assertFalse(a["large"])                    # static
        self.assertEqual(a["incidence_deg"], 35.0)
        self.assertEqual(a["detector_version"], "cfar-v2")
        self.assertAlmostEqual(a["length_m"], 240, delta=25)
        self.assertGreater(a["tophat_db"], 8)
        b = at(300, 203)[0]
        self.assertEqual(b["confidence"], "B")
        self.assertEqual(at(250, 60), [])
        for k in ("valid_fraction", "water_fraction", "threshold_sigma", "guard_px", "outer_px", "hit_pixels",
                  "clusters", "dropped_small", "dropped_vh", "model_counts", "z_t_seed", "exceedance", "excluded"):
            self.assertIn(k, info)
        self.assertGreaterEqual(info["dropped_vh"], 1)
        self.assertEqual(info["ambiguity_check"], "on")

        # lon/lat and GeoJSON
        from sat.grid import Grid
        g = Grid(119.0, 22.0, 119.1, 22.1, n, n)
        d2.add_geo(dets, g, "2026-10-31T10:00:00Z")
        feats = d2.to_features(dets)
        self.assertEqual(feats[0]["geometry"]["type"], "Point")
        self.assertTrue(119.0 < feats[0]["geometry"]["coordinates"][0] < 119.1)
        self.assertEqual(feats[0]["properties"]["acquired_utc"], "2026-10-31T10:00:00Z")

    def test_no_heading_skips_ambiguity(self):
        rng = np.random.default_rng(5)
        vv = 10 * np.log10(VV_MEAN * rng.gamma(4.4, 1 / 4.4, (256, 256)))
        vh = 10 * np.log10(VH_MEAN * rng.gamma(4.4, 1 / 4.4, (256, 256)))
        _dets, info = d2.detect_v2(vv, vh, None, (PX, PX))
        self.assertTrue(info["ambiguity_check"].startswith("skipped"))


@unittest.skipUnless(np is not None, "numpy not installed")
class SyntheticSceneTest(unittest.TestCase):
    """Design 5.1 (c) side lobes / ambiguities handled, no real ship excluded (escorts in column behind
    bright ships, ships crowding the ambiguity positions); (d) length error; seed PFA 1e-7 vs 1e-6."""

    SCENES = (("gamma", "ascending", 35.0, 21), ("K3", "descending", 39.0, 22))

    @classmethod
    def setUpClass(cls):
        cls.res = []
        for kind, direction, lia, seed in cls.SCENES:
            hd = d2.platform_heading(direction, 22.0)
            sc = Scene(kind, hd, lia, seed)
            row = {"scene": sc, "kind": kind, "dir": direction}
            for pfa in (1e-7, 1e-6):
                dets2, info2 = d2.detect_v2(sc.vv, sc.vh, sc.land, (PX, PX), {"pfa_seed": pfa}, lia=lia, heading_deg=hd)
                row[f"v2@{pfa:.0e}"] = sc.score(dets2, True)
                row[f"sec@{pfa:.0e}"] = info2["seconds"]
            row["v2"] = row["v2@1e-07"]
            dets1, _info1 = detect(sc.vv, sc.vh, sc.land, (PX, PX))
            row["v1"] = sc.score(dets1, False)
            cls.res.append(row)
        cls._print()

    @staticmethod
    def _ids(sc, pred):
        return [i for i, s in enumerate(sc.ships, 1) if pred(s)]

    @classmethod
    def _print(cls):
        out = ["", "cfar-v2 PR2 synthetic scenes, 1024x1024 px at 20 m (v1 = sat/detect.py defaults, pfa 1e-5)"]
        bins = (("SCR 3-6", 3, 6), ("SCR 6-10", 6, 10), ("SCR 10-15", 10, 15.01))
        lbins = (("L<50", 0, 50), ("L50-100", 50, 100), ("L>=100", 100, 1e9))
        for r in cls.res:
            sc = r["scene"]
            ships = list(enumerate(sc.ships, 1))
            kinds = {}
            for _i, sh in ships:
                kinds[sh["kind"]] = kinds.get(sh["kind"], 0) + 1
            out.append(f"-- {r['kind']} clutter, {r['dir']} (heading {sc.heading:.1f}), LIA {sc.lia}, "
                       f"delta {sc.delta:.0f} m, ships {kinds}, {len(sc.lobes)} side-lobe pieces, "
                       f"{len(sc.ambs)} ambiguities; v2 {r['sec@1e-07']} s")
            for name in ("v1", "v2@1e-07", "v2@1e-06"):
                s = r[name]
                cells = []
                for lab, lo, hi in bins + lbins:
                    key = "scr" if lab.startswith("SCR") else "L"
                    ids = [i for i, sh in ships if lo <= sh[key] < hi and sh["kind"] not in ("bright",)]
                    cells.append(f"{lab} {sum(i in s['detected'] for i in ids)}/{len(ids)}")
                pl = [i for i, sh in ships if sh["kind"] == "plain" and sh["L"] >= 100]
                cells.append(f"plain L>=100 {sum(i in s['detected'] for i in pl)}/{len(pl)}")
                le = np.array(s["len_err"]) if s["len_err"] else np.array([np.nan])
                out.append(f"  {name}: detected {len(s['detected'])}/{len(ships)} | " + ", ".join(cells))
                out.append(f"      length error (m): median {np.median(le):+.0f}, median |e| {np.median(np.abs(le)):.0f}, "
                           f"fragments {s['fragments']}, other detections {s['other']}")
                if name == "v1":
                    out.append(f"      v1 outputs {len(s['lobes_hit'])}/{len(sc.lobes)} side-lobe pieces and "
                               f"{len(s['ambs_hit'])}/{len(sc.ambs)} ambiguities as ships")
                else:
                    esc = cls._ids(sc, lambda x: x["kind"] == "escort")
                    crw = cls._ids(sc, lambda x: x["kind"] == "crowd_amb")
                    fe = set(s["false_excl"])
                    out.append(f"      side lobes: {len(s['lobes_excl'])} excluded + {len(s['lobes_flag'])} flagged "
                               f"/ {len(s['lobes_hit'])} detected; ambiguities: {len(s['ambs_excl'])} excluded + "
                               f"{len(s['ambs_flag'])} flagged / {len(s['ambs_hit'])} detected")
                    out.append(f"      real ships excluded: {len(fe)} (escorts {len(fe & set(esc))}/{len(esc)}, "
                               f"crowd {len(fe & set(crw))}/{len(crw)}); escorts detected "
                               f"{sum(i in s['detected'] for i in esc)}/{len(esc)}, crowd detected "
                               f"{sum(i in s['detected'] for i in crw)}/{len(crw)}; real ships flagged suspect "
                               f"{len(s['suspect_ships'])}")
        print("\n".join(out), file=sys.stderr)

    def test_sidelobe_and_ambiguity_handled(self):
        for r in self.res:
            s, sc = r["v2"], r["scene"]
            self.assertGreaterEqual(len(s["lobes_hit"]), 10, "test must exercise side lobes")
            self.assertGreaterEqual(len(s["ambs_hit"]), 8, "test must exercise ambiguities")
            # every detected piece the rule applies to (>= 13 dB down on the near arm, >= 20 dB on the
            # far arm) is excluded or flagged; far pieces only 14-20 dB down stay unflagged by design
            due = {k for k in s["lobes_hit"]
                   if sc.lobes[k - 1]["drop_db"] >= (13.5 if sc.lobes[k - 1]["dist"] < 280 else 20.5)}
            self.assertGreaterEqual(len(due & (s["lobes_excl"] | s["lobes_flag"])) / len(due), 0.95)
            self.assertGreaterEqual(len(s["ambs_excl"] | s["ambs_flag"]) / len(s["ambs_hit"]), 0.95)
            # small pieces >= 20 dB below the parent are excluded (>= 95%)
            weak = {k for k in s["lobes_hit"] if sc.lobes[k - 1]["drop_db"] >= 20.5}
            self.assertGreaterEqual(len(weak & s["lobes_excl"]) / len(weak), 0.95)
            self.assertGreaterEqual(len(s["ambs_excl"]) / len(s["ambs_hit"]), 0.8)

    def test_no_real_ship_excluded(self):
        for r in self.res:
            sc = r["scene"]
            for key in ("v2@1e-07", "v2@1e-06"):
                self.assertEqual(r[key]["false_excl"], [], (r["kind"], key))
            for kind in ("decoy_lobe", "decoy_amb", "escort", "crowd_amb"):
                ids = self._ids(sc, lambda x: x["kind"] == kind)
                self.assertTrue(ids, kind)
            if r["kind"] == "gamma":
                esc = self._ids(sc, lambda x: x["kind"] == "escort")
                self.assertGreaterEqual(sum(i in r["v2"]["detected"] for i in esc) / len(esc), 0.9)

    def test_detection_and_length(self):
        for r in self.res:
            ships = r["scene"].ships
            strong = [i for i, s in enumerate(ships, 1) if s["scr"] >= 10 and s["L"] >= 50 and s["kind"] == "plain"]
            self.assertGreaterEqual(sum(i in r["v2"]["detected"] for i in strong) / len(strong), 0.9)
            big = [e for i, e in zip(r["v2"]["detected"], r["v2"]["len_err"]) if ships[i - 1]["L"] >= 100]
            self.assertLessEqual(float(np.median(np.abs(big))), 40.0)

    def test_seed_pfa_is_configurable(self):
        for r in self.res:
            a, b = r["v2@1e-07"]["detected"], r["v2@1e-06"]["detected"]
            self.assertLessEqual(len(a), len(b) + 1)


@unittest.skipUnless(np is not None, "numpy not installed")
class BackgroundOnlyTest(unittest.TestCase):
    """False detections on clutter with no ships: VV and VH share the sea texture (as real sea does),
    so v1's VH check does not remove texture spikes."""

    def test_v2_fewer_false_detections_than_v1(self):
        out = []
        tot1 = tot2 = 0
        for i, nu in enumerate((10.0, 3.0, 1.5)):
            rng = np.random.default_rng(300 + i)
            shape = (1024, 1024)
            tex = rng.gamma(nu, 1 / nu, shape)
            vv = 10 * np.log10(VV_MEAN * tex * rng.gamma(4.4, 1 / 4.4, shape))
            vh = 10 * np.log10(VH_MEAN * tex * rng.gamma(4.4, 1 / 4.4, shape))
            d1, _ = detect(vv, vh, None if False else np.zeros(shape, bool), (PX, PX))
            dv, _ = d2.detect_v2(vv, vh, None, (PX, PX), None, lia=37.0, heading_deg=347.6)
            n2 = len([d for d in d2.kept(dv) if d["confidence"] == "A"])
            out.append(f"K{nu:g}: v1 {len(d1)}, v2 (A) {n2}, v2 (all kept) {len(d2.kept(dv))}")
            tot1 += len(d1)
            tot2 += n2
        print("\nbackground only, 1024x1024 at 20 m (VV/VH shared texture): " + "; ".join(out), file=sys.stderr)
        self.assertGreater(tot1, 0, "the test must make v1 fire")
        self.assertLessEqual(tot2, tot1 / 2)


@unittest.skipUnless(np is not None, "numpy not installed")
class Cfar2ReviewFixTest(unittest.TestCase):
    """1部 PR1 review: smoothing/borrowing never pick a model without params; borrowing distance cap."""

    def _fake_fits(self, failed_tile):
        calls = {"n": 0}
        prm = {"lognormal": (0.0, 0.5), "weibull": (2.0, 0.0), "loggamma": (4.4, 0.0),
               "logweibull3": (-3.0, 4.0, 3.0), "loggengamma": (3.0, 1.0, 0.0)}

        def fake(r, cfg=None, rng=None):
            i = calls["n"]
            calls["n"] += 1
            out = {m: {"params": prm[m], "ll": 0.0, "aic": 10.0, "tail_rmse": 0.1} for m in c2.MODELS}
            if i == failed_tile:
                out["logweibull3"] = {"params": None, "ll": None, "aic": math.inf, "tail_rmse": math.inf}
                out["lognormal"]["aic"] = 0.0
            else:
                out["logweibull3"]["aic"] = 0.0
            return out
        return fake

    def test_smoothing_skips_models_without_params(self):
        rng = np.random.default_rng(9)
        db = 10 * np.log10(VV_MEAN * rng.gamma(4.4, 1 / 4.4, (384, 384)))
        orig = c2.fit_models
        c2.fit_models = self._fake_fits(4)              # centre tile of 3x3: logweibull3 fit failed
        try:
            res = c2.cfar2(db, None, (PX, PX))
        finally:
            c2.fit_models = orig
        centre = [t for t in res["tiles"] if (t["ty"], t["tx"]) == (1, 1)][0]
        self.assertEqual(centre["model"], "lognormal")
        self.assertIsNotNone(centre["params"])
        for t in res["tiles"]:
            self.assertIsNotNone(t["params"])

    def test_borrowing_distance_cap(self):
        rng = np.random.default_rng(10)
        db = 10 * np.log10(VV_MEAN * rng.gamma(4.4, 1 / 4.4, (128, 128 * 6)))
        land = np.zeros(db.shape, bool)
        land[:, 128:] = True                              # only tile column 0 is water
        res = c2.cfar2(db, land, (PX, PX))
        models = {t["tx"]: t["model"] for t in res["tiles"]}
        self.assertIsNotNone(models[3])                   # 3 tiles away: borrows
        self.assertIsNone(models[4])                      # 4 tiles away: no model
        self.assertTrue(np.isnan(c2.threshold_image(res, 1e-7)[:, 128 * 4:]).all())
        res = c2.cfar2(db, land, (PX, PX), {"borrow_max_tiles": 10})
        self.assertIsNotNone(res["tiles"][5]["model"])


@unittest.skipUnless(np is not None, "numpy not installed")
class ExistingPipelineUnchangedTest(unittest.TestCase):
    def test_run_and_v1_do_not_use_detect2(self):
        for name in ("run.py", "detect.py", "quay.py", "quay_est.py", "baseline.py", "render.py"):
            text = (ROOT / "sat" / name).read_text(encoding="utf-8")
            self.assertNotIn("detect2", text, name)
            self.assertNotIn("cfar2", text, name)
        for wf in (ROOT / ".github" / "workflows").glob("*.yml"):
            text = wf.read_text(encoding="utf-8")
            self.assertNotIn("detect2", text, wf.name)

    def test_detect2_dependencies(self):
        src = (ROOT / "sat" / "detect2.py").read_text(encoding="utf-8")
        imports = {ln.split()[1] for ln in src.splitlines() if ln.startswith(("import ", "from "))}
        self.assertLessEqual(imports, {"__future__", "math", "time", "numpy", ".", ".grid"})


if __name__ == "__main__":
    unittest.main()
