"""Number of ships alongside, as a distribution (est-1, trial): narrows the quay-v1 range with the
lengths of the PLA Navy's ships and the kinds of ships each pier usually takes.

For every occupied stretch of quay-v1 (sat/quay.py):

1. Observed length L_obs = stretch length - trim (20 m: the 3x3 mean widens it).
2. n ships moored bow to stern have a total length S_n = sum(l_i) + sum(g_j) (n - 1 gaps,
   g ~ U(10, 40) m). The l_i are drawn from the ship-class table (sat/ship_classes.json: length,
   number in the Southern Theatre, weight w for Zhanjiang; each length + N(0, 3 m)), weighted
   for the stretch's zone (sat/quay_zones.json: per-pier multipliers by length group, mixed
   0.7 / 0.3 with the overall mix) and by the zone's Dirichlet over six length classes
   (updated from clean single-ship stretches, see ZonePrior). 20 000 draws per n (fixed seed).
3. Likelihood of n: N(L_obs; S_n, 20 m) (a histogram of S_n smoothed by the Gaussian), times
   P(k | n) for the k pieces quay-v1 found in the dVV profile (0.95 if n >= k, 0.05^(k-n) if n < k).
   Prior of n uniform on 1..16.
4. Outboard ship (rafted): the outer band's occupied length is estimated the same way and added.
5. Stretches within 60 m of each other (both sides of a pier, a corner): every member but the lead
   is the same ship's smear (0 ships) with probability q (0.5 under 150 m, 0.2 otherwise).
6. Small craft: stretches under 60 m with peak dVH under 6 dB get the T group's weight x3.
   "All" counts every vessel, "warship" only those of 60 m or more.
7. Site total = convolution of the stretches (independent). Structural uncertainty (gaps, trim, q)
   is a mixture of equally weighted scenarios.

Trial: the class table, weights and pier multipliers are all estimates; the accuracy has not
been measured.
"""
from __future__ import annotations

import json
import math
from itertools import product
from pathlib import Path

import numpy as np

EST_VERSION = "est-1"
HERE = Path(__file__).resolve().parent
CLASSES_PATH = HERE / "ship_classes.json"
ZONES_PATH = HERE / "quay_zones.json"

N_MAX = 16
DRAWS = 20000
SEED = 20261030
SIGMA_M = 20.0
BIN_M = 5.0
L_MAX_M = 16 * 260 + 15 * 60 + 200
P_K_OK = 0.95
P_K_MISS = 0.05
WARSHIP_MIN_M = 60.0
SMALL_LEN_M = 60.0
SMALL_DVH_DB = 6.0
SMALL_T_BOOST = 3.0
MIX_GLOBAL = 0.3
GROUPS = ("X", "L", "M", "S", "T")
LENGTH_BINS_M = [0.0, 60.0, 100.0, 145.0, 170.0, 200.0, math.inf]
BIN_NAMES = ["<60", "60-100", "100-145", "145-170", "170-200", ">200"]
KAPPA = 10.0
FORGET = 0.955
CLEAN_MIN_M, CLEAN_MAX_M = 40.0, 260.0
# Structural scenarios (equal weights): gaps between ships, trim of the stretch length, q of the smear
GAPS_M = {"short": (5.0, 15.0), "base": (10.0, 40.0), "long": (20.0, 60.0)}
TRIMS_M = (10.0, 20.0, 30.0)
Q_SETS = {"low": (0.3, 0.1), "base": (0.5, 0.2), "high": (0.7, 0.4)}  # (q under 150 m, q from 150 m)
Q_SPLIT_M = 150.0
P_RAFT = (0.5, 0.8, 1.0)  # probability that an outer-band occupation is really a ship moored outboard
BASE_GAP, BASE_TRIM, BASE_Q = "base", 20.0, "base"
# n = 0 (a false alarm: cargo-handling gear, a structure, a smear) has prior P0, with a stretch length
# uniform on 0..FALSE_MAX_M and k as for one object. A placeholder (no ground truth yet), see settings().
P0 = 0.05
FALSE_MAX_M = 150.0


def load_classes(path: Path = CLASSES_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_zones(path: Path = ZONES_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


DEFAULT_SITE = "ZJG"   # the table of a site without its own weights (as before: every site used Zhanjiang's)
ROLE_ZONE = {          # a site not in quay_zones.json: the zone type from the sector's role (never silently shore)
    "naval": ("NV", "medium"),
    "other": ("SH", "shore"),
}


def resolve_classes(table: dict, site_id: str | None = None) -> tuple[list[dict], str]:
    """The class rows of a site: [{..., count, w}] in the order of the table, and the site key used.

    est-1b (classes + fleets + sites): count from the site's fleet, w from the site; a class without a
    weight for the site is left out. The old flat form (count and w in every class) is used as it is."""
    if "sites" not in table:
        return [dict(c) for c in table["classes"]], DEFAULT_SITE
    key = site_id if site_id in table["sites"] else DEFAULT_SITE
    st = table["sites"][key]
    fleet = table["fleets"][st["fleet"]]
    out = []
    for c in table["classes"]:
        w = st["w"].get(c["id"])
        f = fleet.get(c["id"])
        if w is None or f is None or not (float(w) > 0 and float(f["count"]) > 0):
            continue
        det = (st.get("w_detail") or {}).get(c["id"]) or {}  # per-class credibility (2部), else the site's
        out.append({**c, "count": f["count"], "count_conf": f.get("conf", "推定"), "w": w,
                    "w_conf": det.get("conf") or st.get("w_conf", "推定")})
    return out, key


def classes_version(path: Path = CLASSES_PATH) -> str:
    return load_classes(path).get("version", "")


def site_weights_info(site_id: str, path: Path = CLASSES_PATH) -> dict:
    """Where a site's est-1 weights come from (meta.json sites[].quay.class_weights)."""
    t = load_classes(path)
    rows, key = resolve_classes(t, site_id)
    st = (t.get("sites") or {}).get(key, {})
    return {"site_key": key, "own": key == site_id, "fleet": st.get("fleet"), "w_conf": st.get("w_conf"),
            "status": st.get("status", "estimate"), "basis": st.get("basis"),
            "classes": [c["id"] for c in rows]}


def length_bin(length_m: float) -> int:
    for b in range(len(LENGTH_BINS_M) - 1):
        if LENGTH_BINS_M[b] <= length_m < LENGTH_BINS_M[b + 1]:
            return b
    return len(LENGTH_BINS_M) - 2


class Model:
    """Class table + zone multipliers; class weights per zone; cached simulation tables."""

    def __init__(self, classes: dict | None = None, zones: dict | None = None, use_zones: bool = True,
                 site_id: str | None = None, sector_roles: dict | None = None):
        self.table = classes or load_classes()
        self.zcfg = zones or load_zones()
        self.use_zones = use_zones
        self.classes, self.class_site = resolve_classes(self.table, site_id)
        self.sector_roles = dict(sector_roles or {})
        self.warnings: list[str] = []
        cl = self.classes
        self.lengths = np.array([c["length_m"] for c in cl], float)
        self.groups = [c["group"] for c in cl]
        self.kinds = [c.get("kind") for c in cl]
        self.sd = float(self.table.get("length_sd_m", 3.0))
        w = np.array([c["count"] * c["w"] for c in cl], float)
        self.global_w = w / w.sum()
        self.bin_of = np.array([length_bin(x) for x in self.lengths])
        self._cache: dict = {}

    # --- zones -------------------------------------------------------------------------------
    def zone_of(self, site_id: str, way_id, sector) -> tuple[str, str]:
        """(zone id, zone type) of a stretch."""
        s = self.zcfg["sites"].get(site_id)
        if s is None:
            role = self.sector_roles.get(sector)
            if role is None:
                return "SH", "shore"
            if role not in ROLE_ZONE:
                self._warn(f"{site_id}: 区画 {sector} の role {role} に桟橋の型が無いため shore とした")
                return "SH", "shore"
            return ROLE_ZONE[role]
        ov = (s.get("sector_override") or {}).get(sector)
        if ov:
            return ov["id"], ov["type"]
        for z in s["zones"]:
            if way_id in z["ways"]:
                return z["id"], z["type"]
        return s["default"]["id"], s["default"]["type"]

    def _warn(self, msg: str) -> None:
        if msg not in self.warnings:
            self.warnings.append(msg)

    def zone_ids(self, site_id: str) -> dict[str, str]:
        s = self.zcfg["sites"].get(site_id)
        if s is None:
            out = {"SH": "shore"}
            for role in set(self.sector_roles.values()):
                if role in ROLE_ZONE:
                    zid, zt = ROLE_ZONE[role]
                    out[zid] = zt
            return out
        out = {z["id"]: z["type"] for z in s["zones"]}
        out[s["default"]["id"]] = s["default"]["type"]
        for ov in (s.get("sector_override") or {}).values():
            out[ov["id"]] = ov["type"]
        return out

    def zone_weights(self, ztype: str) -> np.ndarray:
        """pi_z' = 0.7 pi_z + 0.3 pi_g, pi_z proportional to pi_g x the group multiplier."""
        if not self.use_zones:
            return self.global_w.copy()
        t = self.zcfg["types"][ztype]
        mult = t["mult"]
        by_kind = t.get("mult_kind") or {}  # e.g. submarine: the submarines x3, every other class mult_other
        if by_kind:
            m = np.array([by_kind[k] if k in by_kind else float(t.get("mult_other", mult[g]))
                          for g, k in zip(self.groups, self.kinds)], float)
        else:
            m = np.array([mult[g] for g in self.groups])
        pz = self.global_w * m
        pz = pz / pz.sum()
        return (1 - MIX_GLOBAL) * pz + MIX_GLOBAL * self.global_w

    def bin_mass(self, w: np.ndarray) -> np.ndarray:
        return np.bincount(self.bin_of, weights=w, minlength=len(BIN_NAMES))

    def reweight(self, w: np.ndarray, alpha) -> np.ndarray:
        """Scale the classes so that each length class has the Dirichlet mean share."""
        if alpha is None:
            return w
        a = np.asarray(alpha, float)
        share = a / a.sum()
        mass = self.bin_mass(w)
        f = np.where(mass > 0, share / np.where(mass > 0, mass, 1), 0.0)
        out = w * f[self.bin_of]
        return out / out.sum() if out.sum() > 0 else w

    def small_boost(self, w: np.ndarray) -> np.ndarray:
        out = w * np.array([SMALL_T_BOOST if g == "T" else 1.0 for g in self.groups])
        return out / out.sum()

    # --- simulation --------------------------------------------------------------------------
    def table_for(self, w: np.ndarray, gap: str):
        """Smoothed P(L, n_warship | n) on a 5 m grid: array (N_MAX, N_MAX + 1, bins)."""
        key = (gap, np.round(w, 6).tobytes())
        if key in self._cache:
            return self._cache[key]
        rng = np.random.default_rng(SEED + list(GAPS_M).index(gap))
        cls = rng.choice(len(w), size=(DRAWS, N_MAX), p=w / w.sum())
        ln = self.lengths[cls] + rng.normal(0.0, self.sd, size=cls.shape)
        lo, hi = GAPS_M[gap]
        gp = rng.uniform(lo, hi, size=(DRAWS, N_MAX - 1))
        s = np.cumsum(ln, 1)
        s[:, 1:] += np.cumsum(gp, 1)
        war = np.cumsum(ln >= WARSHIP_MIN_M, 1)
        nb = int(L_MAX_M // BIN_M) + 1
        idx = np.clip((s / BIN_M).astype(int), 0, nb - 1)
        h = np.zeros((N_MAX, N_MAX + 1, nb))
        for n in range(N_MAX):
            np.add.at(h[n], (war[:, n], idx[:, n]), 1.0)
        h /= DRAWS
        # Gaussian N(0, SIGMA) along L, as a density per metre
        r = int(math.ceil(4 * SIGMA_M / BIN_M))
        x = np.arange(-r, r + 1) * BIN_M
        kern = np.exp(-0.5 * (x / SIGMA_M) ** 2) / (SIGMA_M * math.sqrt(2 * math.pi))
        flat = h.reshape(-1, nb)
        sm = np.stack([np.convolve(row, kern, mode="same") for row in flat]).reshape(h.shape)
        self._cache[key] = sm
        return sm

    def posterior(self, l_obs: float, k: int, w: np.ndarray, gap: str, p0: float = P0):
        """(P(n), P(n_warship)) for one occupied run; arrays indexed by count 0..N_MAX.
        n = 0 (false alarm) has prior odds p0 : (1 - p0) against n = 1; n = 1..N_MAX are uniform."""
        t = self.table_for(w, gap)
        b = int(np.clip(round(l_obs / BIN_M), 0, t.shape[2] - 1))
        like = t[:, :, b]  # (n-1, n_war)
        n = np.arange(1, N_MAX + 1)
        pk = np.where(n >= k, P_K_OK, P_K_MISS ** np.maximum(k - n, 0))
        joint = like * pk[:, None] / N_MAX
        pk0 = P_K_OK if k <= 1 else P_K_MISS ** (k - 1)  # a false alarm is one object
        # prior odds of n = 0 against n = 1 are p0 : (1 - p0) (not p0 of the whole uniform prior,
        # which would make short stretches mostly "false" only because 1..16 spreads the ships' prior)
        zero = (p0 / (1 - p0) / N_MAX) * pk0 * (1.0 / FALSE_MAX_M if 0 <= l_obs <= FALSE_MAX_M else 0.0)
        tot = joint.sum() + zero
        if not tot > 0:  # far outside every S_n (e.g. a very long run): fall back to the nearest n
            joint = np.zeros_like(like)
            mean_len = (self.lengths * w).sum() / w.sum()
            nn = int(np.clip(round((l_obs + 25) / (mean_len + 25)), max(1, k), N_MAX))
            joint[nn - 1, nn] = 1.0
            tot, zero = 1.0, 0.0
        joint = joint / tot
        pn = np.concatenate([[zero / tot], joint.sum(1)])
        pw = joint.sum(0)
        pw[0] += zero / tot
        return pn, pw

    def single_predictive(self, w: np.ndarray):
        """Mixture of N(l_c, sqrt(sd^2 + SIGMA^2)) for one ship: (means, sd, weights)."""
        return self.lengths, math.hypot(self.sd, SIGMA_M), w / w.sum()


def _ncdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _quant(p: np.ndarray, q: float) -> int:
    c = np.cumsum(p)
    return int(np.searchsorted(c, q * c[-1] - 1e-12))


def stats(p: np.ndarray) -> dict:
    p = np.asarray(p, float)
    p = p / p.sum()
    x = np.arange(len(p))
    return {"mode": int(np.argmax(p)), "p10": _quant(p, 0.1), "p90": _quant(p, 0.9), "mean": round(float((x * p).sum()), 1)}


def _conv(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.convolve(a, b)


def _smear(p: np.ndarray, q: float) -> np.ndarray:
    out = (1 - q) * p
    out[0] += q
    return out


class ZonePrior:
    """Per-zone Dirichlet over the six length classes, updated from clean single-ship stretches.

    alpha <- lambda * alpha + (1 - lambda) * alpha0 + r once per scene (r = soft counts of the
    scene's clean stretches): old scenes fade, but the prior alpha0 (kappa = 10) is never forgotten.
    """

    def __init__(self, model: Model, site_id: str, state: dict | None = None):
        self.model, self.site_id = model, site_id
        self.state = state or {}
        zones = self.state.setdefault("zones", {})
        for zid, ztype in model.zone_ids(site_id).items():
            if zid not in zones:
                a0 = (KAPPA * model.bin_mass(model.zone_weights(ztype))).round(4).tolist()
                zones[zid] = {"type": ztype, "alpha0": a0, "alpha": list(a0), "updates": 0, "scenes": 0}
        self.state.update(version=EST_VERSION, site_id=site_id, kappa=KAPPA, forget=FORGET, bins_m=BIN_NAMES)
        self.state.setdefault("applied", [])

    @classmethod
    def load(cls, model: Model, site_id: str, path: Path | None):
        st = json.loads(path.read_text(encoding="utf-8")) if path is not None and path.exists() else None
        return cls(model, site_id, st)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.state, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    def alpha(self, zid: str):
        z = self.state["zones"].get(zid)
        return z["alpha"] if z else None

    def weights(self, zid: str, ztype: str) -> np.ndarray:
        m = self.model
        return m.reweight(m.zone_weights(ztype), self.alpha(zid) if m.use_zones else None)

    def check(self, zid: str, ztype: str, l_obs: float):
        """(outside the prior's 95 % interval for one ship?, soft counts over the length classes)."""
        means, sd, w = self.model.single_predictive(self.weights(zid, ztype))
        cdf = float(sum(wi * _ncdf((l_obs - mu) / sd) for mu, wi in zip(means, w)))
        dens = w * np.exp(-0.5 * ((l_obs - means) / sd) ** 2)
        r = np.bincount(self.model.bin_of, weights=dens, minlength=len(BIN_NAMES))
        r = r / r.sum() if r.sum() > 0 else r
        return (cdf < 0.025 or cdf > 0.975), r

    def digest(self) -> str:
        """Short hash of every zone's alpha (the prior a scene was estimated with)."""
        import hashlib
        z = {k: v["alpha"] for k, v in sorted(self.state["zones"].items())}
        return hashlib.sha256(json.dumps(z, sort_keys=True).encode()).hexdigest()[:12]

    def record(self, acquired_utc: str, used: bool, note: str = "") -> None:
        """History: the alpha each scene was estimated with (to re-compute later)."""
        h = [x for x in self.state.setdefault("history", []) if x["acquired_utc"] != acquired_utc]
        h.append({"acquired_utc": acquired_utc, "prior_hash": self.digest(), "updated": used, "note": note,
                  "alpha": {k: v["alpha"] for k, v in sorted(self.state["zones"].items())}})
        self.state["history"] = sorted(h, key=lambda x: x["acquired_utc"])[-400:]

    def update(self, acquired_utc: str, soft: dict[str, list[np.ndarray]]) -> bool:
        if acquired_utc in self.state["applied"]:
            return False
        for zid, z in self.state["zones"].items():
            a = np.array(z["alpha"], float)
            a0 = np.array(z["alpha0"], float)
            a = FORGET * a + (1 - FORGET) * a0
            for r in soft.get(zid, []):
                a = a + r
                z["updates"] += 1
            if soft.get(zid):
                z["scenes"] += 1
            z["alpha"] = a.round(4).tolist()
        self.state["applied"] = sorted(self.state["applied"] + [acquired_utc])[-400:]
        self.state["last_update_utc"] = acquired_utc
        return True


def estimate(segments: list[dict], site_id: str, model: Model | None = None, prior: ZonePrior | None = None,
             sectors: tuple = ("A", "B"), update_prior: bool = True) -> dict:
    """Distribution of the number of ships for one scene's stretches (quay.count()["segments"]).

    Adds to every segment: zone, face, k, n_mode, n_p10, n_p90, n_mean, warship_mode, prior_conflict,
    clean (used for the prior update). Returns totals {"all", "warship", "by_sector": {id: stats}},
    "soft" (per-zone soft counts for ZonePrior.update) and "scenarios".
    """
    model = model or Model()
    prior = prior or ZonePrior(model, site_id)
    seg_info = []
    soft: dict[str, list] = {}
    for sg in segments:
        zid, ztype = model.zone_of(site_id, sg.get("way_id"), sg.get("sector"))
        sg["zone"] = zid
        sg["face"] = f"{sg.get('way_id')}{'+' if sg.get('side', 1) > 0 else '-'}"
        w = prior.weights(zid, ztype)
        small = sg["length_m"] < SMALL_LEN_M and sg.get("peak_dvh_db") is not None and sg["peak_dvh_db"] < SMALL_DVH_DB
        if small:
            w = model.small_boost(w)
        k = int(sg.get("min", 1))
        sg["k"] = k
        lead = sg.get("cluster_lead", True) is not False
        seg_info.append((sg, w, k, lead))
    scen = list(product(GAPS_M, TRIMS_M, Q_SETS, P_RAFT))
    tot_all, tot_war = [], []
    tot_sec = {s: [] for s in sectors}
    seg_mix = [[np.zeros(1), np.zeros(1)] for _ in seg_info]
    capped = [False] * len(seg_info)
    for gap, trim, qk, p_raft in scen:
        q_short, q_long = Q_SETS[qk]
        ta, tw = np.ones(1), np.ones(1)
        ts = {s: np.ones(1) for s in sectors}
        for i, (sg, w, k, lead) in enumerate(seg_info):
            l_obs = max(sg["length_m"] - trim, 0.0)
            pn, pw = model.posterior(l_obs, k, w, gap)
            if _quant(pn, 0.9) >= N_MAX:
                capped[i] = True
            out_len = float(sg.get("outer_len_m") or 0)
            if out_len > 0:
                on, ow = model.posterior(max(out_len - trim, 0.0), 1, w, gap, p0=0.0)
                if _quant(on, 0.9) >= N_MAX:
                    capped[i] = True
                on, ow = _smear(on, 1 - p_raft), _smear(ow, 1 - p_raft)  # 0 outboard with 1 - p_raft
                pn, pw = _conv(pn, on), _conv(pw, ow)
            if not lead:
                q = q_short if sg["length_m"] < Q_SPLIT_M else q_long
                pn, pw = _smear(pn, q), _smear(pw, q)
            for j, p in enumerate((pn, pw)):
                cur = seg_mix[i][j]
                if len(cur) < len(p):
                    cur = np.pad(cur, (0, len(p) - len(cur)))
                cur[:len(p)] += p / len(scen)
                seg_mix[i][j] = cur
            ta, tw = _conv(ta, pn), _conv(tw, pw)
            sec = sg.get("sector")
            if sec in ts:
                ts[sec] = _conv(ts[sec], pn)
        tot_all.append(ta)
        tot_war.append(tw)
        for s in sectors:
            tot_sec[s].append(ts[s])

    def mix(ps):
        m = max(len(p) for p in ps)
        return sum(np.pad(p, (0, m - len(p))) for p in ps) / len(ps)

    n_conflict = 0
    for i, (sg, w, k, lead) in enumerate(seg_info):
        sn, sw = stats(seg_mix[i][0]), stats(seg_mix[i][1])
        sg.update(n_mode=sn["mode"], n_p10=sn["p10"], n_p90=sn["p90"], n_mean=sn["mean"], warship_mode=sw["mode"])
        conflict = []
        if k > sn["p90"]:
            conflict.append("valleys")
        if capped[i]:
            conflict.append("cap")  # the 80 % interval reaches the cap of 16 ships: the count may be higher
        # clean single ship: updates the zone's prior unless it is outside the prior's 95 % interval
        l_obs = sg["length_m"] - BASE_TRIM
        clean = (update_prior and k == 1 and not (sg.get("outer_len_m") or 0) and lead
                 and CLEAN_MIN_M <= l_obs <= CLEAN_MAX_M)
        sg["clean"] = False
        if clean:
            zid, ztype = model.zone_of(site_id, sg.get("way_id"), sg.get("sector"))
            out, r = prior.check(zid, ztype, l_obs)
            if out:
                conflict.append("length")
            else:
                soft.setdefault(zid, []).append(r)
                sg["clean"] = True
        sg["prior_conflict"] = ",".join(conflict)
        n_conflict += bool(conflict)
    res = {
        "all": stats(mix(tot_all)) if seg_info else {"mode": 0, "p10": 0, "p90": 0, "mean": 0.0},
        "warship": stats(mix(tot_war)) if seg_info else {"mode": 0, "p10": 0, "p90": 0, "mean": 0.0},
        "by_sector": {s: (stats(mix(tot_sec[s])) if seg_info else {"mode": 0, "p10": 0, "p90": 0, "mean": 0.0}) for s in sectors},
        "conflicts": n_conflict, "capped": int(sum(capped)), "soft": soft, "scenarios": len(scen),
        "prior_hash": prior.digest(),
    }
    return res


def cols_hint(depth_m) -> int | None:
    """Ships abreast from the occupied depth: round((W - 20) / 18), 1..3 (a clue only, not used)."""
    if depth_m is None:
        return None
    return int(min(3, max(1, round((depth_m - 20) / 18))))


LIMITS = [
    "80% 区間はモデル内の区間で、次のものは入っていない（実数は区間の外にもあり得る）",
    f"横抱き：外側の帯の占有が本当に船である確率 p_raft を {'／'.join(f'{x:g}' for x in P_RAFT)} の混合で入れたが、この値自体は未検証（光学画像で確かめていない）",
    f"誤検出：n＝0 は n＝1 に対するオッズ {P0:g} の仮置きで、{FALSE_MAX_M:g} m を超える区間は誤検出と見ない。長い区間の一部だけが誤検出（荷役機械と船が並ぶ等）の場合は入っていない",
    "数えられない区間：基準期間の約 9 割以上同じ場所に居た船・常に埋まっていた岸壁（blind_stretches）の船は区間そのものが無く、推定に入らない",
    "OSM に無い岸壁・桟橋に横付けされた船は数えない",
    "艦種表（隻数・湛江への重み）と桟橋の倍率の誤り：すべて推定で、誤っていれば最頻値も区間もずれる。054A＋と掃海艇の全長は未確認、タグ・作業船は出典なし",
    "区間どうしの独立の仮定：同じ桟橋の区間・両側の区間の相関（例：同じ艦隊がまとまって在泊）は入れていない。合計の区間は狭く出やすい",
    f"上限：1 区間 {N_MAX} 隻まで（外側の帯も同じ）。80% 区間の上端が上限に達した区間は prior_conflict＝cap",
    "精度は測っていない。2026-10-26 以前の撮影は quay-v1 の閾値の調整に使った標本内の値（事前分布の更新にも使わない）",
]


def class_table(site_id: str | None = None) -> list[dict]:
    """The class rows of a site (default: Zhanjiang, as in est-1)."""
    rows, _key = resolve_classes(load_classes(), site_id)
    return [{"id": c["id"], "name": c["name"], "length_m": c["length_m"], "length_conf": c["length_conf"],
             "count": c["count"], "count_conf": c.get("count_conf", "推定"), "w": c["w"], "w_conf": c.get("w_conf", "推定"),
             "group": c["group"], "source": c["source"], **({"kind": c["kind"]} if c.get("kind") else {}),
             **({"note": c["note"]} if c.get("note") else {})} for c in rows]


def site_class_tables() -> dict:
    """Every site with its own weights: {site: {status, basis, sources (if any), classes}} (meta.json).
    A class's w_conf is its own credibility (w_detail) when the site has one."""
    t = load_classes()
    return {sid: {"fleet": st.get("fleet"), "w_conf": st.get("w_conf"), "status": st.get("status", "estimate"),
                  "basis": st.get("basis"), **({"sources": st["sources"]} if st.get("sources") else {}),
                  "classes": class_table(sid)} for sid, st in (t.get("sites") or {}).items()}


def zone_table() -> dict:
    z = load_zones()
    return {"types": {k: {**v, "conf": "推定", "basis": v.get("basis", "")} for k, v in z["types"].items()},
            "sites": z["sites"]}


def settings() -> dict:
    m = Model()
    gw = {c["id"]: round(float(x), 4) for c, x in zip(m.classes, m.global_w)}
    return {
        "version": EST_VERSION,
        "status": "試行（精度未測定）。艦種表・重み・桟橋の倍率はすべて推定を含む。湛江 v2 の調整に使った撮影（標本内）で確かめただけ",
        "limits": LIMITS,
        "class_table_file": "sat/ship_classes.json（全長は各艦種の Wikipedia 英語版、隻数・湛江への重み w は推定）",
        "zones_file": "sat/quay_zones.json（区画＝OSM の way_id、面＝way_id＋側。桟橋の型ごとの群の倍率は推定）",
        "global_mix": gw,
        "per_stretch": (f"L_obs＝区間長−{BASE_TRIM:g} m。n 隻を縦に並べた全長 S_n＝Σℓ＋Σg（ℓ は艦種表から区画の重みで引き N(0,3 m) を足す、g は隻の間隔 U(10,40) m）を"
                        f"{DRAWS} 本（種 {SEED}）で作り、尤度 N(L_obs; S_n, {SIGMA_M:g} m)。断面の谷の数 k（quay-v1 の最小の片の数）は P(k|n)＝{P_K_OK}（n≥k）、{P_K_MISS}^(k−n)（n<k）。"
                        f"n の事前は 1〜{N_MAX} の一様"),
        "zone_prior": (f"π_z ∝ π_g × 群の倍率、π_z'＝{1 - MIX_GLOBAL:g} π_z＋{MIX_GLOBAL:g} π_g。さらに区画ごとの長さ 6 階級（{'、'.join(BIN_NAMES)} m）の"
                       f"ディリクレ（初期 κ＝{KAPPA:g}）の平均に合わせて艦種の重みを階級ごとに掛け直す"),
        "zone_update": (f"きれいな単艦の区間（k＝1、横抱きなし、まとまりの代表、L_obs {CLEAN_MIN_M:g}〜{CLEAN_MAX_M:g} m）の長さの階級の事後確率を軟らかい数として足す。"
                        f"撮影ごとに α←{FORGET} α＋{1 - FORGET:.3f} α0＋r（古い撮影は薄れるが初期値 α0 は残す）。単艦の事前予測の 95% 区間を外れた区間は prior_conflict＝length として更新から外す。"
                        "推定はその撮影の更新の前の事前分布で行う（自分のデータを二重に使わない）。状態は quay/<地点>-zone-prior.json"),
        "rafted": "外側の帯の占有の長さ outer_len_m を同じ方式（k＝1）で推定し、内側に畳み込む",
        "cluster": f"60 m 以内のまとまりの代表以外は確率 q（区間長 {Q_SPLIT_M:g} m 未満 0.5、以上 0.2）で同じ艦のにじみ（0 隻）",
        "small_craft": f"区間長 {SMALL_LEN_M:g} m 未満で ΔVH の最大が {SMALL_DVH_DB:g} dB 未満なら T 群（50 m 未満）の重みを {SMALL_T_BOOST:g} 倍。全船と艦艇（{WARSHIP_MIN_M:g} m 以上）を別に出す",
        "total": "区間の分布を畳み込む（区間どうしは独立と仮定）。A・B は区間の中央の区画",
        "scenarios": ("区間に入れている不確かさ：区間長と艦の長さの照合（σ 20 m）、艦種の混ざり方（区画の事前分布）、谷の数の数え違い（P(k|n)）、"
                      "そして等重みの 81 通りの混合＝隻の間隔 U(5,15)／U(10,40)／U(20,60) m × 長さの削り幅 10／20／30 m × にじみの確率 q（0.3/0.1、0.5/0.2、0.7/0.4）"
                      f" × 横抱きが本当に船である確率 p_raft（{'／'.join(f'{x:g}' for x in P_RAFT)}）。入れていないものは limits"),
        "false_alarm": (f"n＝0（誤検出：荷役機械・構造物・にじみ）を、n＝1 に対する事前のオッズ {P0:g}：{1 - P0:g} で入れる。誤検出の区間の長さは 0〜{FALSE_MAX_M:g} m の一様、谷は 1 つの物として扱う。"
                        "値は仮置き：正解データが無く、quay-v1 は ΔVV 12 dB の種と今回の VV −10 dB で風・にじみを抑えているため、残る誤検出を区間の 1 割未満と見た（未検証）。"
                        "全体の事前の 5% とすると、一様事前（1〜16）が船の側の事前を薄めるため短い区間の大半が 0 隻になってしまうので、n＝1 との比にした"),
        "interval": "80% 区間はモデル内の区間（このモデルと前提が正しいとしたときの p10〜p90）で、実数がこの区間に入る確率ではない",
        "output": "最頻値 mode（モデル上の最頻値）、モデル内の 80% 区間 p10〜p90（累積 10%・90% に達する最小の数）、平均 mean",
        "warship": f"全長 {WARSHIP_MIN_M:g} m 以上の船の数（艦種表の長さによる区別で、艦艇以外の船を含み得る）",
        "prior_conflict": (f"valleys＝谷の数 k が n の 80% 区間の上端を超えた、length＝単艦として区画の事前分布の 95% 区間の外、"
                           f"cap＝区間の（または外側の帯の）80% 区間の上端が上限 {N_MAX} 隻に達した（実数はもっと多い可能性）"),
        "class_table": class_table(),
        "site_class_tables": site_class_tables(),
        "class_table_version": classes_version(),
        "zone_multipliers": zone_table(),
        "depth": "depth_m＝内側の帯で占有のある点で、水際から法線方向 0〜120 m（5 m おき。feat-1 の前は 0〜90 m）に占有が続く奥行きの中央値。cols_hint＝round((depth−20)/18)（1〜3）は横に並ぶ列数の手がかりとして出すだけで推定には使わない（3×3 平均のにじみと艦幅の不確かさが大きく未検証）",
    }
