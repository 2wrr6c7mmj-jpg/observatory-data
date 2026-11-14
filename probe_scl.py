import json, numpy as np
from sat import s2, cdse
from sat.grid import Grid
from sat.run import load_sites
from sat.osm_mask import MASK_DIR, load_mask
client = cdse.Client()
sites = {s["id"]: s for s in load_sites()}
out = []
for sid in ("QGZ", "QDH", "QJG"):
    site = sites[sid]; grid = Grid.for_site(site)
    in_a = s2.sector_index(grid, site)
    mask, _ = load_mask(sid, int(site.get("mask_version", 1)), MASK_DIR)
    for date in ("2026-09-04", "2026-10-27", "2026-11-11"):
        scl, pu = s2.precheck(client, grid, date)
        a = s2.resample_mask(in_a, scl.shape)
        land = s2.resample_mask(np.asarray(mask, bool), scl.shape)
        valid = scl != s2.NODATA
        def share(cls, where):
            w = where & valid
            return round(float((np.isin(scl, cls) & w).sum() / max(w.sum(), 1)), 4)
        allm = np.ones_like(valid)
        r = {"site": sid, "date": date, "pu": pu,
             "cloud_all": share(s2.CLOUD_SCL, allm), "shadow_all": share(s2.SHADOW_SCL, allm),
             "cloud_A": share(s2.CLOUD_SCL, a) if a is not None else None, "shadow_A": share(s2.SHADOW_SCL, a) if a is not None else None,
             "shadow_A_land": share(s2.SHADOW_SCL, a & land) if a is not None else None, "shadow_A_water": share(s2.SHADOW_SCL, a & ~land) if a is not None else None,
             "cloud_A_land": share(s2.CLOUD_SCL, a & land) if a is not None else None, "cloud_A_water": share(s2.CLOUD_SCL, a & ~land) if a is not None else None,
             "classes_A": {int(k): int(v) for k, v in zip(*np.unique(scl[a & valid], return_counts=True))} if a is not None else None}
        out.append(r); print(json.dumps(r))
print("::notice title=scl-breakdown::" + json.dumps(out).replace("\n", " "))
