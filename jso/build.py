"""Build the public 統幕 (Japan Joint Staff Office) transit datasets.

Reads jso/source/jso.sqlite (parsed press releases, see jso/HANDOFF.md) and
writes docs/data/jso/ for GitHub Pages:

  passages.csv / .json   one row per release x strait x day (ship passages)
  round_trips.csv        outbound + return pairs; method = stated | hull-match
  context.csv            exercises and political events used for matching
  summary.json           yearly / monthly counts for the Observatory site
  meta.json, status.json

Changes from the original build_datasets.py:
  - Context events are matched only within +-WINDOW days (the original took
    the nearest event at any distance, e.g. 5,039 days away).
  - Return passages are also inferred from matching hull numbers, flagged
    method=hull-match and kept apart from pairs the releases state.
  - Each passage carries the Taiwan MND daily counts for the same date, when
    that dataset covers it (around-Taiwan format only).

  python -m jso.build
"""
from __future__ import annotations

import collections
import csv
import datetime as dt
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "jso" / "source" / "jso.sqlite"
OUT = ROOT / "docs" / "data" / "jso"
MND = ROOT / "docs" / "data" / "taiwan-mnd" / "daily.csv"
WINDOW = 14        # days before/after a passage for context matching
PAIR_MAX_DAYS = 90  # longest stay considered for a hull-number match
DUP_DAYS = 2        # passages this close are treated as one (see dedupe())
HOME = {"CN": {"東シナ海"}, "RU": {"日本海"}, "CNRU": {"東シナ海", "日本海"}}  # each navy's home side
NAMES = {'miyako': '宮古海峡', 'osumi': '大隅海峡', 'tsushima': '対馬海峡', 'tsugaru': '津軽海峡', 'soya': '宗谷海峡',
         'yonaguni_taiwan': '与那国－台湾', 'yonaguni_iriomote': '与那国－西表', 'amami_yokoate': '奄美－横当', 'tokara': 'トカラ',
         'okinawa_kume': '沖縄－久米', 'ishigaki_iriomote': '石垣－西表', 'miyako_ishigaki': '宮古－石垣',
         'izu_mikura_hachijo': '伊豆諸島', 'ogasawara': '小笠原', 'nemuro': '根室海峡', 'etorofu': '択捉・国後',
         'okinawa_iheya': '沖縄－伊平屋', 'kikai': '喜界', 'yakushima_tanegashima': '屋久－種子'}
D = dt.date.fromisoformat


def sea(name: str) -> str:
    """Basin name without the east/west qualifier, e.g. 太平洋(東) -> 太平洋."""
    return (name or "").split("(")[0].strip()


def context_matchers():
    pla = lambda a, cat, reg: a == 'CN' and cat == 'exercise' and reg in ('taiwan_strait', 'multi')
    allied = lambda a, cat, reg: a in ('JP', 'US', 'USJP', 'MULTI', 'AU', 'PH', 'KR', 'UK', 'FR', 'DE', 'IN') and cat == 'exercise'
    pol = lambda a, cat, reg: cat in ('election', 'inauguration', 'summit', 'visit', 'statement', 'policy', 'sanction',
                                      'legislation', 'incident')
    return {"pla_ex": pla, "allied_ex": allied, "pol": pol}


def nearest(day: dt.date, events, match, window=WINDOW):
    """Closest matching event within +-window days.

    offset = event day - passage day: negative = the event came before the
    passage, 0 = the passage fell inside the event, positive = event after.
    """
    best = None
    for ds, de, actor, cat, name, reg in events:
        if not match(actor, cat, reg):
            continue
        off = 0 if ds <= day <= de else ((ds - day).days if day < ds else -(day - de).days)
        if abs(off) <= window and (best is None or abs(off) < abs(best[0])):
            best = (off, name)
    return best or (None, None)


def load_passages(c: sqlite3.Cursor) -> list[dict]:
    rows = c.execute("""
      select s.sighting_id, r.release_id, r.nation, r.pub_date, s.observed_date, s.observed_date_end, s.strait, s.strait_dir,
             s.waters_from, s.waters_to, s.strait_inferred, s.activity, s.unit_count, r.title, r.url,
             (select group_concat(class_jp || coalesce('(' || hull_no || ')', '') || case when count > 1 then '×' || count else '' end, ' / ')
                from sighting_units u where u.sighting_id = s.sighting_id),
             (select group_concat(hull_no, ' ') from sighting_units u where u.sighting_id = s.sighting_id and hull_no is not null),
             (select count(*) from sighting_units u join vessel_classes v on v.class_key = u.class_key where u.sighting_id = s.sighting_id and v.type_cat = 'CV'),
             (select count(*) from sighting_units u join vessel_classes v on v.class_key = u.class_key where u.sighting_id = s.sighting_id and v.type_cat in ('AOR','AKS')),
             (select count(*) from sighting_units u join vessel_classes v on v.class_key = u.class_key where u.sighting_id = s.sighting_id and v.type_cat in ('AGI','AGS','AGM')),
             (select count(*) from sighting_units u where u.sighting_id = s.sighting_id and u.kind = 'sub'),
             (select count(*) from sighting_units u join vessel_classes v on v.class_key = u.class_key where u.sighting_id = s.sighting_id
                and v.type_cat in ('DDG','DD','CG','CGN','FFG','FF','FFL','PGG')),
             s.same_as_release
        from sightings s join releases r using (release_id)
       where r.domain = 'ship' and s.strait is not null and s.observed_date is not null
       order by s.observed_date, s.sighting_id""").fetchall()
    seen: dict[tuple, dict] = {}
    for (sid, rid, nat, pub, d, de, st, sd, wf, wt, inf, act, n, title, url, units, hulls, cv, aor, agi, sub, comb, same) in rows:
        key = (rid, st, d)
        if key in seen and not (seen[key]["inferred"] and not inf):
            continue  # keep one row per release x strait x day; a stated strait beats an inferred one
        seen[key] = dict(sighting_id=sid, release_id=rid, nation=nat, pub_date=pub, date=d, date_end=de or None,
                         strait=st, strait_jp=NAMES.get(st, st), dir=sd or None, waters_from=wf or None, waters_to=wt or None,
                         inferred=int(inf or 0), activity=act or None, n_units=n, units=units or None,
                         hulls=" ".join(sorted(set((hulls or "").split()))) or None,
                         has_cv=cv, has_aor=aor, has_agi=agi, has_sub=sub, n_combat=comb, linked_release=same or None,
                         title=title, url=url)
    return list(seen.values())


def hulls_match(a: str | None, b: str | None) -> bool:
    """At least half of the smaller hull set is shared (the hull_pairs rule)."""
    ha, hb = set((a or "").split()), set((b or "").split())
    shared = ha & hb
    return bool(shared) and len(shared) * 2 >= min(len(ha), len(hb))


def same_way(a: dict, b: dict) -> bool:
    """Same direction: the same destination basin when both name one, else the same compass heading."""
    if a["waters_to"] and b["waters_to"]:
        return sea(a["waters_to"]) == sea(b["waters_to"])
    return a["dir"] == b["dir"]


def same_navy(a: str, b: str) -> bool:
    """CN and RU never match (hull numbers overlap, e.g. 548); a joint CNRU release matches either."""
    return a == b or "CNRU" in (a, b)


def dedupe(passages: list[dict]) -> tuple[list[dict], dict[int, int], dict[str, int]]:
    """Drop passages counted twice (2部 検品 A1, A2).

    A1  An inferred strait is usually the first sighting, not the passage: drop it
        when the same release states a passage within +-DUP_DAYS days.
    A2  A later release repeating an earlier passage: same hulls, strait and
        direction (same_way) within +-DUP_DAYS days -> keep the earliest release's row.
        A joint CN-RU release repeating a CN or RU release counts (same_navy).
    Returns the kept rows, {dropped sighting -> kept sighting} for A2 (the same
    passage, so round trips stated on the repeat still apply) and counts.
    """
    stated_days = collections.defaultdict(list)
    for p in passages:
        if not p["inferred"]:
            stated_days[p["release_id"]].append(D(p["date"]))
    kept = [p for p in passages if not (p["inferred"] and any(
        abs((D(p["date"]) - d).days) <= DUP_DAYS for d in stated_days[p["release_id"]]))]
    a1 = len(passages) - len(kept)
    kept.sort(key=lambda p: (p["pub_date"] or "", p["release_id"], p["date"]))
    out, alias = [], {}
    for p in kept:
        twin = next((q for q in out if p["hulls"] and q["release_id"] != p["release_id"] and same_navy(q["nation"], p["nation"])
                     and q["strait"] == p["strait"] and same_way(q, p)
                     and abs((D(q["date"]) - D(p["date"])).days) <= DUP_DAYS and hulls_match(q["hulls"], p["hulls"])), None)
        if twin:
            alias[p["sighting_id"]] = twin["sighting_id"]
        else:
            out.append(p)
    out.sort(key=lambda p: (p["date"], p["sighting_id"]))
    return out, alias, {"inferred_near_stated": a1, "repeated_in_later_release": len(alias)}


def direction(nation: str, out_waters_to: str | None) -> str:
    """進出→帰投 when the outbound leg leaves home waters; 帰投→再進出 when it heads home (2部 検品 A3)."""
    return "帰投→再進出" if sea(out_waters_to) in HOME.get(nation, set()) else "進出→帰投"


def stated_pairs(c: sqlite3.Cursor) -> list[dict]:
    out = []
    for o_sid, i_sid, od, ost, idt, ist, days, nat, units, aor, cv, orel, irel, url, oto in c.execute("""
        select o.sighting_id, i.sighting_id, o.observed_date, o.strait, i.observed_date, i.strait, p.days_away, r.nation,
               (select group_concat(class_jp || coalesce('(' || hull_no || ')', ''), ' / ') from sighting_units u where u.sighting_id = o.sighting_id),
               (select count(*) from sighting_units u join vessel_classes v on v.class_key = u.class_key where u.sighting_id = o.sighting_id and v.type_cat in ('AOR','AKS')),
               (select count(*) from sighting_units u join vessel_classes v on v.class_key = u.class_key where u.sighting_id = o.sighting_id and v.type_cat = 'CV'),
               o.release_id, i.release_id, r.url, o.waters_to
          from transit_pairs p join sightings o on o.sighting_id = p.out_sighting_id
          join sightings i on i.sighting_id = p.in_sighting_id join releases r on r.release_id = i.release_id
         where p.days_away between 0 and 200 order by 3"""):
        out.append(dict(method="stated", out_sighting=o_sid, in_sighting=i_sid, out_date=od, out_strait=ost,
                        out_strait_jp=NAMES.get(ost, ost), in_date=idt, in_strait=ist, in_strait_jp=NAMES.get(ist, ist),
                        days_away=days, direction=direction(nat, oto), nation=nat, units=units, with_aor=int(bool(aor)),
                        has_cv=int(bool(cv)), out_release=orel, in_release=irel, in_url=url, shared_hulls=None))
    return out


def hull_pairs(passages: list[dict], stated: list[dict]) -> list[dict]:
    """Infer return passages from hull numbers.

    A passage B is the return of A when the same nation's ships (at least half
    of the smaller hull set) leave the basin A entered (sea(B.from) ==
    sea(A.to)), B is the first such passage after A, and no pair is already
    stated for either side.
    """
    used = {p["out_sighting"] for p in stated} | {p["in_sighting"] for p in stated}
    hs = lambda p: set((p["hulls"] or "").split())
    ordered = sorted((p for p in passages if p["hulls"] and p["waters_from"] and p["waters_to"]), key=lambda p: p["date"])
    out, taken = [], set()
    for i, a in enumerate(ordered):
        if a["sighting_id"] in used or a["sighting_id"] in taken:
            continue
        ha = hs(a)
        for b in ordered[i + 1:]:
            days = (D(b["date"]) - D(a["date"])).days
            if days > PAIR_MAX_DAYS:
                break
            hb = hs(b)
            shared = ha & hb
            if b["nation"] != a["nation"] or not shared or len(shared) * 2 < min(len(ha), len(hb)):
                continue
            if days < 1:
                continue  # same-day sightings of the same group (another strait or report)
            # The next sighting of these ships decides: a return only if it leaves the basin they entered.
            if sea(b["waters_from"]) == sea(a["waters_to"]) and sea(b["waters_to"]) != sea(a["waters_to"]) \
                    and b["sighting_id"] not in used and b["sighting_id"] not in taken:
                taken |= {a["sighting_id"], b["sighting_id"]}
                out.append(dict(method="hull-match", out_sighting=a["sighting_id"], in_sighting=b["sighting_id"],
                                out_date=a["date"], out_strait=a["strait"], out_strait_jp=a["strait_jp"], in_date=b["date"],
                                in_strait=b["strait"], in_strait_jp=b["strait_jp"], days_away=days,
                                direction=direction(a["nation"], a["waters_to"]), nation=a["nation"],
                                units=a["units"], with_aor=int(bool(a["has_aor"])), has_cv=int(bool(a["has_cv"])),
                                out_release=a["release_id"], in_release=b["release_id"], in_url=b["url"],
                                shared_hulls=" ".join(sorted(shared))))
            break
    return out


def mnd_daily() -> dict[str, dict]:
    if not MND.exists():
        return {}
    with open(MND, encoding="utf-8") as f:
        return {r["report_date"]: r for r in csv.DictReader(f) if r.get("format") == "around-taiwan"}


def main() -> int:
    con = sqlite3.connect(DB)
    c = con.cursor()
    events = [(D(a), D(b or a), actor, cat, name, reg) for a, b, actor, cat, name, reg in
              c.execute("select date_start, date_end, actor, category, name, region from context_events")]
    raw = load_passages(c)
    passages, alias, removed = dedupe(raw)
    stated = stated_pairs(c)
    inferred = hull_pairs(passages, stated)
    pairs = sorted(stated + inferred, key=lambda p: (p["out_date"], p["method"]))
    stay = {}
    for p in pairs:  # days away is a stay only when the pair left home waters first
        if p["direction"] == "進出→帰投":
            stay[alias.get(p["in_sighting"], p["in_sighting"])] = (p["days_away"], p["method"])
    mnd = mnd_daily()
    for p in passages:
        day = D(p["date"])
        for key, match in context_matchers().items():
            p[key + "_offset"], p[key + "_name"] = nearest(day, events, match)
        p["days_away"], p["pair_method"] = stay.get(p["sighting_id"], (None, None))
        m = mnd.get(p["date"])
        p["mnd_aircraft"] = int(m["aircraft"]) if m and m["aircraft"] else None
        p["mnd_plan_vessels"] = int(m["plan_vessels"]) if m and m["plan_vessels"] else None

    OUT.mkdir(parents=True, exist_ok=True)
    pcols = list(passages[0])
    write_csv(OUT / "passages.csv", pcols, passages)
    (OUT / "passages.json").write_text(json.dumps({"fields": pcols, "rows": [[p[k] for k in pcols] for p in passages]},
                                                  ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    write_csv(OUT / "round_trips.csv", list(pairs[0]), pairs)
    ctx = c.execute("select date_start, coalesce(date_end, date_start), actor, category, name, region, notes, source_url "
                    "from context_events order by date_start").fetchall()
    write_csv(OUT / "context.csv", ["date_start", "date_end", "actor", "category", "name", "region", "notes", "source_url"],
              [dict(zip(["date_start", "date_end", "actor", "category", "name", "region", "notes", "source_url"], r)) for r in ctx])

    by_year = collections.defaultdict(lambda: collections.Counter())
    by_month = collections.defaultdict(lambda: collections.Counter())
    for p in passages:
        by_year[p["nation"]][(p["date"][:4], p["strait"])] += 1
        by_month[p["nation"]][int(p["date"][5:7])] += 1
    summary = {
        "by_year_strait": {n: [{"year": y, "strait": s, "count": k} for (y, s), k in sorted(v.items())] for n, v in by_year.items()},
        "by_month": {n: [v.get(m, 0) for m in range(1, 13)] for n, v in by_month.items()},
        "stay_days": {meth: sorted(p["days_away"] for p in pairs if p["method"] == meth and p["direction"] == "進出→帰投")
                      for meth in ("stated", "hull-match")},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    releases = c.execute("select count(*), min(pub_date), max(pub_date) from releases").fetchone()
    status = {
        "built_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "releases": releases[0], "first_release": releases[1], "last_release": releases[2],
        "passages": len(passages), "by_nation": dict(collections.Counter(p["nation"] for p in passages)),
        "inferred_strait": sum(p["inferred"] for p in passages),
        "inferred_strait_before_dedupe": sum(p["inferred"] for p in raw),
        "removed_duplicates": removed,
        "round_trips": {"stated": len(stated), "hull-match": len(inferred)},
        "round_trips_by_direction": dict(collections.Counter(p["direction"] for p in pairs)),
        "context_window_days": WINDOW,
        "with_mnd_counts": sum(1 for p in passages if p["mnd_aircraft"] is not None),
    }
    (OUT / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False))
    return 0


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
