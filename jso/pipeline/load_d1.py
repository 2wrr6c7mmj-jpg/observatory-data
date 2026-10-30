#!/usr/bin/env python3
"""Build SQL batches for D1 from data/parsed.json + data/jso_texts.json + ref/context_*.csv.
Writes data/sql/NN.sql (each < ~90KB) to be executed in order."""
import json, re, csv, os, sys, datetime as dt, collections
COMPACT = "--compact" in sys.argv
OUT = "data/sql_compact" if COMPACT else "data/sql"
CAP = 24000 if COMPACT else 85000

os.makedirs(OUT, exist_ok=True)
out = json.load(open("data/parsed.json", encoding="utf-8"))
docs = {d["key"]: d for d in json.load(open("data/jso_texts.json", encoding="utf-8"))}

def q(v):
    if v is None or v == "":
        return "NULL"
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, (int, float)):
        return str(v)
    return "'" + str(v).replace("'", "''") + "'"

# ---------- resolve "同一である" links -> same_as_release ----------
# index: (strait, date) -> release ids ; also (date) -> releases for the same nation
by_date = collections.defaultdict(list)
for r in out:
    rid = r["release"]["release_id"]
    for s in r["sightings"]:
        if s["observed_date"]:
            by_date[s["observed_date"]].append((rid, s.get("strait"), r["release"]["nation"]))
        if s.get("observed_date_end"):
            by_date[s["observed_date_end"]].append((rid, s.get("strait"), r["release"]["nation"]))

import importlib.util
spec = importlib.util.spec_from_file_location("jsoparse", "jsoparse.py"); jp = importlib.util.module_from_spec(spec); spec.loader.exec_module(jp)

def resolve_same(r):
    txt = r.get("same_as_text")
    if not txt:
        return None
    pub = dt.date.fromisoformat(r["release"]["pub_date"])
    ctx = jp.DateCtx(pub)
    d, _ = ctx.resolve(txt)
    if not d:
        return None
    strait = None
    for st in jp.STRAITS:
        if st["_rx"].search(txt):
            strait = st["strait_key"]; break
    cands = []
    for delta in (0, 1, -1, 2):
        dd = (d + dt.timedelta(days=delta)).isoformat()
        for rid, stk, nat in by_date.get(dd, []):
            if rid == r["release"]["release_id"]:
                continue
            if nat != r["release"]["nation"] and not (nat == "CNRU" or r["release"]["nation"] == "CNRU"):
                continue
            score = (2 if (strait and stk == strait) else 0) + (1 if delta == 0 else 0)
            cands.append((score, rid))
    if not cands:
        return None
    cands.sort(reverse=True)
    return cands[0][1] if cands[0][0] > 0 or len(cands) == 1 else cands[0][1]

stmts = ["DELETE FROM sighting_units;", "DELETE FROM sightings;", "DELETE FROM jp_units;", "DELETE FROM transit_pairs;"]
# ---------- releases ----------
rows = []
for r in out:
    rel = r["release"]; k = rel["release_id"]; d = docs[k]
    title = rel.get("title") or d["t"]
    status = rel.get("parse_status") or ("ok" if r["sightings"] else "partial")
    same = resolve_same(r)
    r["_same"] = same
    rows.append(f"({q(k)},{q(rel['pub_date'])},{q(rel['nation'])},{q(rel['domain'])},{q(title)},{q('https://www.mod.go.jp/js/pdf/'+k+'.pdf')},{q(d['n'])},{q(rel.get('has_track_map',0))},{q(rel.get('corrects'))},{q(None) if COMPACT else q(d['x'])},{q(dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'))},{q(status)})")
def flush(prefix, cols, rows, chunk=60, cap=None):
    cap = cap or CAP
    head = f"INSERT OR REPLACE INTO {prefix}({cols}) VALUES "
    cur = []; size = len(head)
    for row in rows:
        if cur and (size + len(row) + 1 > cap or len(cur) >= chunk):
            stmts.append(head + ",".join(cur) + ";"); cur = []; size = len(head)
        cur.append(row); size += len(row) + 1
    if cur:
        stmts.append(head + ",".join(cur) + ";")
flush("releases", "release_id,pub_date,nation,domain,title,url,pages,has_track_map,corrects,raw_text,parsed_at,parse_status", rows, 40)

# ---------- sightings (explicit ids so units can reference) ----------
sid = 0; srows = []; urows = []; jrows = []
for r in out:
    k = r["release"]["release_id"]
    for s in r["sightings"]:
        sid += 1
        srows.append(f"({sid},{q(k)},{s['seq']},{q(s['observed_date'])},{q(s.get('observed_date_end'))},{q(s.get('observed_time'))},{q(s.get('ref_point'))},{q(s.get('bearing_jp'))},{q(s.get('distance_km'))},{q(s.get('lat'))},{q(s.get('lon'))},{q(s.get('heading_jp'))},{q(s.get('heading_deg'))},{q(s.get('strait'))},{q(s.get('strait_dir'))},{q(s.get('waters_from'))},{q(s.get('waters_to'))},{q(s.get('activity'))},{q(s.get('activity_count'))},{q(r['_same'] if s['seq']==len(r['sightings']) or s.get('same_as_text') else None)},{q(s.get('same_as_text') or (r.get('same_as_text') if s['seq']==len(r['sightings']) else None))},{q(s.get('unit_count'))},{q(None) if COMPACT else q(s.get('sentence'))})")
        for u in s["units"]:
            urows.append(f"({sid},{q(u['nation'])},{q(u['kind'])},{q(u.get('class_jp'))},{q(u.get('class_key'))},{q(u.get('hull_no'))},{q(u.get('name'))},{q(u.get('count',1))})")
    for ju in r.get("jp_units", []):
        jrows.append(f"({q(k)},{q(ju.get('unit_kind'))},{q(ju.get('unit_name'))},{q(ju.get('parent_unit'))},{q(ju.get('base'))})")
flush("sightings", "sighting_id,release_id,seq,observed_date,observed_date_end,observed_time,ref_point,bearing_jp,distance_km,lat,lon,heading_jp,heading_deg,strait,strait_dir,waters_from,waters_to,activity,activity_count,same_as_release,same_as_text,unit_count,sentence", srows, 80)
flush("sighting_units", "sighting_id,nation,kind,class_jp,class_key,hull_no,name,count", urows, 300)
flush("jp_units", "release_id,unit_kind,unit_name,parent_unit,base", jrows, 300)

# ---------- context events ----------
crow = []
for f in ("context_cn_ru_military.csv", "context_allied_exercises.csv", "context_political.csv"):
    p = os.path.join("ref", f)
    if not os.path.exists(p):
        continue
    for row in csv.DictReader(open(p, encoding="utf-8")):
        crow.append(f"({q(row['date_start'])},{q(row.get('date_end'))},{q(row['actor'])},{q(row['category'])},{q(row['name'])},{q(row.get('region'))},{q(row.get('lat') or None)},{q(row.get('lon') or None)},{q(row.get('notes'))},{q(row.get('source_url'))})")
stmts.append("DELETE FROM context_events;")
flush("context_events", "date_start,date_end,actor,category,name,region,lat,lon,notes,source_url", crow, 150)

# ---------- write batches ----------
if COMPACT:
    batches = [s + "\n" for s in stmts]
else:
    batches = []; cur = ""
    for s in stmts:
        if len(cur) + len(s) > 90000 and cur:
            batches.append(cur); cur = ""
        cur += s + "\n"
    if cur: batches.append(cur)
for f in os.listdir(OUT):
    os.remove(os.path.join(OUT, f))
for i, b in enumerate(batches):
    open(f"{OUT}/{i:03d}.sql", "w", encoding="utf-8").write(b)
print("releases", len(rows), "sightings", len(srows), "units", len(urows), "jp_units", len(jrows), "context", len(crow), "batches", len(batches))
print("same_as resolved:", sum(1 for r in out if r.get('_same')), "of", sum(1 for r in out if r.get('same_as_text')))
