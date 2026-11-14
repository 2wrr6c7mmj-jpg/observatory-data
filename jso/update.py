"""Keep the 統幕 (Japan Joint Staff Office) transit data current.

The base database jso/source/jso.sqlite holds the releases collected by hand up
to 2026-10-27 (see jso/HANDOFF.md). New releases are added here, automatically:

  fetch   read the press-release list (https://www.mod.go.jp/js/press/index.html),
          pick the Chinese / Russian ship and aircraft releases that are not yet
          stored, download each PDF once and append its text to
          jso/source/updates.jsonl (one line per release, append-only, committed).
  apply   parse the updates.jsonl releases with jso/pipeline/jsoparse.py and add
          them to a copy of the base database (jso.build does this itself, so the
          committed jso.sqlite is never rewritten).
  d1      add the releases missing from the Cloudflare D1 database
          "jso-naval-transits" (same schema; no full text, as the original
          compact load), then rebuild its transit_pairs and vessels tables.
          Needs CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_D1_TOKEN; skipped without them.

The text is extracted with poppler's pdftotext. On 23 releases (2024-2026) it
gives the same parse as the browser pdf.js text the base database was built
from (same sightings, units, Japanese units and "同一である" phrase).

mod.go.jp is fetched through collectors.polite (robots.txt checked; a bot
challenge stops the run, it is never worked around).

  python -m jso.update fetch
  python -m jso.update stale   (exit 0: docs/data/jso needs a rebuild)
  python -m jso.update d1
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import html
import importlib.util
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE_DB = ROOT / "jso" / "source" / "jso.sqlite"
UPDATES = ROOT / "jso" / "source" / "updates.jsonl"
POST_LOAD = ROOT / "jso" / "pipeline" / "post_load.sql"
LIST_URL = "https://www.mod.go.jp/js/press/index.html"
PDF_URL = "https://www.mod.go.jp/js/pdf/{key}.pdf"
UA = "observatory-data/1 (+https://github.com/2wrr6c7mmj-jpg/observatory-data)"
D1_DATABASE_ID = "a8d099ec-b941-4f3f-a8f2-f2c441454d9e"  # jso-naval-transits (APAC)
LOOKBACK_DAYS = 30   # list entries older than (newest stored release - this) are not considered
MAX_PER_RUN = 40     # PDFs fetched in one run at most
PAGE_SEP = "\n=====PAGE=====\n"

_jp = None


def jsoparse():
    """The parser of the original pipeline (jso/pipeline/jsoparse.py), loaded once."""
    global _jp
    if _jp is None:
        spec = importlib.util.spec_from_file_location("jsoparse", ROOT / "jso" / "pipeline" / "jsoparse.py")
        _jp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_jp)
    return _jp


# ------------------------------------------------------------------ list page
ITEM_RX = re.compile(
    r'<a href="/js/pdf/(\d{4}/p(\d{4})(\d{2})(\d{2})[^"/]*?)\.pdf"[^>]*>\s*<time[^>]*>([^<]*)</time>\s*'
    r'<span data-tag="([^"]*)">[^<]*</span>\s*<h5>(.*?)</h5>', re.S)


def list_items(page: str) -> list[dict]:
    """Entries of the press-release list. The title is stored as in the base database:
    "<date text> <tag> <heading>" (e.g. "2026年10月27日 公表 中国軍機の動向について（Y-9 沖宮）").
    The publication date is taken from the file name (pYYYYMMDD...): the page's datetime attribute is
    sometimes malformed (e.g. "2026-05-7"), and some names carry suffixes (_01n, _01-2)."""
    out, seen = [], set()
    for m in ITEM_RX.finditer(page):
        if m.group(1) in seen:
            continue
        seen.add(m.group(1))
        heading = html.unescape(re.sub(r"<[^>]+>", "", m.group(7))).strip()
        # date text as in the base database (zero-padded, e.g. "2026年05月07日"), from the file name
        out.append({"release_id": m.group(1), "pub_date": f"{m.group(2)}-{m.group(3)}-{m.group(4)}",
                    "title": f"{m.group(2)}年{m.group(3)}月{m.group(4)}日 {m.group(6)} {heading}"})
    return out


def is_target(title: str) -> bool:
    """Chinese / Russian ship or aircraft movements, and corrections of them (as the base collection)."""
    return bool(re.search(r"(中国|ロシア|中露)", title) and re.search(r"(動向|飛行|訂正)", title))


def category(title: str) -> tuple[str, str]:
    """(nation, domain) from the title, as jso/pipeline/run_all.py."""
    if re.search(r"中国.*艦艇", title) and "ロシア" in title: return "CNRU", "ship"
    if re.search(r"中国.*艦艇", title): return "CN", "ship"
    if re.search(r"ロシア.*艦艇", title): return "RU", "ship"
    if re.search(r"中国.*(軍機|機)", title) and "ロシア" in title: return "CNRU", "air"
    if re.search(r"中国.*(軍機|機)", title): return "CN", "air"
    if re.search(r"ロシア.*(軍機|機)", title): return "RU", "air"
    if "訂正" in title: return "OTHER", "correction"
    return "OTHER", "other"


def stored_ids(db: Path = BASE_DB, updates: Path = UPDATES) -> tuple[set[str], str]:
    """Release ids already stored (base database + updates.jsonl) and the newest pub_date."""
    con = sqlite3.connect(db)
    ids = {r[0] for r in con.execute("SELECT release_id FROM releases")}
    newest = con.execute("SELECT max(pub_date) FROM releases").fetchone()[0]
    con.close()
    for rec in read_updates(updates):
        ids.add(rec["release_id"])
        newest = max(newest, rec["pub_date"])
    return ids, newest


def to_fetch(items: list[dict], ids: set[str], newest: str) -> list[dict]:
    since = (dt.date.fromisoformat(newest) - dt.timedelta(days=LOOKBACK_DAYS)).isoformat()
    new = [i for i in items if is_target(i["title"]) and i["release_id"] not in ids and i["pub_date"] >= since]
    return sorted(new, key=lambda i: i["release_id"])


# ------------------------------------------------------------------ PDF text
def pdf_text(data: bytes) -> tuple[str, int]:
    """Text of each page joined with the page separator of the base database, and the page count."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "r.pdf"
        p.write_bytes(data)
        out = subprocess.run(["pdftotext", "-enc", "UTF-8", str(p), "-"], capture_output=True, check=True).stdout
    pages = out.decode("utf-8", errors="replace").split("\f")
    while pages and not pages[-1].strip():
        pages.pop()
    return PAGE_SEP.join(p.strip() for p in pages), len(pages)


def read_updates(path: Path = UPDATES) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def fetch(pause: float = 3.0, getter=None, updates: Path = UPDATES, db: Path = BASE_DB) -> int:
    """Append the new releases to updates.jsonl. A release that cannot be fetched or read is skipped with a
    warning and tried again next run; robots.txt refusal or a bot challenge stops the run (never worked around)."""
    from collectors import polite
    get = getter or (lambda url: polite.get(url, UA))
    ids, newest = stored_ids(db, updates)
    page = get(LIST_URL)
    if page.status != 200:
        raise RuntimeError(f"list page: HTTP {page.status}")
    items = list_items(page.body.decode("utf-8", errors="replace"))
    if len(items) < 100:  # the page lists thousands; far fewer means the layout changed
        raise RuntimeError(f"list page: only {len(items)} entries matched; the page layout may have changed")
    todo = to_fetch(items, ids, newest)
    since = (dt.date.fromisoformat(newest) - dt.timedelta(days=LOOKBACK_DAYS)).isoformat()
    aged = [i["release_id"] for i in items if is_target(i["title"]) and i["release_id"] not in ids
            and i["pub_date"] < since and i["pub_date"] >= (dt.date.fromisoformat(since) - dt.timedelta(days=30)).isoformat()]
    if aged:  # a release that kept failing is no longer tried once the window moves past it
        print(f"::warning title=jso-fetch::not stored and now outside the {LOOKBACK_DAYS}-day window: {' '.join(aged)}")
    added, failed, scanned = [], [], []
    with updates.open("a", encoding="utf-8") as f:
        for item in todo[:MAX_PER_RUN]:
            time.sleep(pause)
            k = item["release_id"]
            try:
                r = get(PDF_URL.format(key=k))
                if r.status != 200 or not r.body.startswith(b"%PDF"):
                    raise ValueError(f"HTTP {r.status}, not a PDF")
                text, pages = pdf_text(r.body)
            except polite.SourceBlocked:
                raise
            except (OSError, ValueError, subprocess.CalledProcessError) as e:  # URLError is an OSError
                failed.append(k)
                print(f"::warning title=jso-fetch::{k}: {type(e).__name__}: {str(e)[:200]}; tried again next run")
                continue
            rec = dict(item, pages=pages, text=text,
                       fetched_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            added.append(k)
            if len(text.replace(PAGE_SEP, "").strip()) < 80:
                scanned.append(k)
    print(f"::notice title=jso-fetch::list {len(items)} entries; {len(todo)} new; added {len(added)}: "
          f"{' '.join(added)[:900] or '-'}{'; failed: ' + ' '.join(failed) if failed else ''}"
          f"{'; more next run: ' + str(len(todo) - MAX_PER_RUN) if len(todo) > MAX_PER_RUN else ''}")
    if scanned:
        print(f"::warning title=jso-fetch::no text (image PDF?), stored as scanned, check by hand: {' '.join(scanned)}")
    return len(added)


# ------------------------------------------------------------------ apply to sqlite
def parse_record(rec: dict) -> dict:
    """One updates.jsonl record -> the parser's release dict (as jso/pipeline/run_all.py)."""
    jp = jsoparse()
    nation, domain = category(rec["title"])
    pub = dt.date.fromisoformat(rec["pub_date"])
    text = rec["text"].replace("=====PAGE=====", "\n")
    if len(text.strip()) < 80:
        r = {"release": {"release_id": rec["release_id"], "pub_date": rec["pub_date"], "nation": nation,
                         "domain": domain, "parse_status": "scanned"}, "sightings": [], "jp_units": []}
    else:
        r = jp.parse_release(rec["release_id"], pub, text, domain_hint=domain, nation_hint=nation)
    r["release"].update(title=rec["title"], pages=rec["pages"], url=PDF_URL.format(key=rec["release_id"]))
    return r


def resolve_same(r: dict, by_date: dict) -> str | None:
    """Release referred to by "〜と同一である" (same rule as jso/pipeline/load_d1.py)."""
    jp = jsoparse()
    txt = r.get("same_as_text")
    if not txt:
        return None
    d, _ = jp.DateCtx(dt.date.fromisoformat(r["release"]["pub_date"])).resolve(txt)
    if not d:
        return None
    strait = next((st["strait_key"] for st in jp.STRAITS if st["_rx"].search(txt)), None)
    me, nation = r["release"]["release_id"], r["release"]["nation"]
    cands = []
    for delta in (0, 1, -1, 2):
        for rid, stk, nat in by_date.get((d + dt.timedelta(days=delta)).isoformat(), []):
            if rid == me or (nat != nation and not (nat == "CNRU" or nation == "CNRU")):
                continue
            cands.append(((2 if (strait and stk == strait) else 0) + (1 if delta == 0 else 0), rid))
    return max(cands)[1] if cands else None


def rows_for(r: dict, same: str | None, parsed_at: str):
    """(release row, [(sighting row, [unit rows])], [jp_unit rows]) in column order of the tables."""
    rel = r["release"]
    k = rel["release_id"]
    status = rel.get("parse_status") or ("ok" if r["sightings"] else "partial")
    release = dict(release_id=k, pub_date=rel["pub_date"], nation=rel["nation"], domain=rel["domain"],
                   title=rel["title"], url=rel["url"], pages=rel["pages"], has_track_map=rel.get("has_track_map", 0),
                   corrects=rel.get("corrects"), parsed_at=parsed_at, parse_status=status)
    sightings = []
    n = len(r["sightings"])
    for s in r["sightings"]:
        last = s["seq"] == n
        row = dict(release_id=k, seq=s["seq"], observed_date=s["observed_date"],
                   observed_date_end=s.get("observed_date_end"), observed_time=s.get("observed_time"),
                   ref_point=s.get("ref_point"), bearing_jp=s.get("bearing_jp"), distance_km=s.get("distance_km"),
                   lat=s.get("lat"), lon=s.get("lon"), heading_jp=s.get("heading_jp"), heading_deg=s.get("heading_deg"),
                   strait=s.get("strait"), strait_dir=s.get("strait_dir"), waters_from=s.get("waters_from"),
                   waters_to=s.get("waters_to"), activity=s.get("activity"), activity_count=s.get("activity_count"),
                   same_as_release=same if (last or s.get("same_as_text")) else None,
                   same_as_text=s.get("same_as_text") or (r.get("same_as_text") if last else None),
                   unit_count=s.get("unit_count"), sentence=s.get("sentence"))
        units = [dict(nation=u["nation"], kind=u["kind"], class_jp=u.get("class_jp"), class_key=u.get("class_key"),
                      hull_no=u.get("hull_no"), name=u.get("name"), count=u.get("count", 1)) for u in s["units"]]
        sightings.append((row, units))
    jp_units = [dict(release_id=k, unit_kind=j.get("unit_kind"), unit_name=j.get("unit_name"),
                     parent_unit=j.get("parent_unit"), base=j.get("base")) for j in r.get("jp_units", [])]
    return release, sightings, jp_units


def _insert(con, table: str, row: dict):
    cols = ",".join(row)
    return con.execute(f"INSERT INTO {table}({cols}) VALUES ({','.join('?' * len(row))})", list(row.values()))


def post_load_statements(include_view: bool = False) -> list[str]:
    """The transit_pairs and vessels rebuild of jso/pipeline/post_load.sql (statement by statement)."""
    sql = re.sub(r"--[^\n]*", "", POST_LOAD.read_text(encoding="utf-8"))
    stmts = [s.strip() for s in sql.split(";") if s.strip()]
    return [s for s in stmts if include_view or not s.upper().startswith("CREATE VIEW")]


def apply(con: sqlite3.Connection, records: list[dict]) -> list[str]:
    """Add the records not yet in the database; rebuild transit_pairs and vessels. Returns the added ids."""
    have = {r[0] for r in con.execute("SELECT release_id FROM releases")}
    new, seen = [], set(have)
    for rec in sorted(records, key=lambda x: x["release_id"]):
        if rec["release_id"] not in seen:  # the first record of a release wins
            seen.add(rec["release_id"])
            new.append(rec)
    if not new:
        return []
    parsed = [parse_record(rec) for rec in new]
    by_date = collections.defaultdict(list)
    for d, d_end, rid, strait, nation in con.execute(
            "SELECT s.observed_date, s.observed_date_end, s.release_id, s.strait, r.nation "
            "FROM sightings s JOIN releases r USING(release_id)"):
        for x in (d, d_end):
            if x:
                by_date[x].append((rid, strait, nation))
    for r in parsed:  # the new releases can refer to each other too
        for s in r["sightings"]:
            for x in (s["observed_date"], s.get("observed_date_end")):
                if x:
                    by_date[x].append((r["release"]["release_id"], s.get("strait"), r["release"]["nation"]))
    parsed_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    texts = {rec["release_id"]: rec["text"] for rec in new}
    with con:
        for r in parsed:
            release, sightings, jp_units = rows_for(r, resolve_same(r, by_date), parsed_at)
            _insert(con, "releases", dict(release, raw_text=texts[release["release_id"]]))
            for row, units in sightings:
                sid = _insert(con, "sightings", row).lastrowid
                for u in units:
                    _insert(con, "sighting_units", dict(u, sighting_id=sid))
            for j in jp_units:
                _insert(con, "jp_units", j)
        for stmt in post_load_statements():
            con.execute(stmt)
    return [r["release"]["release_id"] for r in parsed]


def working_db(dest: Path) -> Path:
    """A copy of the base database with updates.jsonl applied (used by jso.build)."""
    shutil.copyfile(BASE_DB, dest)
    con = sqlite3.connect(dest)
    try:
        apply(con, read_updates())
    finally:
        con.close()
    return dest


# ------------------------------------------------------------------ Cloudflare D1
def q(v) -> str:
    """SQL literal (as jso/pipeline/load_d1.py)."""
    if v is None or v == "":
        return "NULL"
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, (int, float)):
        return repr(v) if isinstance(v, float) else str(v)
    return "'" + str(v).replace("'", "''") + "'"


def _values(row: dict) -> tuple[str, str]:
    return ",".join(row), ",".join(q(v) for v in row.values())


def d1_release_sql(con: sqlite3.Connection, release_id: str) -> str:
    """Statements that (re)write one release in D1, without full text and source sentences (compact load).
    Partial rows of an earlier failed attempt are removed first; the request runs as one batch."""
    k = q(release_id)
    out = [f"DELETE FROM sighting_units WHERE sighting_id IN (SELECT sighting_id FROM sightings WHERE release_id={k})",
           f"DELETE FROM sightings WHERE release_id={k}", f"DELETE FROM jp_units WHERE release_id={k}",
           f"DELETE FROM releases WHERE release_id={k}"]
    con.row_factory = sqlite3.Row
    rel = dict(con.execute("SELECT * FROM releases WHERE release_id=?", (release_id,)).fetchone())
    rel["raw_text"] = None
    c, v = _values(rel)
    out.append(f"INSERT INTO releases({c}) VALUES ({v})")
    for s in con.execute("SELECT * FROM sightings WHERE release_id=? ORDER BY seq", (release_id,)).fetchall():
        s = dict(s)
        sid = s.pop("sighting_id")
        s["sentence"] = None
        c, v = _values(s)
        out.append(f"INSERT INTO sightings({c}) VALUES ({v})")
        for u in con.execute("SELECT * FROM sighting_units WHERE sighting_id=? ORDER BY id", (sid,)).fetchall():
            u = dict(u)
            u.pop("id"), u.pop("sighting_id")
            c, v = _values(u)
            # the sighting inserted just above (ids are AUTOINCREMENT, so it is the largest)
            out.append(f"INSERT INTO sighting_units(sighting_id,{c}) VALUES ((SELECT max(sighting_id) FROM sightings),{v})")
    for j in con.execute("SELECT * FROM jp_units WHERE release_id=? ORDER BY id", (release_id,)).fetchall():
        j = dict(j)
        j.pop("id")
        c, v = _values(j)
        out.append(f"INSERT INTO jp_units({c}) VALUES ({v})")
    con.row_factory = None
    return ";\n".join(out) + ";"


class D1:
    def __init__(self, account: str, token: str, database: str = D1_DATABASE_ID):
        self.url = f"https://api.cloudflare.com/client/v4/accounts/{account}/d1/database/{database}/query"
        self.token = token

    def query(self, sql: str, params: list | None = None) -> list[dict]:
        body = json.dumps({"sql": sql, **({"params": params} if params else {})}).encode()
        req = urllib.request.Request(self.url, data=body, method="POST", headers={
            "Authorization": f"Bearer {self.token}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                res = json.load(r)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"D1 HTTP {e.code}: {e.read()[:300]!r}") from None
        if not res.get("success"):
            raise RuntimeError(f"D1 error: {res.get('errors')}")
        return res["result"]


def d1_sync(con: sqlite3.Connection, d1: D1, records: list[dict]) -> list[str]:
    """Bring the updates.jsonl releases in D1 to the local rows; rebuild its transit_pairs and vessels.

    A release is rewritten when D1 does not have it or has another number of sightings (an earlier run that
    stopped half-way). transit_pairs and vessels are rebuilt when a release was written, or when their row
    counts differ from the local database (a rebuild that failed earlier). Each step can be run again."""
    if not records:
        return []
    since = min(r["pub_date"] for r in records)
    ids = sorted({r["release_id"] for r in records})
    local = dict(con.execute("SELECT r.release_id, count(s.sighting_id) FROM releases r LEFT JOIN sightings s "
                             "USING(release_id) WHERE r.pub_date >= ? GROUP BY r.release_id", (since,)).fetchall())
    remote = {row["release_id"]: row["n"] for row in d1.query(
        "SELECT r.release_id, count(s.sighting_id) AS n FROM releases r LEFT JOIN sightings s USING(release_id) "
        "WHERE r.pub_date >= ? GROUP BY r.release_id", [since])[0]["results"]}
    todo = [k for k in ids if k in local and remote.get(k) != local[k]]
    for k in todo:
        d1.query(d1_release_sql(con, k))
        got = d1.query("SELECT count(*) AS n FROM sightings WHERE release_id = ?", [k])[0]["results"][0]["n"]
        if got != local[k]:
            raise RuntimeError(f"D1 {k}: {got} sightings written, expected {local[k]}")
    counts = "SELECT (SELECT count(*) FROM transit_pairs) AS tp, (SELECT count(*) FROM vessels) AS v"
    want = dict(zip(("tp", "v"), con.execute(counts).fetchone()))
    if todo or d1.query(counts)[0]["results"][0] != want:
        d1.query(";\n".join(post_load_statements()) + ";")
        got = d1.query(counts)[0]["results"][0]
        if got != want:
            raise RuntimeError(f"D1 transit_pairs/vessels {got}, expected {want}")
    return todo


def check_built(out: Path = ROOT / "docs" / "data" / "jso") -> list[str]:
    """The rebuilt data covers every stored release, and the new ship releases were read (warnings only for those)."""
    problems = []
    ids, newest = stored_ids()
    status = json.loads((out / "status.json").read_text(encoding="utf-8"))
    if status.get("releases") != len(ids):
        problems.append(f"status.json releases {status.get('releases')}, stored {len(ids)}")
    if (status.get("last_release") or "") < newest:
        problems.append(f"status.json last_release {status.get('last_release')}, newest stored {newest}")
    with tempfile.TemporaryDirectory() as d:
        con = sqlite3.connect(working_db(Path(d) / "jso.sqlite"))
        empty = [r[0] for r in con.execute(
            "SELECT r.release_id FROM releases r WHERE r.domain='ship' AND r.release_id IN (%s) AND NOT EXISTS "
            "(SELECT 1 FROM sightings s WHERE s.release_id=r.release_id)" % ",".join("?" * len(ids)),
            sorted(i for i in ids if i in {x["release_id"] for x in read_updates()}))]
        con.close()
    if empty:
        print(f"::warning title=jso-check::ship releases without any sighting read (check the text): {' '.join(empty)}")
    return problems


# ------------------------------------------------------------------ CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="fetch new releases into updates.jsonl")
    f.add_argument("--pause", type=float, default=float(os.environ.get("JSO_PAUSE_SECONDS", "3")))
    sub.add_parser("d1", help="add missing releases to Cloudflare D1")
    sub.add_parser("stale", help="exit 0 when docs/data/jso does not cover every stored release (build needed)")
    sub.add_parser("check", help="check the built docs/data/jso against the stored releases (exit 1 on a mismatch)")
    a = ap.parse_args(argv)
    if a.cmd == "stale":
        return 0 if is_stale() else 1
    if a.cmd == "check":
        problems = check_built()
        for p in problems:
            print(f"::error title=jso-check::{p}")
        return 1 if problems else 0
    if a.cmd == "fetch":
        n = fetch(a.pause)
        print(f"added {n}")
        return 0
    account, token = os.environ.get("CLOUDFLARE_ACCOUNT_ID"), os.environ.get("CLOUDFLARE_D1_TOKEN")
    if not (account and token):
        print("::notice title=jso-d1::CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_D1_TOKEN not set; D1 not updated")
        return 0
    with tempfile.TemporaryDirectory() as d:
        con = sqlite3.connect(working_db(Path(d) / "jso.sqlite"))
        added = d1_sync(con, D1(account, token), read_updates())
        con.close()
    print(f"::notice title=jso-d1::{len(added)} release(s) added to D1: {' '.join(added) or '-'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
