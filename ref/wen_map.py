"""One-time import of Joseph Wen's PLA bases map (Google My Maps) into GeoJSON.

溫約瑟（Joseph Wen）「中國人民解放軍基地及設施（持續更新）」. The map's
description says 「歡迎大家取用」 (everyone is welcome to use it); the data is
used with the source stated. The original KML is never stored in the repo:
it is read in the runner's temporary directory and only the converted
features (name, description, coordinates, folder) are written.

    python -m ref.wen_map fetch OUT.geojson SUMMARY.txt   # robots.txt, then one GET
    python -m ref.wen_map convert KML OUT.geojson         # offline conversion

robots.txt is evaluated with RFC 9309 longest-match semantics (Google's own):
``urllib.robotparser`` uses the first matching line, which for google.com
would wrongly let ``Disallow: /maps/`` shadow ``Allow: /maps/d/``.
"""
from __future__ import annotations

import json
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from html import unescape

MID = "19Q8BraU1Nmnk23TzMb5rhXFuIAnOpTTq"
VIEWER = f"https://www.google.com/maps/d/viewer?mid={MID}"
KML_URL = f"https://www.google.com/maps/d/kml?mid={MID}&forcekml=1"
USER_AGENT = "observatory-data/1.0 (+https://github.com/2wrr6c7mmj-jpg/observatory-data)"
SOURCE = ("溫約瑟（Joseph Wen）『中國人民解放軍基地及設施（持續更新）』Google My Maps, " + VIEWER)
TERMS = ("地図の説明に「歡迎大家取用」（誰でも自由に使ってよい）とある。出典を明記して利用する。"
         "元の KML は保存・再配布せず、変換した地物（名前・説明・座標・フォルダ）のみを置く。")


# ---- robots.txt (RFC 9309) ----

def _rule_re(path: str) -> re.Pattern:
    end = path.endswith("$")
    body = re.escape(path[:-1] if end else path).replace(r"\*", ".*")
    return re.compile("^" + body + ("$" if end else ""))


def robots_allows(robots: str, path: str, agent: str = "*") -> tuple[bool, str]:
    """Longest matching rule in the group for ``agent`` (or ``*``); ties go to Allow."""
    groups: list[tuple[list[str], list[tuple[str, str]]]] = []
    agents: list[str] = []
    rules: list[tuple[str, str]] = []
    in_rules = False
    for line in robots.splitlines():
        line = line.split("#", 1)[0].strip()
        if ":" not in line:
            continue
        k, v = (x.strip() for x in line.split(":", 1))
        k = k.lower()
        if k == "user-agent":
            if in_rules:
                groups.append((agents, rules))
                agents, rules, in_rules = [], [], False
            agents.append(v.lower())
        elif k in ("allow", "disallow"):
            in_rules = True
            if v:
                rules.append((k, v))
    if agents:
        groups.append((agents, rules))
    chosen = [r for a, r in groups if agent.lower() in a] or [r for a, r in groups if "*" in a]
    best = None
    for rs in chosen:
        for k, v in rs:
            if _rule_re(v).match(path):
                key = (len(v), k == "allow")
                if best is None or key > best[0]:
                    best = (key, k, v)
    if best is None:
        return True, "no matching rule"
    return best[1] == "allow", f"{best[1]}: {best[2]}"


# ---- KML -> GeoJSON ----

def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child(el, name):
    for c in el:
        if _local(c.tag) == name:
            return c
    return None


def _text(el, name) -> str:
    c = _child(el, name)
    return (c.text or "").strip() if c is not None else ""


def _coords(s: str) -> list[list[float]]:
    out = []
    for tok in s.split():
        parts = tok.split(",")
        if len(parts) >= 2:
            out.append([round(float(parts[0]), 6), round(float(parts[1]), 6)])
    return out


def _geom(el):
    name = _local(el.tag)
    if name == "Point":
        c = _coords(_text(el, "coordinates"))
        return {"type": "Point", "coordinates": c[0]} if c else None
    if name == "LineString":
        return {"type": "LineString", "coordinates": _coords(_text(el, "coordinates"))}
    if name == "Polygon":
        rings = []
        for b in el.iter():
            if _local(b.tag) in ("outerBoundaryIs", "innerBoundaryIs"):
                ring = next((r for r in b.iter() if _local(r.tag) == "coordinates"), None)
                if ring is not None:
                    rings.append(_coords(ring.text or ""))
        return {"type": "Polygon", "coordinates": rings}
    if name == "MultiGeometry":
        gs = [g for g in (_geom(c) for c in el) if g]
        return {"type": "GeometryCollection", "geometries": gs}
    return None


def plain(desc: str) -> str:
    """Description as plain text (My Maps puts <br> and <img> in it)."""
    s = re.sub(r"(?i)<br\s*/?>", "\n", desc or "")
    s = re.sub(r"(?is)<img[^>]*>", "", s)
    s = re.sub(r"(?s)<[^>]+>", "", s)
    return unescape(s).strip()


def convert(kml: bytes, fetched_utc: str) -> dict:
    root = ET.fromstring(kml)
    doc = next((e for e in root.iter() if _local(e.tag) == "Document"), root)
    features = []

    def walk(el, folders):
        for c in el:
            t = _local(c.tag)
            if t == "Folder":
                walk(c, folders + [_text(c, "name")])
            elif t == "Placemark":
                g = None
                for gc in c:
                    g = _geom(gc)
                    if g:
                        break
                props = {"name": _text(c, "name"), "description": plain(_text(c, "description")),
                         "folder": " / ".join(folders)}
                ext = _child(c, "ExtendedData")
                if ext is not None:
                    data = {}
                    for d in ext.iter():
                        if _local(d.tag) == "Data" and d.get("name"):
                            v = _text(d, "value")
                            if v and d.get("name") not in ("description", "名稱", "name"):
                                data[d.get("name")] = v
                    if data:
                        props["data"] = data
                features.append({"type": "Feature", "geometry": g, "properties": props})
            elif t == "Document":
                walk(c, folders)

    walk(doc, [])
    return {
        "type": "FeatureCollection",
        "meta": {
            "source": SOURCE,
            "source_url": VIEWER,
            "map_title": _text(doc, "name"),
            "map_description": plain(_text(doc, "description")),
            "terms": TERMS,
            "fetched_utc": fetched_utc,
            "fetched_from": KML_URL,
            "updates": "一度だけの取り込み（定期更新はしない）",
            "feature_count": len(features),
            "folders": dict(Counter(f["properties"]["folder"] for f in features)),
        },
        "features": features,
    }


def summary(fc: dict) -> str:
    m = fc["meta"]
    lines = [f"map: {m['map_title']}", f"features: {m['feature_count']}", f"fetched: {m['fetched_utc']}",
             "description: " + m["map_description"][:600], "folders:"]
    lines += [f"  {n:4d}  {k}" for k, n in m["folders"].items()]
    return "\n".join(lines) + "\n"


def write(fc: dict, out: str) -> None:
    with open(out, "w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False, indent=1)
        f.write("\n")


def _get(url: str) -> tuple[int, str, bytes]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.status, r.headers.get("content-type", ""), r.read(50_000_000)


def fetch(out: str, summary_path: str) -> int:
    _, _, robots = _get("https://www.google.com/robots.txt")
    ok, why = robots_allows(robots.decode("utf-8", "replace"), "/maps/d/kml?mid=" + MID + "&forcekml=1")
    print(f"robots.txt: {'allowed' if ok else 'DISALLOWED'} ({why})")
    if not ok:
        return 2
    fetched = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    status, ctype, body = _get(KML_URL)
    print(f"KML: HTTP {status}, {ctype}, {len(body)} bytes")
    if b"<kml" not in body[:2000]:
        print(body[:500].decode("utf-8", "replace"))
        return 3
    fc = convert(body, fetched)
    write(fc, out)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"robots.txt: allowed ({why})\n" + summary(fc))
    print(summary(fc))
    return 0


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if argv[0] == "fetch":
        return fetch(argv[1], argv[2])
    if argv[0] == "convert":
        with open(argv[1], "rb") as f:
            fc = convert(f.read(), datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
        write(fc, argv[2])
        print(summary(fc))
        return 0
    raise SystemExit(__doc__)


if __name__ == "__main__":
    raise SystemExit(main())
