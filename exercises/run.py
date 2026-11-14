"""Collect, read and merge Indo-Pacific military exercises.

    python -m exercises.run --minutes 40              # on main: writes docs/data/exercises
    python -m exercises.run --probe --out DIR --limit 8   # probe: temporary directory, notices

Needs a llama.cpp server (OpenAI API) at --server, as translate.yml starts it.

Writes (docs/data/exercises/):
  current.json        events now under way, coming, or finished in the last 30 days (the site's screen)
  events/<year>.json  every event, by the year it started (or was first reported): the history search
  analyses.json       one line per document already read (so it is not read again), kept 60 days
  status.json         the last run: candidates, read, sources and their state
  meta.json           rules, areas, attribution
"""
import argparse
import json
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from collectors import polite

from . import extract, merge, regions, sources
from .text import html_to_text, pdf_to_text

OUT = "docs/data/exercises"
KEEP_ANALYSES_DAYS = 60
CURRENT_DONE_DAYS = 30
OFFICIAL_SINCE_DAYS = 45
DAY = 86400000
# The checking rules. When they change, what was read before is read again (the stored answers were kept
# under the old rules): 1 = first version; 2 = countries named in the text, the area named in the text,
# numbers of the forces in the text, names checked (descriptions, earlier exercises, generic names).
RULES = 3
# 3 = model's country names in Chinese/English mapped to the table, China's interior and other areas from the
#     place words, unnamed reports of neighbouring areas merged.


def load(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, ValueError):
        return default


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, separators=(",", ":"))
        f.write("\n")


def fetch(url):
    r = polite.get(url, sources.UA, timeout=30)
    return r.status, r.body


def body_text(doc, log):
    """The document's text; empty when robots.txt, a challenge or an error stops it."""
    try:
        status, data = fetch(doc["url"])
    except polite.SourceBlocked as e:
        log["blocked"] = log.get("blocked", 0) + 1
        return "", str(e)[:120]
    except Exception as e:  # noqa: BLE001
        log["errors"] = log.get("errors", 0) + 1
        return "", str(e)[:120]
    if status != 200:
        log["errors"] = log.get("errors", 0) + 1
        return "", f"http {status}"
    if doc["url"].lower().endswith(".pdf") or data[:5] == b"%PDF-":
        return pdf_to_text(data), "pdf"
    return html_to_text(data.decode("utf-8", "replace")), "html"


def fit(text):
    """Keeps the prompt inside the model's window (8k tokens): CJK text uses about
    one token per character, Latin text about one per four."""
    cjk = sum(1 for ch in text[:2000] if "\u3040" <= ch <= "\u9fff")
    return text[:2500] if cjk > 400 else text[:4500]


def doc_key(doc):
    return doc["id"] + ":" + (doc.get("fingerprint") or "")[:16]


def day_of(ms):
    return datetime.fromtimestamp((ms or 0) / 1000, timezone.utc).date().isoformat()


def stored_event(e):
    """Events keep their internal 'official' flag between runs."""
    e = dict(e)
    e["_official"] = bool(e.pop("official", False)) or any(s.get("kind") in merge.OFFICIAL_KINDS for s in e["sources"])
    for k in ("id", "regionName", "phase", "confidence", "lat", "lon"):
        e.pop(k, None)
    return e


def year_of(e):
    return (e.get("start") or day_of(e["firstReported"]))[:4]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default="")
    ap.add_argument("--minutes", type=float, default=40)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--probe", action="store_true", help="print samples as notices")
    ap.add_argument("--limit", type=int, default=0, help="read at most N documents")
    a = ap.parse_args(argv)
    started = time.time()
    now = int(started * 1000)
    today = datetime.now(timezone.utc).date()
    out = Path(a.out)

    stored = load(out / "analyses.json", {"items": {}})
    fresh_rules = stored.get("rules") != RULES
    analyses = {} if fresh_rules else stored["items"]
    previous = []
    for p in sorted((out / "events").glob("[0-9][0-9][0-9][0-9].json")) if (out / "events").exists() and not fresh_rules else []:
        previous += load(p, {"events": []})["events"]
    previous = [stored_event(e) for e in previous]

    log = {"sources": []}
    feed_docs = []
    try:
        feed = sources.get_json(sources.FEED)
        feed_docs = sources.feed_candidates(feed)
        log["sources"].append({"source": "site-feed", "status": "ok", "records": len(feed.get("records", [])),
                               "picked": len(feed_docs)})
    except Exception as e:  # noqa: BLE001
        log["sources"].append({"source": "site-feed", "status": "error", "reason": str(e)[:200]})
    since = (today - timedelta(days=OFFICIAL_SINCE_DAYS)).isoformat()
    official = sources.official_candidates(fetch, since, log["sources"])
    def due(d):
        hit = analyses.get(doc_key(d))
        return hit is None or (not hit.get("ok") and now - hit.get("at", 0) > DAY)
    todo = [d for d in official + feed_docs if due(d)]
    log["candidates"] = len(official) + len(feed_docs)
    log["new"] = len(todo)

    fresh, read, failed, body_log, samples, seconds = [], 0, 0, {}, [], []
    for doc in todo:
        if time.time() - started > a.minutes * 60 or (a.limit and read >= a.limit):
            break
        t0 = time.time()
        text, how = body_text(doc, body_log)
        text = fit(text)
        published = day_of(doc.get("publishedAt") or now)
        try:
            raw = extract.ask(a.server, doc, text, published)
            ans = extract.validate(raw, doc, text, published)
            ok = True
        except Exception as e:  # noqa: BLE001 - one bad answer must not stop the run
            print(f"read failed: {doc['url']}: {e}", file=sys.stderr)
            ans, ok = {"relevant": False}, False
            failed += 1
        read += 1
        seconds.append(time.time() - t0)
        meta = {k: doc.get(k) for k in ("url", "title", "titleJa", "source", "sourceName", "kind", "publishedAt")}
        entry = {"at": now, "ok": ok, "body": how if text else "none"}
        if ans.get("relevant"):
            entry.update(doc=meta, answer=ans)
            fresh.append(merge.event_from({**meta, "analyzedAt": now}, ans))
        analyses[doc_key(doc)] = entry
        if a.probe:
            samples.append((doc, ans, round(time.time() - t0, 1), len(text), getattr(extract.ask, "last_timings", {})))

    events = merge.merge_all(previous + fresh)
    cutoff = now - KEEP_ANALYSES_DAYS * DAY
    analyses = {k: v for k, v in analyses.items() if v.get("at", 0) >= cutoff}

    finished = [merge.finish(e, today) for e in events]
    for f, e in zip(finished, events):
        f["official"] = e["_official"]
    finished.sort(key=lambda e: (e.get("start") or day_of(e["firstReported"])), reverse=True)
    done_cut = (today - timedelta(days=CURRENT_DONE_DAYS)).isoformat()
    current = [e for e in finished if e["phase"] in ("now", "next") or
               (e.get("end") or e.get("start") or day_of(e["lastReported"])) >= done_cut]

    status = {"generatedAt": now, "model": a.model, "rules": RULES, "reread": fresh_rules, "minutes": round((time.time() - started) / 60, 1),
              "read": read, "failed": failed, "relevantNew": len(fresh), "events": len(finished),
              "current": len(current), "left": max(0, len(todo) - read),
              "secondsPerDocument": round(sum(seconds) / len(seconds), 1) if seconds else None,
              "bodies": body_log, **log}
    if not a.probe or a.out != OUT:
        by_year = {}
        for e in finished:
            by_year.setdefault(year_of(e), []).append(e)
        for y, evs in by_year.items():
            write(out / "events" / f"{y}.json", {"year": int(y), "rules": RULES, "events": evs})
        write(out / "events" / "index.json", {"years": sorted(int(y) for y in by_year)})
        write(out / "current.json", {"generatedAt": now, "today": today.isoformat(), "events": current})
        write(out / "analyses.json", {"rules": RULES, "items": analyses})
        write(out / "status.json", status)
        write(out / "meta.json", meta_doc())

    print(f"::notice title=exercises::candidates {log['candidates']}, new {len(todo)}, read {read} "
          f"(failed {failed}), relevant {len(fresh)}, events {len(finished)}, current {len(current)}, "
          f"left {status['left']}, s/doc {status['secondsPerDocument']}")
    for s in log["sources"]:
        print(f"::notice title=source {s['source']}::{json.dumps(s, ensure_ascii=False)[:300]}")
    for doc, ans, secs, n, tm in samples:
        brief = {k: ans.get(k) for k in ("kind", "name", "countries", "region", "place", "forces", "start", "end",
                                         "status")} if ans.get("relevant") else "not relevant"
        print(f"::notice title=sample {secs}s {n}c::{doc['sourceName']}: {doc['title'][:90]} → "
              f"{json.dumps(brief, ensure_ascii=False)[:600]} {json.dumps(tm)}")
    return 0


def meta_doc():
    return {
        "schemaVersion": 1,
        "description": "インド太平洋の軍事演習・訓練・実弾射撃・大規模展開。ニュースと公式発表を公開の言語モデルで読み、原文と照合して事象にまとめたもの。",
        "rules": {
            "version": RULES,
            "dates": "原文に期間の記述があるときだけ。記述（dateEvidence）が原文に無い日付は捨てる",
            "countries": "原文にその国の名（略称・漢字の略記を含む）があるときだけ",
            "region": "場所の言葉が名指しする地域。無ければ見出しの言葉、さらに無ければモデルの選んだ地域を原文にその地名があるときだけ。それ以外は『その他』",
            "forces": "戦力の記述の中の 3 以上の数は原文の数（数字か数詞）にあるときだけ。無ければ戦力の記述ごと捨てる",
            "counts": "艦艇・航空機・人員の数は原文に数字としてあるときだけ",
            "name": "演習名は原文に現れるときだけ。一般名（『日米共同訓練』『米軍・日本軍共同訓練』）、説明文、日付入り、年が事象と違う過去の演習名は名前として扱わない",
            "phase": "now 実施中・next 予定・done 終了。期間があれば期間から、無ければ報道の時点の状態から",
            "confidence": "高：公式発表あり／中：2つ以上の異なる媒体／低：1媒体のみ",
            "null": "null は原文に記述が無いこと",
        },
        "regions": [{"id": r[0], "name": r[1], "lat": r[2], "lon": r[3]} for r in regions.REGIONS],
        "sources": {"feed": sources.FEED, "official": sources.OFFICIAL},
        "model": "Qwen3-8B (Apache-2.0) via llama.cpp on GitHub Actions",
    }


if __name__ == "__main__":
    sys.exit(main())
