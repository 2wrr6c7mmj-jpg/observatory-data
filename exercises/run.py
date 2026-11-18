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
import re
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
RULES = 7
# 7 = country words that name the country (not "us", "plan", 披露, the South China Sea, インド太平洋 …), counts and
#     numbers not taken from dates or years, month names as whole words and 「10日間」 as a length, a named report
#     without a date joins only an event near it in time, bodies that could not be fetched are fetched again.
#     Events of documents that are not read again are kept (from rules 7 on, a change of rules keeps the history).
# 6 = year files of events that no longer have events are removed (a stale events/2027.json from rules 3 brought
#     back a wrong date); read everything again once.
# 5 = an official release announcing its own country's part adds that country; an undated activity is over 7 days
#     after its last report (was 14).
# 4 = dates read from the quoted words by rule (exercises/dates.py), publication day from the page when the list has
#     none, the end of an activity (merge.phase), confidence with its reason, no current state without a date.
# 3 = model's country names in Chinese/English mapped to the table, China's interior and other areas from the
#     place words, unnamed reports of neighbouring areas merged.


def load(path, default):
    """The file's JSON, or `default` when there is no file. A file that cannot be read stops the run (raises):
    starting from the default would overwrite or remove the history it holds."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default
    except ValueError as e:
        raise SystemExit(f"{path}: not valid JSON ({e}); fix or restore it from git before running again") from e


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, separators=(",", ":"))
        f.write("\n")


def fetch(url):
    r = polite.get(url, sources.UA, timeout=30)
    return r.status, r.body


FETCH_TRIES = 3   # a body that cannot be fetched is tried on this many days; then the headline is read alone


def body_text(doc, log):
    """(text, how, state). state: "ok"; "blocked" when robots.txt disallows the URL or the page is a bot
    challenge (collectors/polite.py: skip, do not retry); "failed" when robots.txt could not be checked, the
    fetch failed or the server answered an error (fetched again on a later day)."""
    try:
        status, data = fetch(doc["url"])
    except polite.Challenge as e:
        log["blocked"] = log.get("blocked", 0) + 1
        return "", str(e)[:120], "blocked"
    except polite.SourceBlocked as e:
        # robots.txt read and it disallows (verified): for good. Not checkable (network, 5xx, HTML): later.
        verified = polite.check_robots(doc["url"], sources.UA).verified   # cached by polite: no new request
        log["blocked" if verified else "errors"] = log.get("blocked" if verified else "errors", 0) + 1
        return "", str(e)[:120], "blocked" if verified else "failed"
    except Exception as e:  # noqa: BLE001
        log["errors"] = log.get("errors", 0) + 1
        return "", str(e)[:120], "failed"
    if status != 200:
        log["errors"] = log.get("errors", 0) + 1
        return "", f"http {status}", "failed"
    if doc["url"].lower().endswith(".pdf") or data[:5] == b"%PDF-":
        return pdf_to_text(data), "pdf", "ok"
    page = data.decode("utf-8", "replace")
    if not doc.get("publishedAt"):
        day = page_day(page)
        if day:
            doc["publishedAt"] = int(datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp() * 1000)
    return html_to_text(page), "html", "ok"


_META_DAY = re.compile(r"<meta[^>]+(?:name|property)=[\"'](?:pubdate|publishdate|PubDate|article:published_time|og:release_date|"
                       r"dc\.date|date)[\"'][^>]+content=[\"'](20\d{2})[-/.年](\d{1,2})[-/.月](\d{1,2})", re.I)
_TEXT_DAY = re.compile(r"(?:发布时间|發布時間|时间|日期|Source:.{0,60}?|Editor:.{0,60}?)\s*[:：]?\s*(20\d{2})[-/.年](\d{1,2})[-/.月](\d{1,2})")


def page_day(page):
    """The page's own publication day (meta tag, or the 发布时间 line), YYYY-MM-DD, or None."""
    m = _META_DAY.search(page) or _TEXT_DAY.search(page)
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3))).isoformat()
    except ValueError:
        return None


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
    e["_pub"] = e.pop("pubKnown", True)
    for k in ("id", "regionName", "phase", "confidence", "confidenceBasis", "lat", "lon"):
        e.pop(k, None)
    return e


LEDGER = Path(__file__).with_name("ledger.json")


def ledger_events():
    """The hand-checked major exercises of the past (ledger.json), as events: merged with the stored ones by name each run."""
    if not LEDGER.exists():
        return []
    out = []
    for x in json.loads(LEDGER.read_text(encoding="utf-8"))["events"]:
        first, *rest = x["sources"]
        e = merge.event_from(first, {k: v for k, v in x.items() if k not in ("sources", "nameEn")})
        for s in rest:
            e = merge._merge(e, merge.event_from(s, {k: v for k, v in x.items() if k not in ("sources", "nameEn")}))
        out.append(e)
    return out


def without_sources(events, urls):
    """The stored events with the sources of `urls` taken out (those documents were read again in this run and
    their new answers replace the old); an event left with no source is dropped."""
    out = []
    for e in events:
        srcs = [s for s in e["sources"] if s.get("url") not in urls]
        if len(srcs) == len(e["sources"]):
            out.append(e)
        elif srcs:
            out.append({**e, "sources": srcs})
    return out


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
    # The stored events are kept whatever the rules: a document read under older rules that is no longer among the
    # candidates cannot be read again, and dropping its event would lose the history. A document read again in this
    # run replaces its old answer (without_sources below).
    previous = []
    for p in sorted((out / "events").glob("[0-9][0-9][0-9][0-9].json")) if (out / "events").exists() else []:
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
    old_items = stored.get("items") or {}

    def due(d):
        hit = analyses.get(doc_key(d))
        return hit is None or (not hit.get("ok") and now - hit.get("at", 0) > DAY)
    todo = [d for d in official + feed_docs if due(d)]
    log["candidates"] = len(official) + len(feed_docs)
    log["new"] = len(todo)

    fresh, read, failed, body_log, samples, seconds, reread = [], 0, 0, {}, [], [], set()
    fetch_failed = 0
    for doc in todo:
        if time.time() - started > a.minutes * 60 or (a.limit and read >= a.limit):
            break
        t0 = time.time()
        text, how, state = body_text(doc, body_log)
        before = analyses.get(doc_key(doc)) or old_items.get(doc_key(doc)) or {}
        tries = (before.get("fetchTries") or 0) + 1 if state == "failed" else 0
        if state == "failed" and tries < FETCH_TRIES:
            # Not read from the headline alone yet: the body is fetched again on a later day.
            analyses[doc_key(doc)] = {"at": now, "ok": False, "body": "none", "fetch": how, "fetchTries": tries}
            fetch_failed += 1
            continue
        text = fit(text)
        published = day_of(doc["publishedAt"]) if doc.get("publishedAt") else None
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
        if state != "ok":
            entry.update(fetch=how, **({"fetchTries": tries} if tries else {}))
        if ok:
            reread.add(doc["url"])
        if ans.get("relevant"):
            entry.update(doc=meta, answer=ans)
            fresh.append(merge.event_from({**meta, "analyzedAt": now}, ans))
        analyses[doc_key(doc)] = entry
        if a.probe:
            samples.append((doc, ans, round(time.time() - t0, 1), len(text), getattr(extract.ask, "last_timings", {})))

    events = merge.merge_all(without_sources(previous, reread) + ledger_events() + fresh)
    cutoff = now - KEEP_ANALYSES_DAYS * DAY
    analyses = {k: v for k, v in analyses.items() if v.get("at", 0) >= cutoff}

    events = [e for e in events if e.get("region") in regions.IN_SCOPE]   # the site's scope (regions.OUT_OF_SCOPE)
    finished = [merge.finish(e, today) for e in events]
    for f, e in zip(finished, events):
        f["official"] = e["_official"]
        f["pubKnown"] = e.get("_pub", True)
    finished.sort(key=lambda e: (e.get("start") or day_of(e["firstReported"])), reverse=True)
    done_cut = (today - timedelta(days=CURRENT_DONE_DAYS)).isoformat()
    current = [e for e in finished if e["phase"] in ("now", "next") or
               (e["phase"] == "done" and (e.get("end") or e.get("start") or day_of(e["lastReported"])) >= done_cut)]

    status = {"generatedAt": now, "model": a.model, "rules": RULES, "reread": fresh_rules, "minutes": round((time.time() - started) / 60, 1),
              "read": read, "failed": failed, "fetchFailed": fetch_failed, "relevantNew": len(fresh), "events": len(finished),
              "current": len(current), "left": max(0, len(todo) - read - fetch_failed),
              "secondsPerDocument": round(sum(seconds) / len(seconds), 1) if seconds else None,
              "bodies": body_log, **log}
    if not a.probe or a.out != OUT:
        by_year = {}
        for e in finished:
            by_year.setdefault(year_of(e), []).append(e)
        for y, evs in by_year.items():
            write(out / "events" / f"{y}.json", {"year": int(y), "rules": RULES, "events": evs})
        for p in (out / "events").glob("[0-9][0-9][0-9][0-9].json"):
            if p.stem not in by_year:
                # Every stored event was loaded above, so a year left without events had them moved to another year
                # or replaced by a new reading of their documents (otherwise its old events come back next run).
                p.unlink()
        write(out / "events" / "index.json", {"years": sorted(int(y) for y in by_year)})
        write(out / "current.json", {"generatedAt": now, "today": today.isoformat(), "events": current})
        write(out / "analyses.json", {"rules": RULES, "items": analyses})
        write(out / "status.json", status)
        write(out / "meta.json", meta_doc())

    print(f"::notice title=exercises::candidates {log['candidates']}, new {len(todo)}, read {read} "
          f"(failed {failed}, body not fetched {fetch_failed}), relevant {len(fresh)}, events {len(finished)}, current {len(current)}, "
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
            "dates": "モデルが写した原文の言葉（dateText）を規則で読む（exercises/dates.py）。日・日の幅（『11月3日から10日間』は 11月3日〜12日）・月の上旬/中旬/下旬だけを日付にし、月名は語として現れるときだけ（marines の mar、decks の dec は月ではない）。『先日』『今月初め』『火曜』などの相対表現と、ページの掲載時刻は日付にしない。年が無いときは掲載日の年（掲載日が分からなければ日付にしない）。datePrecision：day＝日まで、part＝月の上中下旬",
            "phase": "終わりの日がある→その日を過ぎたら終了。始まりだけ→報道が終了と書くか、最後の報道から 14 日で終了。日付なし→実施中にはしない。予定と書かれ最後の報道から 7 日以内なら予定、それ以外は終了。掲載日も分からないものは unknown（現況に出さない）",
            "scope": "東アジア・西太平洋と、米国の太平洋地域の共同訓練。ベンガル湾・インド・インド洋・ロシア西部・米本土・その他の地域の事象は載せない",
            "confidence": "高：期間の書かれた公式の発表（実施国か観測した国の政府・軍。期間の無い公式の発表は中）／中：独立した 2 系列以上の報道（同じ国の国営は 1 系列）／低：1 系列だけ。confidenceBasis にその内訳",
            "countries": "原文にその国の名（略称・漢字の略記を含む）があるときだけ。普通の語や国名を含む地名は国の名としない（代名詞の us、plan、披露・露出・暴露、南シナ海・East China Sea、フィリピン海・Philippine Sea、日本海・Sea of Japan、台湾海峡、インド太平洋、中国地方、内モンゴル、通信社の AFP、『11日米軍』の日米）。US・PLA・PLAN・ROK・ADF などの略称は大文字のときだけ",
            "region": "場所の言葉が名指しする地域。無ければ見出しの言葉、さらに無ければモデルの選んだ地域を原文にその地名があるときだけ。それ以外は『その他』",
            "forces": "戦力の記述の中の 3 以上の数は原文の数（数字か数詞）にあるときだけ。日付・年の数字（11月12日、2026年、Nov. 12）は数として数えない。無ければ戦力の記述ごと捨てる",
            "counts": "艦艇・航空機・人員の数は原文に数字としてあるときだけ。日付・年の一部の数字（『11月12日』の 12、2026 年の 2026）は照合に使わない",
            "merge": "同じ名前の報道は期間が重なるとき 1 つの事象。片方に日付が無いときは、その報道が他方の開始の 60 日前から終了の 30 日後までに掲載されたときだけ。両方に日付が無いときは掲載が 45 日以内のときだけ（翌年の同名演習を前年の事象にまとめない）",
            "history": "規則の版が変わっても、読み直せない過去の文書の事象は残す。読み直した文書はその新しい答えで置き換える。本文を取得できなかった文書は後日取り直す（robots.txt が禁止・ボットチャレンジは取り直さず見出しと要約だけで読む。3 日取れなければ見出しと要約だけで読む）",
            "name": "演習名は原文に現れるときだけ。一般名（『日米共同訓練』『米軍・日本軍共同訓練』）、説明文、日付入り、年が事象と違う過去の演習名は名前として扱わない",
            "null": "null は原文に記述が無いこと",
        },
        "regions": [{"id": r[0], "name": r[1], "lat": r[2], "lon": r[3]} for r in regions.REGIONS],
        "sources": {"feed": sources.FEED, "official": sources.OFFICIAL},
        "model": "Qwen3-8B (Apache-2.0) via llama.cpp on GitHub Actions",
    }


if __name__ == "__main__":
    sys.exit(main())
