"""Where candidates come from.

1. The Observatory site's news feed (the site collects ~48 sources; this reads
   its public /api/feeds, it does not fetch those feeds again).
2. Official press-release lists fetched here (robots.txt checked by
   collectors/polite.py; a challenge or 403 stops that source for the run).
"""
import json
import re
import time
import urllib.request
from urllib.parse import urljoin

FEED = "https://observatory-watch.pages.dev/api/feeds"
UA = "observatory-data/1"

# Official lists. kind is shown on the site as the source's character.
OFFICIAL = [
    {"id": "msdf", "name": "海上幕僚監部 お知らせ", "kind": "公式", "country": "日本",
     "list": "https://www.mod.go.jp/msdf/release/", "href": r"/msdf/release/\d{6}/\d{8}[^\"']*\.pdf"},
    {"id": "jso", "name": "統合幕僚監部 報道発表", "kind": "公式", "country": "日本",
     "list": "https://www.mod.go.jp/js/press/index.html", "href": r"/js/pdf/\d{4}/p\d{8}_\d+\.pdf"},
]

# Words that make a headline a candidate (any language the feed carries).
ACTIVITY = re.compile(
    r"exercis|drill|war ?game|live[- ]fire|manoeuv|maneuv|joint patrol|combat readiness|training|"
    r"keen sword|keen edge|rimpac|balikatan|talisman sabre|malabar|freedom (shield|edge)|ulchi|valiant shield|"
    r"resolute dragon|orient shield|cope north|pitch black|kamandag|sama sama|pacific vanguard|"
    r"joint sea|northern/interaction|strait thunder|joint sword|justice mission|vostok|ocean-20|"
    r"演習|訓練|射撃|演练|演习|演训|演訓|操演|军演|軍演|训练|实弹|实战化|战备警巡|учени|манёвр|маневр|훈련", re.I)
# The region of interest: Indo-Pacific places and actors.
AREA = re.compile(
    r"pacific|indo-|japan|okinawa|korea|taiwan|china|chinese|pla\b|philippin|guam|hawaii|australia|"
    r"india|south china sea|east china sea|sea of japan|east sea|yellow sea|russia|russian|north korea|"
    r"7th fleet|seventh fleet|indopacom|iii mef|marianas|luzon|spratly|scarborough|indonesia|vietnam|"
    r"singapore|malaysia|thailand|new zealand|solomon|pacific islands|"
    r"日本|自衛隊|海自|陸自|空自|米軍|中国|台湾|韓国|北朝鮮|ロシア|フィリピン|豪|インド|太平洋|東シナ海|南シナ海|日本海|"
    r"解放军|台岛|东部战区|南部战区|北部战区|俄|Росси|Тихоокеан", re.I)


def is_candidate(title: str, summary: str = "") -> bool:
    text = f"{title} {summary}"
    return bool(ACTIVITY.search(text)) and bool(AREA.search(text))


def get_json(url: str, timeout: int = 60):
    req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def feed_candidates(feed: dict) -> list:
    """News records that look like an activity in the region, newest first."""
    kinds = {s.get("id"): (s.get("kind") or "報道", s.get("name") or s.get("id")) for s in feed.get("sources", [])}
    out, seen = [], set()
    for r in sorted(feed.get("records", []), key=lambda r: r.get("publishedAt") or 0, reverse=True):
        url = r.get("url")
        if not url or url in seen:
            continue
        if not is_candidate(r.get("title") or "", r.get("summary") or ""):
            continue
        seen.add(url)
        kind, sname = kinds.get(r.get("source"), ("報道", r.get("source")))
        out.append({"id": "feed:" + str(r.get("id")), "fingerprint": r.get("fingerprint") or "",
                    "url": url, "title": r.get("title") or "", "titleJa": r.get("titleJa"),
                    "summary": r.get("summary") or "", "source": r.get("source"), "sourceName": sname,
                    "kind": kind, "publishedAt": r.get("publishedAt")})
    return out


_A = re.compile(r"<a\b[^>]*href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>", re.S | re.I)


def list_links(html_text: str, base: str, href_pattern: str) -> list:
    """(url, link text, YYYY-MM-DD) for links on an official list page."""
    pat = re.compile(href_pattern)
    out, seen = [], set()
    for href, inner in _A.findall(html_text):
        if not pat.search(href):
            continue
        url = urljoin(base, href)
        if url in seen:
            continue
        seen.add(url)
        text = re.sub(r"<[^>]+>", " ", inner)
        text = re.sub(r"\s+", " ", text).strip()
        m = re.search(r"(\d{4})(\d{2})(\d{2})", url)
        day = f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else None
        out.append((url, text, day))
    return out


def official_candidates(fetch, since_day: str, log: list) -> list:
    """fetch(url) -> (status, bytes) via collectors.polite; returns candidates
    whose link text mentions training/exercises, newer than since_day."""
    out = []
    for src in OFFICIAL:
        try:
            status, body = fetch(src["list"])
        except Exception as e:  # noqa: BLE001 - one source must not stop the run
            log.append({"source": src["id"], "status": "skipped", "reason": str(e)[:200]})
            continue
        if status != 200:
            log.append({"source": src["id"], "status": f"http {status}"})
            continue
        links = list_links(body.decode("utf-8", "replace"), src["list"], src["href"])
        picked = [(u, t, d) for u, t, d in links if (d or "") >= since_day and re.search(r"訓練|演習|射撃", t)]
        log.append({"source": src["id"], "status": "ok", "links": len(links), "picked": len(picked)})
        for url, title, day in picked:
            published = int(time.mktime(time.strptime(day, "%Y-%m-%d")) * 1000) if day else None
            out.append({"id": f"{src['id']}:{url.rsplit('/', 1)[-1]}", "fingerprint": "", "url": url,
                        "title": title, "titleJa": title, "summary": "", "source": src["id"],
                        "sourceName": src["name"], "kind": src["kind"], "publishedAt": published,
                        "official": True})
    return out
