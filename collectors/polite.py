"""Shared politeness checks for collectors: robots.txt and bot-challenge pages.

Every collector should call ``require_allowed(url)`` before fetching ``url``
and fetch through ``get()`` (or pass the response to ``check_challenge()``).

robots.txt is read with our own User-Agent and parsed by
``urllib.robotparser``. The outcome is one of:

- allowed            robots.txt read and the URL is not disallowed, or the
                     server answers 404 / 410 (no robots.txt: RFC 9309 2.3.1.3)
- disallowed         the URL is disallowed, or robots.txt answers 401 / 403
                     (treated as "disallow all", as urllib.robotparser does)
- unverifiable       network error, 429, 5xx, an HTML page instead of a
                     robots.txt, or a bot challenge. The source is skipped.

A bot challenge ("Just a moment...", Cloudflare / Incapsula / DDoS-Guard
interstitials) raises ``Challenge``; collectors must stop, not work around it.

Explicit permission (``collectors/permissions.json``): when the maker of a
source has allowed our collection, the permission is recorded there per source
(who allowed it, when it was received, how, the URLs and frequency covered,
where the evidence is, and a review date). ``require_allowed(url, ua,
source=ID)`` then lets a URL through even though robots.txt disallows it, but
only when all of these hold:

- the caller names the source (``source=``) and it has an entry;
- the URL is listed in that entry's ``scope.urls`` (exact match);
- the review date (``review_by``, inclusive, UTC) has not passed;
- robots.txt was read and its *rules* disallow the URL. A robots.txt that
  answers 401 / 403, cannot be read, or is a challenge is never overridden,
  and a bot-challenge page on the URL itself still stops the run.

Sources without a valid entry behave exactly as before (disallowed: skipped).
"""
from __future__ import annotations

import http.client
import json
import time
import urllib.error
import urllib.request
import urllib.robotparser
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

PERMISSIONS_PATH = Path(__file__).resolve().parent / "permissions.json"
PERMISSION_FIELDS = ("granted_by", "received", "channel", "scope", "evidence", "review_by")

ROBOTS_MAX_BYTES = 512 * 1024

# Specific markers of interstitial challenge pages (lower case).
CHALLENGE_MARKERS = (
    "<title>just a moment",
    "cf-chl-",
    "cf_chl_opt",
    "/cdn-cgi/challenge-platform/",
    "attention required! | cloudflare",
    "_incapsula_resource",
    "ddos-guard",
    "checking your browser before accessing",
    "please enable javascript and cookies to continue",
)
# Weaker markers, only counted on a blocking status code.
CHALLENGE_MARKERS_ON_BLOCK = ("captcha", "are you a robot", "access denied")
BLOCK_STATUSES = (403, 429, 503)


class SourceBlocked(Exception):
    """The source must not be fetched in this run (robots.txt or a challenge)."""


class Challenge(SourceBlocked):
    """A bot-challenge page was returned. Stop; do not try to get around it."""


@dataclass(frozen=True)
class Response:
    url: str
    status: int
    headers: dict  # lower-case names
    body: bytes


@dataclass(frozen=True)
class RobotsResult:
    allowed: bool
    verified: bool  # False: robots.txt could not be checked
    reason: str
    by_rules: bool = False  # the answer comes from the rules of a robots.txt that was read (200)
    permission: dict | None = None  # set when allowed only by an explicit permission

    @property
    def access(self) -> str:
        """"robots" (robots.txt allows) or "permission" (explicit permission overrides a disallow)."""
        return "permission" if self.permission else "robots"


def _headers(msg) -> dict:
    return {k.lower(): v for k, v in (msg.items() if msg is not None else [])}


def raw_get(url: str, user_agent: str, timeout: float = 30, headers: dict | None = None,
            max_bytes: int = 20_000_000) -> Response:
    """GET without raising on HTTP error statuses (the body is kept for checks)."""
    req = urllib.request.Request(url, headers={"User-Agent": user_agent, **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return Response(r.geturl(), r.status, _headers(r.headers), r.read(max_bytes))
    except urllib.error.HTTPError as e:
        try:
            body = e.read(max_bytes) if e.fp is not None else b""
        except Exception:  # noqa: BLE001 - body is only used for diagnostics
            body = b""
        return Response(url, e.code, _headers(e.headers), body)


def is_challenge(resp: Response) -> bool:
    if resp.headers.get("cf-mitigated", "").lower() == "challenge":
        return True
    ctype = resp.headers.get("content-type", "").lower()
    head = resp.body[:30000]
    if "html" not in ctype and not head.lstrip()[:1] == b"<":
        return False
    text = head.decode("utf-8", errors="replace").lower()
    if any(m in text for m in CHALLENGE_MARKERS):
        return True
    return resp.status in BLOCK_STATUSES and any(m in text for m in CHALLENGE_MARKERS_ON_BLOCK)


def check_challenge(resp: Response) -> Response:
    if is_challenge(resp):
        raise Challenge(f"{resp.url}: bot challenge page (HTTP {resp.status}); stopping without retrying")
    return resp


_cache: dict[str, tuple[float, object]] = {}
CACHE_SECONDS = 6 * 3600


def robots_url(url: str) -> str:
    p = urlsplit(url)
    return f"{p.scheme}://{p.netloc}/robots.txt"


def _load_robots(url: str, user_agent: str, getter) -> object:
    """A RobotFileParser, or a RobotsResult when the answer does not depend on the path."""
    rurl = robots_url(url)
    try:
        resp = getter(rurl, user_agent, 30, None, ROBOTS_MAX_BYTES)
    except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException, OSError) as e:
        return RobotsResult(False, False, f"{rurl}: could not be read ({e})")
    if is_challenge(resp):
        return RobotsResult(False, False, f"{rurl}: bot challenge page (HTTP {resp.status})")
    if resp.status in (401, 403):
        return RobotsResult(False, True, f"{rurl}: HTTP {resp.status}, treated as disallow all")
    if resp.status in (404, 410):
        return RobotsResult(True, True, f"{rurl}: HTTP {resp.status}, no robots.txt (all allowed)")
    if resp.status != 200:
        return RobotsResult(False, False, f"{rurl}: HTTP {resp.status}")
    text = resp.body.decode("utf-8", errors="replace")
    ctype = resp.headers.get("content-type", "").lower()
    if "html" in ctype or text.lstrip()[:1] == "<":
        return RobotsResult(False, False, f"{rurl}: an HTML page was returned instead of robots.txt")
    rp = urllib.robotparser.RobotFileParser(rurl)
    rp.parse(text.splitlines())
    return rp


def check_robots(url: str, user_agent: str, getter=None) -> RobotsResult:
    getter = getter or raw_get
    origin = robots_url(url)
    hit = _cache.get(origin)
    if hit is None or time.time() - hit[0] > CACHE_SECONDS:
        hit = (time.time(), _load_robots(url, user_agent, getter))
        _cache[origin] = hit
    loaded = hit[1]
    if isinstance(loaded, RobotsResult):
        return loaded
    if loaded.can_fetch(user_agent, url):
        return RobotsResult(True, True, f"{origin}: allowed", by_rules=True)
    return RobotsResult(False, True, f"{origin}: disallowed for this URL", by_rules=True)


# ---------------------------------------------------------------- explicit permissions

def load_permissions(path: Path | None = None) -> dict:
    """The ``sources`` of permissions.json ({} when the file is missing)."""
    path = path or PERMISSIONS_PATH
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("sources") or {}


def _today_utc() -> date:
    return datetime.now(timezone.utc).date()


def find_permission(source: str | None, url: str, today: date | None = None,
                    permissions: dict | None = None) -> tuple[dict | None, str]:
    """(entry, "") when ``source`` has a valid permission covering ``url``; else (None, why not)."""
    if not source:
        return None, "no source named"
    perms = load_permissions() if permissions is None else permissions
    entry = perms.get(source)
    if not entry:
        return None, f"no permission recorded for {source!r}"
    missing = [k for k in PERMISSION_FIELDS if entry.get(k) in (None, "", {}, [])]
    if missing:
        return None, f"permission for {source!r} is incomplete (missing {missing})"
    urls = (entry.get("scope") or {}).get("urls") or []
    if url not in urls:
        return None, f"{url} is outside the permission for {source!r}"
    try:
        review_by = date.fromisoformat(str(entry["review_by"]))
    except ValueError:
        return None, f"permission for {source!r}: bad review_by {entry['review_by']!r}"
    today = today or _today_utc()
    if today > review_by:
        return None, f"permission for {source!r} expired (review_by {review_by.isoformat()}); review it before relying on it"
    return entry, ""


def permission_summary(source: str, entry: dict) -> dict:
    """What status.json records about a fetch made under permission."""
    scope = entry.get("scope") or {}
    return {"source": source, "file": "collectors/permissions.json", "granted_by": entry.get("granted_by"),
            "received": entry.get("received"), "channel": entry.get("channel"),
            "evidence": entry.get("evidence"), "review_by": entry.get("review_by"),
            "max_requests_per_day": scope.get("max_requests_per_day")}


def require_allowed(url: str, user_agent: str, getter=None, source: str | None = None,
                    today: date | None = None, permissions: dict | None = None) -> RobotsResult:
    r = check_robots(url, user_agent, getter)
    if r.allowed:
        return r
    if r.verified and r.by_rules and source:
        entry, why = find_permission(source, url, today, permissions)
        if entry is not None:
            return RobotsResult(True, True, f"{r.reason}; overridden by explicit permission "
                                f"(collectors/permissions.json: {source}, granted by {entry['granted_by']}, "
                                f"evidence {entry['evidence']}, review by {entry['review_by']})",
                                by_rules=True, permission=permission_summary(source, entry))
        raise SourceBlocked(f"{r.reason} ({why})")
    raise SourceBlocked(r.reason)


def get(url: str, user_agent: str, timeout: float = 30, headers: dict | None = None,
        getter=None, source: str | None = None) -> Response:
    """robots.txt check (and explicit permission for ``source``), then GET;
    raises SourceBlocked / Challenge as described above."""
    getter = getter or raw_get
    require_allowed(url, user_agent, getter, source=source)
    return check_challenge(getter(url, user_agent, timeout, headers, 20_000_000))
