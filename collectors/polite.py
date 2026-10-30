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
"""
from __future__ import annotations

import http.client
import time
import urllib.error
import urllib.request
import urllib.robotparser
from dataclasses import dataclass
from urllib.parse import urlsplit

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
        return RobotsResult(True, True, f"{origin}: allowed")
    return RobotsResult(False, True, f"{origin}: disallowed for this URL")


def require_allowed(url: str, user_agent: str, getter=None) -> RobotsResult:
    r = check_robots(url, user_agent, getter)
    if not r.allowed:
        raise SourceBlocked(r.reason)
    return r


def get(url: str, user_agent: str, timeout: float = 30, headers: dict | None = None,
        getter=None) -> Response:
    """robots.txt check, then GET; raises SourceBlocked / Challenge as described above."""
    getter = getter or raw_get
    require_allowed(url, user_agent, getter)
    return check_challenge(getter(url, user_agent, timeout, headers, 20_000_000))
