import json
import unittest
import unittest.mock
import urllib.error

from collectors import polite

UA = "observatory-data/1"


def getter_for(responses):
    calls = []

    def g(url, ua, timeout, headers, max_bytes):
        calls.append(url)
        r = responses[url]
        if isinstance(r, Exception):
            raise r
        status, ctype, body = r
        return polite.Response(url, status, {"content-type": ctype}, body)
    g.calls = calls
    return g


class Robots(unittest.TestCase):
    def setUp(self):
        polite._cache.clear()

    def check(self, robots, url="https://ex.jp/mission/a.pdf"):
        return polite.check_robots(url, UA, getter_for({"https://ex.jp/robots.txt": robots}))

    def test_rules_are_applied_to_the_path(self):
        robots = (200, "text/plain", b"User-agent: *\nDisallow: /private/\n")
        self.assertTrue(self.check(robots).allowed)
        polite._cache.clear()
        r = self.check(robots, "https://ex.jp/private/x.pdf")
        self.assertEqual((r.allowed, r.verified), (False, True))

    def test_rules_for_our_agent(self):
        robots = (200, "text/plain", b"User-agent: observatory-data\nDisallow: /\n\nUser-agent: *\nAllow: /\n")
        self.assertFalse(self.check(robots).allowed)

    def test_missing_robots_allows(self):
        r = self.check((404, "text/html", b"<html>not found</html>"))
        self.assertEqual((r.allowed, r.verified), (True, True))

    def test_forbidden_robots_disallows(self):
        self.assertEqual(self.check((403, "text/plain", b"")).allowed, False)

    def test_unverifiable_is_not_allowed(self):
        for robots in [(500, "text/plain", b""), (429, "text/plain", b""),
                       (200, "text/html", b"<!doctype html><title>top page</title>"),
                       urllib.error.URLError("timed out"), TimeoutError()]:
            polite._cache.clear()
            r = self.check(robots)
            self.assertEqual((r.allowed, r.verified), (False, False), robots)

    def test_challenge_on_robots_is_unverifiable(self):
        r = self.check((503, "text/html", b"<html><head><title>Just a moment...</title>"))
        self.assertEqual((r.allowed, r.verified), (False, False))

    def test_robots_is_read_once_per_origin(self):
        g = getter_for({"https://ex.jp/robots.txt": (200, "text/plain", b"User-agent: *\nAllow: /\n")})
        polite.check_robots("https://ex.jp/a", UA, g)
        polite.check_robots("https://ex.jp/b", UA, g)
        self.assertEqual(g.calls, ["https://ex.jp/robots.txt"])


class Get(unittest.TestCase):
    def setUp(self):
        polite._cache.clear()

    def test_challenge_page_raises(self):
        g = getter_for({"https://ex.jp/robots.txt": (404, "text/plain", b""),
                        "https://ex.jp/a": (403, "text/html", b"<html><script src='/cdn-cgi/challenge-platform/x'>")})
        with self.assertRaises(polite.Challenge):
            polite.get("https://ex.jp/a", UA, getter=g)

    def test_disallowed_url_is_not_fetched(self):
        g = getter_for({"https://ex.jp/robots.txt": (200, "text/plain", b"User-agent: *\nDisallow: /\n")})
        with self.assertRaises(polite.SourceBlocked):
            polite.get("https://ex.jp/a", UA, getter=g)
        self.assertEqual(g.calls, ["https://ex.jp/robots.txt"])

    def test_ordinary_pages_pass(self):
        g = getter_for({"https://ex.jp/robots.txt": (404, "text/plain", b""),
                        "https://ex.jp/a": (200, "text/html", b"<html><form>captcha</form></html>"),
                        "https://ex.jp/b.pdf": (200, "application/pdf", b"%PDF-1.4 just a moment")})
        self.assertEqual(polite.get("https://ex.jp/a", UA, getter=g).status, 200)
        self.assertEqual(polite.get("https://ex.jp/b.pdf", UA, getter=g).status, 200)



class Permission(unittest.TestCase):
    """Explicit permission (collectors/permissions.json) overrides only a robots.txt disallow, only for its source."""
    URL = "https://ex.jp/data/records.csv"
    DISALLOW = (200, "text/plain", b"User-agent: *\nDisallow: /\n")
    TODAY = polite.date(2026, 11, 1)

    def perms(self, **kw):
        entry = {"granted_by": "maker", "received": "2026-10-31T19:20+09:00", "channel": "user report",
                 "scope": {"urls": [self.URL], "max_requests_per_day": 1}, "evidence": "pending",
                 "review_by": "2026-11-30"}
        entry.update(kw)
        return {"ex": entry}

    def setUp(self):
        polite._cache.clear()

    def req(self, robots=None, url=None, source="ex", perms=None, today=None):
        g = getter_for({"https://ex.jp/robots.txt": robots or self.DISALLOW})
        return polite.require_allowed(url or self.URL, UA, g, source=source,
                                      today=today or self.TODAY, permissions=self.perms() if perms is None else perms)

    def test_permission_overrides_a_disallow(self):
        r = self.req()
        self.assertTrue(r.allowed)
        self.assertEqual(r.access, "permission")
        self.assertEqual(r.permission["evidence"], "pending")
        self.assertEqual(r.permission["max_requests_per_day"], 1)
        self.assertIn("disallowed for this URL; overridden by explicit permission", r.reason)

    def test_no_permission_is_skipped_as_before(self):
        with self.assertRaises(polite.SourceBlocked):
            self.req(perms={})
        with self.assertRaises(polite.SourceBlocked):
            self.req(source=None)

    def test_permission_of_another_source_does_not_apply(self):
        with self.assertRaisesRegex(polite.SourceBlocked, "no permission recorded"):
            self.req(source="other")

    def test_url_outside_the_scope_is_not_overridden(self):
        with self.assertRaisesRegex(polite.SourceBlocked, "outside the permission"):
            self.req(url="https://ex.jp/en/about")

    def test_expired_permission_is_not_used(self):
        self.assertTrue(self.req(today=polite.date(2026, 11, 30)).allowed)  # the review day itself is included
        polite._cache.clear()
        with self.assertRaisesRegex(polite.SourceBlocked, "expired"):
            self.req(today=polite.date(2026, 12, 1))

    def test_incomplete_permission_is_not_used(self):
        with self.assertRaisesRegex(polite.SourceBlocked, "incomplete"):
            self.req(perms=self.perms(evidence=""))
        polite._cache.clear()
        with self.assertRaisesRegex(polite.SourceBlocked, "bad review_by"):
            self.req(perms=self.perms(review_by="soon"))

    def test_forbidden_or_unverifiable_robots_is_not_overridden(self):
        for robots in [(403, "text/plain", b""), (503, "text/plain", b""),
                       (200, "text/html", b"<html><title>Just a moment...</title></html>"),
                       urllib.error.URLError("down")]:
            polite._cache.clear()
            with self.subTest(robots=robots), self.assertRaises(polite.SourceBlocked):
                self.req(robots=robots)

    def test_allowed_robots_is_access_robots(self):
        r = self.req(robots=(200, "text/plain", b"User-agent: *\nAllow: /\n"))
        self.assertEqual((r.allowed, r.access, r.permission), (True, "robots", None))

    def test_challenge_on_the_url_still_stops(self):
        g = getter_for({"https://ex.jp/robots.txt": self.DISALLOW,
                        self.URL: (403, "text/html", b"<title>Just a moment...</title>")})
        with unittest.mock.patch.object(polite, "load_permissions", return_value=self.perms()), \
                unittest.mock.patch.object(polite, "_today_utc", return_value=self.TODAY):
            with self.assertRaises(polite.Challenge):
                polite.get(self.URL, UA, getter=g, source="ex")

    def test_get_without_source_ignores_permissions(self):
        g = getter_for({"https://ex.jp/robots.txt": self.DISALLOW})
        with unittest.mock.patch.object(polite, "load_permissions", return_value=self.perms()):
            with self.assertRaises(polite.SourceBlocked):
                polite.get(self.URL, UA, getter=g)
        self.assertEqual(g.calls, ["https://ex.jp/robots.txt"])


class PermissionsFile(unittest.TestCase):
    def test_every_entry_is_complete(self):
        perms = polite.load_permissions()
        self.assertIn("pla-tracker", perms)
        for sid, e in perms.items():
            with self.subTest(source=sid):
                for k in polite.PERMISSION_FIELDS:
                    self.assertTrue(e.get(k), k)
                polite.date.fromisoformat(e["review_by"])
                self.assertTrue(e["scope"]["urls"])
                self.assertLessEqual(e["scope"].get("max_requests_per_day", 1), 1)

    def test_pla_tracker_permission_covers_only_the_csv(self):
        from collectors import pla_tracker as P
        e = polite.load_permissions()["pla-tracker"]
        self.assertEqual(e["scope"]["urls"], [P.CSV_URL])
        self.assertTrue(e["evidence"].startswith("受領済み"))
        # nothing about the internal set-up or the private message beyond the agreed wording
        self.assertEqual(e["channel"], "2026-10-31、運営者経由")
        self.assertEqual(e["received"], "2026-10-31")
        text = json.dumps(e, ensure_ascii=False)
        for word in ("ユーザー", "司令部", "19:2", "ボット対策", "軍"):
            self.assertNotIn(word, text)
        self.assertEqual(e["scope"]["user_agent"], P.UA)
        # a plain bot name (no contact, by the user's decision) and never a browser-like string
        self.assertTrue(P.UA.startswith("observatory-data/"))
        self.assertNotIn("Mozilla", P.UA)


class UserAgents(unittest.TestCase):
    """Every collector sends the plain bot name observatory-data/1: no contact details (decided
    2026-10-31) and never a browser-like string."""

    def test_collectors_use_the_plain_bot_name(self):
        import importlib
        for mod, attr in [("collectors.pla_tracker", "UA"), ("collectors.jcg_senkaku", "UA"),
                          ("collectors.taiwan_mnd", "UA"), ("collectors.ll2", "UA"),
                          ("sat.osm_mask", "USER_AGENT"), ("sat.cdse", "USER_AGENT"), ("ocean.coverage_probe", "USER_AGENT")]:
            with self.subTest(mod=mod):
                self.assertEqual(getattr(importlib.import_module(mod), attr), "observatory-data/1")

    def test_no_contact_or_browser_string_in_user_agents(self):
        import re
        from pathlib import Path
        root = Path(__file__).resolve().parent.parent
        pat = re.compile(r"""(?:\bUA|USER_AGENT)\s*=\s*['"]([^'"]*)['"]|['"]User-Agent['"]\s*:\s*['"]([^'"]*)['"]""")
        found = 0
        for p in list(root.glob("collectors/*.py")) + list(root.glob("sat/*.py")) + list(root.glob("ocean/*.py")) \
                + list(root.glob("analysis/*.py")) + list(root.glob("notam/*.mjs")):
            for m in pat.finditer(p.read_text(encoding="utf-8")):
                ua = m.group(1) if m.group(1) is not None else m.group(2)
                found += 1
                with self.subTest(file=p.name, ua=ua):
                    self.assertEqual(ua, "observatory-data/1")
        self.assertGreaterEqual(found, 9)


if __name__ == "__main__":
    unittest.main()
