import unittest
import urllib.error

from collectors import polite

UA = "observatory-data/1 (+https://github.com/2wrr6c7mmj-jpg/observatory-data)"


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


if __name__ == "__main__":
    unittest.main()
