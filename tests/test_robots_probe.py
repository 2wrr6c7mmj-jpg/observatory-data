import unittest

from collectors import polite, robots_probe


def getter_for(status, body, ctype="text/plain"):
    def g(url, ua, timeout, headers, max_bytes):
        return polite.Response(url, status, {"content-type": ctype}, body.encode())
    return g


class RobotsProbeTest(unittest.TestCase):
    def setUp(self):
        polite._cache.clear()

    def test_allowed_and_disallowed_paths(self):
        body = "User-agent: *\nDisallow: /api/\nCrawl-delay: 5\nSitemap: https://x.example/s.xml\n"
        r = robots_probe.check_host("x.example", ["/", "/api/v2/a"], getter_for(200, body))
        self.assertEqual(r["paths"], {"/": True, "/api/v2/a": False})
        self.assertEqual(r["verdict"], "一部可")
        self.assertIn("Disallow: /api/", r["summary"])
        self.assertIn("Sitemap 1 件", r["summary"])
        self.assertIn("Crawl-delay 5", robots_probe.describe(r))
        # the parse is cached for polite.get
        self.assertTrue(polite.check_robots("https://x.example/", robots_probe.UA, getter=lambda *a: 1/0).allowed)

    def test_disallow_all(self):
        r = robots_probe.check_host("x.example", ["/"], getter_for(200, "User-agent: *\nDisallow: /\n"))
        self.assertEqual(r["verdict"], "不可")

    def test_statuses(self):
        self.assertEqual(robots_probe.check_host("a", ["/"], getter_for(404, ""))["verdict"], "可（robots.txt なし）")
        self.assertEqual(robots_probe.check_host("a", ["/"], getter_for(403, "no"))["verdict"], "不可")
        self.assertTrue(robots_probe.check_host("a", ["/"], getter_for(200, "<html>x</html>", "text/html"))
                        ["verdict"].startswith("確認できない"))
        self.assertTrue(robots_probe.check_host("a", ["/"], getter_for(503, "<title>Just a moment...</title>", "text/html"))
                        ["result"].startswith("ボット対策"))

    def test_unreachable(self):
        def boom(*a):
            raise TimeoutError("timed out")
        r = robots_probe.check_host("a", ["/"], boom)
        self.assertIn("届かない", r["result"])

    def test_dataset_summary(self):
        page = ("<html><title>測試資料集 | 政府資料開放平臺</title><body><dt>提供機關</dt><dd>國防部</dd>"
                "<dt>授權方式</dt><dd>政府資料開放授權條款-第1版</dd>"
                '<a href="https://example.gov.tw/files/a.csv">CSV</a></body></html>')
        s = robots_probe.dataset_summary(page)
        self.assertIn("提供機關：國防部", s)
        self.assertIn("政府資料開放授權條款", s)
        self.assertIn("https://example.gov.tw/files/a.csv", s)


if __name__ == "__main__":
    unittest.main()
