"""sat/pres3/fetch.py: a refusal (HTTP 4xx, 403 and 429 included) is never sent again; only a transient failure (no
connection, cut off, timeout, 5xx) is retried, a fixed number of times. Stubs only: no request leaves the machine.

What is guarded:
  1. search(): one request on a 4xx; the re-sends and their delays on a transient failure; the paging still works
  2. fetch_one() (the window of a scene in fetch.main): the same rule around the reads
  3. gdal_env(): GDAL re-sends only 5xx and failures without an HTTP status (GDAL_HTTP_RETRY_CODES, GDAL >= 3.10, as GDAL
     matches it: a substring test); no retry at all with an older GDAL
  4. where rasterio is installed (not in the Tests workflow): GDAL itself, against a server on 127.0.0.1 that answers 403 /
     429 / 503, makes the number of requests the settings say
numpy only (fetch.py imports it); skipped where numpy is missing (the collect workflows).
"""
from __future__ import annotations

import http.client
import importlib.util
import io
import json
import ssl
import threading
import unittest
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    import numpy as np
except ImportError:  # the collect workflows do not install numpy; Tests does
    np = None

if np is not None:
    from sat.pres3 import fetch


def http_error(code):
    return urllib.error.HTTPError(fetch.SEARCH_URL if np is not None else "", code, "x", {}, io.BytesIO(b""))


class Stub:
    """urlopen: answers (an exception to raise, or a dict to return as JSON) one per call; records the calls."""

    def __init__(self, *answers):
        self.answers = list(answers); self.calls = []

    def __call__(self, req, timeout=None):
        self.calls.append(json.loads(req.data)); a = self.answers.pop(0)
        if isinstance(a, BaseException): raise a
        return io.BytesIO(json.dumps(a).encode())


PAGE = {"features": [{"id": "a"}], "links": []}


@unittest.skipUnless(np is not None, "numpy not installed")
class SearchRetryTest(unittest.TestCase):
    def run_search(self, *answers):
        stub = Stub(*answers); slept = []
        try: return fetch.search({"bbox": [0, 0, 1, 1]}, urlopen=stub, sleep=slept.append), stub, slept
        except Exception as e: return e, stub, slept

    def test_a_refusal_is_not_sent_again(self):
        for code in (400, 401, 403, 404, 418, 429, 451, 499):
            got, stub, slept = self.run_search(http_error(code), PAGE)
            self.assertIsInstance(got, urllib.error.HTTPError, code); self.assertEqual(got.code, code)
            self.assertEqual((len(stub.calls), slept), (1, []), code)

    def test_a_transient_failure_is_sent_again(self):
        for err in (http_error(500), http_error(502), http_error(503), http_error(504), urllib.error.URLError(ConnectionRefusedError()),
                    urllib.error.URLError("timed out"), TimeoutError(), ConnectionResetError(), http.client.RemoteDisconnected(),
                    http.client.IncompleteRead(b"{")):
            got, stub, slept = self.run_search(err, PAGE)
            self.assertEqual(got, [{"id": "a"}], repr(err)); self.assertEqual((len(stub.calls), slept), (2, [5]), repr(err))

    def test_the_number_of_re_sends_and_their_delays(self):
        self.assertEqual((fetch.SEARCH_TRIES, fetch.SEARCH_DELAYS_S), (4, (5, 10, 15)))
        got, stub, slept = self.run_search(*[http_error(503)] * 5)
        self.assertIsInstance(got, urllib.error.HTTPError); self.assertEqual((len(stub.calls), slept), (4, [5, 10, 15]))
        got, stub, slept = self.run_search(http_error(503), http_error(503), http_error(403), PAGE)      # a refusal after a 5xx
        self.assertEqual(got.code, 403); self.assertEqual((len(stub.calls), slept), (3, [5, 10]))

    def test_a_proxy_refusal_or_a_certificate_is_not_sent_again(self):
        # what urllib raises when the proxy refuses the tunnel (no HTTPError: the status is only in the reason), and when the
        # certificate does not verify
        cert = ssl.SSLCertVerificationError(1, "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate")
        for err in (urllib.error.URLError(OSError("Tunnel connection failed: 403 Forbidden")),
                    urllib.error.URLError(OSError("Tunnel connection failed: 407 Proxy Authentication Required")),
                    urllib.error.URLError(OSError("Tunnel connection failed: 429 Too Many Requests")),
                    urllib.error.URLError("Tunnel connection failed: 403"),
                    urllib.error.URLError(cert), cert,
                    urllib.error.URLError(ssl.SSLError(1, "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed (_ssl.c:1006)")),
                    urllib.error.URLError("certificate has expired")):
            got, stub, slept = self.run_search(err, PAGE)
            self.assertIs(got, err, repr(err)); self.assertEqual((len(stub.calls), slept), (1, []), repr(err))
            self.assertFalse(fetch.transient(err), repr(err))
        # still sent again: a proxy's 5xx, and a failed connection whose text holds a port or an address with 4 in it
        for err in (urllib.error.URLError(OSError("Tunnel connection failed: 502 Bad Gateway")),
                    urllib.error.URLError(OSError("Tunnel connection failed: 503 Service Unavailable")),
                    urllib.error.URLError(ConnectionRefusedError(111, "Connection refused to earth-search.aws.element84.com:443")),
                    urllib.error.URLError(OSError(101, "Network is unreachable 10.0.0.4")),
                    urllib.error.URLError(OSError(-3, "Temporary failure in name resolution")), urllib.error.URLError("[Errno 104] reset"),
                    # curl's text of a failed connection: "port 443" is a port, not a status
                    urllib.error.URLError(OSError("Failed to connect to earth-search.aws.element84.com port 443 after 3 ms: Couldn't connect to server")),
                    urllib.error.URLError("Failed to connect to 127.0.0.1 port 403 after 0 ms: Connection refused")):
            got, stub, slept = self.run_search(err, PAGE)
            self.assertEqual(got, [{"id": "a"}], repr(err)); self.assertEqual((len(stub.calls), slept), (2, [5]), repr(err))

    def test_a_bad_answer_is_not_sent_again(self):
        got, stub, slept = self.run_search(ValueError("not JSON"), PAGE)
        self.assertIsInstance(got, ValueError); self.assertEqual((len(stub.calls), slept), (1, []))

    def test_pages(self):
        first = {"features": [{"id": "a"}], "links": [{"rel": "next", "body": {"token": "t"}}]}
        got, stub, slept = self.run_search(first, http_error(503), PAGE)
        self.assertEqual(got, [{"id": "a"}, {"id": "a"}]); self.assertEqual(slept, [0.5, 5])
        self.assertEqual(stub.calls[0]["collections"], ["sentinel-2-l2a"]); self.assertEqual(stub.calls[1], {"token": "t"})


@unittest.skipUnless(np is not None, "numpy not installed")
class WindowRetryTest(unittest.TestCase):
    def run_one(self, *errors):
        calls = []; slept = []; errors = list(errors)

        def read(work, port, f, bb):
            calls.append(f["id"]); e = errors.pop(0) if errors else None
            if e: raise e
            return "ok"
        return fetch.fetch_one("w", "QDH", {"id": "S"}, [0, 0, 1, 1], read=read, sleep=slept.append), calls, slept

    def test_a_refusal_is_not_read_again(self):
        # what rasterio raises when GDAL gets a refusal (the text GDAL writes), and urllib's own
        for e in (Exception("HTTP response code: 403"), Exception("'/vsicurl/https://x/B04.tif' not recognized ... HTTP response code: 429"),
                  Exception("HTTP response code: 429 - CURL error: timed out"),           # a 4xx wins over the transient words
                  http_error(403), Exception("not recognized as a supported file format")):
            got, calls, slept = self.run_one(e)
            self.assertTrue(got.startswith("ERR S "), e); self.assertEqual((len(calls), slept), (1, []), e)
        # a certificate that does not verify, as curl inside GDAL writes it (it says "CURL error" too) and as urllib does
        for e in (Exception("CURL error: SSL certificate problem: unable to get local issuer certificate"),
                  urllib.error.URLError(OSError("Tunnel connection failed: 403 Forbidden")),
                  Exception("CURL error: CONNECT tunnel failed, response 403"),                  # curl's text of a proxy's 403
                  Exception("CURL error: Received HTTP code 403 from proxy after CONNECT")):
            got, calls, slept = self.run_one(e)
            self.assertTrue(got.startswith("ERR S "), e); self.assertEqual((len(calls), slept), (1, []), e)

    def test_a_transient_failure_is_read_again(self):
        self.assertEqual((fetch.WINDOW_TRIES, fetch.WINDOW_DELAYS_S), (3, (4, 8)))
        for e in (Exception("HTTP response code: 503"), Exception("CURL error: Connection reset by peer"), Exception("Operation timed out"), TimeoutError(),
                  # curl inside GDAL, a failed connection: "port 443" is a port, not a status
                  Exception("CURL error: Failed to connect to sentinel-cogs.s3.us-west-2.amazonaws.com port 443 after 3 ms: Couldn't connect to server"),
                  Exception("Failed to connect to sentinel-cogs.s3.us-west-2.amazonaws.com port 443 after 3 ms: Couldn't connect to server")):
            got, calls, slept = self.run_one(e)
            self.assertEqual((got, len(calls), slept), ("ok", 2, [4]), e)
        got, calls, slept = self.run_one(*[Exception("HTTP response code: 502")] * 3)
        self.assertTrue(got.startswith("ERR S ")); self.assertEqual((len(calls), slept), (3, [4, 8]))


def gdal_retries(code, codes):
    """GDAL 3.12 port/cpl_http.cpp CPLHTTPGetNewRetryDelay with RETRY_CODES set: EQUAL(codes, "ALL") || strstr(codes, "%d" % code)."""
    return codes.upper() == "ALL" or str(code) in codes


@unittest.skipUnless(np is not None, "numpy not installed")
class GdalEnvTest(unittest.TestCase):
    def test_gdal_retries_only_5xx_and_failed_connections(self):
        env = fetch.gdal_env("3.12.2")
        self.assertEqual((env["GDAL_HTTP_MAX_RETRY"], env["GDAL_HTTP_RETRY_DELAY"], env["GDAL_HTTP_RETRY_CODES"]), ("3", "3", "500,502,503,504"))
        codes = env["GDAL_HTTP_RETRY_CODES"]
        self.assertEqual([c for c in range(100, 600) if gdal_retries(c, codes)], [500, 502, 503, 504])
        self.assertTrue(gdal_retries(0, codes))                    # curl failed, no HTTP status (no connection, cut off)
        self.assertEqual(fetch.gdal_env("3.10.0")["GDAL_HTTP_MAX_RETRY"], "3")

    def test_an_older_gdal_does_not_retry(self):
        # before 3.10 GDAL_HTTP_RETRY_CODES is not read and GDAL retries 429 by default: then no retry at all
        for v in ("3.9.3", "3.8.0", "2.4.4"):
            self.assertEqual(fetch.gdal_env(v)["GDAL_HTTP_MAX_RETRY"], "0", v)


class Refuser(BaseHTTPRequestHandler):
    """Answers every request with the server's code; counts the requests."""

    def do_HEAD(self):
        self.server.n += 1; self.send_response(self.server.code); self.send_header("Content-Length", "0"); self.end_headers()

    do_GET = do_HEAD

    def log_message(self, *a):
        pass


@unittest.skipUnless(np is not None and importlib.util.find_spec("rasterio"), "rasterio not installed (the Tests workflow does not install it)")
class GdalAgainstALocalServerTest(unittest.TestCase):
    """GDAL itself, with fetch.gdal_env(), against 127.0.0.1 (no request leaves the machine)."""

    def requests_for(self, code):
        import rasterio
        srv = ThreadingHTTPServer(("127.0.0.1", 0), Refuser); srv.n = 0; srv.code = code
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        env = dict(fetch.gdal_env(), GDAL_HTTP_RETRY_DELAY="0.05")         # 127.0.0.1 is not sent through a proxy (NO_PROXY)
        try:
            with rasterio.Env(**env):
                with self.assertRaises(Exception) as cm:
                    rasterio.open(f"http://127.0.0.1:{srv.server_port}/x{code}/B04.tif")
        finally:
            srv.shutdown(); srv.server_close()
        self.assertEqual(fetch.transient(cm.exception), code >= 500, (code, str(cm.exception)))     # and fetch_one reads it the same way
        return srv.n

    def test_requests_per_refusal_and_per_5xx(self):
        import rasterio
        if tuple(int(x) for x in rasterio.__gdal_version__.split(".")[:2]) < (3, 10): self.skipTest("GDAL < 3.10")
        n403, n429, n404 = self.requests_for(403), self.requests_for(429), self.requests_for(404)
        n503 = self.requests_for(503)
        self.assertEqual((n403, n429, n404), (n404, n404, n404))                  # a refusal: as few requests as a missing file
        self.assertEqual(n503, n404 * (1 + fetch.GDAL_MAX_RETRY))                # a 5xx: each request sent again GDAL_MAX_RETRY times


if __name__ == "__main__":
    unittest.main()
