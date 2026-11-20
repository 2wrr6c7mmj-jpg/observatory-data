"""The pictures of the queue page (sat/confirm_img.py) and the queue's use of them (confirm-1, PR-4)."""
from __future__ import annotations

import io
import json
import re
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    from PIL import Image
except ImportError:  # the collect workflows do not install numpy and Pillow; Tests does
    np = None

from sat import confirm as cf
from sat import confirm_img as ci

ROOT = Path(__file__).resolve().parent.parent
GRID = {"west": 120.0, "south": 36.0, "east": 120.004, "north": 36.003}
SEC = {"id": "A", "way": 1, "kind": "pier", "side": 1, "sector": "A", "length_m": 150, "line": [[120.001, 36.001], [120.002, 36.001]],
       "frame": [[120.001, 36.001], [120.002, 36.001], [120.002, 36.002], [120.001, 36.002]]}


def cell(date, p="0.5"):
    return {"site_id": "QDH", "section_id": "A", "date": date, "p": p, "model_version": "t", "cloud_shadow": "0", "nodata": "0"}


@unittest.skipUnless(np is not None, "numpy / Pillow not installed")
class ConfirmImgTest(unittest.TestCase):
    def setUp(self):
        t = tempfile.TemporaryDirectory()
        self.addCleanup(t.cleanup)
        self.out = Path(t.name)
        cf.sync_sections(self.out, "QDH", [SEC], "2026-06-01T00:00:00Z")
        self.refl = np.zeros((33, 36, 4), np.uint16)
        self.refl[..., 0], self.refl[..., 1], self.refl[..., 2] = 300, 600, 3000     # B2, B3, B4: red

    def test_picture_and_index(self):
        p = ci.write_image(self.out, "QDH", "2026-11-01", self.refl, GRID, name="青島港内")
        self.assertEqual(p, self.out / "confirm" / "img" / "QDH" / "2026-11-01.webp")
        self.assertEqual(p.read_bytes()[8:12], b"WEBP")
        im = Image.open(io.BytesIO(p.read_bytes())).convert("RGB")
        self.assertGreater(im.height, 33)                              # the strip with the credit is below the chip
        r, g, b = im.getpixel((10, 10))
        self.assertTrue(r > 240 and g < r and b < g, (r, g, b))        # reflectance 0.3 in B4 is full red; the same look as s2nc
        ix = json.loads((p.parent / "index.json").read_text(encoding="utf-8"))
        self.assertEqual(ix["days"]["2026-11-01"], {"image": "2026-11-01.webp", "grid": {**GRID, "width": 36, "height": 33}})

    def test_queue_refers_to_the_pictures_and_nothing_is_fetched(self):
        c = cf.load_config()
        cf.run([cell("2026-11-01")], self.out, c, "2026-11-02T06:00:00Z")
        q = json.loads((self.out / "confirm" / "queue.json").read_text(encoding="utf-8"))
        self.assertIsNone(q["cells"][0]["image"])
        self.assertEqual(ci.needed(self.out, q), [("QDH", "2026-11-01")])
        ci.write_image(self.out, "QDH", "2026-11-01", self.refl, GRID)
        cf.run([], self.out, c, "2026-11-02T07:00:00Z")
        q = json.loads((self.out / "confirm" / "queue.json").read_text(encoding="utf-8"))
        self.assertEqual(q["cells"][0]["image"], "QDH|2026-11-01")
        self.assertEqual(q["images"]["QDH|2026-11-01"]["src"], "confirm/img/QDH/2026-11-01.webp")
        self.assertEqual(q["images"]["QDH|2026-11-01"]["grid"]["width"], 36)
        self.assertEqual(ci.needed(self.out, q), [])
        self.assertTrue((self.out / q["images"]["QDH|2026-11-01"]["src"]).exists())

    def test_prune_keeps_what_the_queue_uses(self):
        c = cf.load_config()
        for d in ("2026-06-01", "2026-06-10", "2026-10-20", "2026-11-01"):
            ci.write_image(self.out, "QDH", d, self.refl, GRID)
        cf.run([cell("2026-06-10")], self.out, c, "2026-06-11T06:00:00Z")
        f = self.out / "dl.csv"
        f.write_text(",".join(cf.ANSWER_FIELDS) + "\nQDH,A,2026-06-10,1,1,owner,,2026-06-12T00:00:00Z,q\n", encoding="utf-8")
        self.assertEqual(cf.merge(f, self.out, c, "2026-06-12T00:00:00Z")["added"], 1)
        cf.run([cell("2026-11-01")], self.out, c, "2026-11-02T06:00:00Z")
        q = json.loads((self.out / "confirm" / "queue.json").read_text(encoding="utf-8"))
        self.assertEqual(q["cells"][0]["prev"], {"date": "2026-06-10", "answer": 1, "image": "QDH|2026-06-10"})
        gone = ci.prune(self.out, q, "2026-11-02", c["image_keep_days"])
        self.assertEqual(gone, ["QDH/2026-06-01"])                    # old and not referred to; 06-10 is the day of the last answer
        left = sorted(p.name for p in (self.out / "confirm" / "img" / "QDH").glob("*.webp"))
        self.assertEqual(left, ["2026-06-10.webp", "2026-10-20.webp", "2026-11-01.webp"])
        self.assertEqual(sorted(ci.load_index(self.out, "QDH")["days"]), ["2026-06-10", "2026-10-20", "2026-11-01"])


class PageTest(unittest.TestCase):
    """docs/sat/label/index.html: the queue view reads what sat/confirm.py writes, and the old views are still there."""
    HTML = (ROOT / "docs" / "sat" / "label" / "index.html").read_text(encoding="utf-8")

    def test_csv_header_is_merges_fields(self):
        self.assertIn("'" + ",".join(cf.ANSWER_FIELDS) + "'", self.HTML)
        self.assertIn("confirm/queue.json", self.HTML)
        self.assertIn(f"'confirm:{cf.QUEUE_VERSION}'", self.HTML)

    def test_the_old_views_keep_their_csv_and_keys(self):
        self.assertIn("'site_id,section_id,date,verified,verified_occupied,verified_c,verified_color,verified_by,verified_note'", self.HTML)
        self.assertIn("'site_id,section_id,s1_acquired_utc,relative_orbit,s2_date,dt_h,verified,verified_occupied,verified_by,verified_note'", self.HTML)
        self.assertIn("(S1 ? 's1rev:' : 's2rev:') + SITE + ':' + R.kind + ':' + R.version", self.HTML)
        self.assertIn("if (QUEUE) loadQueue(); else load();", self.HTML)

    def test_the_page_has_no_place_for_the_judge(self):
        js = self.HTML[self.HTML.index("// ---------------------------------------------------------------- the queue"):]
        for word in ("c.p", ".score", "audit", "v1_state", "model_version"):
            self.assertIsNone(re.search(r"\b" + re.escape(word) + r"\b", js), word)


if __name__ == "__main__":
    unittest.main()
