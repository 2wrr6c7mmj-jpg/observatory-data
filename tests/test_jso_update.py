import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path

from jso import update

LIST = """
<ul class="list-news">
  <li>
    <a href="/js/pdf/2026/p20261113_07.pdf" target="_blank" rel="noreferrer">
      <time datetime="2026-11-13">2026年11月13日</time>
      <span data-tag="公表">公表</span>
      <h5>仏軍主催人道支援・災害救援（ＨＡ／ＤＲ）多国間訓練「赤道２６」への参加について</h5>
    </a>
  </li>
  <li>
    <a href="/js/pdf/2026/p20261113_01.pdf" target="_blank" rel="noreferrer">
      <time datetime="2026-11-13">2026年11月13日</time>
      <span data-tag="公表">公表</span>
      <h5>中国海軍艦艇の動向について（ルーヤンⅢ、フユ／沖縄本島－宮古島間南東進）</h5>
    </a>
  </li>
  <li>
    <a href="/js/pdf/2026/p20261027_03.pdf" target="_blank" rel="noreferrer">
      <time datetime="2026-10-27">2026年10月27日</time>
      <span data-tag="公表">公表</span>
      <h5>中国海軍艦艇の動向について（ルーヤンⅢ、ジャンカイⅡ／沖縄本島－宮古島間南東進）</h5>
    </a>
  </li>
  <li>
    <a href="/js/pdf/2026/p20261102_01n.pdf" target="_blank" rel="noreferrer">
      <time datetime="2026-11-2">2026年11月2日</time>
      <span data-tag="公表">公表</span>
      <h5>推定中国機の動向について</h5>
    </a>
  </li>
  <li>
    <a href="/js/pdf/2018/p20181112_01.pdf" target="_blank" rel="noreferrer">
      <time datetime="2018-11-12">2018年11月12日</time>
      <span data-tag="公表">公表</span>
      <h5>ロシア海軍艦艇との海賊対処訓練の実施について</h5>
    </a>
  </li>
</ul>
"""

COLS = ("seq,observed_date,observed_date_end,observed_time,ref_point,bearing_jp,distance_km,lat,lon,heading_jp,"
        "heading_deg,strait,strait_dir,waters_from,waters_to,activity,activity_count,same_as_release,same_as_text,"
        "unit_count,sentence")
UNITS = ("SELECT u.nation,kind,class_jp,class_key,hull_no,name,count FROM sighting_units u "
         "JOIN sightings s USING(sighting_id) WHERE release_id=? ORDER BY seq,u.id")


def remove(con, k):
    con.execute("DELETE FROM sighting_units WHERE sighting_id IN (SELECT sighting_id FROM sightings WHERE release_id=?)", (k,))
    for t in ("sightings", "jp_units", "releases"):
        con.execute(f"DELETE FROM {t} WHERE release_id=?", (k,))
    con.commit()


class ListPage(unittest.TestCase):
    def test_items_and_targets(self):
        items = update.list_items(LIST)
        self.assertEqual([i["release_id"] for i in items],
                         ["2026/p20261113_07", "2026/p20261113_01", "2026/p20261027_03", "2026/p20261102_01n",
                          "2018/p20181112_01"])
        self.assertEqual(items[1]["title"],
                         "2026年11月13日 公表 中国海軍艦艇の動向について（ルーヤンⅢ、フユ／沖縄本島－宮古島間南東進）")
        self.assertEqual([update.is_target(i["title"]) for i in items], [False, True, True, True, False])
        self.assertEqual(items[3]["pub_date"], "2026-11-02")  # from the file name; the datetime attribute is malformed

    def test_only_new_recent_targets_are_fetched(self):
        items = update.list_items(LIST)
        todo = update.to_fetch(items, {"2026/p20261027_03"}, "2026-10-27")
        self.assertEqual([i["release_id"] for i in todo], ["2026/p20261102_01n", "2026/p20261113_01"])
        self.assertEqual([update.category(i["title"]) for i in todo], [("CN", "air"), ("CN", "ship")])


class Apply(unittest.TestCase):
    """A stored release removed and added again from its text gives the same rows (the text here is the
    pdf.js text of the base database; the pdftotext text was checked the same way on 23 releases)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "jso.sqlite"
        shutil.copyfile(update.BASE_DB, self.db)
        self.ref = sqlite3.connect(update.BASE_DB)

    def tearDown(self):
        self.ref.close()
        self.tmp.cleanup()

    def record(self, k):
        pub, title, pages, text = self.ref.execute(
            "SELECT pub_date,title,pages,raw_text FROM releases WHERE release_id=?", (k,)).fetchone()
        return dict(release_id=k, pub_date=pub, title=title, pages=pages, text=text)

    def test_round_trip(self):
        keys = ["2026/p20261027_01", "2026/p20261027_02", "2026/p20261027_03", "2025/p20250507_02"]
        con = sqlite3.connect(self.db)
        for k in keys:
            remove(con, k)
        self.assertEqual(sorted(update.apply(con, [self.record(k) for k in keys])), sorted(keys))
        for k in keys:
            q = f"SELECT {COLS} FROM sightings WHERE release_id=? ORDER BY seq"
            self.assertEqual(con.execute(q, (k,)).fetchall(), self.ref.execute(q, (k,)).fetchall(), k)
            self.assertEqual(con.execute(UNITS, (k,)).fetchall(), self.ref.execute(UNITS, (k,)).fetchall(), k)
        for t in ("transit_pairs", "vessels"):
            q = f"SELECT count(*) FROM {t}"
            self.assertEqual(con.execute(q).fetchone(), self.ref.execute(q).fetchone(), t)
        self.assertEqual(update.apply(con, [self.record(k) for k in keys]), [])  # already there
        con.close()

    def test_post_load_reproduces_the_base_tables(self):
        con = sqlite3.connect(self.db)
        for stmt in update.post_load_statements():
            con.execute(stmt)
        for q in ("SELECT out_sighting_id,in_sighting_id,method,days_away FROM transit_pairs ORDER BY 1,2",
                  "SELECT * FROM vessels ORDER BY nation,hull_no"):
            self.assertEqual(con.execute(q).fetchall(), self.ref.execute(q).fetchall())
        con.close()

    def test_d1_statements_rewrite_one_release(self):
        """The D1 statements, run on a copy without the release (as D1 before the sync), add it without the full
        text and sentences; run again they replace it rather than adding twice."""
        k = "2026/p20261027_03"
        target = sqlite3.connect(self.db)
        remove(target, k)
        sql = update.d1_release_sql(self.ref, k)
        for _ in range(2):
            target.executescript(sql)
        q = ("SELECT seq,observed_date,strait,strait_dir,waters_from,waters_to,same_as_release,unit_count "
             "FROM sightings WHERE release_id=? ORDER BY seq")
        self.assertEqual(target.execute(q, (k,)).fetchall(), self.ref.execute(q, (k,)).fetchall())
        self.assertEqual(target.execute(UNITS, (k,)).fetchall(), self.ref.execute(UNITS, (k,)).fetchall())
        self.assertEqual(target.execute("SELECT raw_text FROM releases WHERE release_id=?", (k,)).fetchone(), (None,))
        self.assertEqual(target.execute("SELECT count(*) FROM sightings WHERE release_id=? AND sentence IS NOT NULL",
                                        (k,)).fetchone(), (0,))
        target.close()


if __name__ == "__main__":
    unittest.main()


class Resp:
    def __init__(self, status, body):
        self.status, self.body = status, body


class Fetch(unittest.TestCase):
    """fetch() with a fake web: a failed release is skipped (tried again next run), a challenge stops the run."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.updates = Path(self.tmp.name) / "updates.jsonl"
        self.text = update.pdf_text
        update.pdf_text = lambda body: ("中国海軍艦艇の動向について" * 10, 2)

    def tearDown(self):
        update.pdf_text = self.text
        self.tmp.cleanup()

    def page(self, n):
        rows = "".join(f'<a href="/js/pdf/2026/p202611{d:02d}_0{i}.pdf"><time datetime="x">x</time><span data-tag="公表">公表</span>'
                       f'<h5>中国海軍艦艇の動向について</h5></a>' for d in range(1, n + 1) for i in (1, 2))
        filler = "".join(f'<a href="/js/pdf/2025/p20250101_{i:03d}.pdf"><time datetime="x">x</time><span data-tag="公表">公表</span>'
                         f'<h5>統合幕僚長の会見</h5></a>' for i in range(100))  # the page lists many other releases
        return (rows + LIST + filler).encode()

    def test_failed_release_is_skipped_and_the_cap_is_kept(self):
        def get(url):
            if url == update.LIST_URL:
                return Resp(200, self.page(3))
            return Resp(404, b"<html>") if url.endswith("p20261102_01.pdf") else Resp(200, b"%PDF-1.4 x")
        n = update.fetch(0, get, self.updates)
        recs = update.read_updates(self.updates)
        self.assertEqual(n, 7)  # 3 days x 2 + 2026/p20261113_01 + 2026/p20261102_01n, less the 404
        self.assertNotIn("2026/p20261102_01", [r["release_id"] for r in recs])
        self.assertEqual(len({r["release_id"] for r in recs}), len(recs))
        old = update.MAX_PER_RUN
        update.MAX_PER_RUN = 1
        try:
            ok = lambda url: Resp(200, self.page(3)) if url == update.LIST_URL else Resp(200, b"%PDF-1.4 x")
            self.assertEqual(update.fetch(0, ok, self.updates), 1)  # the next run tries the failed one again
        finally:
            update.MAX_PER_RUN = old

    def test_challenge_stops_the_run(self):
        from collectors import polite
        def get(url):
            if url == update.LIST_URL:
                return Resp(200, self.page(1))
            raise polite.Challenge("challenge")
        with self.assertRaises(polite.Challenge):
            update.fetch(0, get, self.updates)


class FakeD1:
    """D1 played by a SQLite database (the REST endpoint runs a request of several statements as one batch)."""

    def __init__(self, path):
        self.con = sqlite3.connect(path)
        self.con.row_factory = sqlite3.Row
        self.fail_post_load = False

    def query(self, sql, params=None):
        if sql.lstrip().upper().startswith("SELECT"):
            return [{"results": [dict(r) for r in self.con.execute(sql, params or [])]}]
        if self.fail_post_load and "transit_pairs" in sql and "DELETE FROM transit_pairs" in sql:
            self.con.execute("DELETE FROM transit_pairs")
            self.con.commit()
            raise RuntimeError("D1 error (simulated)")
        self.con.executescript(sql)
        return [{"results": []}]


class D1Sync(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.local = Path(self.tmp.name) / "local.sqlite"
        self.remote = Path(self.tmp.name) / "remote.sqlite"
        shutil.copyfile(update.BASE_DB, self.local)
        shutil.copyfile(update.BASE_DB, self.remote)
        ref = sqlite3.connect(update.BASE_DB)
        self.keys = ["2026/p20261027_02", "2026/p20261027_03"]
        self.recs = []
        for k in self.keys:
            pub, title, pages, text = ref.execute("SELECT pub_date,title,pages,raw_text FROM releases WHERE release_id=?", (k,)).fetchone()
            self.recs.append(dict(release_id=k, pub_date=pub, title=title, pages=pages, text=text))
        ref.close()
        for path in (self.local, self.remote):  # as if both were collected after the base database
            con = sqlite3.connect(path)
            for k in self.keys:
                remove(con, k)
            con.close()
        con = sqlite3.connect(self.local)
        update.apply(con, self.recs)
        con.close()

    def tearDown(self):
        self.tmp.cleanup()

    def counts(self, con, k):
        return con.execute("SELECT count(*) FROM sightings WHERE release_id=?", (k,)).fetchone()[0]

    def test_missing_releases_are_added_once(self):
        local, d1 = sqlite3.connect(self.local), FakeD1(self.remote)
        self.assertEqual(update.d1_sync(local, d1, self.recs), self.keys)
        for k in self.keys:
            self.assertEqual(self.counts(d1.con, k), self.counts(local, k))
        q = "SELECT count(*) FROM transit_pairs"
        self.assertEqual(d1.con.execute(q).fetchone()[0], local.execute(q).fetchone()[0])
        self.assertEqual(update.d1_sync(local, d1, self.recs), [])  # nothing left to do

    def test_a_half_written_release_and_a_failed_rebuild_are_repaired_next_run(self):
        local, d1 = sqlite3.connect(self.local), FakeD1(self.remote)
        d1.fail_post_load = True
        with self.assertRaises(RuntimeError):
            update.d1_sync(local, d1, self.recs)
        self.assertEqual(d1.con.execute("SELECT count(*) FROM transit_pairs").fetchone()[0], 0)
        d1.con.execute("DELETE FROM sighting_units WHERE sighting_id IN (SELECT sighting_id FROM sightings WHERE release_id=?)", (self.keys[0],))
        d1.con.execute("DELETE FROM sightings WHERE release_id=?", (self.keys[0],))
        d1.con.commit()
        d1.fail_post_load = False
        self.assertEqual(update.d1_sync(local, d1, self.recs), [self.keys[0]])
        q = "SELECT count(*) FROM transit_pairs"
        self.assertEqual(d1.con.execute(q).fetchone()[0], local.execute(q).fetchone()[0])
        self.assertEqual(self.counts(d1.con, self.keys[0]), self.counts(local, self.keys[0]))
