import json
import unittest

from tools import port_roster as pr


class PortRosterTest(unittest.TestCase):
    def setUp(self):
        self.rows = pr.load_rows()
        self.data = pr.build(self.rows)

    def test_file_is_built_from_the_csv(self):
        self.assertEqual(pr.OUT.read_text(encoding='utf-8'), pr.render(self.data))

    def test_roster_checks_of_revision_3(self):
        # 検算（改訂3）の艦種別の隻数
        def n(cond):
            return sum(r['count'] for r in self.rows if cond(r))
        self.assertEqual(n(lambda r: r['class'] == 'Type 055'), 10)
        self.assertEqual(n(lambda r: r['class'] == 'Type 052D'), 35)
        self.assertEqual(n(lambda r: r['class'] in ('Type 054A', 'Type 054AG')), 46)
        self.assertEqual(n(lambda r: r['class'] == 'Type 056A'), 50)
        self.assertEqual(n(lambda r: r['type'] == 'SSK'), 49)
        self.assertEqual(n(lambda r: r['class'] == 'Type 022'), 60)
        pennants = [r['pennant'] for r in self.rows if r['pennant']]
        self.assertEqual(len(pennants), len(set(pennants)))

    def test_destroyer_flotillas_match_their_size(self):
        # 中国語版 Wikipedia の規模（○駆○護）
        size = {'1st': (11, 5), '10th': (8, 5), '3rd': (10, 5), '6th': (11, 6), '2nd': (9, 6), '9th': (12, 5)}
        for k, (dd, ff) in size.items():
            mine = [r for r in self.rows if r['unit'] == f'{k} Destroyer Flotilla']
            self.assertEqual(sum(r['count'] for r in mine if r['type'] == 'DDG'), dd, k)
            self.assertEqual(sum(r['count'] for r in mine if r['type'] == 'FFG'), ff, k)

    def test_assigned_counts(self):
        got = {k: v['assigned'] for k, v in self.data['sites'].items()}
        self.assertEqual(got, {'ZJG': 15, 'SYL': 14, 'SLP': 30, 'QGZ': 17, 'QDH': 9, 'QJG': 3, 'LSN': 23,
                               'ZDH': 32, 'XSN': 6, 'SWS': 8, 'XMN': 6, 'STU': 0})
        stu = self.data['sites']['STU']['units'][0]
        self.assertEqual(stu['short'], 12)

    def test_every_site_is_a_watched_site(self):
        with open(pr.ROOT / 'sat/sites.json', encoding='utf-8') as f:
            sites = json.load(f)
        ids = {s['id'] for s in sites['sites']}
        self.assertTrue(set(self.data['sites']) <= ids)


if __name__ == '__main__':
    unittest.main()
