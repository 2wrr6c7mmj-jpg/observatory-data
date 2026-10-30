import json
import math
import os
import random
import tempfile
import unittest
from datetime import date, timedelta
from unittest import mock

from analysis import proto


def make_rows(values, start=date(2024, 1, 1), metric='plan_vessels'):
    rows = {}
    for i, v in enumerate(values):
        if v is ...:  # 日報なしの日
            continue
        rec = {m: None for m in proto.METRICS + ['balloons']}
        rec[metric] = v
        rows[start + timedelta(days=i)] = rec
    return rows


class Basics(unittest.TestCase):
    def test_median_quantile_mad(self):
        self.assertEqual(proto.median([3, 1, 2]), 2)
        self.assertEqual(proto.median([4, 1, 2, 3]), 2.5)
        self.assertIsNone(proto.median([]))
        self.assertAlmostEqual(proto.quantile([1, 2, 3, 4, 5], 0.9), 4.6)
        self.assertAlmostEqual(proto.mad_scale([1, 2, 3, 4, 100]), 1.4826)

    def test_wilson(self):
        p, lo, hi = proto.wilson(5, 10)
        self.assertAlmostEqual(p, 0.5)
        self.assertAlmostEqual(lo, 0.2366, places=3)
        self.assertAlmostEqual(hi, 0.7634, places=3)
        self.assertEqual(proto.wilson(0, 0), (None, None, None))
        self.assertEqual(proto.wilson(0, 20)[1], 0.0)

    def test_tail_poisson_and_nb(self):
        # ポアソン(2) で P(Y>=5) = 1 - Σ_{k<5}
        exact = 1 - sum(math.exp(-2) * 2 ** k / math.factorial(k) for k in range(5))
        self.assertAlmostEqual(proto.nb_upper_tail(5, 2.0, 2.0), exact, places=10)
        self.assertEqual(proto.nb_upper_tail(0, 3.0, 9.0), 1.0)
        # 負の二項（size=2, mu=4）の P(Y>=1) = 1 - (2/6)^2
        self.assertAlmostEqual(proto.nb_upper_tail(1, 4.0, 4 + 16 / 2), 1 - (2 / 6) ** 2, places=10)
        # 裾が重いほど大きな値が起こりやすい
        self.assertGreater(proto.nb_upper_tail(30, 5.0, 50.0), proto.nb_upper_tail(30, 5.0, 5.0))

    def test_fit_nb_size(self):
        poissonish = [(5.0, y) for y in [3, 4, 5, 5, 6, 7, 4, 5, 6, 5] * 10]
        dispersed = [(5.0, y) for y in [0, 0, 1, 2, 5, 9, 15, 0, 1, 17] * 10]
        self.assertGreater(proto.fit_nb_size(poissonish), proto.fit_nb_size(dispersed))


class ChangePoints(unittest.TestCase):
    def test_single_step(self):
        rng = random.Random(3)
        x = [10 + rng.gauss(0, 1) for i in range(40)] + [20 + rng.gauss(0, 1) for i in range(40)]
        self.assertEqual(proto.optimal_partition(x, 3 * math.log(80)), [40])
        res = proto.changepoints(x, n_boot=20)
        self.assertEqual([c['index'] for c in res['changepoints']], [40])
        c = res['changepoints'][0]
        self.assertLessEqual(c['lo'], 40)
        self.assertGreaterEqual(c['hi'], 40)
        self.assertGreater(c['redetect'], 0.9)

    def test_flat_has_no_change(self):
        rng = random.Random(4)
        x = [5 + rng.gauss(0, 1) for i in range(100)]
        self.assertEqual(proto.changepoints(x, n_boot=5)['changepoints'], [])

    def test_weekly_median_and_missing(self):
        vals = [1, 2, 3, 4, 100, None, ...] + [5] * 7  # 2024-01-01 は月曜
        w = proto.weekly(make_rows(vals), 'plan_vessels')
        self.assertEqual(w[0]['value'], 3)  # 中央値なので 100 に引きずられない
        self.assertAlmostEqual(w[0]['missing_share'], 2 / 7, places=3)
        self.assertEqual(w[1]['value'], 5)


class Persistence(unittest.TestCase):
    def test_runs(self):
        vals = [5] * 200 + [9, 9, 9, 5, 9, None, 9, 9]
        R = proto.runs_above(make_rows(vals), window=365, q=0.9, min_obs=180)
        self.assertEqual([r['length'] for r in R['runs']], [3, 1, 2])
        self.assertEqual([r['ended_by'] for r in R['runs']], ['below', 'missing', 'ongoing'])
        self.assertEqual(R['current_run'], 2)

    def test_weekly_share_skips_partial_days(self):
        rows = make_rows([10, 10, 10], metric='aircraft')
        for i, x in enumerate(sorted(rows)):
            rows[x]['median_line_or_adiz'] = [5, None, 2][i]
        w = proto.weekly_share(rows)['weeks'][0]
        self.assertEqual((w['k'], w['n'], w['days']), (7, 20, 2))


class Baseline(unittest.TestCase):
    def test_spike_is_flagged_and_rare(self):
        vals = [6 + (i % 3) for i in range(500)]
        vals[450] = 40
        B = proto.build_baseline(make_rows(vals), 'plan_vessels')
        day = B['days'][450]
        self.assertTrue(day['out'])
        self.assertLess(day['p'], 1e-4)
        self.assertLessEqual(B['n_out'], 3)
        # その日の値は自分の平常値に入らない（過去だけを使う）
        self.assertLess(day['exp'], 10)

    def test_nulls_are_not_zero(self):
        vals = [5] * 120 + [None] * 30 + [5] * 60
        B = proto.build_baseline(make_rows(vals), 'plan_vessels')
        self.assertTrue(all('p' not in d for d in B['days'][120:150]))
        self.assertFalse(any(d.get('out') for d in B['days']))


class Gaps(unittest.TestCase):
    def test_classification(self):
        rows = make_rows([3, ..., ..., 4])
        reports = {'1': {'report_date': '2024-01-02', 'parsed': False}, '2': {'report_date': '2024-01-04', 'parsed': True},
                   '3': {'report_date': '2024-01-04', 'parsed': True}}
        g = proto.build_gaps(rows, reports, [])
        self.assertEqual([c['status'] for c in g['days']], ['parsed', 'unparsed_only', 'no_report', 'parsed'])
        self.assertEqual(g['days'][1]['unparsed'], ['1'])
        self.assertEqual(g['totals']['multiple'], 1)
        self.assertEqual(g['per_metric']['plan_vessels']['value'], 2)
        self.assertEqual(g['per_metric']['aircraft']['null'], 2)


class Events(unittest.TestCase):
    def test_day0_is_next_report(self):
        vals = list(range(200))
        ex = [{'id': 'a', 'start': '2024-03-01', 'sources': []}, {'id': 'b', 'start': '2024-05-01', 'sources': []}]
        E = proto.event_study(make_rows(vals), ex, metrics=['plan_vessels'])
        self.assertEqual(E['events'][0]['day0_report'], '2024-03-02')
        M = E['metrics']['plan_vessels']
        j = M['rel'].index(0)
        self.assertEqual(M['matrix'][0][j], (date(2024, 3, 2) - date(2024, 1, 1)).days)
        self.assertEqual(len(M['loo']), 2)

    def test_summary_needs_three_cases(self):
        vals = list(range(300))
        two = [{'id': c, 'start': s, 'sources': []} for c, s in (('a', '2024-03-01'), ('b', '2024-05-01'))]
        M = proto.event_study(make_rows(vals), two, metrics=['plan_vessels'])['metrics']['plan_vessels']
        self.assertEqual(M['n_cases'], 2)
        self.assertIsNone(M['median_pre_ratio'])
        self.assertIsNone(M['placebo']['percentile_of_observed'])
        self.assertTrue(all(v is None for v in M['median']))
        three = two + [{'id': 'c', 'start': '2024-07-01', 'sources': []}]
        M = proto.event_study(make_rows(vals), three, metrics=['plan_vessels'])['metrics']['plan_vessels']
        self.assertEqual(M['n_cases'], 3)
        self.assertIsNotNone(M['median_pre_ratio'])
        self.assertTrue(all(x['median_pre_ratio'] is None for x in M['loo']))  # 1件除くと2件

    def test_exercise_list_is_sourced(self):
        with open(proto.EXERCISES, encoding='utf-8') as f:
            ex = json.load(f)['exercises']
        self.assertGreaterEqual(len(ex), 4)
        for e in ex:
            date.fromisoformat(e['start'])
            self.assertLessEqual(e['start'], e['end'])
            self.assertTrue(e['sources'] and all(u.startswith('https://') for u in e['sources']), e['id'])


class Output(unittest.TestCase):
    def test_write_if_changed_ignores_timestamp(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(proto, 'OUT', tmp):
            self.assertTrue(proto.write_if_changed('x.json', {'a': [1, 2.5, None]}))
            self.assertFalse(proto.write_if_changed('x.json', {'a': [1, 2.5, None]}))
            self.assertTrue(proto.write_if_changed('x.json', {'a': [1]}))
            self.assertEqual(os.listdir(tmp), ['x.json'])

    def test_real_data_smoke(self):
        rows = proto.load_daily()
        self.assertGreater(len(rows), 100)
        B = proto.build_baseline(rows, 'aircraft')
        self.assertGreater(B['n_evaluated'], 100)
        self.assertTrue(all(0 <= d['p'] <= 1 for d in B['days'] if 'p' in d))


if __name__ == '__main__':
    unittest.main()
