import unittest
from datetime import date, timedelta

from analysis import ew


def flat(days, value, end=date(2026, 6, 30)):
    return {end - timedelta(days=i): value for i in range(days)}


class ScoreTest(unittest.TestCase):
    def test_cuts(self):
        self.assertIsNone(ew.score_from_z(None))
        self.assertEqual([ew.score_from_z(z) for z in (-3, 1.9, 2, 2.9, 3, 4.9, 5, 9)], [0, 0, 1, 1, 2, 2, 3, 3])

    def test_flat_series_is_zero(self):
        a = ew.assess_count(flat(400, 10), date(2026, 6, 30))
        self.assertEqual(a['score'], 0)
        self.assertEqual(a['baseline_median'], 70)

    def test_spike_scores_three(self):
        s = flat(400, 10)
        for i in range(3):
            s[date(2026, 6, 30) - timedelta(days=i)] = 60
        self.assertEqual(ew.assess_count(s, date(2026, 6, 30))['score'], 3)

    def test_drop_is_not_scored(self):
        s = flat(400, 10)
        for i in range(7):
            s[date(2026, 6, 30) - timedelta(days=i)] = 0
        self.assertEqual(ew.assess_count(s, date(2026, 6, 30))['score'], 0)

    def test_short_history_is_null(self):
        a = ew.assess_count(flat(60, 10), date(2026, 6, 30))
        self.assertIsNone(a['score'])
        self.assertIn('週が足りない', a['reason'])

    def test_missing_recent_days_is_null(self):
        s = flat(400, 10)
        for i in range(4):
            s.pop(date(2026, 6, 30) - timedelta(days=i))
        self.assertIsNone(ew.assess_count(s, date(2026, 6, 30))['score'])

    def test_recent_spike_not_in_baseline(self):
        # 直近 14 日の高まりは平常値に入らない
        s = flat(400, 10)
        for i in range(14):
            s[date(2026, 6, 30) - timedelta(days=i)] = 40
        self.assertEqual(ew.assess_count(s, date(2026, 6, 30))['baseline_median'], 70)

    def test_since_limits_baseline(self):
        s = flat(400, 10)
        a = ew.assess_count(s, date(2026, 6, 30), since=date(2026, 5, 1))
        self.assertIsNone(a['score'])

    def test_zero_fill(self):
        s = {date(2025, 6, 1): 1, date(2026, 6, 29): 3}
        a = ew.assess_count(s, date(2026, 6, 30), zero_fill=True)
        self.assertEqual(a['value'], 3)
        self.assertEqual(a['score'], 2)


class JudgeTest(unittest.TestCase):
    def ind(self, layer, cat, score, status='collected'):
        return {'layer': layer, 'id': f'{layer}-{cat}-{score}', 'category': cat, 'score': score, 'status': status}

    def test_normal(self):
        j = ew.judge([self.ind('action', 'military', 1)])
        self.assertEqual(j['verdict'], 'normal')
        self.assertEqual(j['ceiling'], 'exercise')

    def test_exercise(self):
        self.assertEqual(ew.judge([self.ind('action', 'military', 2)])['verdict'], 'exercise')

    def test_one_logistics_category_is_not_alert(self):
        j = ew.judge([self.ind('mobilization', 'medical', 3), self.ind('strategic', 'medical', 2)])
        self.assertEqual(j['verdict'], 'normal')

    def test_two_logistics_categories_alert(self):
        j = ew.judge([self.ind('mobilization', 'medical', 2), self.ind('strategic', 'money', 3)])
        self.assertEqual(j['verdict'], 'alert')
        self.assertIsNone(j['ceiling'])

    def test_planned_null(self):
        j = ew.judge([self.ind('strategic', 'money', None, 'planned')])
        self.assertIsNone(j['layers'][0]['max'])
        self.assertEqual(j['layers'][0]['collected'], 0)


class ExerciseIndicatorTest(unittest.TestCase):
    def test_named_window(self):
        named = [{'name': 'X', 'start': '2026-04-01', 'end': '2026-04-02'}]
        self.assertEqual(ew.exercise_indicator(date(2026, 4, 3), named, [])['score'], 3)
        self.assertEqual(ew.exercise_indicator(date(2026, 4, 4), named, [])['score'], 0)

    def test_event_confidence_and_region(self):
        ev = [{'countries': ['中国'], 'region': 'taiwan', 'start': '2026-04-01', 'end': None, 'confidence': '高', 'titleJa': 'A'},
              {'countries': ['中国'], 'region': 'europe', 'start': '2026-04-01', 'confidence': '高'},
              {'countries': ['日本'], 'region': 'taiwan', 'start': '2026-04-01', 'confidence': '高'}]
        self.assertEqual(ew.exercise_indicator(date(2026, 4, 2), [], ev)['score'], 2)
        self.assertEqual(ew.exercise_indicator(date(2026, 4, 3), [], ev)['score'], 0)


class RealDataTest(unittest.TestCase):
    def test_builds_on_repository_data(self):
        series, named, events = ew.load_inputs()
        inds = ew.build_indicators(series, named, events, date(2026, 11, 1))
        ids = [i['id'] for i in inds]
        self.assertEqual(len(ids), len(set(ids)))
        for i in inds:
            self.assertIn(i['layer'], [l[0] for l in ew.LAYERS])
            self.assertIn(i['category'], ew.CATEGORY_JA)
            if i['status'] == 'planned':
                self.assertIsNone(i['score'])
        self.assertIn(ew.judge(inds)['verdict'], ('normal', 'exercise', 'alert'))


if __name__ == '__main__':
    unittest.main()
