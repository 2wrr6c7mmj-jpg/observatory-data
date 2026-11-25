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

    def test_unknown_without_taiwan_area(self):
        t = dict(self.ind('action', 'military', None), group='taiwan')
        j = ew.judge([t, self.ind('action', 'law_enforcement', 0), self.ind('action', 'military', 0)])
        self.assertEqual(j['verdict'], 'unknown')

    def test_unknown_when_action_layer_thin(self):
        self.assertEqual(ew.judge([self.ind('action', 'military', 3)])['verdict'], 'unknown')

    def test_normal(self):
        j = ew.judge([self.ind('action', 'military', 1), self.ind('action', 'law_enforcement', 0)])
        self.assertEqual(j['verdict'], 'normal')
        self.assertEqual(j['ceiling'], 'exercise')

    def test_exercise(self):
        self.assertEqual(ew.judge([self.ind('action', 'military', 2), self.ind('action', 'law_enforcement', 0)])['verdict'], 'exercise')

    def test_one_logistics_category_is_not_alert(self):
        j = ew.judge([self.ind('mobilization', 'medical', 3), self.ind('strategic', 'medical', 2),
                      self.ind('action', 'military', 0), self.ind('action', 'law_enforcement', 0)])
        self.assertEqual(j['verdict'], 'normal')

    def test_two_logistics_categories_alert(self):
        j = ew.judge([self.ind('mobilization', 'personnel', 2), self.ind('strategic', 'money', 3)])
        self.assertEqual(j['verdict'], 'alert')
        self.assertIsNone(j['ceiling'])

    def test_planned_null(self):
        j = ew.judge([self.ind('strategic', 'money', None, 'planned')])
        self.assertIsNone(j['layers'][0]['max'])
        self.assertEqual(j['layers'][0]['collected'], 0)


def ms(day):
    from datetime import datetime, timezone
    return int(datetime(day.year, day.month, day.day, 12, tzinfo=timezone.utc).timestamp() * 1000)


class ExerciseIndicatorTest(unittest.TestCase):
    def test_named_window(self):
        named = [{'name': 'X', 'start': '2026-04-01', 'end': '2026-04-02'}]
        self.assertEqual(ew.exercise_indicator(date(2026, 4, 3), named, [])['score'], 3)

    def test_before_collection_is_null(self):
        self.assertIsNone(ew.exercise_indicator(date(2026, 4, 4), [], [])['score'])
        self.assertEqual(ew.exercise_indicator(date(2026, 12, 1), [], [])['score'], 0)

    def test_event_confidence_region_and_first_report(self):
        day = date(2026, 12, 1)
        ev = [{'countries': ['中国'], 'region': 'taiwan', 'start': '2026-12-01', 'end': None, 'confidence': '高', 'titleJa': 'A', 'firstReported': ms(day)},
              {'countries': ['中国'], 'region': 'europe', 'start': '2026-12-01', 'confidence': '高', 'firstReported': ms(day)},
              {'countries': ['日本'], 'region': 'taiwan', 'start': '2026-12-01', 'confidence': '高', 'firstReported': ms(day)}]
        self.assertEqual(ew.exercise_indicator(date(2026, 12, 2), [], ev)['score'], 2)
        self.assertEqual(ew.exercise_indicator(date(2026, 12, 3), [], ev)['score'], 0)
        late = [dict(ev[0], firstReported=ms(date(2026, 12, 20)))]
        self.assertEqual(ew.exercise_indicator(date(2026, 12, 2), [], late)['score'], 0)
        self.assertEqual(ew.exercise_indicator(date(2026, 12, 2), [], [dict(ev[0], firstReported=None)])['score'], 0)


class JsoTest(unittest.TestCase):
    def src(self, rows, last=None):
        return {'jso_rows': rows, 'jso_last_release': last}

    def test_only_published_rows(self):
        rows = [(date(2026, 1, 1), date(2026, 1, 2)), (date(2026, 1, 5), date(2026, 1, 9))]
        dd = ew.jso_daily(self.src(rows), date(2026, 1, 7))
        self.assertEqual(dd[date(2026, 1, 1)], 1)
        self.assertEqual(dd[date(2026, 1, 5)], 0)
        self.assertEqual(ew.jso_daily(self.src(rows), date(2026, 1, 9))[date(2026, 1, 5)], 1)

    def test_fill_stops_at_last_release(self):
        rows = [(date(2026, 1, 1), date(2026, 1, 1))]
        dd = ew.jso_daily(self.src(rows, date(2026, 1, 3)), date(2026, 1, 10))
        self.assertEqual(max(dd), date(2026, 1, 3))
        self.assertIsNone(dd.get(date(2026, 1, 5)))

    def test_weekly_series_uses_filled_series(self):
        rows = [(date(2025, 1, 1) + timedelta(days=30 * k), date(2025, 1, 1) + timedelta(days=30 * k)) for k in range(18)]
        src = {'aircraft': {}, 'crossings': {}, 'navy': {}, 'ships': {}, 'senkaku': {}, **self.src(rows)}
        inds = ew.build_indicators(src, [], [], date(2026, 6, 1))
        j = [i for i in inds if i['id'] == 'jso-transits'][0]
        self.assertTrue(all(v is not None for _, v in j['series']))


class LagTest(unittest.TestCase):
    def src(self, last):
        s = {k: {} for k in ('crossings', 'navy', 'ships', 'senkaku')}
        s['aircraft'] = {last - timedelta(days=i): 10 for i in range(400)}
        s.update({'jso_rows': [], 'jso_last_release': None})
        return s

    def test_lag_within_three_days_moves_as_of(self):
        inds = ew.build_indicators(self.src(date(2026, 6, 1)), [], [], date(2026, 6, 4), with_series=False)
        a = inds[0]
        self.assertEqual(a['as_of'], '2026-06-01')
        self.assertEqual(a['score'], 0)

    def test_older_data_is_null(self):
        inds = ew.build_indicators(self.src(date(2026, 6, 1)), [], [], date(2026, 6, 10), with_series=False)
        self.assertIsNone(inds[0]['score'])


class NoAnswerLeakTest(unittest.TestCase):
    def test_backtest_and_false_alarms_ignore_named_list(self):
        src = {k: {} for k in ('aircraft', 'crossings', 'navy', 'ships', 'senkaku')}
        src.update({'jso_rows': [], 'jso_last_release': None})
        named = [{'id': 'x', 'name': 'X', 'start': '2026-12-10', 'end': '2026-12-11'}]
        rows = ew.backtest(src, named, [])
        self.assertTrue(all(p['scores']['exercises'] != 3 for p in rows[0]['points']))
        fa = ew.false_alarms(src, [], [], date(2026, 12, 11), days=3)
        self.assertEqual(fa['flagged'], 0)
        self.assertEqual(fa['unknown_days'], 3)


class WriteTest(unittest.TestCase):
    def test_generated_utc_only_change_is_not_written(self):
        import tempfile
        old = ew.OUT
        with tempfile.TemporaryDirectory() as tmp:
            ew.OUT = tmp
            try:
                self.assertTrue(ew.write_if_changed('a.json', {'generated_utc': '1', 'x': 1}))
                self.assertFalse(ew.write_if_changed('a.json', {'generated_utc': '2', 'x': 1}))
                self.assertTrue(ew.write_if_changed('a.json', {'generated_utc': '3', 'x': 2}))
            finally:
                ew.OUT = old


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
