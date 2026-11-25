import io
import unittest
import zipfile
from datetime import date, datetime, timedelta, timezone

from analysis import ew
from collectors import ew2


def xlsx(rows):
    """A tiny xlsx with inline shared strings, enough for xlsx_rows."""
    strings, sheet = [], []
    for r, row in enumerate(rows, 1):
        cells = []
        for c, v in enumerate(row):
            strings.append(v)
            cells.append(f'<c r="{chr(65 + c)}{r}" t="s"><v>{len(strings) - 1}</v></c>')
        sheet.append(f'<row r="{r}">{"".join(cells)}</row>')
    ns = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as z:
        z.writestr('xl/sharedStrings.xml', f'<sst xmlns="{ns}">' + ''.join(f'<si><t>{s}</t></si>' for s in strings) + '</sst>')
        z.writestr('xl/worksheets/sheet1.xml', f'<worksheet xmlns="{ns}"><sheetData>{"".join(sheet)}</sheetData></worksheet>')
    return buf.getvalue()


class CollectorParseTest(unittest.TestCase):
    def test_gold_table(self):
        rows = ew2.xlsx_rows(xlsx([['项目', '2026.01', '', '2026.02', ''], ['4. 黄金', '3695.82', '2674.51', '3875.88', '2819.30'],
                                   ['', '7419万盎司', '7419万盎司', '7422万盎司', '7422万盎司']]))
        self.assertEqual(ew2.parse_gold(rows), {'2026-01': 7419, '2026-02': 7422})
        self.assertEqual(ew2.parse_gold([['x']]), {})

    def test_jodi_imports_only_numbers(self):
        text = ('REF_AREA,TIME_PERIOD,ENERGY_PRODUCT,FLOW_BREAKDOWN,UNIT_MEASURE,OBS_VALUE,ASSESSMENT_CODE\n'
                'CN,2026-01,CRUDEOIL,TOTIMPSB,KBD,12010.9394,3\nCN,2026-02,CRUDEOIL,TOTIMPSB,KBD,x,3\n'
                'CN,2026-01,CRUDEOIL,INDPROD,KBD,4427,3\nJP,2026-01,CRUDEOIL,TOTIMPSB,KBD,2000,1\n')
        self.assertEqual(ew2.parse_jodi(text), {'2026-01': 12010.9})

    def test_rmrb_both_formats_and_page_titles_skipped(self):
        class R:
            def __init__(self, status, body=b''):
                self.status, self.body = status, body
        pc = ('<a href="node_02.html">02版</a><a href="../../../content/202611/24/content_1.html">第01版：要闻 PDF下载</a>'
              '<a href="../../../content/202611/24/content_2.html">坚决反对台独分裂</a>').encode()
        old = '<a href="nw.D110000renmrb_20240126_1-01.htm">两岸融合</a>'.encode()

        def get_new(url):
            return R(200, pc) if 'node_01' in url else R(404)
        heads = ew2.rmrb_day(date(2026, 11, 24), get_new)
        self.assertEqual([h['title'] for h in heads], ['坚决反对台独分裂'])

        def get_old(url):
            if '/pc/' in url:
                return R(404)
            return R(200, old) if '_01.htm' in url else R(404)
        self.assertEqual([h['title'] for h in ew2.rmrb_day(date(2024, 1, 26), get_old)], ['两岸融合'])
        self.assertIsNone(ew2.rmrb_day(date(2024, 1, 27), lambda u: R(404)))

    def test_feed_needs_place_and_word(self):
        ms = int(datetime(2026, 11, 25, tzinfo=timezone.utc).timestamp() * 1000)
        feed = {'records': [{'title': '福建民兵拉动演练', 'publishedAt': ms, 'url': 'a'},
                            {'title': 'Taiwan says undersea cable cut near Matsu', 'publishedAt': ms, 'url': 'b'},
                            {'title': '征兵 in Germany', 'publishedAt': ms, 'url': 'c'}]}
        m = ew2.feed_matches(feed)
        self.assertEqual([i['url'] for i in m['mobilization']], ['a'])
        self.assertEqual([i['url'] for i in m['cables']], ['b'])
        self.assertEqual(m['conscription'], [])

    def test_feed_false_positives_are_left_out(self):
        ms = int(datetime(2026, 11, 25, tzinfo=timezone.utc).timestamp() * 1000)
        recs = [('Places to visit explains militia history', 'p1'),     # "PLA" inside words
                ('福建拉动消费', 'p2'),                                   # 拉动 in an economy story
                ('Taiwan extends conscription to one year', 'p3'),       # Taiwan's own conscription
                ('台湾国军征召后备军人', 'p4'),
                ('福建退役军人事务厅', 'p5'),                              # 退役 alone
                ('Taiwan extends conscription amid PLA pressure', 'p6'),  # Taiwan with PLA
                ('浙江召开征兵工作会议', 'p7')]                             # the routine twice-a-year call-up
        m = ew2.feed_matches({'records': [{'title': t, 'publishedAt': ms, 'url': u} for t, u in recs]})
        self.assertEqual(m['mobilization'] + m['conscription'], [])
        m = ew2.feed_matches({'records': [{'title': '解放军福建民兵演练', 'publishedAt': ms, 'url': 'q'}]})
        self.assertEqual([i['url'] for i in m['mobilization']], ['q'])

    def test_merge_ranges(self):
        self.assertEqual(ew2.merge_ranges([['2026-01-05', '2026-01-09'], ['2026-01-01', '2026-01-04'], ['2026-02-01', '2026-02-02']]),
                         [['2026-01-01', '2026-01-09'], ['2026-02-01', '2026-02-02']])


class ScoringTest(unittest.TestCase):
    def test_monthly_waits_for_publication_and_diff(self):
        months = {f'2025-{m:02d}': 7000 + 3 * m for m in range(1, 13)}
        months.update({f'2024-{m:02d}': 6950 + 3 * m for m in range(1, 13)})
        months['2026-01'] = 7300
        a = ew.assess_monthly(months, date(2026, 2, 6), 7, diff=True)
        self.assertEqual(a['month'], '2025-12')
        b = ew.assess_monthly(months, date(2026, 2, 8), 7, diff=True)
        self.assertEqual(b['month'], '2026-01')
        self.assertEqual(b['score'], 3)

    def test_monthly_short_history_is_null(self):
        a = ew.assess_monthly({'2026-01': 1, '2026-02': 2}, date(2026, 6, 1), 7)
        self.assertIsNone(a['score'])

    def test_samples(self):
        s = {date(2026, 6, 30) - timedelta(days=6 * i): 4 for i in range(60)}
        self.assertEqual(ew.assess_samples(s, date(2026, 6, 30))['score'], 0)
        s.update({date(2026, 6, 30) - timedelta(days=i): 20 for i in range(0, 25, 6)})
        self.assertEqual(ew.assess_samples(s, date(2026, 6, 30))['score'], 3)
        self.assertIsNone(ew.assess_samples({}, date(2026, 6, 30))['score'])

    def test_down_direction(self):
        s = {date(2026, 6, 30) - timedelta(days=i): 100 for i in range(400)}
        for i in range(7):
            s[date(2026, 6, 30) - timedelta(days=i)] = 10
        self.assertEqual(ew.assess_count(s, date(2026, 6, 30), down=True)['score'], 3)
        self.assertEqual(ew.assess_count(s, date(2026, 6, 30))['score'], 0)

    def test_feed_event_needs_the_whole_window_covered(self):
        doc = {'covered': [['2026-10-01T00:00Z', '2026-11-30T23:59Z']], 'items': {'cables': [{'date': '2026-11-26', 'title': 'x'}]}}
        self.assertIsNone(ew.feed_event(doc, 'cables', date(2026, 10, 15))['score'])   # window starts 09-16: not covered
        self.assertEqual(ew.feed_event(doc, 'cables', date(2026, 11, 27))['score'], 1)
        self.assertEqual(ew.feed_event(doc, 'mobilization', date(2026, 11, 27))['score'], 0)
        gap = {'covered': [['2026-10-01T00:00Z', '2026-11-10T00:00Z'], ['2026-11-11T06:00Z', '2026-11-30T23:59Z']], 'items': {}}
        self.assertIsNone(ew.feed_event(gap, 'cables', date(2026, 11, 27))['score'])

    def test_feed_event_on_today_uses_the_last_complete_day(self):
        now = datetime(2026, 11, 27, 6, 41, tzinfo=timezone.utc)
        spans = [['2026-10-20T00:00Z', ew2.iso(now)]]
        doc = {'covered': spans, 'items': {'cables': [{'date': '2026-11-26', 'title': 'x'}]}}
        r = ew.feed_event(doc, 'cables', date(2026, 11, 27))
        self.assertEqual((r['score'], r['as_of']), (1, '2026-11-26'))

    def test_covered_spans(self):
        sp = [['2026-11-01T00:00Z', '2026-11-03T00:00Z'], ['2026-11-03T00:00Z', '2026-11-05T12:00Z']]
        self.assertTrue(ew.covered(sp, date(2026, 11, 1), date(2026, 11, 4)))
        self.assertFalse(ew.covered(sp, date(2026, 11, 1), date(2026, 11, 5)))
        self.assertTrue(ew.covered([['2026-11-01', '2026-11-05']], date(2026, 11, 2), date(2026, 11, 5)))
        self.assertEqual(ew2.merge_spans([['2026-11-01T00:00Z', '2026-11-02T00:00Z'], ['2026-11-02T00:00Z', '2026-11-03T00:00Z'],
                                          ['2026-11-04T00:00Z', '2026-11-05T00:00Z']]),
                         [['2026-11-01T00:00Z', '2026-11-03T00:00Z'], ['2026-11-04T00:00Z', '2026-11-05T00:00Z']])

    def test_ew2_indicators_without_data_are_null_not_zero(self):
        inds = ew.ew2_indicators({}, date(2026, 11, 25))
        self.assertTrue(inds)
        self.assertTrue(all(i['score'] is None for i in inds))
        self.assertEqual(len({i['id'] for i in inds}), len(inds))

    def test_jwc_and_ioda(self):
        t = int(datetime(2026, 11, 24, 3, tzinfo=timezone.utc).timestamp())
        src = {'ew2': {'jwc': {'circulars': {'JWLA-036': {'first_seen': '2026-11-20', 'east_asia_terms': ['Taiwan']}}},
                       'ioda': {'verified': {'at': 'x'}, 'covered': [['2026-11-01T00:00Z', '2026-11-30T00:00Z']],
                                'events': [{'start': t, 'duration': 7200}]}}}
        inds = {i['id']: i for i in ew.ew2_indicators(src, date(2026, 11, 25))}
        self.assertEqual(inds['war-risk']['score'], 3)
        self.assertEqual(inds['cables']['score'], 2)
        self.assertIsNone({i['id']: i for i in ew.ew2_indicators(src, date(2026, 11, 10))}['war-risk']['score'])
        src['ew2']['ioda'].pop('verified')
        self.assertIsNone({i['id']: i for i in ew.ew2_indicators(src, date(2026, 11, 25))}['cables']['score'])

    def test_gold_alone_is_one_point_and_trial_indicators_do_not_judge(self):
        months = {f'2024-{m:02d}': 7000 + m for m in range(1, 13)}
        months.update({f'2025-{m:02d}': 7020 + m for m in range(1, 12)})
        months['2025-12'] = 7200
        src = {'ew2': {'gold': {'months': {k: {'value': v} for k, v in months.items()}}}}
        g = {i['id']: i for i in ew.ew2_indicators(src, date(2026, 1, 10))}['gold']
        self.assertEqual(g['score'], 1)
        self.assertFalse(g['judged'])
        mk = lambda layer, cat, sc, **kw: {'layer': layer, 'id': f'{layer}{cat}{sc}', 'category': cat, 'score': sc, 'status': 'collected', **kw}  # noqa: E731
        base = [mk('action', 'military', 0), mk('action', 'law_enforcement', 0)]
        j = ew.judge(base + [mk('strategic', 'money', 3, judged=False), mk('deployment', 'transport', 3, judged=False)])
        self.assertEqual(j['verdict'], 'normal')
        self.assertEqual(len(j['trial_hot']), 2)
        self.assertEqual(j['ceiling'], 'exercise')

    def test_monthly_stale_is_null(self):
        months = {f'2024-{m:02d}': m for m in range(1, 13)}
        months.update({f'2025-{m:02d}': m for m in range(1, 13)})
        self.assertIsNone(ew.assess_monthly(months, date(2026, 6, 1), 7)['score'])
        self.assertIsNotNone(ew.assess_monthly(months, date(2026, 1, 20), 7)['score'])

    def test_amphib_takes_the_highest_yard(self):
        base = {date(2026, 6, 30) - timedelta(days=6 * i): 4 for i in range(60)}
        hot = dict(base)
        hot.update({date(2026, 6, 30) - timedelta(days=i): 20 for i in range(0, 25, 6)})
        src = {'maw': base, 'yards': {'LXD': hot, 'HDZ': {}, 'SJN': base}}
        a = {i['id']: i for i in ew.ew2_indicators(src, date(2026, 6, 30))}['amphib-build']
        self.assertEqual((a['score'], a['site']), (3, '広州・龍穴島'))
        self.assertIsNone(a['parts']['HDZ']['score'])
        self.assertFalse(a['judged'])

    def test_alert_from_logistics(self):
        mk = lambda layer, cat, sc: {'layer': layer, 'id': f'{layer}{cat}{sc}', 'category': cat, 'score': sc, 'status': 'collected'}  # noqa: E731
        base = [mk('action', 'military', 0), mk('action', 'law_enforcement', 0)]
        self.assertEqual(ew.judge(base + [mk('strategic', 'money', 2), mk('deployment', 'transport', 3)])['verdict'], 'alert')
        self.assertEqual(ew.judge(base + [mk('strategic', 'money', 2), mk('mobilization', 'money', 3)])['verdict'], 'normal')
        self.assertIsNone(ew.judge(base + [mk('strategic', 'money', 0)])['ceiling'])


if __name__ == '__main__':
    unittest.main()
