import os
import tempfile
import unittest
from datetime import date, timedelta
from unittest import mock

from analysis import proto, proto2


def passage(sid, nation, day, strait='soya', to='オホーツク海', frm='日本海', dir_='E', inferred=False, units='', title='', release=None, nations=None):
    return {'sighting_id': str(sid), 'release_id': release or f'r{sid}', 'nation': nation, 'date': day, 'strait': strait,
            'strait_jp': strait, 'dir': dir_, 'waters_from': frm, 'waters_to': to, 'inferred': inferred, 'units': units,
            'title': title, 'url': 'u', 'nations': nations or ([nation] if nation != 'CNRU' else ['CN', 'RU'])}


def jso(passages, trips=(), first='2020-01-01', last='2020-12-31'):
    return {'passages': list(passages), 'round_trips': list(trips), 'meta': {},
            'status': {'first_release': first, 'last_release': last}}


class Stats(unittest.TestCase):
    def test_clopper_pearson_known_values(self):
        lo, hi = proto2.clopper_pearson(5, 10)
        self.assertAlmostEqual(lo, 0.1871, places=3)
        self.assertAlmostEqual(hi, 0.8129, places=3)
        lo, hi = proto2.clopper_pearson(0, 10)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 0.3085, places=3)

    def test_rate_ratio_exposure_and_zero(self):
        q = proto2.rate_ratio(20, 100, 10, 200)  # 率 0.2 と 0.05
        self.assertAlmostEqual(q['rr'], 4.0)
        self.assertLess(q['lo'], 4.0)
        self.assertGreater(q['hi'], 4.0)
        q = proto2.rate_ratio(3, 365, 0, 365)  # 前年 0 件：比は無限大、区間の下限は出る
        self.assertIsNone(q['rr'])
        self.assertIsNone(q['hi'])
        self.assertGreater(q['lo'], 0)
        self.assertIsNone(proto2.rate_ratio(0, 365, 0, 365)['rr'])

    def test_kaplan_meier_with_censoring(self):
        km = proto2.kaplan_meier([(2, 1), (4, 1), (5, 0), (6, 1)])
        # S(2)=3/4, S(4)=3/4*2/3=1/2, 5 は打ち切り, S(6)=0
        self.assertEqual([s for _, s in km['steps']], [1.0, 0.75, 0.5, 0.0])
        self.assertEqual(km['median'], 4)
        self.assertEqual(km['censored'], 1)
        km = proto2.kaplan_meier([(3, 1), (10, 0), (10, 0)])
        self.assertIsNone(km['median'])  # 中央値に届かない

    def test_midrank_percentile_ties(self):
        ref = [0] * 90 + [1] * 10
        self.assertAlmostEqual(proto2.midrank_percentile(0, ref), 45.0)  # ふつうの 0 は上振れにならない
        self.assertAlmostEqual(proto2.midrank_percentile(1, ref), 95.0)
        self.assertIsNone(proto2.midrank_percentile(None, ref))


class Nations(unittest.TestCase):
    def test_cnru_rows_are_split_by_class(self):
        cls = [('ルーヤンⅢ級ミサイル駆逐艦', 'CN'), ('ウダロイⅠ級駆逐艦', 'RU')]
        self.assertEqual(proto2.unit_nations('ウダロイⅠ級駆逐艦(564) / アルタイ改級補給艦', 'CNRU', cls), {'RU'})
        self.assertEqual(proto2.unit_nations('ルーヤンⅢ級ミサイル駆逐艦(134) / ウダロイⅠ級駆逐艦(564)', 'CNRU', cls), {'CN', 'RU'})
        self.assertEqual(proto2.unit_nations('不明な艦', 'CNRU', cls), set())  # 判定できなければ両国とせず空
        self.assertEqual(proto2.unit_nations('', 'CN', cls), {'CN'})

    def test_real_class_table_covers_cnru_rows(self):
        J = proto2.load_jso()
        cnru = [p for p in J['passages'] if p['nation'] == 'CNRU']
        self.assertTrue(cnru)
        self.assertTrue(any(len(p['nations']) == 1 for p in cnru))


class Heatmap(unittest.TestCase):
    def test_counts_inferred_and_both_nations(self):
        J = jso([passage(1, 'CN', '2020-03-01'), passage(2, 'CN', '2020-03-05', inferred=True),
                 passage(3, 'CNRU', '2020-04-01'), passage(4, 'RU', '2019-04-01')], first='2019-01-01', last='2020-12-31')
        H = proto2.build_heatmap(J)
        cells = {(n, mo, i): k for n, s, mo, i, k in H['cells']}
        self.assertEqual(cells[('CN', '2020-03', 0)], 1)
        self.assertEqual(cells[('CN', '2020-03', 1)], 1)
        self.assertEqual(cells[('CN', '2020-04', 0)], 1)
        self.assertEqual(cells[('RU', '2020-04', 0)], 1)  # 中露の行は両方に数える
        rec = next(x for x in H['rate_ratios'] if x['nation'] == 'CN' and x['strait'] == 'all' and x['inferred'] == 0)
        y20 = next(y for y in rec['years'] if y['year'] == 2020)
        self.assertEqual(y20['n'], 2)  # 明記だけ
        self.assertIsNone(y20['rr'])  # 2019 は 0 件
        self.assertEqual(H['exposure'][2020], 366)


class Stay(unittest.TestCase):
    def test_only_outbound_return_with_strait_and_censoring(self):
        P = [passage(1, 'CN', '2020-01-01', 'miyako', '太平洋', '東シナ海'),
             passage(2, 'CN', '2020-01-11', 'miyako', '東シナ海', '太平洋'),
             passage(3, 'CN', '2020-12-20', 'miyako', '太平洋', '東シナ海'),   # 最近の往路：打ち切り
             passage(4, 'CN', '2020-02-01', 'osumi', '太平洋', '東シナ海'),    # 古い往路：曲線に入れない
             passage(5, 'CN', '2020-03-01', 'miyako', '東シナ海', '太平洋')]   # 帰り（往路ではない）
        trips = [
            {'method': 'stated', 'out_sighting': '1', 'in_sighting': '2', 'out_date': '2020-01-01', 'in_date': '2020-01-11',
             'out_strait_jp': 'm', 'in_strait': 'miyako', 'in_strait_jp': 'm', 'days_away': '10', 'direction': '進出→帰投',
             'nation': 'CN', 'units': '', 'with_aor': '0', 'has_cv': '0'},
            {'method': 'stated', 'out_sighting': '9', 'in_sighting': '10', 'out_date': '2020-05-01', 'in_date': '2020-05-03',
             'out_strait_jp': 'm', 'in_strait': '', 'in_strait_jp': '', 'days_away': '2', 'direction': '進出→帰投',
             'nation': 'CN', 'units': '', 'with_aor': '0', 'has_cv': '0'},  # 復路に海峡なし
            {'method': 'hull-match', 'out_sighting': '11', 'in_sighting': '12', 'out_date': '2020-06-01', 'in_date': '2020-06-05',
             'out_strait_jp': 'm', 'in_strait': 'miyako', 'in_strait_jp': 'm', 'days_away': '4', 'direction': '帰投→再進出',
             'nation': 'CN', 'units': '', 'with_aor': '0', 'has_cv': '0'},
        ]
        S = proto2.build_stay(jso(P, trips))
        self.assertEqual(len(S['trips']), 1)
        self.assertEqual(S['trips'][0]['days'], 10)
        self.assertEqual(S['trips'][0]['to'], '太平洋')
        self.assertEqual(sorted((o['date'], o['recent']) for o in S['open_out']), [('2020-02-01', False), ('2020-12-20', True)])
        c = next(c for c in S['curves'] if c['nation'] == 'CN' and c['dest'] == 'all' and c['method'] == 'all' and c['year'] == 2020)
        self.assertEqual((c['events'], c['censored']), (1, 1))
        # 行先で絞る：組（太平洋行き）と打ち切り（太平洋行き）は「太平洋」に入り、「太平洋以外」には無い
        c = next(c for c in S['curves'] if c['nation'] == 'CN' and c['dest'] == 'pacific' and c['method'] == 'all' and c['year'] == 2020)
        self.assertEqual((c['events'], c['censored']), (1, 1))
        self.assertFalse(any(c['dest'] == 'other' and c['nation'] == 'CN' and c['year'] == 2020 for c in S['curves']))
        self.assertEqual(S['dest_counts'], {'CN:太平洋': 1})


class Joint(unittest.TestCase):
    def test_tiers_and_direction(self):
        P = [passage(1, 'CN', '2020-07-01'), passage(2, 'RU', '2020-07-03'),            # 候補（+2日）
             passage(3, 'CN', '2020-08-01'), passage(4, 'RU', '2020-08-04'),            # 3日差：対象外
             passage(5, 'CN', '2020-09-01'), passage(6, 'RU', '2020-09-01', to='日本海', dir_='W'),  # 逆向き
             passage(7, 'CN', '2020-10-01', to='', dir_=''), passage(8, 'RU', '2020-10-01'),  # 向き不明
             passage(9, 'CNRU', '2020-11-01', title='（中露共同航行／宗谷海峡西進）'),
             passage(10, 'CNRU', '2020-11-20', title='中国及びロシア海軍艦艇の動向について')]
        R = proto2.build_joint(jso(P))
        self.assertEqual([(c['cn_date'], c['ru_date']) for c in R['candidates']], [('2020-07-01', '2020-07-03')])
        self.assertEqual(len(R['unknown_direction']), 1)
        self.assertEqual([x['date'] for x in R['stated']], ['2020-11-01'])
        self.assertEqual([x['date'] for x in R['together']], ['2020-11-20'])
        self.assertEqual(R['placebo']['observed'], 1)


class IW(unittest.TestCase):
    def test_nulls_are_not_counted(self):
        start = date(2022, 11, 1)
        J = jso([], first='2021-01-01', last='2023-12-31')
        M = {'daily': {}}
        for i in range(-400, 30):
            x = start + timedelta(days=i)
            v = 50 if i == 10 else (i % 5)
            M['daily'][x] = {'aircraft': v, 'median_line_or_adiz': None, 'plan_vessels': v, 'official_ships': None}
        G = {'daily': {}}
        W = proto2.build_iw(J, G, M, start=start)
        tw = next(s for s in W['scenarios'] if s['key'] == 'taiwan')
        self.assertEqual(tw['above'][10], 2)  # 軍機と艦艇
        # 中間線越え・公務船は null なので比べられない。統幕の南西諸島は 0 件だが値はある
        self.assertEqual(tw['available'][10], 3)
        self.assertIsNone(W['series']['mnd_median']['p'][10])
        self.assertIsNone(W['series']['jcg_cz']['p'][10])


class Grayzone(unittest.TestCase):
    def test_empty_jcg_and_nulls(self):
        G = {'daily': {}, 'status': {}, 'available': False}
        M = {'daily': {date(2024, 1, 1): {'official_ships': None}, date(2024, 1, 2): {'official_ships': 3}}}
        Z = proto2.build_grayzone(G, M)
        self.assertFalse(Z['jcg_available'])
        self.assertEqual(Z['days'], [['2024-01-01', None, None, None, 1], ['2024-01-02', None, None, 3, 1]])
        self.assertEqual(Z['jcg_unverified_months'], [])

    def test_unverified_months(self):
        G = {'daily': {date(2014, 2, 1): {'cz': 3, 'ts': None, 'check': 'no_totals'},
                       date(2014, 2, 3): {'cz': None, 'ts': None, 'check': 'no_totals'},
                       date(2014, 3, 1): {'cz': 3, 'ts': 0, 'check': 'match'}}, 'status': {}, 'available': True}
        Z = proto2.build_grayzone(G, {'daily': {}})
        self.assertEqual(Z['jcg_unverified_months'], ['2014-02'])


class RealData(unittest.TestCase):
    def test_main_writes_all(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(proto, 'OUT', tmp):
            self.assertEqual(proto2.main([]), 0)
            self.assertEqual(sorted(os.listdir(tmp)), ['grayzone.json', 'heatmap.json', 'iw.json', 'joint.json', 'stay.json'])


if __name__ == '__main__':
    unittest.main()
