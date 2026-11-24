import unittest

from briefing import cluster as C
from briefing import places as P
from briefing import sources as S

NOW = 1_800_000_000_000
HOUR = 3600 * 1000


def rec(i, source, title, **kw):
    r = {'id': f'id{i}', 'source': source, 'titleJa': title, 'title': title, 'url': f'https://example.com/{i}',
         'publishedAt': NOW - HOUR, 'category': 'military'}
    r.update(kw)
    return r


def filler(n):
    # 互いに語が重ならない見出し（IDF の分母を大きくする）
    kanji = '山川林田島森湖橋港空雲雪花鳥魚虫石砂'
    return [rec(1000 + i, f'src{i}', f'{kanji[i % 18]}{kanji[(i * 7 + 3) % 18]}{kanji[(i * 5 + 1) % 18]}の話題{i}号') for i in range(n)]


class SourcesTable(unittest.TestCase):
    def test_every_site_source_has_family_and_country(self):
        missing = [k for k in S.META if k not in S.SOURCES]
        self.assertEqual(missing, [])

    def test_state_media_are_one_family(self):
        self.assertEqual(S.family('tass'), S.family('rt'))
        self.assertEqual(S.family('cgtn'), S.family('ecns'))
        self.assertEqual(S.family('navy'), S.family('dvids'))
        self.assertNotEqual(S.family('bbc'), S.family('aj'))
        self.assertEqual(S.family('bbc-china'), S.family('bbc'))

    def test_unknown_source_is_kept_apart_and_not_counted_as_a_country(self):
        self.assertEqual(S.family('zzz'), 'zzz')
        self.assertIsNone(S.country('zzz'))
        self.assertEqual(S.kind('zzz'), '不明')

    def test_kinds_from_the_site(self):
        self.assertTrue(S.is_primary_or_state('tass'))
        self.assertTrue(S.is_primary_or_state('dvids'))
        self.assertFalse(S.is_primary_or_state('bbc'))

    def test_country_codes_are_two_letters(self):
        for k, (_, c) in S.SOURCES.items():
            self.assertRegex(c, r'^[A-Z]{2}$', k)


class Window(unittest.TestCase):
    def test_window_and_duplicates(self):
        rs = [rec(1, 'a', 'x'), rec(2, 'a', 'old', publishedAt=NOW - 25 * HOUR), rec(3, 'a', 'far future', publishedAt=NOW + 20 * HOUR),
              rec(4, 'a', 'no time', publishedAt=None), rec(5, 'a', 'dup', url='https://example.com/1'), 'junk']
        self.assertEqual([r['id'] for r in C.in_window(rs, NOW)], ['id1'])


class Clustering(unittest.TestCase):
    def setUp(self):
        self.tok = C.tokens_fallback
        self.recs = filler(40) + [
            rec(1, 'usni', '中国海軍の空母遼寧が宮古海峡を通過'),
            rec(2, 'kyodo', '空母遼寧が宮古海峡を通過 防衛省発表'),
            rec(3, 'tass', '中国海軍の空母遼寧、宮古海峡を通過'),
            rec(4, 'rt', '空母遼寧が宮古海峡通過、中国海軍'),
            rec(5, 'bbc', '台湾で新しい議員が就任の宣誓'),
        ]

    def test_cluster_and_independent_sources(self):
        cl = C.cluster(self.recs, self.tok)
        big = max(cl, key=len)
        self.assertEqual({r['id'] for r in big}, {'id1', 'id2', 'id3', 'id4'})
        d = C.describe(big)
        # tass と rt は 1 系列
        self.assertEqual(sorted(d['families']), ['kyodo', 'ru-state', 'usni'])
        self.assertEqual(d['countries'], ['JP', 'RU', 'US'])
        self.assertEqual(d['size'], 4)

    def test_select_needs_two_families_and_ranks(self):
        cands = C.select(C.cluster(self.recs, self.tok))
        self.assertEqual(len(cands), 1)
        self.assertEqual(len(cands[0]['stats']['families']), 3)

    def test_state_media_only_is_not_selected(self):
        recs = filler(40) + [rec(1, 'tass', '中国海軍の空母遼寧が宮古海峡を通過'), rec(2, 'rt', '空母遼寧が宮古海峡を通過 中国海軍')]
        self.assertEqual(C.select(C.cluster(recs, self.tok)), [])

    def test_rank_prefers_more_families_then_countries(self):
        a = {'families': ['a', 'b'], 'countries': ['US'], 'east_asia': False, 'military': False, 'size': 9}
        b = {'families': ['a', 'b', 'c'], 'countries': ['US'], 'east_asia': False, 'military': False, 'size': 3}
        c = {'families': ['a', 'b', 'c'], 'countries': ['US', 'JP'], 'east_asia': False, 'military': False, 'size': 3}
        self.assertEqual(sorted([a, b, c], key=C.rank_key), [c, b, a])

    def test_representatives_one_per_family_independent_first(self):
        recs = [rec(1, 'tass', 'a'), rec(2, 'rt', 'b', publishedAt=NOW), rec(3, 'bbc', 'c', publishedAt=NOW - 5 * HOUR)]
        reps = C.representatives(recs)
        self.assertEqual([r['source'] for r in reps], ['bbc', 'rt'])

    def test_fallback_tokens(self):
        self.assertEqual(C.tokens_fallback('中国海軍のPLAN 24隻'), {'中国海軍', 'plan', '24'})


class Places(unittest.TestCase):
    def test_place_is_not_taken_from_capital_and_codes_follow(self):
        hits = P.find(['The Kremlin said Ukraine hit a ship in the Mediterranean'])
        self.assertEqual(P.main_place(hits)['name'], '地中海')
        self.assertEqual(P.country_codes(hits)[:2], ['UA', 'RU'])

    def test_no_place_gives_none(self):
        self.assertIsNone(P.main_place(P.find(['Russia and Ukraine talk'])))

    def test_table_is_lon_lat(self):
        for p in P.table():
            lon, lat = p['lonlat']
            self.assertTrue(-180 <= lon <= 180 and -90 <= lat <= 90, p['id'])
            self.assertIn(p['kind'], ('place', 'country', 'capital'))


if __name__ == '__main__':
    unittest.main()
