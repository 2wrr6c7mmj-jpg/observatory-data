import json
import os
import tempfile
import unittest
from unittest import mock

from translate import headlines as H
from translate.headlines import pending_titles, valid

NOW = 1_800_000_000_000  # ms
DAY = 86400000


class PendingTitles(unittest.TestCase):
    def test_selects_untranslated_newest_first(self):
        recs = [
            {'title': 'Old one', 'source': 'usni', 'publishedAt': 1},
            {'title': 'New one', 'source': 'usni', 'publishedAt': 5},
            {'title': 'New one', 'source': 'twz', 'publishedAt': 4},
            {'title': 'Has ja', 'source': 'usni', 'titleJa': '訳あり', 'publishedAt': 3},
            {'title': '防衛省の発表', 'source': 'x', 'publishedAt': 2},
            {'title': 'NHK item', 'source': 'nhk', 'publishedAt': 2},
            {'title': '解放军海军舰艇编队', 'source': 'ecns', 'publishedAt': 0},
        ]
        self.assertEqual(pending_titles(recs), ['New one', 'Old one', '解放军海军舰艇编队'])

    def test_east_asia_and_military_first(self):
        recs = [
            {'title': 'World news', 'source': 'bbc', 'publishedAt': 9, 'category': 'diplomacy'},
            {'title': 'Drone strike', 'source': 'aj', 'publishedAt': 5, 'category': 'conflict'},
            {'title': '國防部記者會', 'source': 'cna', 'publishedAt': 1, 'scope': 'TW', 'category': 'military'},
        ]
        self.assertEqual(pending_titles(recs), ['國防部記者會', 'Drone strike', 'World news'])

    def test_valid(self):
        self.assertTrue(valid('PLA Navy drills', '中国海軍が演習'))
        self.assertFalse(valid('PLA Navy drills', 'PLA Navy drills'))
        self.assertFalse(valid('abc', ''))
        self.assertFalse(valid('解放军', '解放军'))


def answer(pairs):
    """The model's message content for [(n, ja), ...]."""
    return json.dumps({'items': [{'n': n, 'ja': ja} for n, ja in pairs]}, ensure_ascii=False)


class FakeModel:
    """Stands in for the model server: answers each numbered title from ``table``."""

    def __init__(self, table, reverse=False, fail=False):
        self.table, self.reverse, self.fail, self.calls = table, reverse, fail, 0

    def __call__(self, server, body):
        self.calls += 1
        if self.fail:
            raise OSError('connection refused')
        text = body['messages'][1]['content']
        numbered = json.loads(text[text.index('\n') + 1:])
        pairs = [(x['n'], self.table[x['title']]) for x in numbered]
        return answer(pairs[::-1] if self.reverse else pairs)


class MatchBatch(unittest.TestCase):
    TITLES = ['PLA Navy drills', 'Coast Guard patrol', 'Pentagon budget']

    def test_taken_by_number_not_by_position(self):
        got = H.match_batch(self.TITLES, answer([(3, '米国防総省の予算'), (1, '中国海軍が演習'), (2, '海警が巡視')]))
        self.assertEqual(got, ['中国海軍が演習', '海警が巡視', '米国防総省の予算'])

    def test_missing_repeated_and_out_of_range_numbers_are_not_used(self):
        got = H.match_batch(self.TITLES, answer([(1, '中国海軍が演習'), (1, '海警が巡視'), (4, '米国防総省の予算'), (0, 'x')]))
        self.assertEqual(got, [None, None, None])
        got = H.match_batch(self.TITLES, answer([(2, '海警が巡視'), ('3', '米国防総省の予算')]))
        self.assertEqual(got, [None, '海警が巡視', '米国防総省の予算'])

    def test_number_of_another_headline_means_a_swap(self):
        titles = ['25th Infantry Division arrives', 'Two J-16 fighters', 'Type 055 destroyer sails']
        got = H.match_batch(titles, answer([(1, '055型駆逐艦が出航'), (2, 'J-16戦闘機2機'), (3, '第25歩兵師団が到着')]))
        self.assertEqual(got, [None, 'J-16戦闘機2機', None])
        # numbers of the headline itself (also full-width, with separators) are fine
        got = H.match_batch(['10,000 troops', 'F-35 crash'], answer([(1, '兵１００００人'), (2, 'F-35が墜落')]))
        self.assertEqual(got, ['兵１００００人', 'F-35が墜落'])

    def test_unusable_answer(self):
        for content in ('not json', '[]', '{"ja": ["a", "b", "c"]}', '{"items": "x"}', None):
            self.assertIsNone(H.match_batch(self.TITLES, content), content)


class Main(unittest.TestCase):
    TABLE = {'PLA Navy drills': '中国海軍が演習', 'Coast Guard patrol': '海警が巡視', 'Pentagon budget': '米国防総省の予算'}

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.out = tmp.name
        self.cache_path = os.path.join(self.out, 'cache.json')
        self.pub_path = os.path.join(self.out, 'headlines-ja.json')

    def feed(self, titles):
        return {'records': [{'title': t, 'source': 'usni', 'publishedAt': 100 - i} for i, t in enumerate(titles)]}

    def run_main(self, feed, model, now=NOW):
        with mock.patch.object(H, 'get_json', return_value=feed), mock.patch.object(H, 'chat', model), \
                mock.patch.object(H.time, 'time', return_value=now / 1000), mock.patch('builtins.print'):
            return H.main(['--out', self.out, '--model', 'm'])

    def read(self, path):
        with open(path, encoding='utf-8') as f:
            return json.load(f)

    def files(self):
        out = []
        for path in (self.cache_path, self.pub_path):
            with open(path, 'rb') as f:
                out.append(f.read())
        return tuple(out)

    def write_existing(self):
        H.write(self.cache_path, {'items': {
            H.key('Pentagon budget'): {'o': 'Pentagon budget', 't': '米国防総省の予算', 'seen': NOW - 5 * DAY},
            H.key('Old seen'): {'o': 'Old seen', 't': '古い訳', 'seen': NOW - 29 * DAY},
            H.key('Expired'): {'o': 'Expired', 't': '期限切れ', 'seen': NOW - 31 * DAY},
        }})
        H.write(self.pub_path, {'updatedAt': 1, 'model': 'm', 'pending': 1,
                                'items': [{'original': 'Pentagon budget', 'translated': '米国防総省の予算'}]})
        return self.files()

    def test_translates_publishes_and_prunes_the_cache(self):
        self.write_existing()
        model = FakeModel(self.TABLE, reverse=True)  # answers out of order: matched by number
        self.assertEqual(self.run_main(self.feed(['PLA Navy drills', 'Coast Guard patrol', 'Pentagon budget']), model), 0)
        self.assertEqual(model.calls, 1)  # only the two titles not in the cache, in one batch
        cache = self.read(self.cache_path)['items']
        self.assertNotIn(H.key('Expired'), cache)  # unseen for more than 30 days
        self.assertEqual(cache[H.key('Old seen')]['seen'], NOW - 29 * DAY)  # kept, not refreshed
        self.assertEqual(cache[H.key('Pentagon budget')]['seen'], NOW)  # still pending: refreshed
        self.assertEqual(cache[H.key('Coast Guard patrol')], {'o': 'Coast Guard patrol', 't': '海警が巡視', 'seen': NOW})
        pub = self.read(self.pub_path)
        self.assertEqual((pub['updatedAt'], pub['model'], pub['pending']), (NOW, 'm', 3))
        # only the titles the site still lacks, in the feed's order (not "Old seen")
        self.assertEqual(pub['items'], [{'original': t, 'translated': self.TABLE[t]}
                                        for t in ['PLA Navy drills', 'Coast Guard patrol', 'Pentagon budget']])
        self.assertEqual(sorted(os.listdir(self.out)), ['cache.json', 'headlines-ja.json'])  # no .tmp left

    def test_published_items_are_capped(self):
        titles = [f'Headline number {i}' for i in range(H.PUBLISH_MAX + 5)]
        H.write(self.cache_path, {'items': {H.key(t): {'o': t, 't': f'見出し{i}', 'seen': NOW} for i, t in enumerate(titles)}})
        self.assertEqual(self.run_main(self.feed(titles), FakeModel({})), 0)
        self.assertEqual(len(self.read(self.pub_path)['items']), H.PUBLISH_MAX)

    def test_feed_without_records_keeps_the_published_files(self):
        before = self.write_existing()
        for feed in ({}, {'records': []}, {'records': None}, {'records': ['x']}, [], {'error': 'down'}):
            with self.subTest(feed=feed):
                model = FakeModel(self.TABLE)
                self.assertNotEqual(self.run_main(feed, model), 0)
                self.assertEqual(model.calls, 0)
                self.assertEqual(self.files(), before)

    def test_unreadable_feed_keeps_the_published_files(self):
        before = self.write_existing()
        with mock.patch.object(H, 'get_json', side_effect=ValueError('not JSON')), mock.patch('builtins.print'):
            self.assertNotEqual(H.main(['--out', self.out]), 0)
        self.assertEqual(self.files(), before)

    def test_all_batches_failing_keeps_the_published_files(self):
        before = self.write_existing()
        titles = [f'Headline {chr(65 + i)}' for i in range(H.BATCH + 3)]  # two batches
        for model in (FakeModel({}, fail=True), lambda server, body: 'not json',
                      lambda server, body: answer([(1, 'not japanese')])):
            with self.subTest(model=model):
                self.assertEqual(self.run_main(self.feed(titles), model), 1)
                self.assertEqual(self.files(), before)

    def test_one_failed_batch_does_not_stop_the_rest(self):
        titles = [f'Headline {chr(65 + i)}' for i in range(H.BATCH + 2)]
        table = {t: f'見出し{t[-1]}' for t in titles}
        model = FakeModel(table)
        calls = []

        def flaky(server, body):
            calls.append(1)
            if len(calls) == 1:
                raise TimeoutError('timed out')
            return model(server, body)
        self.assertEqual(self.run_main(self.feed(titles), flaky), 0)
        pub = self.read(self.pub_path)
        self.assertEqual([x['original'] for x in pub['items']], titles[H.BATCH:])

    def test_swapped_answers_are_not_cached(self):
        titles = ['25th Infantry Division arrives', 'Type 055 destroyer sails']
        model = lambda server, body: answer([(1, '055型駆逐艦が出航'), (2, '第25歩兵師団が到着')])
        self.assertEqual(self.run_main(self.feed(titles), model), 1)  # nothing usable: nothing written
        self.assertFalse(os.path.exists(self.cache_path))
        titles.append('Coast Guard patrol')
        model = lambda server, body: answer([(1, '055型駆逐艦が出航'), (2, '第25歩兵師団が到着'), (3, '海警が巡視')])
        self.assertEqual(self.run_main(self.feed(titles), model), 0)
        self.assertEqual([v['o'] for v in self.read(self.cache_path)['items'].values()], ['Coast Guard patrol'])


if __name__ == '__main__':
    unittest.main()
