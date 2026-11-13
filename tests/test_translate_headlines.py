import unittest
from translate.headlines import pending_titles, valid


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

    def test_valid(self):
        self.assertTrue(valid('PLA Navy drills', '中国海軍が演習'))
        self.assertFalse(valid('PLA Navy drills', 'PLA Navy drills'))
        self.assertFalse(valid('abc', ''))
        self.assertFalse(valid('解放军', '解放军'))


if __name__ == '__main__':
    unittest.main()
