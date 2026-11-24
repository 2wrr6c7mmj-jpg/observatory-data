import io
import json
import os
import tempfile
import unittest
import wave
from unittest import mock

from briefing import audio as A
from briefing import build as B
from briefing import script as T

NOW = 1_800_000_000_000
HOUR = 3600 * 1000


def wav(sec, rate=24000):
    b = io.BytesIO()
    with wave.open(b, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b'\x01\x00' * int(rate * sec))
    return b.getvalue()


class Audio(unittest.TestCase):
    def test_assemble_records_starts(self):
        starts, pcm = A.assemble([('a', 'x', 0.5), ('b', 'yy', 1.0), ('c', 'z', 0)], lambda t: wav(len(t)))
        self.assertEqual(starts, {'a': 0.0, 'b': 1.5, 'c': 4.5})
        self.assertAlmostEqual(pcm.seconds, 5.5, 2)

    def test_rate_change_is_refused(self):
        pcm = A.Pcm()
        pcm.add_wav(wav(0.1))
        with self.assertRaises(ValueError):
            pcm.add_wav(wav(0.1, 16000))

    def test_voice_and_readings_files(self):
        v = A.load_voice()
        self.assertIsInstance(v['style_id'], int)
        self.assertIn('VOICEVOX', v['credit'])
        for surface, pron, accent in A.load_readings():
            self.assertRegex(pron, r'^[ァ-ヶー]+$', surface)
            self.assertIsInstance(accent, int)


def feed(n_fill=40):
    kanji = '山川林田島森湖橋港空雲雪花鳥魚虫石砂'
    recs = [{'id': f'f{i}', 'source': f'src{i}', 'title': 'filler', 'titleJa': f'{kanji[i % 18]}{kanji[(i * 7 + 3 + 2 * (i // 18)) % 18]}{kanji[(i * 5 + 1 + i // 18) % 18]}の話題{i}号',
             'url': f'https://e.com/f{i}', 'publishedAt': NOW - HOUR, 'category': 'diplomacy'} for i in range(n_fill)]
    for i, (s, t) in enumerate([('usni', '中国海軍の空母遼寧が宮古海峡を通過'), ('kyodo', '空母遼寧が宮古海峡を通過 防衛省発表'),
                                ('bbc', '中国海軍の空母遼寧、宮古海峡を通過')]):
        recs.append({'id': f'm{i}', 'source': s, 'title': t, 'titleJa': t, 'summary': 'Liaoning passed the Miyako Strait.',
                     'url': f'https://e.com/m{i}', 'publishedAt': NOW - HOUR, 'category': 'military', 'scope': 'JP'})
    return {'records': recs}


ANSWER = json.dumps({'headline': '遼寧が宮古海峡を通過', 'lines': [
    {'text': '防衛省は、中国海軍の空母遼寧が宮古海峡を通過したと発表しました。', 'sub': '通過'},
    {'text': 'USNIとBBCも、同じ通過を報じています。', 'sub': '報道'}]}, ensure_ascii=False)


class Build(unittest.TestCase):
    def run_main(self, min_events=1, answer=ANSWER):
        d = tempfile.mkdtemp()
        p = os.path.join(d, 'feed.json')
        with open(p, 'w') as f:
            json.dump(feed(), f)
        out = os.path.join(d, 'out')
        with mock.patch.object(T, 'chat', lambda s, m: answer), mock.patch.object(B, 'MIN_EVENTS', min_events):
            rc = B.main(['--feed', p, '--out', out, '--now-ms', str(NOW), '--no-audio'])
        with open(os.path.join(out, 'latest.json'), encoding='utf-8') as f:
            return rc, json.load(f), out

    def test_json_shape(self):
        rc, doc, out = self.run_main()
        self.assertEqual(rc, 0)
        for k in ('version', 'createdAt', 'windowHours', 'hold', 'generated', 'model', 'voice', 'audio', 'counts', 'events', 'dropped', 'intro', 'outro'):
            self.assertIn(k, doc)
        self.assertEqual((doc['version'], doc['windowHours'], doc['generated'], doc['hold']), (1, 24, 'machine', False))
        self.assertEqual(set(doc['voice']) >= {'credit', 'styleId'}, True)
        self.assertEqual(len(doc['events']), 1)
        e = doc['events'][0]
        for k in ('start', 'end', 'category', 'label', 'country', 'countries', 'place', 'headline', 'lines', 'sources'):
            self.assertIn(k, e)
        self.assertEqual(e['category'], 'military')
        self.assertEqual(e['country'], 'JP')
        self.assertEqual(e['place']['name'], '宮古海峡')
        self.assertEqual(e['sources']['count'], 3)
        self.assertEqual(e['sources']['countries'], 3)
        self.assertLessEqual(len(e['sources']['names']), 3)
        self.assertLessEqual(len(e['sources']['items']), 6)
        self.assertTrue(all(i['url'].startswith('https://') for i in e['sources']['items']))
        self.assertLessEqual(len(e['countries']), 3)
        self.assertEqual(len(e['lines']), 2)
        self.assertFalse(any('報道機関' in ln['text'] for ln in e['lines']))   # 出典の数は読まない
        self.assertTrue(all(ln['sub'] == '' or isinstance(ln['sub'], str) for ln in e['lines']))
        self.assertEqual(doc['counts']['records'], 43)
        self.assertTrue(os.path.exists(os.path.join(out, 'credits.txt')))

    def test_too_few_events_holds(self):
        rc, doc, out = self.run_main(min_events=3)
        self.assertTrue(doc['hold'])
        self.assertIn('件', doc['holdReason'])
        self.assertIsNone(doc['audio'])
        self.assertFalse(os.path.exists(os.path.join(out, 'latest.mp3')))

    def test_failed_check_goes_to_dropped(self):
        bad = json.dumps({'headline': 'x', 'lines': [{'text': '中国の挑発的な行動を、日本は強く非難しています。', 'sub': ''},
                                                     {'text': '日本の報道は、艦艇を12隻としています。', 'sub': ''}]}, ensure_ascii=False)
        rc, doc, _ = self.run_main(answer=bad)
        self.assertEqual(doc['events'], [])
        self.assertTrue(doc['hold'])
        self.assertEqual(len(doc['dropped']), 1)
        self.assertIn('評価', doc['dropped'][0]['reason'])

    def test_unsafe_url_is_dropped(self):
        self.assertIsNone(B.safe_url('javascript:alert(1)'))
        self.assertEqual(B.safe_url('https://a.example/x'), 'https://a.example/x')


class Meta(unittest.TestCase):
    def test_category_follows_the_words(self):
        def cat(*titles):
            reps = [{'title': t, 'summary': ''} for t in titles]
            return B.category_of([dict(r, category='diplomacy') for r in reps], reps)
        self.assertEqual(cat('Israeli airstrike kills 3 in Gaza', 'Three killed in strike on Khan Younis', 'Gaza talks'), 'conflict')
        self.assertEqual(cat('Iran vows to counter US naval blockade', 'Navy blockade of Iranian oil'), 'military')
        self.assertEqual(cat('US imposes sanctions on Iran-linked banks', 'Treasury sanctions network'), 'diplomacy')

    def test_trim_to_length_drops_the_lowest_ranked(self):
        ev = lambda n: {'rank': n, 'script': {'headline': 'x', 'lines': [{'text': 'あ' * 55}, {'text': 'い' * 55}]}}
        keep, cut = B.trim_to_length([ev(i) for i in range(1, 9)])
        self.assertLessEqual(B.estimate_seconds(keep), B.TARGET_SECONDS)
        self.assertEqual([e['rank'] for e in keep], list(range(1, len(keep) + 1)))
        self.assertEqual(len(keep) + len(cut), 8)
        self.assertTrue(cut)
        short = [{'rank': n, 'script': {'headline': 'x', 'lines': [{'text': 'あ' * 40}]}} for n in range(1, 9)]
        self.assertEqual(B.trim_to_length(short)[1], [])


if __name__ == '__main__':
    unittest.main()
