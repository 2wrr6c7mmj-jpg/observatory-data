import io
import json
import os
import re
import tempfile
import unittest
import wave
from unittest import mock

from briefing import audio as A
from briefing import build as B
from briefing import script as T
from briefing import editor as E

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


class Twins(unittest.TestCase):
    TOK = staticmethod(lambda h: set(h.split()))

    def ev(self, cat, countries, place, *heads):
        return {'meta': {'category': cat, 'countries': countries, 'place': place and {'name': place}}, 'heads': list(heads)}

    def test_same_event(self):
        a = self.ev('military', ['RU', 'KR', 'JP'], None, '韓国 ロシア 軍機 防空識別圏 警報')
        b = self.ev('military', ['RU', 'KR'], None, 'ロシア 軍機 韓国 防空識別圏 進入')
        self.assertTrue(B.same_event(a, b, self.TOK))
        # 国の組が同じでも、地点が違えば別の出来事
        c = self.ev('conflict', ['UA', 'RU', 'LY'], None, 'ウクライナ リビア沖 ロシア 軍艦 撃沈')
        d = self.ev('conflict', ['UA', 'RU'], 'ザポリージャ', 'ロシア ウクライナ ザポリージャ 攻撃')
        self.assertFalse(B.same_event(c, d, self.TOK))
        # 分類が違えば別（封鎖への対抗 と 制裁）
        e = self.ev('conflict', ['IR', 'US'], None, 'イラン 米国 封鎖 対抗')
        f = self.ev('diplomacy', ['IR', 'US'], None, '米国 イラン 制裁 発表')
        self.assertFalse(B.same_event(e, f, self.TOK))
        # 国と分類が同じでも、見出しの語が重ならなければ別
        g = self.ev('diplomacy', ['CN', 'US'], None, '首脳 会談 開催 合意')
        h = self.ev('diplomacy', ['CN', 'US'], None, '関税 引き上げ 発表 半導体')
        self.assertFalse(B.same_event(g, h, self.TOK))

    def test_place_names_become_fixed_translations(self):
        terms = T.glossary_for([{'title': 'Russian strike on Zaporizhzhia kills two', 'summary': '', 'titleJa': ''}], B.place_terms())
        self.assertIn('ザポリージャ', [t['ja'] for t in terms])
        self.assertTrue(all(len(s) > 3 or not s.isascii() for t in B.place_terms() for s in t['src']))


class Twins2(unittest.TestCase):
    def test_country_names_do_not_count_as_overlap(self):
        tok = lambda h: set(h.split())
        ev = lambda countries, *heads: {'meta': {'category': 'conflict', 'countries': countries, 'place': None}, 'heads': list(heads)}
        a = ev(['UA', 'RU', 'LY'], 'ウクライナ リビア沖 ロシア 軍艦 撃沈 ドローン')
        b = ev(['RU', 'UA', 'PL'], 'ロシア ドローン ポーランド 空域 侵入 ウクライナ 巡航ミサイル')
        self.assertFalse(B.same_event(a, b, tok))


class Twins3(unittest.TestCase):
    def test_same_script(self):
        tok = lambda h: set(h.replace('、', ' ').replace('。', ' ').split())
        ev = lambda cat, countries, head, *lines: {'meta': {'category': cat, 'countries': countries, 'place': None},
                                                   'script': {'headline': head, 'lines': [{'text': t} for t in lines]}}
        a = ev('military', ['RU', 'KR', 'JP'], '韓国 ロシア 軍機 接近 空襲警報', '韓国軍 は ロシア 軍機 が 防空識別圏 に 進入 した と 発表 しました。')
        b = ev('military', ['KR', 'RU', 'US'], 'ロシア 軍機 が 韓国 の 防空識別圏 に 進入', '韓国 は ロシア 軍機 が 防空識別圏 に 進入 した と 発表 しました。')
        self.assertTrue(B.same_script(a, b, tok))
        c = ev('military', ['RU', 'KR'], 'ロシア と 韓国 が 漁業 協定 を 協議', '両国 は 漁業 協定 の 改定 を 協議 する と 発表 しました。')
        self.assertFalse(B.same_script(a, c, tok))
        d = ev('conflict', ['KR', 'RU'], 'ロシア 軍機 が 韓国 の 防空識別圏 に 進入', '韓国 は ロシア 軍機 が 防空識別圏 に 進入 した と 発表 しました。')
        self.assertFalse(B.same_script(a, d, tok))   # 分類が違う


class Editor(unittest.TestCase):
    """台本を人（毎朝の Claude）が書く経路：候補の出力、確かめ、人の台本での公開。"""
    LINES = ['防衛省は、中国海軍の空母遼寧が宮古海峡を通過したと発表しました。', 'USNIとBBCも、同じ通過を報じています。']

    def setUp(self):
        self.d = tempfile.mkdtemp()
        doc = feed()
        for r in doc['records']:
            r['id'] = (r['id'].encode().hex() + '0' * 24)[:24]   # 本物の id のように長くする（key は頭 12 字）
        self.feed = os.path.join(self.d, 'feed.json')
        with open(self.feed, 'w') as f:
            json.dump(doc, f)

    def script(self, lines=None, key=('6d3000000000', '6d3100000000')):
        p = os.path.join(self.d, 'script.json')
        with open(p, 'w', encoding='utf-8') as f:
            json.dump({'version': 1, 'events': [{'key': list(key), 'headline': '遼寧が宮古海峡を通過', 'lines': lines or self.LINES}]}, f, ensure_ascii=False)
        return p

    def test_candidates_file(self):
        out = os.path.join(self.d, 'cands.json')
        self.assertEqual(E.main(['candidates', '--feed', self.feed, '--out', out, '--now-ms', str(NOW)]), 0)
        with open(out, encoding='utf-8') as f:
            d = json.load(f)
        self.assertEqual(len(d['candidates']), 1)
        c = d['candidates'][0]
        self.assertEqual((c['sources']['count'], c['place'], c['needAttribution']), (3, '宮古海峡', False))
        self.assertTrue(c['key'] and all(len(k) == E.KEY_LEN for k in c['key']))
        self.assertEqual(len(c['articles']), 3)
        self.assertIn('bodyChars', d['limits'])

    def test_check_passes_and_fails(self):
        run = lambda path: E.main(['check', '--feed', self.feed, '--script', path, '--now-ms', str(NOW)])
        with mock.patch.object(B, 'MIN_EVENTS', 1):
            self.assertEqual(run(self.script()), 0)
            self.assertEqual(run(self.script(lines=['防衛省は、中国海軍の空母遼寧が宮古海峡を通過したと発表した。'])), 1)   # だ・である調
            self.assertEqual(run(self.script(key=('ffffffffffff',))), 1)                                              # 記事が無い

    def test_publish_with_a_written_script(self):
        out = os.path.join(self.d, 'out')
        with mock.patch.object(B, 'MIN_EVENTS', 1):
            rc = B.main(['--feed', self.feed, '--script', self.script(), '--out', out, '--now-ms', str(NOW), '--no-audio'])
        with open(os.path.join(out, 'latest.json'), encoding='utf-8') as f:
            doc = json.load(f)
        self.assertEqual((rc, doc['model'], doc['hold'], len(doc['events'])), (0, 'Claude', False, 1))
        e = doc['events'][0]
        self.assertEqual((e['sources']['count'], e['place']['name'], e['category']), (3, '宮古海峡', 'military'))   # 記事から取り直した値
        self.assertEqual([ln['text'] for ln in e['lines']], self.LINES)
        self.assertEqual([ln['sub'] for ln in e['lines']], ['', ''])
        with open(os.path.join(out, 'credits.txt'), encoding='utf-8') as f:
            self.assertIn('Claude', f.read())

    def test_brackets_are_not_read_aloud(self):
        self.assertEqual(A.spoken('宇宙船「神舟24号」の乗組員'), '宇宙船神舟24号の乗組員')


class TuningAndMap(unittest.TestCase):
    """その日の読みの指定（調声）と、地図の行き先の指定（アニメーション）。"""
    def test_extra_readings_keep_only_well_formed_words(self):
        doc = {'readings': [['神舟', 'シンシュウ', 0], ['天宮', 'てんきゅう', 0], ['x' * 30, 'ア', 0], ['酒泉', 'シュセン', 9], ['鬱陵島', 'ウルルンド', 3],
                            'bad', ['a', 'ア'], ['危険<b>', 'キケン', 0]]}
        self.assertEqual(B.extra_readings(doc), [('神舟', 'シンシュウ', 0), ('鬱陵島', 'ウルルンド', 3)])
        self.assertEqual(B.extra_readings({}), [])
        self.assertEqual(len(B.extra_readings({'readings': [['語%d' % i, 'ゴ', 0] for i in range(60)]})), 40)

    def test_map_hint(self):
        ev = {'map': {'countries': ['ly', 'UA', 'RU', 'DE', 'x1', 5], 'name': 'リビア',
                      'place': {'name': 'ベンガジ', 'lon': 20.0712, 'lat': 32.1234}}}
        self.assertEqual(B.map_hint(ev), {'countries': ['LY', 'UA', 'RU'], 'name': 'リビア', 'place': {'name': 'ベンガジ', 'lon': 20.071, 'lat': 32.123}})
        self.assertEqual(B.map_hint({'map': {'place': {'name': 'どこか', 'lon': 200, 'lat': 10}, 'name': 'https://x'}}), {})
        self.assertEqual(B.map_hint({'map': 'LY'}), {})
        self.assertEqual(B.map_hint({}), {})

    def test_map_hint_overrides_the_machine_values(self):
        meta = {'category': 'conflict', 'label': '紛争 · ウクライナ', 'country': 'UA', 'countries': ['UA', 'RU'], 'place': None}
        out = B.apply_map_hint(dict(meta), {'countries': ['LY', 'UA'], 'name': 'リビア', 'place': {'name': 'ベンガジ', 'lon': 20.07, 'lat': 32.12}})
        self.assertEqual((out['country'], out['countries'], out['label'], out['place']['name']), ('LY', ['LY', 'UA'], '紛争 · リビア', 'ベンガジ'))
        self.assertEqual(B.apply_map_hint(dict(meta), {}), meta)

    def test_written_script_carries_the_map(self):
        d = tempfile.mkdtemp()
        doc = feed()
        for r in doc['records']:
            r['id'] = (r['id'].encode().hex() + '0' * 24)[:24]
        p, sp, out = os.path.join(d, 'feed.json'), os.path.join(d, 'script.json'), os.path.join(d, 'out')
        with open(p, 'w') as f:
            json.dump(doc, f)
        with open(sp, 'w', encoding='utf-8') as f:
            json.dump({'version': 1, 'readings': [['遼寧', 'リョーニン', 0]], 'events': [{
                'key': ['6d3000000000'], 'headline': '遼寧が宮古海峡を通過', 'lines': Editor.LINES,
                'map': {'countries': ['JP', 'CN'], 'name': '沖縄', 'place': {'name': '宮古島', 'lon': 125.3, 'lat': 24.8}}}]}, f, ensure_ascii=False)
        with mock.patch.object(B, 'MIN_EVENTS', 1):
            B.main(['--feed', p, '--script', sp, '--out', out, '--now-ms', str(NOW), '--no-audio'])
        with open(os.path.join(out, 'latest.json'), encoding='utf-8') as f:
            e = json.load(f)['events'][0]
        self.assertEqual((e['label'], e['countries'], e['place']), ('軍事 · 沖縄', ['JP', 'CN'], {'name': '宮古島', 'lon': 125.3, 'lat': 24.8}))


class Intonation(unittest.TestCase):
    """調声のカナ（アクセントの下がり目・句の区切り）と、声の設定。"""
    AUTO = "ウクラ'イナワ、カモツ'センオ/ゲ_キチン'/_シタ'ト/ハッピョオ'/シマ'_シタ"
    FIXED = "ウクラ'イナワ、カモツセンオ'/ゲ_キチン'/_シタ'ト/ハッピョオ'/シマ'_シタ"

    def test_bare_reading_is_what_must_not_change(self):
        self.assertEqual(A.bare(self.AUTO), A.bare(self.FIXED))
        self.assertEqual(A.bare("ニホ'ンカイ/ジョオクウノ'"), 'ニホンカイジョオクウノ')
        self.assertNotEqual(A.bare(self.AUTO), A.bare(self.FIXED.replace('カモツセン', 'グンカン')))

    def test_script_line(self):
        self.assertEqual(B.script_line(' 文です。 '), {'text': '文です。', 'sub': ''})
        self.assertEqual(B.script_line({'text': '文です。', 'kana': self.FIXED}), {'text': '文です。', 'sub': '', 'kana': self.FIXED})
        self.assertEqual(B.script_line({'text': '文です。', 'kana': 'かな <b>'}), {'text': '文です。', 'sub': ''})   # 記法に無い字は捨てる
        self.assertIsNone(B.script_line({'kana': self.FIXED}))
        self.assertIsNone(B.script_line(5))

    def test_parse_tune(self):
        self.assertEqual(B.parse_tune('speed=1.08,intonation=0.9,volume=3,pitch=x'), {'speed': 1.08, 'intonation': 0.9})
        self.assertEqual(B.parse_tune(''), {})

    def test_voice_file_tune_is_in_range(self):
        tune = A.load_voice().get('tune') or {}
        for k, v in tune.items():
            self.assertIn(k, A.TUNE)
            self.assertTrue(A.TUNE_RANGE[k][0] <= v <= A.TUNE_RANGE[k][1], (k, v))


def phrase(moras, accent, pause=False, ask=False):
    """試験用のアクセント句。moras = 'シ5.75 マ5.79 シ0 タ5.57'（0 は声の出ない拍）。"""
    ms = [mock.Mock(text=re.match(r'\D+', x).group(), pitch=float(re.search(r'[\d.]+$', x).group())) for x in moras.split()]
    return mock.Mock(moras=ms, accent=accent, pause_mora=object() if pause else None, is_interrogative=ask)


def pitches(ph):
    return [round(m.pitch, 2) for m in ph.moras]


class Ending(unittest.TestCase):
    """文末の落ち着かせ：文末の句の山と最後の拍を、どの文も同じ高さへ下げる（上げない）。"""
    CONF = {'peak': 0.17, 'open': 0.10, 'end': 0.0}

    def test_high_ending_is_brought_down(self):
        # 〜と発表しました。：文末の句が高く持ち上がったまま、最後の「た」が下がりきらない
        ps = [phrase('シ0 タ5.88 ト5.64', 2), phrase('ハ5.38 ッ0 ピョ5.72 オ5.74', 4), phrase('シ5.75 マ5.79 シ0 タ5.57', 2)]
        down, drop = A.settle_ending(ps, base=5.47, **self.CONF)
        self.assertAlmostEqual(down, 0.15, 2)
        self.assertEqual(pitches(ps[0]), [0, 5.88, 5.64])                 # 前の前の句は動かない
        self.assertEqual(pitches(ps[2]), [5.62, 5.64, 0, 5.47])           # 山は 低いところ+0.17、最後の拍は 低いところ
        self.assertEqual(pitches(ps[1]), [5.35, 0, 5.64, 5.64])           # 前の句は山へ向かって少しずつ（頭はほとんど動かない）

    def test_open_ending_gets_the_lower_limit(self):
        # 〜に搭乗します。：「す」は声が出ないので、核の「ま」で終わる。上がったまま終わらせない
        ps = [phrase('ト5.46 オ5.56 ジョ5.62 オ5.68', 4), phrase('シ5.69 マ5.73 ス0', 2)]
        A.settle_ending(ps, base=5.47, **self.CONF)
        self.assertEqual(pitches(ps[1]), [5.56, 5.57, 0])
        self.assertLess(ps[0].moras[0].pitch, 5.46)

    def test_low_ending_is_left_and_nothing_goes_up(self):
        ps = [phrase('コ5.72 オ5.80 ギ5.65', 1), phrase('シ5.39 マ5.53 シ0 タ5.50', 2)]
        down, drop = A.settle_ending(ps, base=5.47, **self.CONF)
        self.assertEqual((down, round(drop, 2)), (0.0, 0.03))
        self.assertEqual(pitches(ps[0]) + pitches(ps[1]), [5.72, 5.80, 5.65, 5.39, 5.53, 0, 5.47])
        ps = [phrase('シ0 テ5.40', 2), phrase('イ5.41 マ5.44 ス0', 2)]      # もう低い文末：触らない
        self.assertEqual(A.settle_ending(ps, base=5.47, **self.CONF), (0.0, 0.0))
        self.assertEqual(pitches(ps[1]), [5.41, 5.44, 0])

    def test_last_mora_does_not_sink_below_the_floor(self):
        # 山を下げたぶんで「た」が低いところを下回るなら、そこで止める
        ps = [phrase('ア5.50 ゲ5.70', 2), phrase('マ5.90 シ0 タ5.50', 1)]
        A.settle_ending(ps, base=5.47, **self.CONF)
        self.assertEqual(pitches(ps[1]), [5.64, 0, 5.47])

    def test_pause_question_and_short(self):
        ps = [phrase('ハ5.40 ッ0 ピョ5.72 オ5.74', 4, pause=True), phrase('シ5.75 マ5.79 シ0 タ5.57', 2)]
        A.settle_ending(ps, base=5.47, **self.CONF)
        self.assertEqual(pitches(ps[0]), [5.40, 0, 5.72, 5.74])           # 間（読点）の前の句は動かさない
        ps = [phrase('イ5.50 キ5.80 マ5.85 ス5.90 カ5.95', 3, ask=True)]
        self.assertEqual(A.settle_ending(ps, base=5.47, **self.CONF), (0.0, 0.0))   # 疑問の上がりは残す
        self.assertEqual(A.settle_ending([phrase('ハ5.80 イ5.90', 1)], **self.CONF), (0.0, 0.0))   # 拍が少なく基準が無い
        self.assertEqual(A.settle_ending([], base=5.47, **self.CONF), (0.0, 0.0))

    def test_base_from_the_sentence_itself(self):
        self.assertEqual(A.low_pitch([0, 5.9, 5.3, 5.5, 5.7, 0, 5.6, 5.8, 5.4, 5.45, 5.65, 5.75]), 5.4)
        self.assertIsNone(A.low_pitch([0, 5.5, 5.6]))

    def test_conf(self):
        self.assertEqual(A.ending_of(self.CONF), self.CONF)
        for bad in (None, {}, {'peak': 0.17, 'open': 0.10}, {'peak': 0.17, 'open': 0.10, 'end': 2}, {'peak': True, 'open': 0.1, 'end': 0}):
            self.assertIsNone(A.ending_of(bad))
        self.assertEqual(A.ending_of(A.load_voice().get('ending')), A.load_voice()['ending'])   # voice.json の値は範囲の中

    def test_site_name_is_read_in_two_phrases_with_the_same_reading(self):
        self.assertEqual(A.bare(B.NAME_KANA[0]), A.bare(B.NAME_KANA[1]))
        self.assertIn('/', B.NAME_KANA[1])

    def test_intro_and_outro_are_spoken_sentence_by_sentence(self):
        self.assertEqual(B.spread('outro', '以上です。機械で読み上げました。', 0.5),
                         [('outro', '以上です。', A.SAMPLE_PAUSE['line']), ('outro+1', '機械で読み上げました。', 0.5)])
        self.assertEqual(B.spread('intro', '一文だけ', 0.8), [('intro', '一文だけ', 0.8)])


if __name__ == '__main__':
    unittest.main()
