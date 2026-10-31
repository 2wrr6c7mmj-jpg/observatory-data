import csv
import io
import json
import os
import tempfile
import unittest

from tools import wen_map

KML = '''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>試験の地図</name>
    <description><![CDATA[歡迎大家取用。<br><br>歡迎來信:<br>https://www.facebook.com/joseph.wen.37/<br>編者：Joseph.W]]></description>
    <Folder>
      <name>解放軍海軍、海軍陸戰隊基地及設施</name>
      <Placemark>
        <name>湛江基地‎</name>
        <description><![CDATA[南海艦隊司令部所在地<br>說明&amp;註 <img src="https://example.invalid/a.png" height="200"/> https://zh.wikipedia.org/wiki/X]]></description>
        <Point><coordinates>
          110.4387291234,21.2370763456,0
        </coordinates></Point>
      </Placemark>
      <Placemark>
        <name>某港 北營</name>
        <description><![CDATA[來源 https://twitter.com/someone/status/1?s=20&t=AbC 聯絡 Email: foo.bar@example.com 電話：0912-345-678 mailto:x@y.org<br>製圖及資料提供者為軍友 ZI Hao Huang(FaceBook用戶名) https://example.org/a?id=5&fbclid=IwAR0x&utm_source=fb https://www.youtube.com/watch?v=abc&t=6s https://public.tableau.com/app/profile/p/viz/V/Dashboard1%3Cbr%3E%3Cbr%3E]]></description>
        <Point><coordinates>121.0,38.0,0</coordinates></Point>
      </Placemark>
      <Placemark>
        <name>線</name>
        <LineString><coordinates>1,1,0 2,2,0</coordinates></LineString>
      </Placemark>
    </Folder>
    <Folder>
      <name>解放軍火箭軍 </name>
      <Folder>
        <name>子フォルダ</name>
        <Placemark><name>某旅</name><Point><coordinates>100.1,30.2</coordinates></Point></Placemark>
      </Folder>
    </Folder>
  </Document>
</kml>
'''


def parse():
    return wen_map.parse_kml(io.BytesIO(KML.encode('utf-8')))


class ParseKml(unittest.TestCase):
    def test_layers_and_points(self):
        name, _, pts, stats = parse()
        self.assertEqual(name, '試験の地図')
        self.assertEqual([p['layer'] for p in pts],
                         ['解放軍海軍、海軍陸戰隊基地及設施'] * 2 + ['解放軍火箭軍'])
        self.assertEqual(stats['skipped_non_point'], 1)
        self.assertEqual(pts[0]['name'], '湛江基地')  # 見えない字を除く
        self.assertEqual(pts[1]['name'], '某港 北營')  # NBSP は半角空白に
        self.assertNotIn('\u00a0', ''.join(p['name'] + p['layer'] for p in pts))
        self.assertEqual(pts[0]['coordinates'], [110.438729, 21.237076])  # 小数 6 桁、高度なし
        self.assertEqual(pts[2]['coordinates'], [100.1, 30.2])

    def test_html_removed(self):
        _, _, pts, _ = parse()
        d = pts[0]['description']
        self.assertNotIn('<', d)
        self.assertNotIn('example.invalid', d)
        self.assertIn('說明&註', d)
        self.assertEqual(d.split('\n')[0], '南海艦隊司令部所在地')
        self.assertIn('https://zh.wikipedia.org/wiki/X', d)

    def test_contacts_removed_links_kept(self):
        _, _, pts, stats = parse()
        d = pts[1]['description']
        for s in ('foo.bar@example.com', '0912', 'x@y.org', 'mailto'):
            self.assertNotIn(s, d)
        self.assertIn('https://twitter.com/someone/status/1', d)
        self.assertEqual(stats['contacts_removed'], 3)
        self.assertNotIn('foo.bar', d)
        self.assertEqual(stats['features_with_contacts'], 1)

    def test_third_party_name_replaced(self):
        _, _, pts, stats = parse()
        d = pts[1]['description']
        for s in ('Huang', 'ZI Hao', '用戶名', 'FaceBook', '軍友'):
            self.assertNotIn(s, d)
        self.assertIn(wen_map.THIRD_PARTY_NOTE, d)
        self.assertEqual(stats['third_party_names_replaced'], 1)

    def test_tracking_params_removed(self):
        _, _, pts, stats = parse()
        d = pts[1]['description']
        self.assertIn('https://twitter.com/someone/status/1 ', d)  # t・s を除き ? も残さない
        self.assertNotIn('t=AbC', d)
        self.assertIn('https://example.org/a?id=5 ', d)
        self.assertNotIn('fbclid', d)
        self.assertNotIn('utm_', d)
        self.assertIn('https://www.youtube.com/watch?v=abc&t=6s', d)  # YouTube の t（再生位置）は残す
        self.assertTrue(d.endswith('/viz/V/Dashboard1'))
        self.assertNotIn('%3C', d)
        self.assertEqual(stats['tracking_params_removed'], 4)
        self.assertEqual(wen_map.clean_url('https://x.com/a/status/1?s=20')[0], 'https://x.com/a/status/1')

    def test_contact_cue_url_removed(self):
        _, doc_desc, _, _ = parse()
        text, n = wen_map.scrub_contacts(wen_map.html_to_text(doc_desc))
        self.assertNotIn('facebook', text)
        self.assertNotIn('joseph.wen', text)
        self.assertIn('歡迎大家取用', text)
        self.assertEqual(n, 1)


class Outputs(unittest.TestCase):
    def test_build_and_crosswalk(self):
        with tempfile.TemporaryDirectory() as tmp:
            kml = os.path.join(tmp, 'm.kml')
            with open(kml, 'w', encoding='utf-8') as f:
                f.write(KML)
            out = os.path.join(tmp, 'ref')
            self.assertEqual(wen_map.main(['build', kml, '--retrieved', '2026-10-31', '--out-dir', out]), 0)
            with open(os.path.join(out, wen_map.GEOJSON_NAME), encoding='utf-8') as f:
                fc = json.load(f)
            self.assertEqual([x['id'] for x in fc['features']], ['wen-0001', 'wen-0002', 'wen-0003'])
            self.assertFalse(os.path.exists(os.path.join(out, 'meta.json')))
            with open(os.path.join(out, 'wen-pla-bases.meta.json'), encoding='utf-8') as f:
                meta = json.load(f)
            self.assertEqual(meta['attribution']['retrieved'], '2026-10-31')
            self.assertIn('承認を示すものではない', meta['attribution']['endorsement'])
            self.assertEqual(meta['processing']['contacts_removed'], 3)
            meta_text = json.dumps(meta, ensure_ascii=False)
            self.assertNotIn('facebook', meta_text)
            self.assertNotIn('joseph.wen', meta_text)  # 作者の Facebook の URL を載せない
            self.assertIn('明示のライセンスは無い', meta['license'])
            self.assertIn('取り下げる', meta['license'])
            self.assertIn('空欄', meta['crosswalk']['blank'])
            self.assertEqual(meta['processing']['third_party_names_replaced'], 1)
            with open(os.path.join(out, wen_map.GEOJSON_NAME), encoding='utf-8') as f:
                gj_text = f.read()
            self.assertNotIn('joseph.wen', gj_text)
            self.assertNotIn('Huang', gj_text)
            sites = os.path.join(tmp, 'sites.json')
            with open(sites, 'w', encoding='utf-8') as f:
                json.dump({'sites': [{'id': 'ZJG', 'name': '湛江', 'center_lon': 110.44, 'center_lat': 21.23},
                                     {'id': 'FAR', 'name': '遠い', 'center_lon': 0.0, 'center_lat': 0.0},
                                     {'id': 'RKT', 'name': '某造船所', 'center_lon': 100.12, 'center_lat': 30.2},
                                     {'id': 'RK2', 'name': '某港', 'center_lon': 100.12, 'center_lat': 30.2}]}, f)
            self.assertEqual(wen_map.main(['crosswalk', '--out-dir', out, '--sites', sites]), 0)
            with open(os.path.join(out, wen_map.CROSSWALK_NAME), encoding='utf-8') as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 4)
            self.assertEqual((rows[0]['site_id'], rows[0]['wen_id'], rows[0]['wen_name'], rows[0]['wen_layer']),
                             ('ZJG', 'wen-0001', '湛江基地', '解放軍海軍、海軍陸戰隊基地及設施'))
            self.assertLess(float(rows[0]['distance_km']), 8)
            self.assertEqual((rows[1]['site_id'], rows[1]['wen_id'], rows[1]['wen_layer']), ('FAR', '', ''))  # 該当なしも 1 行
            # 火箭軍層は対象外なので、造船所の地点でも港の地点でも空欄
            self.assertEqual([r['wen_id'] for r in rows[2:]], ['', ''])

    def test_target_layers(self):
        self.assertEqual(wen_map.target_layer_keys({'name': '大連・造船所', 'reef': False}),
                         [wen_map.NAVY_LAYER_KEY, wen_map.INDUSTRY_LAYER_KEY])
        self.assertEqual(wen_map.target_layer_keys({'name': '武漢・武船重工', 'reef': False})[-1], wen_map.INDUSTRY_LAYER_KEY)
        self.assertEqual(wen_map.target_layer_keys({'name': '南沙・永暑礁', 'reef': True}),
                         [wen_map.NAVY_LAYER_KEY, wen_map.AIR_LAYER_KEY])
        self.assertEqual(wen_map.target_layer_keys({'name': '湛江', 'reef': False}), [wen_map.NAVY_LAYER_KEY])

    def test_haversine(self):
        self.assertAlmostEqual(wen_map.haversine_km(0, 0, 0, 1), 111.2, delta=0.1)


class PublishedFiles(unittest.TestCase):
    """公開中の参照データに個人の連絡先が無いこと。"""

    def test_no_contacts_in_published(self):
        path = os.path.join('docs', 'data', 'ref', wen_map.GEOJSON_NAME)
        if not os.path.exists(path):
            self.skipTest('未生成')
        with open(path, encoding='utf-8') as f:
            fc = json.load(f)
        for feat in fc['features']:
            d = feat['properties']['description']
            self.assertEqual(wen_map.scrub_contacts(d)[1], 0, feat['id'])
            self.assertEqual(wen_map.scrub_third_party(d)[1], 0, feat['id'])
            self.assertEqual(wen_map.clean_urls(d)[1], 0, feat['id'])
            self.assertNotIn('<', d)
            self.assertNotIn('%3Cbr', d)
            self.assertNotIn('joseph.wen', d)
            self.assertNotIn('\u00a0', feat['properties']['name'] + feat['properties']['layer'])
        with open(os.path.join('docs', 'data', 'ref', wen_map.META_NAME), encoding='utf-8') as f:
            self.assertNotIn('joseph.wen', f.read())


if __name__ == '__main__':
    unittest.main()
