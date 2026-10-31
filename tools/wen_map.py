"""溫約瑟（Joseph Wen）氏の Google マイマップを参照データに変換する（手動更新、自動取得はしない）。

  python -m tools.wen_map build KML [--retrieved YYYY-MM-DD] [--out-dir docs/data/ref]
  python -m tools.wen_map crosswalk [--out-dir docs/data/ref]

build は KML（マイマップの「KML に書き出し」、KMZ なら中の doc.kml）を読み、
wen-pla-bases.geojson と wen-pla-bases.meta.json を作る。外部へは何も送らない。
crosswalk は sat/sites.json・sat/ports.json の監視地点と海軍層の対応表を作る。
標準ライブラリのみ。
"""
import argparse
import html
import json
import math
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter

KML_NS = '{http://www.opengis.net/kml/2.2}'
GEOJSON_NAME = 'wen-pla-bases.geojson'
META_NAME = 'meta.json'
CROSSWALK_NAME = 'wen-port-crosswalk.csv'
NAVY_LAYER_KEY = '海軍、海軍陸戰隊'
CROSSWALK_KM = 8.0
DECIMALS = 6

MAP_TITLE = '中國人民解放軍基地及設施 (持續更新)'
MAP_URL = 'https://www.google.com/maps/d/viewer?mid=19Q8BraU1Nmnk23TzMb5rhXFuIAnOpTTq'

# 名前・層名に混じる見えない字（左右の向きの印、幅ゼロの空白、BOM）
_INVISIBLE = re.compile('[​-‏‪-‮⁦-⁩﻿]')
_BR = re.compile(r'(?i)<\s*br\s*/?\s*>|</\s*(p|div|li|tr)\s*>')
_TAG = re.compile(r'<[^>]*>')

# 個人の連絡先（公開データに載せない）
_EMAIL = re.compile(r'(?i)(mailto:)?[a-z0-9._%+-]+@[a-z0-9-]+(\.[a-z0-9-]+)+')
_TEL_URI = re.compile(r'(?i)\btel:\s*\+?[\d\-() ]{6,}')
_LABELLED = re.compile(
    r'(?i)(e-?mail|電郵|電子郵件|電子信箱|信箱|郵箱|邮箱|電話|电话|手機|手机|聯絡電話|tel|phone|'
    r'line\s*id|wechat|微信|telegram|whatsapp|qq)\s*[:：]\s*[^\s,，、;；]+')
# 連絡を求める文の後に続く URL（例：「歡迎來信: https://www.facebook.com/…」）
_CONTACT_CUE = re.compile(
    r'(?i)(歡迎來信|欢迎来信|來信|来信|聯絡我|联系我|聯繫我|聯絡方式|联系方式|contact\s*me)\s*[:：]?\s*(https?://\S+)?')


def _clean_inline(s):
    return _INVISIBLE.sub('', s or '').strip()


def scrub_contacts(text):
    """個人の連絡先を除き、(文, 除いた件数) を返す。出典などのリンクは残す。"""
    n = 0
    for pat, rep in ((_CONTACT_CUE, ''), (_EMAIL, ''), (_TEL_URI, ''), (_LABELLED, '')):
        text, k = pat.subn(rep, text)
        n += k
    return text, n


def html_to_text(desc):
    """説明文の HTML を除いて平文にする（改行は残し、行内の連続空白は 1 つに）。"""
    s = _BR.sub('\n', desc or '')
    s = _TAG.sub(' ', s)
    s = html.unescape(s)
    s = _INVISIBLE.sub('', s).replace(' ', ' ').replace('　', ' ')
    lines = [re.sub(r'[ \t\r\f\v]+', ' ', ln).strip() for ln in s.split('\n')]
    out = []
    for ln in lines:
        if ln or (out and out[-1]):
            out.append(ln)
    return '\n'.join(out).strip()


def clean_description(desc):
    text, n = scrub_contacts(html_to_text(desc))
    text = '\n'.join(re.sub(r' {2,}', ' ', ln).strip() for ln in text.split('\n'))
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    return text, n


def _child_text(el, tag):
    c = el.find(KML_NS + tag)
    return (c.text or '') if c is not None else ''


def _point(pm):
    c = pm.find('.//' + KML_NS + 'Point/' + KML_NS + 'coordinates')
    if c is None or not (c.text or '').strip():
        return None
    parts = c.text.split()[0].split(',')
    lon, lat = float(parts[0]), float(parts[1])
    if not (-180 <= lon <= 180 and -90 <= lat <= 90):
        raise ValueError('座標が範囲外: %r' % c.text)
    return [round(lon, DECIMALS), round(lat, DECIMALS)]


def parse_kml(source):
    """KML（パスまたはファイル）を読み、(文書名, 文書の説明, 点の一覧, 集計) を返す。

    層は Document 直下の Folder の名前（入れ子の Folder は最上位の名前に寄せる）。
    Folder の外の Placemark は層名 ''。点以外の形は数えて捨てる。
    """
    root = ET.parse(source).getroot()
    doc = root.find(KML_NS + 'Document')
    if doc is None:
        doc = root
    stats = Counter()
    points = []

    def walk(el, layer):
        for child in el:
            if child.tag == KML_NS + 'Folder':
                walk(child, layer if layer is not None else _clean_inline(_child_text(child, 'name')))
            elif child.tag == KML_NS + 'Placemark':
                pt = _point(child)
                if pt is None:
                    stats['skipped_non_point'] += 1
                    continue
                desc, n = clean_description(_child_text(child, 'description'))
                stats['contacts_removed'] += n
                if n:
                    stats['features_with_contacts'] += 1
                points.append({'layer': layer or '', 'name': _clean_inline(_child_text(child, 'name')),
                               'description': desc, 'coordinates': pt})
            elif child.tag == KML_NS + 'Document':
                walk(child, layer)

    walk(doc, None)
    return _clean_inline(_child_text(doc, 'name')), _child_text(doc, 'description'), points, stats


def to_geojson(points):
    feats = []
    for i, p in enumerate(points, 1):
        feats.append({'type': 'Feature', 'id': 'wen-%04d' % i,
                      'geometry': {'type': 'Point', 'coordinates': p['coordinates']},
                      'properties': {'layer': p['layer'], 'name': p['name'], 'description': p['description']}})
    return {'type': 'FeatureCollection', 'features': feats}


def dump_geojson(fc):
    """1 地物 1 行（差分が読みやすいように）。"""
    head = '{"type":"FeatureCollection","features":[\n'
    body = ',\n'.join(json.dumps(f, ensure_ascii=False, separators=(',', ':')) for f in fc['features'])
    return head + body + '\n]}\n'


def build_meta(doc_name, points, stats, retrieved):
    layers = Counter(p['layer'] for p in points)
    return {
        'dataset': 'ref/wen-pla-bases',
        'kind': '参照データ（他者の地図の写し。うちの観測・推定ではない）',
        'attribution': {
            'author': '溫約瑟（Joseph Wen）',
            'title': MAP_TITLE,
            'url': MAP_URL,
            'retrieved': retrieved,
            'title_in_file': doc_name,
            'credit': '出典：溫約瑟（Joseph Wen）「%s」（Google マイマップ、%s、%s 取得）' % (MAP_TITLE, MAP_URL, retrieved),
            'author_statement_summary': '作者の地図の説明の要旨：軍事愛好者と民衆が中国の軍事の概況を知り、東アジアの安全保障・両岸関係などを議論する助けとなるよう作ったもので、自由に取用することを歓迎する（「歡迎大家取用」）。誤りの指摘も歓迎する、とある。',
            'army_layer_note': '陸軍層（解放軍陸軍、陸軍防空單位、聯勤保障設施）は、情報が極めて少ないため一部の施設が作者の主観的な判読（推定）であり、誤りがありうると作者自身が注記している（層名にも「部分設施為個人推斷」「製作中」とある）。',
            'endorsement': '作者の公式の関与・承認を示すものではない。このデータの誤りは observatory-data の責任で、作者に問い合わせないこと。',
        },
        'processing': {
            'tool': 'tools/wen_map.py（python -m tools.wen_map build）',
            'steps': [
                'ユーザーがマイマップから書き出した KML を KML→GeoJSON に変換（外部へのリクエストはしない）',
                '点（Point）だけを残す。層は最上位のフォルダ名',
                '座標は経度・緯度を小数 6 桁に丸め、高度は捨てた',
                '説明文の HTML（タグ・画像）を除いて平文にし、文字参照を戻した。改行は残す',
                '名前・層名の見えない字（左右の向きの印 U+200E など）を除いた',
                '個人の連絡先（メールアドレス、電話、連絡を求める文に付いた SNS の URL など）を除いた。出典の記事・投稿などのリンクは残した',
                '作者の連絡先（地図の説明にある）は載せていない',
                '地図に付いた画像（gx_media_links）とアイコンの様式は載せていない',
                '地物の id（wen-0001…）は KML の並び順で振った通し番号で、KML を更新すると変わりうる',
            ],
            'contacts_removed': stats.get('contacts_removed', 0),
            'features_with_contacts_removed': stats.get('features_with_contacts', 0),
            'skipped_non_point': stats.get('skipped_non_point', 0),
        },
        'counts': {'features': len(points), 'layers': dict(layers.most_common())},
        'fields': {
            'id': 'wen-NNNN（KML の並び順の通し番号）',
            'layer': '作者の層名（繁体字、原文のまま）',
            'name': '作者の地名・施設名（原文のまま）',
            'description': '作者の説明（HTML を除いた平文、連絡先を除いた）。出典 URL を含むことがある',
        },
        'limits': [
            '作者個人がまとめた地図で、公式の資料ではない。位置・名称・所属は作者の判断による',
            '更新は手動（README の手順）。作者の地図はその後も更新されるので、retrieved の日付時点の写し',
            '分析では単独の根拠にせず、衛星画像・公式発表などと照合する',
        ],
        'files': {
            GEOJSON_NAME: '地物（Point）',
            CROSSWALK_NAME: 'うちの監視地点（sat/sites.json・sat/ports.json）と海軍層の対応（中心から %g km 以内）' % CROSSWALK_KM,
        },
    }


def haversine_km(lon1, lat1, lon2, lat2):
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def load_sites(paths):
    """監視地点を id で合わせる（同じ id は先のファイルの中心を使い、files に両方を記す）。"""
    sites = {}
    for path in paths:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        items = data.get('sites', data) if isinstance(data, dict) else data
        for s in items:
            key = s['id']
            if key in sites:
                sites[key]['files'].append(os.path.basename(path))
                continue
            sites[key] = {'id': key, 'name': s.get('name', ''), 'lon': s['center_lon'], 'lat': s['center_lat'],
                          'files': [os.path.basename(path)]}
    return list(sites.values())


def crosswalk_rows(sites, fc, km=CROSSWALK_KM):
    navy = [f for f in fc['features'] if NAVY_LAYER_KEY in f['properties']['layer']]
    rows = []
    for s in sites:
        near = []
        for f in navy:
            lon, lat = f['geometry']['coordinates'][:2]
            d = haversine_km(s['lon'], s['lat'], lon, lat)
            if d <= km:
                near.append((d, f))
        near.sort(key=lambda x: (x[0], x[1]['id']))
        base = [s['id'], s['name'], '+'.join(s['files'])]
        if not near:
            rows.append(base + ['', '', ''])
        for d, f in near:
            rows.append(base + [f['id'], f['properties']['name'], '%.2f' % d])
    return rows


def write_crosswalk(path, rows):
    import csv
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['site_id', 'site_name', 'site_files', 'wen_id', 'wen_name', 'distance_km'])
        w.writerows(rows)


def cmd_build(args):
    doc_name, _doc_desc, points, stats = parse_kml(args.kml)
    if not points:
        print('点が 1 件も無い。KML を確かめる', file=sys.stderr)
        return 1
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, GEOJSON_NAME), 'w', encoding='utf-8') as f:
        f.write(dump_geojson(to_geojson(points)))
    meta = build_meta(doc_name, points, stats, args.retrieved)
    with open(os.path.join(args.out_dir, META_NAME), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
        f.write('\n')
    print('点 %d、層 %d、除いた連絡先 %d 件（%d 地物）、点以外 %d' % (
        len(points), len(meta['counts']['layers']), stats.get('contacts_removed', 0),
        stats.get('features_with_contacts', 0), stats.get('skipped_non_point', 0)))
    return 0


def cmd_crosswalk(args):
    with open(os.path.join(args.out_dir, GEOJSON_NAME), encoding='utf-8') as f:
        fc = json.load(f)
    rows = crosswalk_rows(load_sites(args.sites), fc, args.km)
    write_crosswalk(os.path.join(args.out_dir, CROSSWALK_NAME), rows)
    print('対応表 %d 行' % len(rows))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build')
    b.add_argument('kml')
    b.add_argument('--retrieved', required=True, help='KML を書き出した日（YYYY-MM-DD）')
    b.add_argument('--out-dir', default='docs/data/ref')
    c = sub.add_parser('crosswalk')
    c.add_argument('--out-dir', default='docs/data/ref')
    c.add_argument('--sites', nargs='+', default=['sat/sites.json', 'sat/ports.json'])
    c.add_argument('--km', type=float, default=CROSSWALK_KM)
    args = ap.parse_args(argv)
    return cmd_build(args) if args.cmd == 'build' else cmd_crosswalk(args)


if __name__ == '__main__':
    sys.exit(main())
