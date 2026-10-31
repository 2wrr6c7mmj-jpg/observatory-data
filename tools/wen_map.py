"""溫約瑟（Joseph Wen）氏の Google マイマップを参照データに変換する（手動更新、自動取得はしない）。

  python -m tools.wen_map build KML --retrieved YYYY-MM-DD [--out-dir docs/data/ref]
  python -m tools.wen_map crosswalk [--out-dir docs/data/ref] [--sites JSON ...] [--km 8]

build は KML（マイマップの「KML/KMZ にエクスポート」、KMZ なら中の doc.kml）を読み、
wen-pla-bases.geojson と wen-pla-bases.meta.json を作る（--retrieved は必須）。
説明文から HTML・個人の連絡先・第三者の記名・リンクの追跡用引数を除く。
crosswalk は build の出力を読み、sat/sites.json・sat/ports.json の監視地点の中心から
8 km 以内の点の対応表を作る（海軍層。造船所の地点は軍工層、人工島の地点は空軍層も）。
外部へは何も送らない。標準ライブラリのみ。
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
META_NAME = 'wen-pla-bases.meta.json'
CROSSWALK_NAME = 'wen-port-crosswalk.csv'
NAVY_LAYER_KEY = '海軍、海軍陸戰隊'
INDUSTRY_LAYER_KEY = '中國軍工及航天產業'
AIR_LAYER_KEY = '空軍、海軍航空兵'
THIRD_PARTY_NOTE = '（作者の記載では第三者の提供による）'
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


# 情報提供者の記名（例：「製圖及資料提供者為軍友 某某(FaceBook用戶名)」）。個人名・SNS のユーザー名は載せない
_SNS_USER = r'[(（]\s*(?:facebook|fb|臉書|脸书|ig|instagram|twitter|推特)?\s*(?:用戶名|用户名|帳號|账号|ID)\s*[)）]'
_THIRD_PARTY = [
    re.compile(r'(?i)(?:製圖|制图)?(?:及|與|与|和)?(?:資料|资料)?提供者?\s*(?:為|为|是|[:：])\s*(?:軍友|网友|網友)?\s*'
               r'[^\s(（。，,；;]+(?:[ \t]+[^\s(（。，,；;]+){0,3}\s*' + _SNS_USER),
    re.compile(r'(?i)[A-Za-z][\w.\-]*(?:[ \t]+[A-Za-z][\w.\-]*){0,3}\s*' + _SNS_USER),
]

# リンクの追跡用引数（どのサイトでも落とすもの・サイト別に落とすもの）
_TRACK_ANY = {'fbclid', 'gclid', 'dclid', 'msclkid', 'igshid', 'mc_cid', 'mc_eid', 'spm', 'yclid', '_ga'}
_TRACK_HOST = {'twitter.com': {'t', 's'}, 'x.com': {'t', 's'}, 'mobile.twitter.com': {'t', 's'}}
_URL = re.compile(r'https?://[^\s<>"\']+')
_TRAIL_BR = re.compile(r'(?i)(?:%3Cbr\s*/?%3E)+$')


def _clean_inline(s):
    s = _INVISIBLE.sub('', s or '').replace('\u00a0', ' ').replace('\u3000', ' ')
    return '\n'.join(re.sub(r'[ \t]+', ' ', ln).strip() for ln in s.split('\n')).strip()


def clean_url(url):
    """末尾の %3Cbr%3E と追跡用引数を除いた URL と、除いた引数の数を返す（他の引数は書き換えない）。"""
    url = _TRAIL_BR.sub('', url)
    base, frag = (url.split('#', 1) + [None])[:2]
    if '?' not in base:
        return url, 0
    head, query = base.split('?', 1)
    host = re.sub(r'^https?://', '', head).split('/', 1)[0].lower()
    host = host[4:] if host.startswith('www.') else host
    drop_host = _TRACK_HOST.get(host, set())
    kept, n = [], 0
    for kv in query.split('&'):
        key = kv.split('=', 1)[0].lower()
        if not kv:
            continue
        if key in _TRACK_ANY or key.startswith('utm_') or key in drop_host:
            n += 1
            continue
        kept.append(kv)
    out = head + ('?' + '&'.join(kept) if kept else '')
    if frag is not None:
        out += '#' + frag
    return out, n


def clean_urls(text):
    total = 0

    def sub(m):
        nonlocal total
        u, n = clean_url(m.group(0))
        total += n
        return u
    return _URL.sub(sub, text), total


def scrub_third_party(text):
    """情報提供者の個人名・SNS のユーザー名を注記に置き換え、(文, 件数) を返す。"""
    n = 0
    for pat in _THIRD_PARTY:
        text, k = pat.subn(THIRD_PARTY_NOTE, text)
        n += k
    return text, n


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
    """(平文, 除いた連絡先, 置き換えた第三者の記名, 除いた追跡用引数) を返す。"""
    text, n_contact = scrub_contacts(html_to_text(desc))
    text, n_third = scrub_third_party(text)
    text, n_track = clean_urls(text)
    text = '\n'.join(re.sub(r' {2,}', ' ', ln).strip() for ln in text.split('\n'))
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    return text, n_contact, n_third, n_track


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
                desc, n, n_third, n_track = clean_description(_child_text(child, 'description'))
                stats['contacts_removed'] += n
                stats['third_party_names_replaced'] += n_third
                stats['tracking_params_removed'] += n_track
                if n:
                    stats['features_with_contacts'] += 1
                if n_third:
                    stats['features_with_third_party'] += 1
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
        'license': '明示のライセンスは無い。作者の説明による利用の歓迎（「歡迎大家取用」）に基づく。作者の申し出があれば取り下げる。',
        'processing': {
            'tool': 'tools/wen_map.py（python -m tools.wen_map build）',
            'steps': [
                'ユーザーがマイマップから書き出した KML を KML→GeoJSON に変換（外部へのリクエストはしない）',
                '点（Point）だけを残す。層は最上位のフォルダ名',
                '座標は経度・緯度を小数 6 桁に丸め、高度は捨てた',
                '説明文の HTML（タグ・画像）を除いて平文にし、文字参照を戻した。改行は残す',
                '名前・層名の見えない字（左右の向きの印 U+200E など）を除いた',
                '個人の連絡先（メールアドレス、電話、連絡を求める文に付いた SNS の URL など）を除いた。出典の記事・投稿などのリンクは残した',
                '情報提供者として記された第三者の個人名・SNS のユーザー名は「%s」に置き換えた（作者名は出典として残す）' % THIRD_PARTY_NOTE,
                'リンクの追跡用の引数（Facebook の fbclid、X/Twitter の t・s、utm_*、spm など）と、URL 末尾に紛れた %3Cbr%3E を除いた。他の引数は変えていない',
                '名前・層名の改行なし空白（U+00A0）と全角空白は半角空白にし、続く空白は 1 つにした',
                '作者の連絡先（地図の説明にある）は載せていない',
                '地図に付いた画像（gx_media_links）とアイコンの様式は載せていない',
                '地物の id（wen-0001…）は KML の並び順で振った通し番号で、KML を更新すると変わりうる',
            ],
            'contacts_removed': stats.get('contacts_removed', 0),
            'features_with_contacts_removed': stats.get('features_with_contacts', 0),
            'third_party_names_replaced': stats.get('third_party_names_replaced', 0),
            'features_with_third_party_replaced': stats.get('features_with_third_party', 0),
            'tracking_params_removed': stats.get('tracking_params_removed', 0),
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
            CROSSWALK_NAME: 'うちの監視地点（sat/sites.json・sat/ports.json）と地図の点の対応（中心から %g km 以内）' % CROSSWALK_KM,
        },
        'crosswalk': {
            'columns': 'site_id, site_name, site_files, wen_layer, wen_id, wen_name, distance_km（地点の中心からの大円距離）',
            'target_layers': {
                'すべての地点': '解放軍海軍、海軍陸戰隊基地及設施',
                '造船所の地点（名前に「造船」「船重工」を含む）': '上記＋中國軍工及航天產業',
                '人工島の地点（sites.json・ports.json に reef の設定がある：MIS・SUB・FCR・WDY）': '上記＋解放軍空軍、海軍航空兵基地及設施',
            },
            'blank': '地点の行の wen_layer・wen_id・wen_name・distance_km が空欄なのは、対象の層に %g km 以内の点が無いという意味（施設が無いという意味ではない）' % CROSSWALK_KM,
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
                sites[key]['reef'] = sites[key]['reef'] or bool(s.get('reef'))
                continue
            sites[key] = {'id': key, 'name': s.get('name', ''), 'lon': s['center_lon'], 'lat': s['center_lat'],
                          'reef': bool(s.get('reef')), 'files': [os.path.basename(path)]}
    return list(sites.values())


def target_layer_keys(site):
    keys = [NAVY_LAYER_KEY]
    if '造船' in site['name'] or '船重工' in site['name']:
        keys.append(INDUSTRY_LAYER_KEY)
    if site.get('reef'):
        keys.append(AIR_LAYER_KEY)
    return keys


def crosswalk_rows(sites, fc, km=CROSSWALK_KM):
    rows = []
    for s in sites:
        keys = target_layer_keys(s)
        near = []
        for f in fc['features']:
            if not any(k in f['properties']['layer'] for k in keys):
                continue
            lon, lat = f['geometry']['coordinates'][:2]
            d = haversine_km(s['lon'], s['lat'], lon, lat)
            if d <= km:
                near.append((d, f))
        near.sort(key=lambda x: (x[0], x[1]['id']))
        base = [s['id'], s['name'], '+'.join(s['files'])]
        if not near:
            rows.append(base + ['', '', '', ''])
        for d, f in near:
            rows.append(base + [f['properties']['layer'], f['id'], f['properties']['name'], '%.2f' % d])
    return rows


def write_crosswalk(path, rows):
    import csv
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['site_id', 'site_name', 'site_files', 'wen_layer', 'wen_id', 'wen_name', 'distance_km'])
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
    print('点 %d、層 %d、除いた連絡先 %d 件（%d 地物）、第三者の記名 %d 件（%d 地物）、追跡用引数 %d、点以外 %d' % (
        len(points), len(meta['counts']['layers']), stats.get('contacts_removed', 0),
        stats.get('features_with_contacts', 0), stats.get('third_party_names_replaced', 0),
        stats.get('features_with_third_party', 0), stats.get('tracking_params_removed', 0),
        stats.get('skipped_non_point', 0)))
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
