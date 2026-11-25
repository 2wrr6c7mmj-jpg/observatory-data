"""早期警戒 ew-2 の追加の情報源（戦略層・動員層・展開層）。書き込むのは docs/data/ew/src/ だけ。

源ごとに 1 つの JSON（docs/data/ew/src/<id>.json）に、前回までの値を残したまま新しい分を足す。
どの源も collectors/polite.py で robots.txt を確かめ、チャレンジ・403 なら止める（回り込まない）。
1 つの源の失敗は他の源を止めない（status.json の sources に結果を書く）。

源（2部の調査 2026-11-25、probe/ew2-robots で取得の形を確認）
- gold     国家外汇管理局「官方储备资产」の年ごとの xlsx（金の保有、万盎司）。月次
- oil      JODI-Oil World Database の年ごとの CSV（中国の原油：生産＋輸入−輸出−精製投入＝見かけの在庫の増え、千バレル/日）。月次。
           中国は在庫（CLOSTLV）を出していない（probe で確認）
- jwc      Lloyd's Market Association の Joint War Committee の Listed Areas（最新の JWLA の PDF の本文に東アジアの語があるか）。事象
- rmrb     人民日報 電子版の 1〜4 版（要聞）の見出しと URL（本文は取らない）。日次の語数
- feed     東アジア情勢観測の /api/feeds（取り直しはしない）から、国防動員・徴兵・海底ケーブルの語の見出しを貯める。事象
- ioda     IODA（Georgia Tech）の地域の障害の事象（連江〔馬祖〕・金門・澎湖）。事象
- ferries  Global Fishing Watch 4Wings の public-global-presence（旅客船の在りかの時間、格子の合計だけ）。日次（約 4 日遅れ）
個人を特定するデータは扱わない（記者名・船の識別子・人名は保存しない）。
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from collectors import polite

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'docs' / 'data' / 'ew' / 'src'
UA = 'observatory-data/1'
PAUSE = 1.5


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def load(name: str, default: dict) -> dict:
    p = OUT / f'{name}.json'
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else default


def save(name: str, doc: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'{name}.json').write_text(json.dumps(doc, ensure_ascii=False, indent=1, sort_keys=True) + '\n', encoding='utf-8')


def fetch(url: str, max_bytes: int = 40_000_000) -> polite.Response:
    polite.require_allowed(url, UA)
    time.sleep(PAUSE)
    return polite.check_challenge(polite.raw_get(url, UA, 90, None, max_bytes))


def links(html: str) -> list[tuple[str, str]]:
    out = re.findall(r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', html, re.S | re.I)
    return [(h, re.sub(r'<[^>]+>|&nbsp;|\s+', ' ', t).strip()) for h, t in out]


# ---------------------------------------------------------------- gold (SAFE)
SAFE = 'https://www.safe.gov.cn'
SAFE_INDEX = SAFE + '/safe/whcb/index.html'


def xlsx_rows(data: bytes) -> list[list[str]]:
    z = zipfile.ZipFile(io.BytesIO(data))
    m = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
    ss = []
    if 'xl/sharedStrings.xml' in z.namelist():
        for si in ET.fromstring(z.read('xl/sharedStrings.xml')).iter(m + 'si'):
            ss.append(''.join(t.text or '' for t in si.iter(m + 't')))
    sheet = sorted(n for n in z.namelist() if re.match(r'xl/worksheets/sheet\d+\.xml$', n))[0]
    rows = []
    for row in ET.fromstring(z.read(sheet)).iter(m + 'row'):
        vals = {}
        for c in row.iter(m + 'c'):
            v = c.find(m + 'v')
            t = v.text if v is not None and v.text is not None else ''
            if c.get('t') == 's' and t:
                t = ss[int(t)]
            col = re.match(r'[A-Z]+', c.get('r', 'A1')).group(0)
            idx = 0
            for ch in col:
                idx = idx * 26 + ord(ch) - 64
            vals[idx - 1] = t.strip()
        rows.append([vals.get(i, '') for i in range(max(vals) + 1)] if vals else [])
    return rows


def parse_gold(rows: list[list[str]]) -> dict[str, int]:
    """{'YYYY-MM': 万盎司} from one year's table (months in the header row, ounces in the row with 万盎司)."""
    head = next((r for r in rows if any(re.fullmatch(r'\d{4}\.\d{2}', c) for c in r)), None)
    oz = next((r for r in rows if any('万盎司' in c for c in r)), None)
    if not head or not oz:
        return {}
    out = {}
    for i, c in enumerate(head):
        if re.fullmatch(r'\d{4}\.\d{2}', c) and i < len(oz):
            m = re.search(r'([\d,]+(?:\.\d+)?)\s*万盎司', oz[i])
            if m:
                out[c.replace('.', '-')] = round(float(m.group(1).replace(',', '')))
    return out


def run_gold(years: int) -> dict:
    doc = load('gold', {'unit': '万盎司', 'months': {}})
    idx = fetch(SAFE_INDEX).body.decode('utf-8', 'replace')
    pages = [(h, t) for h, t in links(idx) if re.search(r'官方储备资产（\d{4}年）', t)]
    this = date.today().year
    for h, t in pages:
        y = int(re.search(r'(\d{4})年', t).group(1))
        if y < this - years:
            continue
        page = fetch(urllib.parse.urljoin(SAFE_INDEX, h)).body.decode('utf-8', 'replace')
        x = [u for u, _ in links(page) if u.lower().endswith('.xlsx')]
        if not x:
            continue
        months = parse_gold(xlsx_rows(fetch(urllib.parse.urljoin(SAFE_INDEX, x[-1])).body))
        for k, v in months.items():
            old = doc['months'].get(k)
            doc['months'][k] = {'value': v, 'first_seen': (old or {}).get('first_seen') or date.today().isoformat()}
    doc.update(source='国家外汇管理局「官方储备资产」（月）', url=SAFE_INDEX, retrieved_utc=now_utc())
    save('gold', doc)
    return {'months': len(doc['months'])}


# ---------------------------------------------------------------- oil (JODI)
JODI = 'https://www.jodidata.org/_resources/files/downloads/oil-data/annual-csv/primary/primaryyear{y}.csv'
FLOWS = {'INDPROD': 1, 'TOTIMPSB': 1, 'TOTEXPSB': -1, 'REFINOBS': -1}


def parse_jodi(text: str) -> dict[str, float]:
    """{'YYYY-MM': 見かけの在庫の増え（千バレル/日）} for China's crude; a month is kept only when all four flows are numbers."""
    parts: dict[str, dict] = {}
    for r in csv.DictReader(io.StringIO(text)):
        if r.get('REF_AREA') != 'CN' or r.get('ENERGY_PRODUCT') != 'CRUDEOIL' or r.get('UNIT_MEASURE') != 'KBD':
            continue
        if r.get('FLOW_BREAKDOWN') in FLOWS:
            try:
                parts.setdefault(r['TIME_PERIOD'], {})[r['FLOW_BREAKDOWN']] = float(r['OBS_VALUE'])
            except (TypeError, ValueError):
                pass
    return {m: round(sum(FLOWS[k] * v for k, v in p.items()), 1) for m, p in parts.items() if len(p) == len(FLOWS)}


def run_oil(years: int) -> dict:
    doc = load('oil', {'unit': '千バレル/日', 'months': {}})
    for y in range(date.today().year - years, date.today().year + 1):
        try:
            text = fetch(JODI.format(y=y)).body.decode('utf-8', 'replace')
        except polite.SourceBlocked:
            raise
        for k, v in parse_jodi(text).items():
            old = doc['months'].get(k)
            doc['months'][k] = {'value': v, 'first_seen': (old or {}).get('first_seen') or date.today().isoformat()}
    doc.update(source='JODI-Oil World Database（中国・原油：生産＋輸入−輸出−精製投入）', url='https://www.jodidata.org/oil/', retrieved_utc=now_utc())
    save('oil', doc)
    return {'months': len(doc['months'])}


# ---------------------------------------------------------------- JWC listed areas
JWC = 'https://lmalloyds.com/specialist-areas/underwriting/listed-areas/'
EAST_ASIA = re.compile(r'Taiwan|East China Sea|South China Sea|Yellow Sea|Bohai|Chin(a|ese) (port|terr|water)|PRC|Korea|Japan', re.I)


def pdf_text(data: bytes) -> str:
    import subprocess
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pdf') as f:
        f.write(data)
        f.flush()
        return subprocess.run(['pdftotext', '-layout', f.name, '-'], capture_output=True, text=True, timeout=120).stdout


def run_jwc() -> dict:
    doc = load('jwc', {'circulars': {}})
    page = fetch(JWC).body.decode('utf-8', 'replace')
    pdfs = sorted({(re.search(r'JWLA-?(\d+)', h, re.I).group(1), urllib.parse.urljoin(JWC, h))
                   for h, _ in links(page) if re.search(r'JWLA-?\d+.*\.pdf$', h, re.I)}, key=lambda x: int(x[0]))
    for num, url in pdfs[-2:]:
        key = f'JWLA-{int(num):03d}'
        if key in doc['circulars']:
            continue
        text = pdf_text(fetch(url).body)
        hits = sorted({m.group(0) for m in EAST_ASIA.finditer(text)})
        doc['circulars'][key] = {'url': url, 'first_seen': date.today().isoformat(), 'east_asia_terms': hits, 'chars': len(text)}
    doc.update(source="Lloyd's Market Association, Joint War Committee — Listed Areas", url=JWC, retrieved_utc=now_utc())
    save('jwc', doc)
    return {'circulars': len(doc['circulars'])}


# ---------------------------------------------------------------- People's Daily headlines
RMRB_PC = 'https://paper.people.com.cn/rmrb/pc/layout/{ym}/{d}/node_{p:02d}.html'
RMRB_OLD = 'http://paper.people.com.cn/rmrb/html/{y}-{m}/{d}/nbs.D110000renmrb_{p:02d}.htm'
PAGES = (1, 2, 3, 4)


def rmrb_day(day: date, get) -> list[dict] | None:
    """Headlines (title, page, url) of pages 1–4 of one day; None when the day has no edition in either format."""
    out = []
    for p in PAGES:
        url = RMRB_PC.format(ym=day.strftime('%Y%m'), d=day.strftime('%d'), p=p)
        r = get(url)
        pat = r'content_\d+\.html?$'
        if r.status == 404:
            url = RMRB_OLD.format(y=day.year, m=day.strftime('%m'), d=day.strftime('%d'), p=p)
            r = get(url)
            pat = r'nw\.D110000renmrb_\d+_\d+-\d+\.htm$'
        if r.status == 404:
            if p == 1:
                return None
            continue
        if r.status != 200:
            raise polite.SourceBlocked(f'{url}: HTTP {r.status}')
        seen = set()
        for h, t in links(r.body.decode('utf-8', 'replace')):
            u = urllib.parse.urljoin(url, h)
            if re.search(pat, h) and t and u not in seen:
                seen.add(u)
                out.append({'page': p, 'title': t, 'url': u})
    return out


def run_rmrb(days: int) -> dict:
    doc = load('rmrb', {'days': {}})
    today = datetime.now(timezone(timedelta(hours=8))).date()
    got = 0
    for i in range(days, -1, -1):
        day = today - timedelta(days=i)
        k = day.isoformat()
        if k in doc['days'] and i > 1:
            continue
        heads = rmrb_day(day, fetch)
        if heads is not None:
            doc['days'][k] = heads
            got += 1
    doc.update(source='人民日報 電子版（1〜4 版の見出し。本文は取らない）', url='https://paper.people.com.cn/', retrieved_utc=now_utc())
    save('rmrb', doc)
    return {'days': len(doc['days']), 'new': got}


# ---------------------------------------------------------------- site feed keywords
FEED = 'https://observatory-watch.pages.dev/api/feeds'
FEED_WORDS = {
    'mobilization': re.compile(r'国防动员|人防|民兵|拉动|征召|支前|戦時動員|国防動員|民兵|national defen[cs]e mobili[sz]ation|militia', re.I),
    'conscription': re.compile(r'征兵|退役|延长服役|暂缓退役|徴兵|除隊|退役延期|conscription|demobili[sz]ation|discharge.{0,20}(delay|postpon|extend)', re.I),
    'cables': re.compile(r'海缆|海纜|海底电缆|海底電纜|海底光缆|海底光纜|海底ケーブル|submarine cable|undersea cable|subsea cable', re.I),
}
FEED_PLACE = re.compile(r'中国|中國|台湾|台灣|臺灣|福建|浙江|广东|廈門|厦门|金门|金門|马祖|馬祖|澎湖|东部战区|東部戦区|China|Chinese|PLA|Taiwan|Matsu|Kinmen|Penghu|Fujian', re.I)


def feed_matches(feed: dict) -> dict[str, list]:
    out = {k: [] for k in FEED_WORDS}
    for r in feed.get('records', []):
        text = f"{r.get('title') or ''} {r.get('summary') or ''}"
        if not FEED_PLACE.search(text):
            continue
        ts = r.get('publishedAt')
        day = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).date().isoformat() if isinstance(ts, (int, float)) else None
        for k, pat in FEED_WORDS.items():
            if pat.search(text) and day:
                out[k].append({'date': day, 'title': (r.get('title') or '')[:200], 'url': r.get('url'), 'source': r.get('source')})
    return out


def run_feed() -> dict:
    doc = load('feed', {'items': {k: [] for k in FEED_WORDS}})
    req = urllib.request.Request(FEED, headers={'Accept': 'application/json', 'User-Agent': UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        feed = json.loads(r.read().decode('utf-8'))
    first = min((datetime.fromtimestamp(x['publishedAt'] / 1000, tz=timezone.utc).date() for x in feed.get('records', [])
                 if isinstance(x.get('publishedAt'), (int, float))), default=None)
    for k, items in feed_matches(feed).items():
        have = {i['url'] for i in doc['items'].setdefault(k, [])}
        doc['items'][k] += [i for i in items if i['url'] not in have]
        doc['items'][k].sort(key=lambda i: i['date'])
    doc.setdefault('covered', [])
    if first:
        doc['covered'].append([first.isoformat(), date.today().isoformat()])
        doc['covered'] = merge_ranges(doc['covered'])
    doc.update(source='東アジア情勢観測の報道の一覧（/api/feeds、約 48 源）の見出しと要約の語', url=FEED, retrieved_utc=now_utc())
    save('feed', doc)
    return {k: len(v) for k, v in doc['items'].items()}


def merge_ranges(rs: list) -> list:
    out = []
    for a, b in sorted(rs):
        if out and a <= (date.fromisoformat(out[-1][1]) + timedelta(days=1)).isoformat():
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


# ---------------------------------------------------------------- IODA
IODA = 'https://api.ioda.inetintel.cc.gatech.edu/v2/outages/events'
IODA_REGIONS = {'4210': '連江（馬祖）', '4209': '金門', '4208': '澎湖'}


def run_ioda(days: int) -> dict:
    doc = load('ioda', {'regions': IODA_REGIONS, 'events': [], 'covered': []})
    until = int(time.time())
    since = until - days * 86400
    have = {(e['region'], e['start'], e['datasource']) for e in doc['events']}
    for code in IODA_REGIONS:
        start = since
        while start < until:
            end = min(until, start + 90 * 86400)
            q = urllib.parse.urlencode({'entityType': 'region', 'entityCode': code, 'from': start, 'until': end, 'limit': 500})
            body = json.loads(fetch(f'{IODA}?{q}').body.decode('utf-8'))
            for e in body.get('data') or []:
                key = (code, int(e['start']), e.get('datasource'))
                if key not in have:
                    have.add(key)
                    doc['events'].append({'region': code, 'start': int(e['start']), 'duration': int(e.get('duration') or 0),
                                          'datasource': e.get('datasource'), 'score': round(float(e.get('score') or 0), 1)})
            start = end
    doc['events'].sort(key=lambda e: (e['start'], e['region']))
    day = lambda t: datetime.fromtimestamp(t, tz=timezone.utc).date().isoformat()  # noqa: E731
    doc['covered'] = merge_ranges(doc['covered'] + [[day(since), day(until)]])
    doc.update(source='IODA（Internet Outage Detection and Analysis, Georgia Tech）地域の障害の事象', url='https://ioda.inetintel.cc.gatech.edu/', retrieved_utc=now_utc())
    save('ioda', doc)
    return {'events': len(doc['events'])}


# ---------------------------------------------------------------- ferries (GFW presence)
GFW = 'https://gateway.api.globalfishingwatch.org/v3/4wings/report'
ZONES = {
    'fujian': {'name': '台湾海峡の西岸（福建沿岸）', 'bbox': [119.0, 24.0, 120.4, 26.6]},
    'bohai': {'name': '渤海の定期航路（煙台〜大連・蓬莱〜旅順）', 'bbox': [120.7, 37.4, 122.1, 39.1]},
}


def gfw_hours(token: str, bbox, start: date, end: date) -> dict[str, float]:
    x0, y0, x1, y1 = bbox
    poly = {'type': 'Polygon', 'coordinates': [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]}
    q = {'spatial-resolution': 'LOW', 'temporal-resolution': 'DAILY', 'datasets[0]': 'public-global-presence:latest',
         'date-range': f'{start.isoformat()},{end.isoformat()}', 'format': 'JSON', 'filters[0]': "vessel_type in ('passenger')"}
    req = urllib.request.Request(GFW + '?' + urllib.parse.urlencode(q), data=json.dumps({'geojson': poly}).encode(), method='POST',
                                 headers={'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json', 'User-Agent': UA})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            doc = json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f'GFW HTTP {e.code}') from None
    out: dict[str, float] = {}
    for ent in doc.get('entries') or []:
        for rows in ent.values():
            for row in rows:
                out[row['date']] = out.get(row['date'], 0) + float(row.get('hours') or 0)
    return out


def run_ferries(days: int) -> dict:
    token = os.environ.get('GFW_API_TOKEN', '').strip()
    if not token:
        raise RuntimeError('GFW_API_TOKEN が無い')
    doc = load('ferries', {'zones': ZONES, 'daily': {}, 'covered': []})
    end = date.today() - timedelta(days=3)
    start = end - timedelta(days=days)
    for zid, z in ZONES.items():
        series = doc['daily'].setdefault(zid, {})
        s = start
        while s <= end:
            e = min(end, s + timedelta(days=180))
            got = gfw_hours(token, z['bbox'], s, e)
            for i in range((e - s).days + 1):
                k = (s + timedelta(days=i)).isoformat()
                series[k] = round(got.get(k, 0.0), 1)   # a day in the range with no row: no passenger hours in the zone
            s = e + timedelta(days=1)
            time.sleep(PAUSE)
    doc['covered'] = merge_ranges(doc['covered'] + [[start.isoformat(), end.isoformat()]])
    doc.update(source='Global Fishing Watch 4Wings public-global-presence:latest（旅客船、格子の合計の時間）', url='https://globalfishingwatch.org/',
               attribution='Powered by Global Fishing Watch. CC BY-NC 4.0', retrieved_utc=now_utc())
    save('ferries', doc)
    return {z: len(v) for z, v in doc['daily'].items()}


# ---------------------------------------------------------------- main
RUNNERS = {
    'gold': lambda a: run_gold(a.years),
    'oil': lambda a: run_oil(a.years),
    'jwc': lambda a: run_jwc(),
    'rmrb': lambda a: run_rmrb(a.days),
    'feed': lambda a: run_feed(),
    'ioda': lambda a: run_ioda(a.days),
    'ferries': lambda a: run_ferries(a.days),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('sources', nargs='*', default=list(RUNNERS))
    ap.add_argument('--days', type=int, default=10, help='さかのぼる日数（rmrb・ioda・ferries）')
    ap.add_argument('--years', type=int, default=1, help='さかのぼる年数（gold・oil）')
    a = ap.parse_args(argv)
    status = load('status', {'sources': {}})
    failed = 0
    for name in a.sources:
        try:
            res = RUNNERS[name](a)
            status['sources'][name] = {'status': 'ok', 'at': now_utc(), **res}
            print(f'::notice title=ew2 {name}::ok {json.dumps(res, ensure_ascii=False)}')
        except polite.Challenge as e:
            status['sources'][name] = {'status': 'challenge', 'at': now_utc(), 'reason': str(e)[:200]}
            print(f'::warning title=ew2 {name}::challenge {e}')
            failed += 1
        except Exception as e:  # noqa: BLE001 - one source must not stop the others
            status['sources'][name] = {'status': 'error', 'at': now_utc(), 'reason': f'{type(e).__name__}: {str(e)[:200]}'}
            print(f'::warning title=ew2 {name}::{type(e).__name__}: {str(e)[:200]}')
            failed += 1
    save('status', status)
    return 0


if __name__ == '__main__':
    sys.exit(main())
