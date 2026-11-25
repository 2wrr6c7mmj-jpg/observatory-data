"""早期警戒 ew-2 の追加の情報源（戦略層・動員層・展開層）。書き込むのは docs/data/ew/src/ だけ。

源ごとに 1 つの JSON（docs/data/ew/src/<id>.json）に、前回までの値を残したまま新しい分を足す。
どの源も collectors/polite.py で robots.txt を確かめ、チャレンジ・403 なら止める（回り込まない）。
1 つの源の失敗は他の源を止めない（status.json の sources に結果を書く）。

源（2部の調査 2026-11-25、probe/ew2-robots で取得の形を確認）
- gold     国家外汇管理局「官方储备资产」の年ごとの xlsx（金の保有、万盎司）。月次
- oil      JODI-Oil World Database の年ごとの CSV（中国の原油の輸入、千バレル/日）。月次。中国は在庫（CLOSTLV）を出しておらず、
           生産＋輸入−輸出−精製投入（見かけの在庫の増え）もほぼ 0 に釣り合っている（probe で確認）ので、備蓄の積み増しの代わりに輸入を見る
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
import http.client
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
    r = polite.check_challenge(polite.raw_get(url, UA, 90, None, max_bytes + 1))
    if len(r.body) > max_bytes:
        raise RuntimeError(f'{url}: larger than {max_bytes} bytes (not cut short silently)')
    return r


def recent(doc: dict, days: int) -> bool:
    """Fetched within ``days`` (monthly sources are not fetched again every day)."""
    t = doc.get('retrieved_utc')
    return bool(t) and datetime.now(timezone.utc) - datetime.fromisoformat(t.replace('Z', '+00:00')) < timedelta(days=days)


def links(html: str) -> list[tuple[str, str]]:
    # the old rmrb layout (to 2024-11) writes href=nw.….htm without quotes
    out = re.findall(r'<a\s[^>]*?href=(?:"([^"]+)"|\'([^\']+)\'|([^\s>"\']+))[^>]*>(.*?)</a>', html, re.S | re.I)
    return [(a or b or c, re.sub(r'<[^>]+>|&nbsp;|\s+', ' ', t).strip()) for a, b, c, t in out]


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


def run_gold(years: int, force: bool = False) -> dict:
    doc = load('gold', {'unit': '万盎司', 'months': {}})
    if not force and recent(doc, 3):
        return {'months': len(doc['months']), 'skipped': 'fetched within 3 days'}
    idx = ok(fetch(SAFE_INDEX), SAFE_INDEX).body.decode('utf-8', 'replace')
    pages = [(h, t) for h, t in links(idx) if re.search(r'官方储备资产（\d{4}年）', t)]
    this = date.today().year
    for h, t in pages:
        y = int(re.search(r'(\d{4})年', t).group(1))
        if y < this - years:
            continue
        u = urllib.parse.urljoin(SAFE_INDEX, h)
        page = ok(fetch(u), u).body.decode('utf-8', 'replace')
        x = [u for u, _ in links(page) if u.lower().endswith('.xlsx')]
        if not x:
            continue
        u = urllib.parse.urljoin(SAFE_INDEX, x[-1])
        months = parse_gold(xlsx_rows(ok(fetch(u), u).body))
        keep(doc, months)
    doc.update(source='国家外汇管理局「官方储备资产」（月）', url=SAFE_INDEX, retrieved_utc=now_utc())
    save('gold', doc)
    return {'months': len(doc['months'])}


# ---------------------------------------------------------------- oil (JODI)
JODI_DIR = 'https://www.jodidata.org/_resources/files/downloads/oil-data/annual-csv/primary/'


def jodi_url(y: int, this_year: int) -> str:
    """The current year is primaryyearYYYY.csv, earlier years YYYY.csv (data-downloads.aspx, 2026-11-25)."""
    return JODI_DIR + (f'primaryyear{y}.csv' if y == this_year else f'{y}.csv')


def ok(r: polite.Response, url: str) -> polite.Response:
    if r.status != 200:
        raise RuntimeError(f'{url}: HTTP {r.status}')
    return r


def parse_jodi(text: str) -> dict[str, float]:
    """{'YYYY-MM': 原油の輸入（千バレル/日）} for China; a month is kept only when the value is a number."""
    out: dict[str, float] = {}
    for r in csv.DictReader(io.StringIO(text)):
        if (r.get('REF_AREA'), r.get('ENERGY_PRODUCT'), r.get('FLOW_BREAKDOWN'), r.get('UNIT_MEASURE')) != ('CN', 'CRUDEOIL', 'TOTIMPSB', 'KBD'):
            continue
        try:
            out[r['TIME_PERIOD']] = round(float(r['OBS_VALUE']), 1)
        except (TypeError, ValueError):
            pass
    return out


def keep(doc: dict, months: dict) -> None:
    """Store the months; the first value seen is kept beside the latest (the sources revise earlier months)."""
    for k, v in months.items():
        old = doc['months'].get(k) or {}
        doc['months'][k] = {'value': v, 'first_seen': old.get('first_seen') or date.today().isoformat(),
                            'first_value': old.get('first_value', old.get('value', v))}


def run_oil(years: int, force: bool = False) -> dict:
    doc = load('oil', {'unit': '千バレル/日', 'months': {}})
    if not force and recent(doc, 3):
        return {'months': len(doc['months']), 'skipped': 'fetched within 3 days'}
    for y in range(date.today().year - years, date.today().year + 1):
        url = jodi_url(y, date.today().year)
        r = fetch(url)
        if r.status == 404 and y == date.today().year:
            doc.setdefault('notes', {})[str(y)] = 'この年のファイルはまだ無い（1 月ごと）'
            continue
        keep(doc, parse_jodi(ok(r, url).body.decode('utf-8', 'replace')))
    doc.update(source='JODI-Oil World Database（中国・原油の輸入）', url='https://www.jodidata.org/oil/', retrieved_utc=now_utc())
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
    page = ok(fetch(JWC), JWC).body.decode('utf-8', 'replace')
    pdfs = sorted({(re.search(r'JWLA-?(\d+)', h, re.I).group(1), urllib.parse.urljoin(JWC, h))
                   for h, _ in links(page) if re.search(r'JWLA-?\d+.*\.pdf$', h, re.I)}, key=lambda x: int(x[0]))
    for num, url in pdfs[-2:]:
        key = f'JWLA-{int(num):03d}'
        if key in doc['circulars']:
            continue
        text = pdf_text(fetch(url).body)
        if not text.strip():
            raise RuntimeError(f'{key}: pdftotext returned no text')
        hits = sorted({m.group(0) for m in EAST_ASIA.finditer(text)})
        doc['circulars'][key] = {'url': url, 'first_seen': date.today().isoformat(), 'east_asia_terms': hits, 'chars': len(text)}
    doc.update(source="Lloyd's Market Association, Joint War Committee — Listed Areas", url=JWC, retrieved_utc=now_utc())
    save('jwc', doc)
    return {'circulars': len(doc['circulars'])}


# ---------------------------------------------------------------- People's Daily headlines
RMRB_PC = 'https://paper.people.com.cn/rmrb/pc/layout/{ym}/{d}/node_{p:02d}.html'
RMRB_OLD = 'https://paper.people.com.cn/rmrb/html/{y}-{m}/{d}/nbs.D110000renmrb_{p:02d}.htm'
PAGES = (1, 2, 3, 4)
MAX_DAYS = 1200   # 2023 からの大規模演習を過去事例の検証に入れるため（初回の取り込みだけ。毎日は 10 日）
RMRB_MINUTES = 200   # 1 回の実行で人民日報に使う時間の上限。残りは次の実行が続きから取る（既にある日は飛ばす）


def rmrb_day(day: date, get) -> list[dict] | None:
    """Headlines (title, page, url) of pages 1–4 of one day; None when the day has no edition in either format."""
    out, old = [], None
    for p in PAGES:
        if not old:
            url = RMRB_PC.format(ym=day.strftime('%Y%m'), d=day.strftime('%d'), p=p)
            r = get(url)
            pat = r'content_\d+\.html?$'
        if old or (r.status == 404 and p == 1):
            old = True   # the format is decided on page 1
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
            if re.search(pat, h) and t and not re.match(r'第\d+版', t) and u not in seen:
                seen.add(u)
                out.append({'page': p, 'title': t, 'url': u})
    return out   # [] : an edition without headlines on pages 1–4 (recorded as such, never counted as 0 matches)


# a broken connection on one day must not throw away the whole backfill (urllib.error.HTTPError is not here: it is a reply)
TRANSIENT = (http.client.IncompleteRead, http.client.RemoteDisconnected, ConnectionError, TimeoutError, urllib.error.URLError)
RMRB_MAX_FAILS = 5   # consecutive failed days: stop, keep what was fetched, and report stopped_before


def rmrb_day_retry(day: date, get=None, wait: float = 30) -> list[dict] | None:
    """rmrb_day, tried once more after a pause when the connection breaks."""
    get = get or fetch
    try:
        return rmrb_day(day, get)
    except TRANSIENT as e:
        if isinstance(e, urllib.error.HTTPError):
            raise
        time.sleep(wait)
        return rmrb_day(day, get)


def run_rmrb(days: int) -> dict:
    doc = load('rmrb', {'days': {}})
    today = datetime.now(timezone(timedelta(hours=8))).date()
    got, stopped, failed, skipped = 0, None, 0, []
    deadline = time.monotonic() + RMRB_MINUTES * 60
    for i in range(min(days, MAX_DAYS), -1, -1):
        day = today - timedelta(days=i)
        k = day.isoformat()
        if k in doc['days'] and i > 1:
            continue
        if time.monotonic() > deadline:
            stopped = k
            break
        try:
            heads = rmrb_day_retry(day)
        except TRANSIENT as e:
            if isinstance(e, urllib.error.HTTPError):
                raise
            failed += 1
            skipped.append(k)
            # retaken by a later run only if it reaches back this far (the daily run looks back --days 10); listed in the result
            print(f'::warning::rmrb {k}: {type(e).__name__}: {e} (skipped)')
            if failed >= RMRB_MAX_FAILS:
                stopped = k
                break
            continue
        failed = 0
        if heads == []:
            doc.setdefault('empty', [])
            if k not in doc['empty']:
                doc['empty'].append(k)
            continue
        if heads is not None:
            doc['days'][k] = heads
            if k in doc.get('empty', []):
                doc['empty'].remove(k)
            got += 1
            if got % 20 == 0:
                save('rmrb', doc)   # a long backfill keeps what it has fetched if the job is stopped
    doc.update(source='人民日報 電子版（1〜4 版の見出し。本文は取らない）', url='https://paper.people.com.cn/', retrieved_utc=now_utc())
    save('rmrb', doc)
    return {'days': len(doc['days']), 'new': got, **({'stopped_before': stopped} if stopped else {}), **({'skipped': skipped} if skipped else {})}


# ---------------------------------------------------------------- site feed keywords
FEED = 'https://observatory-watch.pages.dev/api/feeds'
FEED_WORDS = {
    'mobilization': re.compile(r'国防动员|國防動員|国防動員|人民防空|人防演练|民兵|支前|national defen[cs]e mobili[sz]ation|militia', re.I),
    'conscription': re.compile(r'征兵|徵兵|徴兵|延长服役|暂缓退役|延期退役|退役延期|conscription|demobili[sz]ation (delay|postpon|halt)', re.I),
    'cables': re.compile(r'海缆|海纜|海底电缆|海底電纜|海底光缆|海底光纜|海底ケーブル|submarine cable|undersea cable|subsea cable', re.I),
}
# 中国側の地名（国防動員・徴兵）。台湾側の動員・徴兵の記事は数えない
CHINA_SIDE = re.compile(r'福建|浙江|广东|廣東|江苏|山东|厦门|廈門|福州|宁波|寧波|温州|東部戦区|东部战区|南部战区|解放军|解放軍|中国军|中共|Fujian|Zhejiang|Guangdong|Eastern Theater|\bPLA\b|Chinese military|China\'s military')
# 年 2 回のいつもの徴兵・民兵の整組の報道は数えない
ROUTINE = re.compile(r'征兵工作会议|征兵开始|征兵工作启动|征兵体检|民兵整组|征兵宣传')
TAIWAN_SIDE = re.compile(r'台湾|台灣|臺灣|國軍|国軍|Taiwan', re.I)
CABLE_PLACE = re.compile(r'台湾|台灣|臺灣|金门|金門|马祖|馬祖|澎湖|沖縄|中国|中國|Taiwan|Matsu|Kinmen|Penghu|Okinawa|China|Chinese')


def feed_matches(feed: dict) -> dict[str, list]:
    out = {k: [] for k in FEED_WORDS}
    for r in feed.get('records', []):
        text = f"{r.get('title') or ''} {r.get('summary') or ''}"
        ts = r.get('publishedAt')
        day = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).date().isoformat() if isinstance(ts, (int, float)) else None
        for k, pat in FEED_WORDS.items():
            if k == 'cables':
                place = CABLE_PLACE.search(text)
            else:
                place = CHINA_SIDE.search(text) and not TAIWAN_SIDE.search(text) and not ROUTINE.search(text)
            if pat.search(text) and place and day:
                out[k].append({'date': day, 'title': (r.get('title') or '')[:200], 'url': r.get('url'), 'source': r.get('source')})
    return out


def run_feed() -> dict:
    doc = load('feed', {'items': {k: [] for k in FEED_WORDS}})
    # 自分のサイトの公開の API（exercises/sources.py と同じ読み方。robots.txt は無く、Pages は HTML を返すことがある）
    req = urllib.request.Request(FEED, headers={'Accept': 'application/json', 'User-Agent': UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        feed = json.loads(r.read(30_000_000).decode('utf-8'))
    run = datetime.now(timezone.utc)
    for k, items in feed_matches(feed).items():
        have = {i['url'] for i in doc['items'].setdefault(k, [])}
        doc['items'][k] += [i for i in items if i['url'] not in have]
        doc['items'][k].sort(key=lambda i: i['date'])
    # /api/feeds returns the last 24 hours; this run has seen [run − 24 h, run] (UTC, to the minute)
    doc['covered'] = merge_spans(doc.get('covered', []) + [[iso(run - timedelta(hours=FEED_WINDOW_H)), iso(run)]])
    doc.update(source='東アジア情勢観測の報道の一覧（/api/feeds、約 48 源）の見出しと要約の語', url=FEED, retrieved_utc=now_utc())
    save('feed', doc)
    return {k: len(v) for k, v in doc['items'].items()}


FEED_WINDOW_H = 24


def iso(t: datetime) -> str:
    return t.strftime('%Y-%m-%dT%H:%MZ')


def merge_spans(rs: list) -> list:
    """Merge time spans (ISO, UTC) that overlap or touch; a gap stays a gap."""
    out = []
    for a, b in sorted(rs):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


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


IODA_NAMES = {'4210': 'Lienchiang', '4209': 'Kinmen', '4208': 'Penghu'}


def run_ioda(days: int) -> dict:
    doc = load('ioda', {'regions': IODA_REGIONS, 'events': [], 'covered': []})
    if not doc.get('verified'):
        for code, name in IODA_NAMES.items():
            q = urllib.parse.urlencode({'entityType': 'region', 'search': name})
            u = f'https://api.ioda.inetintel.cc.gatech.edu/v2/entities/query?{q}'
            found = json.loads(ok(fetch(u), u).body.decode('utf-8')).get('data') or []
            if not any(e.get('code') == code and e.get('name') == name and (e.get('attrs') or {}).get('country_code') == 'TW' for e in found):
                raise RuntimeError(f'IODA region {code} is not {name} (TW)')
        doc['verified'] = {'at': now_utc(), 'names': IODA_NAMES}
    until = int(time.time())
    since = until - min(days, MAX_DAYS) * 86400
    have = {(e['region'], e['start'], e['datasource']) for e in doc['events']}
    for code in IODA_REGIONS:
        start = since
        while start < until:
            end = min(until, start + 90 * 86400)
            q = urllib.parse.urlencode({'entityType': 'region', 'entityCode': code, 'from': start, 'until': end, 'limit': 500})
            body = json.loads(ok(fetch(f'{IODA}?{q}'), f'{IODA}?{q}').body.decode('utf-8'))
            if body.get('error'):
                raise RuntimeError(f"IODA error: {str(body['error'])[:120]}")
            for e in body.get('data') or []:
                key = (code, int(e['start']), e.get('datasource'))
                if key not in have:
                    have.add(key)
                    doc['events'].append({'region': code, 'start': int(e['start']), 'duration': int(e.get('duration') or 0),
                                          'datasource': e.get('datasource'), 'score': round(float(e.get('score') or 0), 1)})
            start = end
    doc['events'].sort(key=lambda e: (e['start'], e['region']))
    ts = lambda t: iso(datetime.fromtimestamp(t, tz=timezone.utc))  # noqa: E731
    doc['covered'] = merge_spans(doc['covered'] + [[ts(since), ts(until)]])
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
    end = date.today() - timedelta(days=5)   # GFW presence comes about 4 days late; the last days are not complete
    start = end - timedelta(days=min(days, MAX_DAYS))
    for zid, z in ZONES.items():
        series = doc['daily'].setdefault(zid, {})
        s = start
        while s <= end:
            e = min(end, s + timedelta(days=180))
            got = gfw_hours(token, z['bbox'], s, e)
            if not got:
                raise RuntimeError(f'GFW: no rows for {zid} {s}..{e}')
            for k, v in got.items():
                series[k] = round(v, 1)   # a day without a row is left out (unknown), never written as 0
            s = e + timedelta(days=1)
            time.sleep(PAUSE)
    doc['covered'] = merge_ranges(doc['covered'] + [[start.isoformat(), end.isoformat()]])
    doc.update(source='Global Fishing Watch 4Wings public-global-presence:latest（旅客船、格子の合計の時間）', url='https://globalfishingwatch.org/',
               attribution='Powered by Global Fishing Watch. CC BY-NC 4.0', retrieved_utc=now_utc())
    save('ferries', doc)
    return {z: len(v) for z, v in doc['daily'].items()}


# ---------------------------------------------------------------- main
RUNNERS = {
    'gold': lambda a: run_gold(a.years, a.force),
    'oil': lambda a: run_oil(a.years, a.force),
    'jwc': lambda a: run_jwc(),
    'rmrb': lambda a: run_rmrb(a.days),
    'feed': lambda a: run_feed(),
    'ioda': lambda a: run_ioda(a.days),
    'ferries': lambda a: run_ferries(a.days),
}


def set_pause(seconds: float) -> None:
    global PAUSE
    PAUSE = seconds


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('sources', nargs='*', help='源（空ならすべて）: ' + ' '.join(RUNNERS))
    ap.add_argument('--force', action='store_true', help='月次の源も取り直す（さかのぼり用）')
    ap.add_argument('--pause', type=float, default=PAUSE, help='取得の間隔（秒）。長いさかのぼりでは広げる')
    ap.add_argument('--days', type=int, default=10, help='さかのぼる日数（rmrb・ioda・ferries）')
    ap.add_argument('--years', type=int, default=1, help='さかのぼる年数（gold・oil）')
    a = ap.parse_args(argv)
    set_pause(a.pause)
    a.sources = a.sources or list(RUNNERS)
    bad = [x for x in a.sources if x not in RUNNERS]
    if bad:
        ap.error('unknown sources: ' + ' '.join(bad))
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
        except polite.SourceBlocked as e:
            status['sources'][name] = {'status': 'blocked', 'at': now_utc(), 'reason': str(e)[:200]}
            print(f'::warning title=ew2 {name}::blocked {e}')
            failed += 1
        except Exception as e:  # noqa: BLE001 - one source must not stop the others
            status['sources'][name] = {'status': 'error', 'at': now_utc(), 'reason': f'{type(e).__name__}: {str(e)[:200]}'}
            print(f'::warning title=ew2 {name}::{type(e).__name__}: {str(e)[:200]}')
            failed += 1
    save('status', status)
    return 1 if failed == len(a.sources) else 0   # every source failed: make the run red


if __name__ == '__main__':
    sys.exit(main())
