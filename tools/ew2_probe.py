"""ew-2 の情報源の確認（probe 専用）：各源のデータの形。本文は保存しない。読み取り専用。"""
import csv
import io
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from collectors import polite

UA = 'observatory-data/1'


def note(t, m):
    print(f'::notice title=ew2 {t}::{m[:1900]}')


def fetch(url, n=6_000_000):
    polite.require_allowed(url, UA)
    return polite.check_challenge(polite.raw_get(url, UA, 60, None, n))


def links(url, pat):
    r = fetch(url)
    txt = r.body.decode('utf-8', 'replace')
    out = re.findall(r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', txt, re.S)
    return r.status, [(h, re.sub(r'<[^>]+>|\s+', ' ', t).strip()) for h, t in out if re.search(pat, h + ' ' + t, re.I)]


def safe():
    st, ls = links('https://www.safe.gov.cn/en/ForexReserves/index.html', r'reserve|储备|\d{4}/\d{4}/')
    note('safe-index', f'{st} ' + ' | '.join(f'{t}={h}' for h, t in ls[:30]))
    st, ls = links('https://www.safe.gov.cn/safe/whcb/index.html', r'储备|\d{4}/\d{4}/')
    note('safe-cn-index', f'{st} ' + ' | '.join(f'{t}={h}' for h, t in ls[:30]))


def xlsx_rows(data):
    import zipfile
    import xml.etree.ElementTree as ET
    z = zipfile.ZipFile(io.BytesIO(data))
    ns = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    ss = []
    if 'xl/sharedStrings.xml' in z.namelist():
        for si in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('m:si', ns):
            ss.append(''.join(t.text or '' for t in si.iter('{%s}t' % ns['m'])))
    sheet = sorted(n for n in z.namelist() if n.startswith('xl/worksheets/sheet'))[0]
    rows = []
    for row in ET.fromstring(z.read(sheet)).iter('{%s}row' % ns['m']):
        vals = []
        for c in row.findall('m:c', ns):
            v = c.find('m:v', ns)
            t = v.text if v is not None else ''
            if c.get('t') == 's' and t:
                t = ss[int(t)]
            vals.append(t)
        rows.append(vals)
    return rows


def safe2():
    for page in ('https://www.safe.gov.cn/safe/2026/0205/27113.html', 'https://www.safe.gov.cn/safe/2020/0207/22234.html'):
        st, ls = links(page, r'\.(xlsx?|pdf)')
        note('safe-year', f'{st} {page} ' + ' | '.join(f'{t}={h}' for h, t in ls[:12]))
        x = [h for h, t in ls if h.endswith('.xlsx')]
        if x:
            url = x[-1] if x[-1].startswith('http') else 'https://www.safe.gov.cn' + x[-1]
            rows = xlsx_rows(fetch(url).body)
            hit = [r for r in rows if any(('盎司' in c) or ('ounce' in c.lower()) for c in r)]
            note('safe-oz', f'{len(rows)} rows; ' + ' || '.join(' ; '.join(r[:9]) for r in hit[:4]))
            note('safe-cols', ' ; '.join(rows[3][-6:]) + f' ncols={max(len(r) for r in rows)}')


def jwc():
    st, ls = links('https://lmalloyds.com/committee/joint-war-committee/', r'jwla|listed|\.pdf|circular')
    note('jwc', f'{st} ' + ' | '.join(f'{t}={h}' for h, t in ls[:30]))


def jodi():
    r = fetch('https://www.jodidata.org/_resources/files/downloads/oil-data/annual-csv/primary/primaryyear2026.csv', 40_000_000)
    rows = [x for x in csv.DictReader(io.StringIO(r.body.decode('utf-8', 'replace'))) if x['REF_AREA'] == 'CN']
    keys = sorted({(x['ENERGY_PRODUCT'], x['FLOW_BREAKDOWN'], x['UNIT_MEASURE']) for x in rows})
    note('jodi-keys', f'{len(rows)} rows; ' + ' | '.join('/'.join(k) for k in keys[:40]))
    cs = [x for x in rows if x['ENERGY_PRODUCT'] == 'CRUDEOIL' and x['FLOW_BREAKDOWN'] in ('CLOSTLV', 'STOCKCH', 'INDPROD', 'TOTIMPSB')]
    note('jodi-cn-crude', ' | '.join(f"{x['TIME_PERIOD']} {x['FLOW_BREAKDOWN']} {x['UNIT_MEASURE']} {x['OBS_VALUE']} {x.get('ASSESSMENT_CODE','')}" for x in cs[:40]))


def rmrb():
    st, ls = links('https://paper.people.com.cn/rmrb/pc/layout/202611/24/node_01.html', r'content|node_')
    note('rmrb-01', f'{st} ' + ' | '.join(f'{t}={h}' for h, t in ls[:40]))
    st, ls = links('https://paper.people.com.cn/rmrb/pc/layout/202401/26/node_01.html', r'content|node_')
    note('rmrb-2024', f'{st} {len(ls)} links; ' + ' | '.join(f'{t}={h}' for h, t in ls[:6]))


def ioda():
    for q in ('Lienchiang', 'Kinmen', 'Penghu', 'Taiwan'):
        r = fetch('https://api.ioda.inetintel.cc.gatech.edu/v2/entities/query?' + urllib.parse.urlencode({'entityType': 'region', 'search': q}))
        note(f'ioda-{q}', r.body.decode()[:600])
    r = fetch('https://api.ioda.inetintel.cc.gatech.edu/v2/outages/events?' + urllib.parse.urlencode({'entityType': 'country', 'entityCode': 'TW', 'from': 1735689600, 'until': 1738368000}))
    note('ioda-events-tw-2025-01', r.body.decode()[:900])


def gfw():
    tok = os.environ.get('GFW_API_TOKEN', '').strip()
    if not tok:
        note('gfw', 'no token')
        return
    poly = {'type': 'Polygon', 'coordinates': [[[119.4, 24.2], [120.2, 24.2], [120.2, 25.6], [119.4, 25.6], [119.4, 24.2]]]}
    for flt in ("vessel_type in ('passenger')", None):
        q = {'spatial-resolution': 'LOW', 'temporal-resolution': 'DAILY', 'datasets[0]': 'public-global-presence:latest',
             'date-range': '2026-10-01,2026-10-08', 'format': 'JSON'}
        if flt:
            q['filters[0]'] = flt
        req = urllib.request.Request('https://gateway.api.globalfishingwatch.org/v3/4wings/report?' + urllib.parse.urlencode(q),
                                     data=json.dumps({'geojson': poly}).encode(), method='POST',
                                     headers={'Authorization': 'Bearer ' + tok, 'Content-Type': 'application/json', 'User-Agent': UA})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                doc = json.loads(r.read())
            ent = doc.get('entries') or []
            rows = list(ent[0].values())[0] if ent else []
            note(f'gfw {flt}', f'rows={len(rows)} sample={json.dumps(rows[:3])[:500]}')
        except urllib.error.HTTPError as e:
            note(f'gfw {flt}', f'HTTP {e.code} {e.read()[:300]!r}')


def main():
    fn = {'safe': safe2, 'jwc': jwc, 'jodi': jodi, 'rmrb': rmrb, 'ioda': ioda, 'gfw': gfw}[sys.argv[1]]
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        note(sys.argv[1], f'ERROR {type(e).__name__}: {str(e)[:300]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
