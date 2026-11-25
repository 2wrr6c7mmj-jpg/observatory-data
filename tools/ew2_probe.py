"""ew-2 の情報源の確認（probe 専用）：robots.txt と、1 回だけの取得の結果（状態・種類・大きさ・チャレンジ）を注釈で返す。
本文は保存しない。読み取り専用。"""
import sys
from collectors import polite

UA = 'observatory-data/1'
URLS = [
    ('gold-safe', 'https://www.safe.gov.cn/safe/2026/0307/26776.html'),
    ('gold-safe-pdf', 'https://www.safe.gov.cn/safe/file/file/20260307/cd68437d63ad4627a7a37aeefab6002b.pdf'),
    ('gold-safe-en', 'https://www.safe.gov.cn/en/2018/0517/1487.html'),
    ('gold-bazg', 'https://www.bazg.admin.ch/en/swiss-foreign-trade-statistics-gold-silver-and-coins'),
    ('jodi-csv', 'https://www.jodidata.org/_resources/files/downloads/oil-data/annual-csv/primary/primaryyear2026.csv'),
    ('jodi-page', 'https://www.jodidata.org/oil/database/data-downloads.aspx'),
    ('jwc', 'https://lmalloyds.com/committee/joint-war-committee/'),
    ('rmrb', 'http://paper.people.com.cn/rmrb/html/2026-11/24/nbs.D110000renmrb_01.htm'),
    ('rmrb-https', 'https://paper.people.com.cn/rmrb/html/2026-11/24/nbs.D110000renmrb_01.htm'),
    ('jfjb', 'http://www.81.cn/jfjbmap/content/2026-11/24/node_2.htm'),
    ('gfb', 'http://www.81.cn/gfbmap/content/2026-11/24/node_2.htm'),
    ('airplanes-live', 'https://api.airplanes.live/v2/point/24.5/119.5/150'),
    ('airplanes-terms', 'https://airplanes.live/terms-of-service/'),
    ('ioda', 'https://api.ioda.inetintel.cc.gatech.edu/v2/entities/query?entityType=region&search=Lienchiang'),
    ('ioda-site', 'https://ioda.inetintel.cc.gatech.edu/'),
]


def main():
    for name, url in URLS:
        r = polite.check_robots(url, UA)
        line = f'{name}: robots={"allowed" if r.allowed else "DISALLOWED"} verified={r.verified} ({r.reason})'
        if r.allowed:
            try:
                resp = polite.check_challenge(polite.raw_get(url, UA, 30, None, 2_000_000))
                ctype = resp.headers.get('content-type', '') if hasattr(resp, 'headers') else ''
                body = resp.body if hasattr(resp, 'body') else b''
                line += f' get={resp.status} type={ctype[:40]} bytes={len(body)} head={body[:80]!r}'
            except polite.Challenge as e:
                line += f' get=CHALLENGE {e}'
            except Exception as e:  # noqa: BLE001
                line += f' get=ERROR {type(e).__name__}: {str(e)[:120]}'
        print(f'::notice title=ew2 {name}::{line}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
