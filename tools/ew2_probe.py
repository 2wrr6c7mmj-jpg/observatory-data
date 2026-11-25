"""ew-2 の情報源の確認（probe 専用）：robots.txt と、1 回だけの取得の結果（状態・種類・大きさ・チャレンジ）を注釈で返す。
本文は保存しない。読み取り専用。"""
import sys
from collectors import polite

UA = 'observatory-data/1'
URLS = [
    ('rmrb-pc', 'https://paper.people.com.cn/rmrb/pc/layout/202611/24/node_01.html'),
    ('rmrb-root', 'https://paper.people.com.cn/'),
    ('81-szb', 'http://www.81.cn/szb_223187/index.html'),
    ('81-root', 'http://www.81.cn/'),
    ('gfb-old', 'http://www.81.cn/gfbmap/content/2026-11/24/node_2.htm'),
    ('airplanes-live', 'https://api.airplanes.live/v2/point/24.5/119.5/150'),
    ('airplanes-terms', 'https://airplanes.live/terms-of-service/'),
    ('airplanes-guide', 'https://airplanes.live/api-guide/'),
    ('ioda', 'https://api.ioda.inetintel.cc.gatech.edu/v2/entities/query?entityType=region&search=Lienchiang'),
    ('ioda-site', 'https://ioda.inetintel.cc.gatech.edu/'),
]


def main():
    part = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    for name, url in URLS[part * 8:(part + 1) * 8]:
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
        if name.endswith(('-root', '-szb', '-guide', '-terms')) and r.allowed:
            import re
            try:
                txt = resp.body.decode('utf-8', 'replace')
                links = sorted(set(re.findall(r'href=["\']([^"\']+)', txt)))
                keep = [l for l in links if any(k in l.lower() for k in ('jfjb', 'gfb', 'layout', 'szb', 'term', 'licen', 'api', 'commercial'))][:25]
                print(f'::warning title=ew2 {name} links::{" | ".join(keep)[:900]}')
            except Exception:  # noqa: BLE001
                pass
    return 0


if __name__ == '__main__':
    sys.exit(main())
