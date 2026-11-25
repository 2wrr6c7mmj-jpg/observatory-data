"""ew-2 の情報源の確認（probe 専用）：81.cn の電子版（JS の画面）が読むデータの場所を探す。本文は保存しない。読み取り専用。"""
import re
import sys
from collectors import polite

UA = 'observatory-data/1'


def show(url, pat=None, n=900):
    r = polite.check_robots(url, UA)
    if not r.allowed:
        print(f'::notice title=ew2 {url}::robots {r.reason}')
        return ''
    resp = polite.check_challenge(polite.raw_get(url, UA, 30, None, 3_000_000))
    txt = resp.body.decode('utf-8', 'replace')
    found = sorted(set(re.findall(pat, txt))) if pat else [re.sub(r'\s+', ' ', txt)]
    print(f'::notice title=ew2 {url[:60]}::{resp.status} {" | ".join(found)[:n]}')
    return txt


def main():
    page = show('http://www.81.cn/szb_223187/index.html', None, 1800)
    for src in re.findall(r'src=["\']([^"\']+\.js[^"\']*)', page)[:4]:
        url = src if src.startswith('http') else 'http://www.81.cn' + (src if src.startswith('/') else '/szb_223187/' + src)
        show(url, r'["\'](/?[A-Za-z0-9_./-]*(?:json|api|szb|paper|Date|jfjb|gfb)[A-Za-z0-9_./?=&-]*)["\']', 900)
    show('http://www.81.cn/ysym/jfjbkhdxzym/index.html', r'href=["\']([^"\']+)', 600)
    return 0


if __name__ == '__main__':
    sys.exit(main())
