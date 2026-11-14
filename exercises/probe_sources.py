"""Probe only: can Actions reach more official sources? robots.txt, status, sample links.
Prints notices; changes nothing."""
import re
import sys

from collectors import polite

from .sources import UA, list_links

CANDIDATES = [
    ("陸上幕僚監部", "https://www.mod.go.jp/gsdf/news/press/index.html", r"\.pdf|/press/"),
    ("航空幕僚監部", "https://www.mod.go.jp/asdf/news/release/index.html", r"\.pdf|/release/"),
    ("防衛省 報道資料", "https://www.mod.go.jp/j/press/news/index.html", r"\.pdf|/press/news/"),
    ("中国国防部 发言人", "http://www.mod.gov.cn/gfbw/xwfyr/index.html", r"\.html"),
    ("中国国防部 军事演习", "http://www.mod.gov.cn/gfbw/jsxd/index.html", r"\.html"),
    ("中国軍網 英語", "http://eng.chinamil.com.cn/", r"\.html"),
    ("豪国防省", "https://www.defence.gov.au/news-events/releases", r"/news-events/releases/"),
    ("INDOPACOM", "https://www.pacom.mil/Media/News/", r"/Article/"),
    ("米第7艦隊", "https://www.c7f.navy.mil/Media/News/", r"/Article/"),
    ("Stars and Stripes Pacific", "https://www.stripes.com/theaters/asia_pacific/", r"/theaters/asia_pacific/20"),
    ("フィリピン PNA", "https://www.pna.gov.ph/categories/national", r"/articles/"),
    ("インド PIB 国防 RSS", "https://pib.gov.in/RssMain.aspx?ModId=6&Lang=1&Regid=3", r"PRID="),
    ("韓国 聯合 英語 北朝鮮", "https://en.yna.co.kr/RSS/northkorea.xml", r"/view/"),
]


def main():
    for name, url, href in CANDIDATES:
        try:
            r = polite.get(url, UA, timeout=30)
            links = list_links(r.body.decode("utf-8", "replace"), url, href) if r.status == 200 else []
            sample = " | ".join(f"{t[:40]} {u[-60:]}" for u, t, d in links[:4])
            print(f"::notice title=probe {name}::HTTP {r.status}, {len(r.body)} bytes, {len(links)} links. {sample}"[:900])
        except polite.SourceBlocked as e:
            print(f"::warning title=probe {name}::blocked: {str(e)[:300]}")
        except Exception as e:  # noqa: BLE001
            print(f"::warning title=probe {name}::error: {str(e)[:300]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
