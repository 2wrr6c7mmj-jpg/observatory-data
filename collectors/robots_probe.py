"""One-off check of robots.txt on candidate sources (probe only, no page bodies).

For each host, robots.txt is read once with the collectors' User-Agent and the
outcome is printed as GitHub Actions annotations: HTTP status, a summary of
the rules that apply to us, and whether the main paths may be fetched. Nothing
is written to the repository.

The only page fetched besides robots.txt is the data.gov.tw dataset page
(``DATASET_URL``), once, and only when robots.txt allows it; its provider,
licence and download links are extracted for the annotation.

    python -m collectors.robots_probe            # all hosts
    python -m collectors.robots_probe --no-dataset
"""
from __future__ import annotations

import argparse
import html
import http.client
import re
import socket
import time
import urllib.error
import urllib.robotparser

from collectors import polite
from collectors.taiwan_mnd import UA

TIMEOUT = 25

# host -> main paths to judge
HOSTS = {
    "data.gov.tw": ["/", "/dataset/138118", "/datasets/search", "/api/v2/rest/dataset/138118"],
    "www.msa.gov.cn": ["/", "/html/xxgk/", "/html/hxaq/hxgg/", "/page/outter/weather.jsp"],
    "www.mod.gov.cn": ["/", "/gfbw/", "/gfbw/xwfyr/", "/gfbw/qwfb/"],
    "eng.chinamil.com.cn": ["/", "/CHINA_209163/", "/view/", "/node_"],
    "www.news.cn": ["/", "/english/", "/politics/", "/mil/", "/world/"],
}
DATASET_URL = "https://data.gov.tw/dataset/138118"


def esc(s: str) -> str:
    return s.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def annotate(level: str, title: str, text: str) -> None:
    print(f"::{level} title={title}::{esc(text[:3900])}")


def rule_summary(text: str, max_lines: int = 40) -> str:
    """User-agent groups that matter (``*`` and anything resembling us), plus sitemaps."""
    groups: list[tuple[list[str], list[str]]] = []
    sitemaps: list[str] = []
    agents: list[str] = []
    rules: list[str] = []
    last_was_rule = False
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, val = (x.strip() for x in line.split(":", 1))
        k = key.lower()
        if k == "user-agent":
            if last_was_rule:
                groups.append((agents, rules))
                agents, rules = [], []
            agents.append(val)
            last_was_rule = False
        elif k == "sitemap":
            sitemaps.append(val)
        else:
            rules.append(f"{key}: {val}")
            last_was_rule = True
    if agents or rules:
        groups.append((agents, rules))
    out = [f"グループ {len(groups)}（UA: " + ", ".join(sorted({a for g in groups for a in g[0]}))[:600] + "）"]
    for ags, rs in groups:
        low = [a.lower() for a in ags]
        if "*" in low or any("observatory" in a for a in low):
            out.append(f"[{', '.join(ags)}] " + (" / ".join(rs[:max_lines]) or "（規則なし）")
                       + (f" …他 {len(rs) - max_lines} 行" if len(rs) > max_lines else ""))
    if sitemaps:
        out.append(f"Sitemap {len(sitemaps)} 件：" + ", ".join(sitemaps[:3]))
    return "\n".join(out)


def check_host(host: str, paths: list[str], getter=None) -> dict:
    getter = getter or polite.raw_get
    url = f"https://{host}/robots.txt"
    res = {"host": host, "url": url}
    try:
        r = getter(url, UA, TIMEOUT, None, polite.ROBOTS_MAX_BYTES)
    except (urllib.error.URLError, TimeoutError, socket.timeout, ConnectionError,
            http.client.HTTPException, OSError) as e:
        res.update(result=f"届かない（{type(e).__name__}: {e}）", verdict="確認できない（取得しない）")
        return res
    text = r.body.decode("utf-8", errors="replace")
    ctype = r.headers.get("content-type", "")
    res["http"] = (f"HTTP {r.status}、{ctype or 'content-type なし'}、{len(r.body)} バイト"
                   + (f"、Last-Modified {r.headers['last-modified']}" if r.headers.get("last-modified") else "")
                   + (f"、最終 URL {r.url}" if r.url != url else ""))
    if polite.is_challenge(r):
        res.update(result="ボット対策のチャレンジページ", verdict="確認できない（取得しない）",
                   excerpt=text[:300])
        return res
    if r.status in (401, 403):
        res.update(result=f"HTTP {r.status}（全拒否として扱う）", verdict="不可", excerpt=text[:300])
        return res
    if r.status in (404, 410):
        res.update(result=f"HTTP {r.status}（robots.txt なし＝全許可）",
                   verdict="可（robots.txt なし）", paths={p: True for p in paths})
        return res
    if r.status != 200:
        res.update(result=f"HTTP {r.status}", verdict="確認できない（取得しない）", excerpt=text[:300])
        return res
    if "html" in ctype.lower() or text.lstrip()[:1] == "<":
        res.update(result="robots.txt の代わりに HTML が返った", verdict="確認できない（取得しない）",
                   excerpt=re.sub(r"\s+", " ", text[:400]))
        return res
    rp = urllib.robotparser.RobotFileParser(url)
    rp.parse(text.splitlines())
    # polite.get() below reuses this parse instead of reading robots.txt again
    polite._cache[polite.robots_url(url)] = (time.time(), rp)
    judged = {p: rp.can_fetch(UA, f"https://{host}{p}") for p in paths}
    res.update(result="読めた", summary=rule_summary(text), paths=judged,
               crawl_delay=rp.crawl_delay(UA),
               verdict="可" if all(judged.values()) else ("不可" if not any(judged.values()) else "一部可"))
    return res


def describe(res: dict) -> str:
    lines = [f"{res['url']}：{res['result']}"]
    if res.get("http"):
        lines.append(res["http"])
    lines.append(f"判定（UA {UA}）：{res['verdict']}")
    for p, ok in (res.get("paths") or {}).items():
        lines.append(f"  {p} → {'許可' if ok else '拒否'}")
    if res.get("crawl_delay"):
        lines.append(f"Crawl-delay {res['crawl_delay']}")
    if res.get("summary"):
        lines.append(res["summary"])
    if res.get("excerpt"):
        lines.append("本文の冒頭：" + res["excerpt"])
    return "\n".join(lines)


def _text(s: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s))).strip()


def dataset_summary(body: str) -> str:
    """Provider, licence and download links from the data.gov.tw dataset page."""
    out = []
    t = re.search(r"<title>(.*?)</title>", body, re.S | re.I)
    if t:
        out.append("題名：" + _text(t.group(1))[:200])
    plain = _text(body)
    for label in ("提供機關", "提供機關聯絡人", "授權方式", "授權說明網址", "計費方式", "更新頻率",
                  "資料集描述", "主要欄位說明", "上架日期", "詮釋資料更新時間", "檔案格式"):
        m = re.search(re.escape(label) + r"\s*[:：]?\s*(.{0,160})", plain)
        if m:
            out.append(f"{label}：{m.group(1)}")
    links = sorted(set(re.findall(r"""https?://[^\s"'<>]+""", body)))
    dl = [u for u in links if re.search(r"\.(csv|json|xml|zip|xlsx?|ods)\b|download|odws|opendata", u, re.I)
          and "data.gov.tw/dataset/" not in u]
    out.append(f"ダウンロード先の候補 {len(dl)} 件：" + (" ".join(dl[:15]) or "なし"))
    if len(out) <= 2:
        desc = re.search(r'<meta[^>]+name="description"[^>]+content="([^"]*)"', body, re.I)
        out.append("meta description：" + (html.unescape(desc.group(1))[:400] if desc else "なし"))
        out.append("本文（タグ除去）の冒頭：" + plain[:600])
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-dataset", action="store_true")
    a = ap.parse_args(argv)
    table = []
    results = {}
    for host, paths in HOSTS.items():
        res = check_host(host, paths)
        results[host] = res
        annotate("notice", f"robots {host}", describe(res))
        table.append(f"{host}: {res['verdict']}（{res['result'][:80]}）")
    annotate("notice", "robots 判定一覧", "\n".join(table))
    if a.no_dataset:
        return 0
    ds = results["data.gov.tw"]
    if not (ds.get("paths") or {}).get("/dataset/138118"):
        annotate("warning", "data.gov.tw dataset", f"{DATASET_URL} は robots で許可されていない／確認できないため取得しない")
        return 0
    try:
        r = polite.get(DATASET_URL, UA, TIMEOUT)
    except polite.SourceBlocked as e:
        annotate("warning", "data.gov.tw dataset", f"取得しない：{e}")
        return 0
    except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException, OSError) as e:
        annotate("warning", "data.gov.tw dataset", f"届かない：{type(e).__name__}: {e}")
        return 0
    body = r.body.decode("utf-8", errors="replace")
    annotate("notice", "data.gov.tw dataset",
             f"{DATASET_URL}：HTTP {r.status}、{r.headers.get('content-type', '')}、{len(r.body)} バイト\n"
             + dataset_summary(body))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
