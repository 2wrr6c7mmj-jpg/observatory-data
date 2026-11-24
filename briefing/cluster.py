"""過去 24 時間の記事を出来事ごとに束ね（cluster）、独立した出典の数・国の数で選ぶ。LLM は使わない。

束ね：見出し（日本語訳があればそれ、なければ原文）の名詞・固有名詞・英単語を IDF で重み付けした cos 類似度が
しきい値以上の記事を、union-find で 1 つの束にする。形態素解析は fugashi（あれば）、なければ正規表現の代用
（試験用。本番は fugashi を使い、使った方式を json に書く）。
選び方：独立した出典（系列）の数 → 出典の国の数 → 東アジア → 軍事・紛争 → 記事数 の順。
  - 系列は briefing/sources.py の表。同じ統制下・同じ会社・同じ通信社・同じ発表の再掲は 1 つ。
  - 系列が MIN_FAMILIES 未満の束は選ばない（1 つの出典だけの話は「まとめ」に入れない）。
"""
import collections
import math
import re

from . import sources as S

WINDOW_MS = 24 * 3600 * 1000
FUTURE_SLACK_MS = 12 * 3600 * 1000   # 配信元の時刻の解釈のずれで少し先の時刻が付く記事がある
THRESHOLD = 0.45
MIN_FAMILIES = 2
EAST_ASIA = {'CN', 'HK', 'MO', 'TW', 'JP', 'KR', 'KP', 'PH'}
MAX_DF_RATE = 0.1   # 全記事の 1 割より多くに出る語は、候補の絞り込みに使わない

LATIN = re.compile(r'[A-Za-z][A-Za-z0-9\-]{2,}')
_FALLBACK = re.compile(r'[一-龠々]{2,}|[ァ-ヶー]{2,}|\d{2,}')

_tagger = None


def _fugashi_tagger():
    global _tagger
    if _tagger is None:
        from fugashi import Tagger  # 遅い import
        _tagger = Tagger()
    return _tagger


def tokens_mecab(text):
    out = set()
    for w in _fugashi_tagger()(text):
        f, s = w.feature, w.surface
        if f.pos1 != '名詞':
            continue
        if f.pos2 == '固有名詞' or (f.pos2 == '普通名詞' and len(s) >= 2) or (f.pos2 == '数詞' and s.isdigit() and len(s) >= 2):
            out.add(s)
    out |= {m.group(0).lower() for m in LATIN.finditer(text)}
    return out


def tokens_fallback(text):
    """形態素解析なしの代用：漢字の連なり・カタカナの連なり・2 桁以上の数・英単語。"""
    out = {m.group(0) for m in _FALLBACK.finditer(text)}
    out |= {m.group(0).lower() for m in LATIN.finditer(text)}
    return out


def default_tokenizer():
    """(関数, 方式名)。fugashi があればそれ。"""
    try:
        _fugashi_tagger()
        return tokens_mecab, 'fugashi'
    except Exception:
        return tokens_fallback, 'regex-fallback'


def in_window(records, now_ms):
    """過去 24 時間の記事（URL か id の重複は 1 件）。公開時刻の無い記事は入れない。"""
    out, seen = [], set()
    for r in records:
        if not isinstance(r, dict):
            continue
        t = r.get('publishedAt')
        if not isinstance(t, (int, float)) or isinstance(t, bool):
            continue
        if not now_ms - WINDOW_MS <= t <= now_ms + FUTURE_SLACK_MS:
            continue
        k = r.get('url') or r.get('id')
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def head(r):
    return (r.get('titleJa') or r.get('title') or '').strip()


def cluster(records, tokenizer=None, threshold=THRESHOLD):
    """記事の列 → 束（記事の列）の列。"""
    tok = tokenizer or default_tokenizer()[0]
    docs = [(r, tok(head(r))) for r in records if head(r)]
    n = len(docs)
    df = collections.Counter(t for _, ts in docs for t in ts)
    idf = {t: math.log(n / c) for t, c in df.items()}
    vec = [{t: idf[t] for t in ts} for _, ts in docs]
    norm = [math.sqrt(sum(v * v for v in d.values())) or 1.0 for d in vec]
    post = collections.defaultdict(list)
    for i, d in enumerate(vec):
        for t in d:
            if df[t] <= n * MAX_DF_RATE:
                post[t].append(i)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        cand = collections.Counter(j for t in vec[i] for j in post.get(t, ()) if j > i)
        for j, shared in cand.items():
            if shared < 2:
                continue
            cos = sum(vec[i][t] * vec[j][t] for t in vec[i].keys() & vec[j].keys()) / (norm[i] * norm[j])
            if cos >= threshold:
                parent[find(i)] = find(j)
    groups = collections.defaultdict(list)
    for i in range(n):
        groups[find(i)].append(docs[i][0])
    return list(groups.values())


def describe(recs):
    """束の指標：系列・国・東アジア・軍事。"""
    fams = sorted({S.family(r.get('source')) for r in recs})
    ctry = sorted({c for c in (S.country(r.get('source')) for r in recs) if c})
    scopes = {r.get('scope') for r in recs}
    cats = collections.Counter(r.get('category') for r in recs)
    return {
        'families': fams,
        'countries': ctry,
        'east_asia': bool(scopes & EAST_ASIA),
        'military': bool(cats) and cats.most_common(1)[0][0] in ('military', 'conflict'),
        'size': len(recs),
    }


def rank_key(d):
    return (-len(d['families']), -len(d['countries']), not d['east_asia'], not d['military'], -d['size'])


def select(clusters, min_families=MIN_FAMILIES):
    """選ぶ候補（順位つき）：[{'records':…, 'stats':…}]。系列が min_families 未満の束は除く。"""
    rows = []
    for recs in clusters:
        d = describe(recs)
        if len(d['families']) >= min_families:
            rows.append({'records': recs, 'stats': d})
    rows.sort(key=lambda x: rank_key(x['stats']))
    return rows


def representatives(recs, limit=6):
    """束から台本に渡す記事：系列ごとに最新 1 件、系列数の多い順でなく新しい順、最大 limit 件。
    一次発表・国営の記事は、独立した報道が先に入るように後回しにする（同じ新しさなら）。"""
    best = {}
    for r in sorted(recs, key=lambda r: -(r.get('publishedAt') or 0)):
        best.setdefault(S.family(r.get('source')), r)
    reps = sorted(best.values(), key=lambda r: (S.is_primary_or_state(r.get('source')), -(r.get('publishedAt') or 0)))
    return reps[:limit]


def same_story(a_heads, b_heads, tok, limit=0.3):
    """選んだ束どうしが同じ話の取りこぼし分割かの目安（見出しの語の Jaccard）。"""
    a = set().union(*(tok(h) for h in a_heads)) if a_heads else set()
    b = set().union(*(tok(h) for h in b_heads)) if b_heads else set()
    return bool(a and b) and len(a & b) / len(a | b) >= limit
