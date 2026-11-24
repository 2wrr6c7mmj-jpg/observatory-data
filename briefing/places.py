"""地名と座標：元の記事の文字列に別名が出る地名を、briefing/places.json の表から引く（モデルには決めさせない）。
座標は [経度, 緯度] の 10 進。表に無い地名は出さない（place は null）。
"""
import json
import os

_PATH = os.path.join(os.path.dirname(__file__), 'places.json')
_cache = None


def table():
    global _cache
    if _cache is None:
        with open(_PATH, encoding='utf-8') as f:
            _cache = json.load(f)['places']
    return _cache


def find(texts, places=None):
    """texts（記事ごとの文字列の列、または 1 つの文字列）に別名が出る地名。
    地点（place）を先、国（country）、首都（capital）の順で、各々 出てくる記事の数の多い順、同数なら早く出る順。"""
    if isinstance(texts, str):
        texts = [texts]
    lows = [(t or '').lower() for t in texts]
    order = {'place': 0, 'country': 1, 'capital': 2}
    hits = []
    for p in places or table():
        al = [a.lower() for a in p['aliases'] if a]
        n = sum(1 for t in lows if any(a in t for a in al))
        if n:
            first = min(t.find(a) for t in lows for a in al if a in t)
            hits.append((order[p['kind']], -n, first, p))
    hits.sort(key=lambda x: x[:3])
    return [h[3] for h in hits]


def main_place(hits):
    """画面に出す 1 つ：kind が place のものの先頭。無ければ None（国・首都は地点を示さない）。"""
    for p in hits:
        if p['kind'] == 'place':
            return {'name': p['name'], 'lon': p['lonlat'][0], 'lat': p['lonlat'][1]}
    return None


def country_codes(hits, limit=3):
    """輪郭を出す国（ISO 2 文字、重複なし、最大 limit。具体的な地名・国の順）。"""
    out = []
    for p in hits:
        c = p.get('cc')
        if c and c not in out:
            out.append(c)
    return out[:limit]
