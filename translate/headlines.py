"""Translates the Observatory site's untranslated headlines into Japanese.

Runs on GitHub Actions with an open model served by llama.cpp (llama-server,
OpenAI-compatible API on localhost), so there is no key, account or quota.

Reads   https://observatory-watch.pages.dev/api/feeds (titles still pending)
Keeps   docs/data/translations/cache.json        (every translation made, pruned after 30 days unseen)
Writes  docs/data/translations/headlines-ja.json (only the titles the site still lacks; the site's
        background upkeep takes them from GitHub Pages)
"""
import argparse, hashlib, json, re, sys, time, urllib.request

FEED = 'https://observatory-watch.pages.dev/api/feeds'
OUT_DIR = 'docs/data/translations'
SKIP_SOURCES = {'nhk', 'mofa', 'mod'}
KANA = re.compile(r'[ぁ-んァ-ヶ]')
JAPANESE = re.compile(r'[ぁ-んァ-ヶ一-龠]')
PUBLISH_MAX = 600
KEEP_DAYS = 30
BATCH = 12

SYSTEM = (
    'あなたは安全保障・軍事ニュースの専門翻訳者です。与えられた見出し（英語・中国語・ロシア語・韓国語など）を、'
    '日本の報道の見出しとして自然な日本語に訳してください。固有名詞は日本の報道で通常使われる表記にします'
    '（例：PLA→人民解放軍、PLAN→中国海軍、Ministry of National Defense（台湾）→台湾国防部、Pentagon→米国防総省）。'
    '艦名・人名・数字・日付は落とさず、推測で情報を足さないこと。説明や注記は付けず、訳だけを返します。'
)


def key(title):
    return hashlib.sha256(title.encode('utf-8')).hexdigest()[:24]


def pending_titles(records):
    """Titles the site still shows untranslated, newest first, without duplicates."""
    out, seen = [], set()
    for r in sorted(records, key=lambda r: r.get('publishedAt') or 0, reverse=True):
        t = (r.get('title') or '').strip()
        if not t or t in seen or r.get('source') in SKIP_SOURCES:
            continue
        if r.get('titleJa') or KANA.search(t):
            continue
        seen.add(t)
        out.append(t)
    return out


def valid(original, translated):
    if not isinstance(translated, str):
        return False
    t = translated.strip()
    return bool(t) and bool(JAPANESE.search(t)) and len(t) <= max(400, 4 * len(original)) and t != original


def get_json(url, timeout=60):
    req = urllib.request.Request(url, headers={'Accept': 'application/json', 'User-Agent': 'observatory-data translator'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode('utf-8'))


def translate_batch(server, titles):
    schema = {'type': 'object', 'properties': {'ja': {'type': 'array', 'items': {'type': 'string'},
              'minItems': len(titles), 'maxItems': len(titles)}}, 'required': ['ja']}
    body = {
        'messages': [
            {'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': '次の見出しを順番どおりに訳し、{"ja": [...]} の形で返してください。\n' + json.dumps(titles, ensure_ascii=False)},
        ],
        'chat_template_kwargs': {'enable_thinking': False},
        'temperature': 0.1, 'max_tokens': 90 * len(titles) + 50,
        'response_format': {'type': 'json_object', 'schema': schema},
    }
    req = urllib.request.Request(server + '/v1/chat/completions', data=json.dumps(body).encode('utf-8'),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=900) as r:
        res = json.loads(r.read().decode('utf-8'))
    ja = json.loads(res['choices'][0]['message']['content']).get('ja', [])
    return ja if isinstance(ja, list) and len(ja) == len(titles) else None


def load(path, default):
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, ValueError):
        return default


def write(path, value):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, separators=(',', ':'))
        f.write('\n')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--server', default='http://127.0.0.1:8080')
    ap.add_argument('--model', default='')
    ap.add_argument('--minutes', type=float, default=40)
    ap.add_argument('--out', default=OUT_DIR)
    a = ap.parse_args(argv)
    started, now = time.time(), int(time.time() * 1000)
    cache_path, pub_path = a.out + '/cache.json', a.out + '/headlines-ja.json'
    cache = load(cache_path, {'items': {}})['items']
    titles = pending_titles(get_json(FEED).get('records', []))
    for t in titles:
        if key(t) in cache:
            cache[key(t)]['seen'] = now
    todo = [t for t in titles if key(t) not in cache]
    made = failed = 0
    for i in range(0, len(todo), BATCH):
        if time.time() - started > a.minutes * 60:
            break
        chunk = todo[i:i + BATCH]
        try:
            ja = translate_batch(a.server, chunk)
        except Exception as e:  # one bad batch must not stop the rest
            print(f'batch failed: {e}', file=sys.stderr)
            ja = None
        if ja is None:
            failed += len(chunk)
            continue
        for o, t in zip(chunk, ja):
            if valid(o, t):
                cache[key(o)] = {'o': o, 't': t.strip(), 'seen': now}
                made += 1
            else:
                failed += 1
    cutoff = now - KEEP_DAYS * 86400000
    cache = {k: v for k, v in cache.items() if v.get('seen', 0) >= cutoff}
    items = [{'original': t, 'translated': cache[key(t)]['t']} for t in titles if key(t) in cache][:PUBLISH_MAX]
    write(cache_path, {'items': cache})
    write(pub_path, {'updatedAt': now, 'model': a.model, 'pending': len(titles), 'items': items})
    left = len([t for t in titles if key(t) not in cache])
    print(f'::notice title=translate::pending {len(titles)}, translated now {made}, failed {failed}, published {len(items)}, still untranslated {left}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
