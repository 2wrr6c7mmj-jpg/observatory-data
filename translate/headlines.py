"""Translates the Observatory site's untranslated headlines into Japanese.

Runs on GitHub Actions with an open model served by llama.cpp (llama-server,
OpenAI-compatible API on localhost), so there is no key, account or quota.

Reads   https://observatory-watch.pages.dev/api/feeds (titles still pending)
Keeps   docs/data/translations/cache.json        (every translation made, pruned after 30 days unseen)
Writes  docs/data/translations/headlines-ja.json (only the titles the site still lacks; the site's
        background upkeep takes them from GitHub Pages)

An answer from the feed without a non-empty list of records, or a run in which
every batch failed, writes nothing (the published files stay as they are) and
exits non-zero (a translation dropped one by one is not a failure). The model numbers each translation after its headline; a
translation whose number is missing, repeated or out of range, or that carries a
number found only in another headline of the batch, is not used.
"""
import argparse, hashlib, json, os, re, sys, time, urllib.request

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
    '部隊名・階級は日本の軍事報道の定訳にします（例：25th Infantry Division→第25歩兵師団、Carrier Strike Group→空母打撃群、'
    'Sgt.→軍曹、Service Members→将兵、Coast Guard（中国）→海警、Eastern Theater Command→東部戦区）。'
    '略語（HIMARS、NAS など）は原文のまま残してかまいません。'
    '艦名・人名・数字・日付は落とさず、推測で情報を足さないこと。説明や注記は付けず、訳だけを返します。'
)


def key(title):
    return hashlib.sha256(title.encode('utf-8')).hexdigest()[:24]


# East Asia first: headlines from sources scoped to the region, or classed as
# military/conflict, are translated before the rest (newest first within each).
EAST_ASIA = {'CN', 'HK', 'MO', 'TW', 'JP', 'KR', 'KP', 'PH'}


def priority(r):
    if r.get('scope') in EAST_ASIA:
        return 0
    if r.get('category') in ('military', 'conflict'):
        return 1
    return 2


def pending_titles(records):
    """Titles the site still shows untranslated, East Asia and military first, newest first, without duplicates."""
    out, seen = [], set()
    for r in sorted(records, key=lambda r: (priority(r), -(r.get('publishedAt') or 0))):
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


def chat(server, body):
    """One request to the model server; the message content (a JSON string)."""
    req = urllib.request.Request(server + '/v1/chat/completions', data=json.dumps(body).encode('utf-8'),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=900) as r:
        res = json.loads(r.read().decode('utf-8'))
    return res['choices'][0]['message']['content']


NUMBER = re.compile(r'\d+(?:[.,]\d+)*')
MIN_DIGITS = 3  # shorter numbers (2 sailors, 25th, 10万) are too common, and too often rewritten, to tell a swap
# A headline that writes a number in words or in kanji/hanzi numerals may get digits in its translation
# ("Two sailors" → 水兵2人, 一百五十 → 150): such a headline is not checked for numbers.
NUMBER_WORDS = re.compile(r'\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|'
                          r'sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|'
                          r'hundreds?|thousands?|millions?|billions?|dozens?|first|second|third)\b|[〇零一二三四五六七八九十百千万萬億亿兩两]',
                          re.IGNORECASE)


def numbers(text):
    """The digit runs of a text with at least MIN_DIGITS digits (full-width digits folded, separators dropped)."""
    text = (text or '').translate(str.maketrans('０１２３４５６７８９，．', '0123456789,.'))
    out = set()
    for n in NUMBER.findall(text):
        n = n.replace(',', '')
        if sum(c.isdigit() for c in n) >= MIN_DIGITS:
            out.add(n)
    return out


def match_batch(titles, content):
    """The translation for each title (None where it cannot be used), from the model's numbered answer.

    Each answer item is {"n": number of the headline (1-based), "ja": translation}. Items are
    taken by their number, never by their position; a number that is missing, repeated or out
    of range gives no translation, and so does a translation with a number that appears in
    another headline of the batch but not in its own (a sign that two answers were swapped; only
    numbers of MIN_DIGITS digits or more count, and a headline with numbers in words is not checked).
    Returns None when the answer is not the expected shape at all."""
    try:
        data = json.loads(content)
    except (TypeError, ValueError):
        return None
    items = data.get('items') if isinstance(data, dict) else None
    if not isinstance(items, list):
        return None
    by_n, dup = {}, set()
    for it in items:
        if not isinstance(it, dict):
            continue
        n = it.get('n')
        if isinstance(n, str) and n.strip().isdigit():
            n = int(n.strip())
        if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= len(titles):
            continue
        if n in by_n:
            dup.add(n)
        by_n[n] = it.get('ja')
    out = []
    own = [numbers(t) for t in titles]
    for i, t in enumerate(titles, 1):
        ja = by_n.get(i) if i not in dup else None
        if isinstance(ja, str) and not NUMBER_WORDS.search(t):
            others = set().union(*(own[j] for j in range(len(titles)) if j != i - 1)) - own[i - 1]
            if numbers(ja) & others:
                ja = None
        out.append(ja if isinstance(ja, str) else None)
    return out


def translate_batch(server, titles):
    """Translations of ``titles`` (None for each one that cannot be used), or None when the answer is unusable."""
    k = len(titles)
    schema = {'type': 'object', 'properties': {'items': {'type': 'array', 'minItems': k, 'maxItems': k, 'items': {
        'type': 'object', 'properties': {'n': {'type': 'integer', 'minimum': 1, 'maximum': k}, 'ja': {'type': 'string'}},
        'required': ['n', 'ja']}}}, 'required': ['items']}
    numbered = [{'n': i, 'title': t} for i, t in enumerate(titles, 1)]
    body = {
        'messages': [
            {'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': '次の番号つきの見出しをそれぞれ訳し、{"items": [{"n": 番号, "ja": "訳"}, ...]} の形で返してください。'
                                        'n には訳した見出しの番号をそのまま書き、1 つの番号に 1 つの訳だけを書きます。\n'
                                        + json.dumps(numbered, ensure_ascii=False)},
        ],
        'chat_template_kwargs': {'enable_thinking': False},
        'temperature': 0.1, 'max_tokens': 100 * k + 50,
        'response_format': {'type': 'json_object', 'schema': schema},
    }
    return match_batch(titles, chat(server, body))


def load(path, default):
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, ValueError):
        return default


def write(path, value):
    """Write beside the file and rename, so a killed run never leaves half a file to publish."""
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, separators=(',', ':'))
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def feed_records(feed):
    """The feed's records, or None when the answer is not a non-empty list of records."""
    records = feed.get('records') if isinstance(feed, dict) else None
    if not isinstance(records, list):
        return None
    records = [r for r in records if isinstance(r, dict)]
    return records or None


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
    try:
        records = feed_records(get_json(FEED))
    except Exception as e:  # network error, not JSON: nothing to do this run
        print(f'::error title=translate::the feed could not be read ({type(e).__name__}: {e}); the published translations are kept')
        return 2
    if records is None:
        print('::error title=translate::the feed returned no records; the published translations are kept')
        return 2
    titles = pending_titles(records)
    for t in titles:
        if key(t) in cache:
            cache[key(t)]['seen'] = now
    todo = [t for t in titles if key(t) not in cache]
    made = failed = batches = answered = 0
    for i in range(0, len(todo), BATCH):
        if time.time() - started > a.minutes * 60:
            break
        chunk = todo[i:i + BATCH]
        batches += 1
        try:
            ja = translate_batch(a.server, chunk)
        except Exception as e:  # one bad batch must not stop the rest
            print(f'batch failed: {type(e).__name__}: {e}', file=sys.stderr)
            ja = None
        if ja is None:
            failed += len(chunk)
            continue
        answered += 1
        for o, t in zip(chunk, ja):
            if valid(o, t):
                cache[key(o)] = {'o': o, 't': t.strip(), 'seen': now}
                made += 1
            else:
                failed += 1
    # Non-zero only when the model never gave a usable answer (an error or a wrong shape in every batch).
    # Translations dropped one by one (a headline like "HIMARS" that stays as it is, a swap) are normal:
    # the files are written (seen refreshed, old entries pruned) and the run ends with 0.
    if batches and not answered:
        print(f'::error title=translate::all {batches} batches failed ({failed} titles); the published translations are kept')
        return 1
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
