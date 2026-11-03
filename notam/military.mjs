// East Asian military / launch notices from NOTAM Watch, for the exercise tab
// of 世界情勢観測.
//
//   node notam/military.mjs OUTDIR [SNAPSHOT.json] [--now 2026-11-04T00:00:00Z]
//
// Without SNAPSHOT.json the current snapshot is read from
// https://notam-watch.pages.dev/data/ (manifest + the snapshot it names, the
// same static files and reader as notam/extract.mjs; no Cloudflare Worker is
// involved). NOTAM Watch's public files carry no CORS headers, so the browser
// cannot read them; this script runs in GitHub Actions and publishes a small
// summary to docs/data/notam/ (military.json, meta.json).
//
// Kept: notices that are valid now or become valid within 30 days, lie in
// East Asia (lon 100-150 E, lat 0-50 N) and are about launches, firing,
// exercises, other military activity, GNSS interference (NOTAM Watch's own
// rules, vendored unchanged) or, for aviation, a danger / restricted area by
// Q-code. Outputs are written only after validate() passes.
import {mkdir, readFile, writeFile, rename} from 'node:fs/promises';
import {dirname, join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {classifyAviationActivity, aviationBody} from './vendor/aviation-activity.mjs';
import {classifyMaritimeActivity} from './vendor/maritime-activity.mjs';
import {fetchSnapshot} from './extract.mjs';

export const VERSION = 'military-1';
export const SITE = 'https://notam-watch.pages.dev/';
export const REGION = Object.freeze({lat_min: 0, lat_max: 50, lon_min: 100, lon_max: 150});
export const HORIZON_DAYS = 30;
const TEXT_MAX = 8000;
const FAA_SEARCH = 'https://notams.aim.faa.gov/notamSearch/nsapp.html#/';
// Source countries treated as East Asian when a notice has no drawable area.
const EA_COUNTRIES = new Set(['JP', 'CN', 'TW', 'KR', 'KP', 'PH', 'VN']);

export const CATEGORIES = Object.freeze({
  launch: 'ミサイル・ロケット発射（落下物の危険区域を含む）',
  firing: '射撃・爆撃',
  exercise: '軍事演習・訓練',
  military: '軍の活動・任務',
  gnss: 'GPS等の電波障害',
  danger_area: '危険区域・制限区域の設定（Q コードのみによる。軍事と確定していない）',
});
const ORDER = Object.keys(CATEGORIES);
// Same mapping as NOTAM Watch's military tab (notam-static-trial/src/military-items.mjs).
const MARITIME_KIND = {missile: 'launch', launch: 'launch', firing: 'firing', bombing: 'firing',
  'source-category': 'firing', exercise: 'exercise', military: 'military', drone: 'military'};
// ICAO Q-code subjects (2nd-3rd letters) and the category they imply.
export const Q_SUBJECTS = Object.freeze({
  WM: 'firing',      // missile, gun or rocket firing
  WE: 'exercise',    // exercises
  WF: 'military',    // air refuelling
  WT: 'military',    // mass movement of aircraft
  RD: 'danger_area', // danger area
  RR: 'danger_area', // restricted area
  RP: 'danger_area', // prohibited area
  RT: 'danger_area', // temporary restricted area
  RM: 'danger_area', // military operating area
  RA: 'danger_area', // airspace reservation
});
// Q-code conditions (4th-5th letters) meaning the area is no longer in force.
const Q_ENDED = new Set(['CD', 'CN']); // deactivated, cancelled
const CIVIL = /\bFIREWORKS?\b|\bMODEL\s+ROCKETS?\b|\bLASER\s+(?:LIGHT\s+)?SHOW\b|\bWATER\s+BOMBING\b|\bFIRE\s*FIGHTING\b|\bRAINMAKING\b|\bCLOUD\s+SEEDING\b/;

const iso = ms => (Number.isFinite(ms) ? new Date(ms).toISOString().replace(/\.\d{3}Z$/, 'Z') : null);
const round = v => +Number(v).toFixed(4);

/** Q-line of an ICAO NOTAM: {fir, code, subject, condition, lat, lon} or null. */
export function qLine(text = '') {
  const m = /Q\)\s*([A-Z]{4})\s*\/\s*(Q[A-Z]{4})\s*\/[^\n]*?\/\s*(\d{4})([NS])(\d{5})([EW])\d{0,3}/.exec(String(text));
  if (!m) {
    const c = /Q\)\s*([A-Z]{4})\s*\/\s*(Q[A-Z]{4})\b/.exec(String(text));
    return c ? {fir: c[1], code: c[2], subject: c[2].slice(1, 3), condition: c[2].slice(3, 5), lat: null, lon: null} : null;
  }
  const lat = (+m[3].slice(0, 2) + +m[3].slice(2) / 60) * (m[4] === 'S' ? -1 : 1);
  const lon = (+m[5].slice(0, 3) + +m[5].slice(3) / 60) * (m[6] === 'W' ? -1 : 1);
  return {fir: m[1], code: m[2], subject: m[2].slice(1, 3), condition: m[2].slice(3, 5), lat: round(lat), lon: round(lon)};
}

const pick = kinds => ORDER.filter(k => kinds.includes(k));

/** Categories with their evidence; [] when the notice is not military / launch related. */
export function classify(item, dataset) {
  if (dataset === 'aviation') {
    const activity = classifyAviationActivity(item);
    const basis = activity.evidence.map(e => ({by: 'keyword', category: e.tag, rule: e.rule, matched: e.matched}));
    const q = qLine(item.text);
    const body = aviationBody(item.text);
    const qCat = q && Q_SUBJECTS[q.subject];
    if (qCat && !Q_ENDED.has(q.condition) && !CIVIL.test(body) && !/^\s*NOTAMC\b/m.test(String(item.text).split('\n')[0])) {
      // Q-code only adds a category the E) text did not give, so text evidence wins.
      if (!basis.some(b => b.category === qCat) && !(qCat === 'danger_area' && basis.length)) {
        basis.push({by: 'q-code', category: qCat, rule: `Q-${q.subject}`, matched: q.code});
      }
    }
    return {categories: pick(basis.map(b => b.category)), basis: unique(basis), q};
  }
  const activity = classifyMaritimeActivity(item);
  const basis = activity.evidence.filter(e => MARITIME_KIND[e.tag])
    .map(e => ({by: e.tag === 'source-category' ? 'source-category' : 'keyword', category: MARITIME_KIND[e.tag], rule: e.rule, matched: e.matched}));
  return {categories: pick(basis.map(b => b.category)), basis: unique(basis), q: null};
}
const unique = list => list.filter((b, i) => list.findIndex(o => o.category === b.category && o.rule === b.rule && o.matched === b.matched) === i);

const inRegion = ([lat, lon]) => lat >= REGION.lat_min && lat <= REGION.lat_max && lon >= REGION.lon_min && lon <= REGION.lon_max;

function shapesOf(item) {
  return (Array.isArray(item.shapes) ? item.shapes : [])
    .filter(s => s && ['point', 'line', 'polygon', 'circle'].includes(s.kind) && Array.isArray(s.points) && s.points.length
      && s.points.every(p => Array.isArray(p) && p.length === 2 && p.every(Number.isFinite)))
    .map(s => ({kind: s.kind, points: s.points.map(([a, b]) => [round(a), round(b)]),
      ...(s.kind === 'circle' && Number.isFinite(s.radius) ? {radius_m: s.radius} : {}),
      ...(s.approx || s.provenance === 'provider' ? {approximate: true} : {})}));
}

/** {shapes, basis} when the notice lies in the region, else null. */
export function area(item, q) {
  const shapes = shapesOf(item);
  if (shapes.length) return shapes.some(s => s.points.some(inRegion)) ? {basis: 'shape', shapes} : null;
  if (q && q.lat !== null && inRegion([q.lat, q.lon])) return {basis: 'q-line', shapes: [{kind: 'point', points: [[q.lat, q.lon]], approximate: true}]};
  const country = item.sourceCountry || item.countries?.[0];
  if (country && EA_COUNTRIES.has(country)) return {basis: 'source-country', shapes: []};
  return null;
}

/** active / upcoming / unknown, or null when outside [now, now + 30 days] or cancelled. */
export function timeStatus(item, now) {
  if (item.cancelled) return null;
  const start = Number.isFinite(item.start) ? item.start : null;
  const end = item.permanent ? null : (Number.isFinite(item.end) ? item.end : null);
  if (end !== null && end < now) return null;
  if (start !== null && start > now + HORIZON_DAYS * 86400e3) return null;
  if (start === null && end === null && !item.permanent) return 'unknown';
  if (start !== null && start > now) return 'upcoming';
  return 'active';
}

function source(item, dataset) {
  if (dataset === 'aviation') {
    return {name: 'FAA NOTAM Search（各国の航空当局が出した NOTAM）', url: item.sourceUrl || FAA_SEARCH,
      via: 'NOTAM Watch', note: 'NOTAM Watch の詳細画面の「公式原文」と同じ参照先'};
  }
  return {name: item.sourceName ?? null, url: item.sourceUrl ?? null, country: item.sourceCountry ?? null, via: 'NOTAM Watch'};
}

function original(item) {
  const text = String(item.text ?? '');
  const out = {text: text.slice(0, TEXT_MAX), ...(text.length > TEXT_MAX ? {truncated: true} : {})};
  if (Array.isArray(item.originals) && item.originals.length) {
    out.originals = item.originals.map(o => ({source_name: o.sourceName ?? null, text: String(o.text ?? '').slice(0, TEXT_MAX),
      links: (o.sourceLinks || []).filter(l => typeof l === 'string')}));
  }
  return out;
}

/** Launch history keys (docs/data/launches) -> {notice_key, event_index}. */
export function launchIndex(notices = {}, events = []) {
  const byKey = new Map();
  for (const key of Object.keys(notices?.notices ?? {})) byKey.set(key, {notice_key: key, event_index: null});
  events.forEach((e, i) => {
    for (const key of e.notices || []) {
      const ref = byKey.get(key) ?? {notice_key: key, event_index: null};
      if (ref.event_index === null) ref.event_index = i;
      ref.site = e.site ?? null;
      ref.method = e.method ?? null;
      ref.window_start_utc = e.window_start_utc ?? null;
      ref.window_end_utc = e.window_end_utc ?? null;
      byKey.set(key, ref);
    }
  });
  return byKey;
}

export function build(data, {now = Date.now(), snapshot = {}, launches = new Map()} = {}) {
  const items = [];
  for (const dataset of ['aviation', 'maritime']) {
    for (const item of data?.[dataset]?.items ?? []) {
      if (!item || typeof item.id !== 'string') continue;
      const status = timeStatus(item, now);
      if (!status) continue;
      const {categories, basis, q} = classify(item, dataset);
      if (!categories.length) continue;
      const where = area(item, q);
      if (!where) continue;
      const key = `${dataset}:${item.id}`;
      const end = item.permanent ? null : iso(item.end);
      items.push({
        key, id: item.id, dataset, type: dataset === 'aviation' ? '航空' : '海上',
        issuer: dataset === 'aviation'
          ? {code: item.location ?? q?.fir ?? null, name: null, country: item.countries?.[0] ?? null}
          : {code: null, name: item.sourceName ?? null, country: item.sourceCountry ?? item.countries?.[0] ?? null},
        q_code: q?.code ?? null,
        category: categories[0], categories, basis,
        valid: {start_utc: iso(item.start), end_utc: end, permanent: !!item.permanent, status,
          schedule: dataset === 'aviation' ? (item.staticSchedule?.diagnostic?.raw || null) : null,
          estimated_end: item.staticSchedule?.estimatedEnd === true || undefined},
        area: where,
        original: original(item),
        source: source(item, dataset),
        notam_watch: {url: SITE, dataset, id: item.id, snapshot: snapshot.file ?? null},
        launch: launches.get(key) ?? null,
      });
    }
  }
  for (const it of items) if (it.valid.estimated_end === undefined) delete it.valid.estimated_end;
  items.sort((a, b) => ORDER.indexOf(a.category) - ORDER.indexOf(b.category)
    || String(a.valid.start_utc ?? '').localeCompare(String(b.valid.start_utc ?? '')) || a.key.localeCompare(b.key));
  return items;
}

export function counts(items) {
  const by = f => items.reduce((o, it) => ({...o, [f(it)]: (o[f(it)] || 0) + 1}), {});
  return {total: items.length, by_category: Object.fromEntries(ORDER.map(c => [c, items.filter(i => i.category === c).length])),
    by_dataset: {aviation: 0, maritime: 0, ...by(i => i.dataset)}, by_status: {active: 0, upcoming: 0, unknown: 0, ...by(i => i.valid.status)},
    with_launch_ref: items.filter(i => i.launch).length};
}

export function meta({items, now, snapshot, retrievedAt}) {
  return {
    dataset: 'notam', version: VERSION,
    title: '東アジアの軍事演習・射撃・発射・危険区域の NOTAM と航行警報（NOTAM Watch 経由）',
    retrieved_utc: retrievedAt, evaluated_at_utc: iso(now),
    source_snapshot: {site: SITE, file: snapshot.file ?? null, sha256: snapshot.sha256 ?? null,
      built_utc: snapshot.builtAt ?? null, counts: snapshot.counts ?? null, via: snapshot.via ?? null},
    source: {name: 'NOTAM Watch（notam-watch.pages.dev）が公開する航空 NOTAM と航行警報',
      origin: '航空：各国の航空当局が出した NOTAM（FAA NOTAM Search 経由）。海上：海上保安庁の航行警報、中国海事局の航行警告、台湾・韓国・フィリピン等の水路当局の告知など。各項目の source に NOTAM Watch が示す元の出典を記載',
      position: '公式の告知。演習・射撃・発射などを予告するもので、実施を示すものではない'},
    files: {'military.json': '条件に合う告知の一覧（items）。meta.json：本ファイル'},
    counts: counts(items),
    region: {...REGION, note: '区域の頂点（円は中心）のどれかがこの範囲内にあれば採る。図形がない場合は航空は Q 行の座標、海上は発出元の国（JP・CN・TW・KR・KP・PH・VN）で判断し、area.basis に記す'},
    period: `取得時点で有効なもの（status=active）と、${HORIZON_DAYS}日以内に有効になるもの（upcoming）。開始・終了とも不明なものは unknown。取消済みは除く。時刻は UTC`,
    classification: {
      categories: CATEGORIES,
      rules: [
        'キーワード：NOTAM Watch の軍事タブと同じ判定（notam/vendor/aviation-activity.mjs・maritime-activity.mjs。NOTAM Watch の shared/ から変更なしで複製）。航空は E) 欄、海上は本文で判定',
        `Q コード（航空のみ）：${Object.entries(Q_SUBJECTS).map(([k, v]) => `Q${k}**=${v}`).join('、')}。キーワードで分類できなかった場合だけ追加する。花火・模型ロケット等の民間の語がある NOTAM と NOTAMC は除く`,
        'danger_area は Q コードだけで採ったもの。軍の使用が多いが、軍事と確定したものではない',
        '複数に当たる場合、category は launch > firing > exercise > military > gnss > danger_area の順で最初のもの。categories にすべてを記す。basis に根拠（by=keyword / q-code / source-category）',
      ],
    },
    launch_matching: 'docs/data/launches/notices.json に同じ key（"aviation:B5829/26" など）がある項目に launch を付ける。events.json の発射候補に含まれる場合は event_index（events の添字）と推定発射場を付ける',
    cautions: [
      '0 件は警告や演習がないことを意味しない。NOTAM Watch が取得できた当局・範囲だけが対象で、取得の失敗や未対応の当局もある',
      '告知は予定であり、実施・中止は別に確かめる必要がある',
      '時刻・区域は NOTAM Watch の解釈による。D) 欄の時間帯や日出・日没の条件は valid.schedule と原文を確認すること',
      '原文（original.text）が正。要約や分類は機械判定で誤りがありうる',
      'NOTAM Watch に個別の告知を開く URL はない。notam_watch.url はサイトの入口で、id で検索する',
    ],
    null_rule: 'null は不明を表す（終了時刻が null で permanent=false なら終了不明）',
    schedule: '2時間ごと（UTC の奇数時 50 分）。NOTAM Watch の公開（偶数時 13 分から）を避ける',
  };
}

const ISO = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$/;
/** Throws when the output does not have the published shape. */
export function validate(items, m) {
  const fail = msg => { throw Error(`military.json validation: ${msg}`); };
  if (!Array.isArray(items)) fail('items is not an array');
  const keys = new Set();
  for (const it of items) {
    const at = it?.key ?? '?';
    if (keys.has(it.key)) fail(`duplicate ${at}`);
    keys.add(it.key);
    if (!['aviation', 'maritime'].includes(it.dataset) || it.key !== `${it.dataset}:${it.id}`) fail(`dataset/key ${at}`);
    if (!CATEGORIES[it.category] || it.categories[0] !== it.category || !it.categories.every(c => CATEGORIES[c])) fail(`category ${at}`);
    if (!it.basis.length) fail(`basis ${at}`);
    if (!['active', 'upcoming', 'unknown'].includes(it.valid.status)) fail(`status ${at}`);
    for (const t of [it.valid.start_utc, it.valid.end_utc]) if (t !== null && !ISO.test(t)) fail(`time ${at}`);
    if (!['shape', 'q-line', 'source-country'].includes(it.area.basis)) fail(`area basis ${at}`);
    for (const s of it.area.shapes) {
      if (!['point', 'line', 'polygon', 'circle'].includes(s.kind) || !s.points.length) fail(`shape ${at}`);
      if (!s.points.every(p => Math.abs(p[0]) <= 90 && Math.abs(p[1]) <= 180)) fail(`coordinates ${at}`);
      if (s.kind === 'circle' && !(s.radius_m > 0)) fail(`radius ${at}`);
    }
    if (it.area.basis !== 'source-country' && !it.area.shapes.some(s => s.points.some(inRegion))) fail(`outside region ${at}`);
    if (typeof it.original?.text !== 'string' || !it.original.text) fail(`original text ${at}`);
    if (it.source?.via !== 'NOTAM Watch' || it.notam_watch?.url !== SITE) fail(`source ${at}`);
  }
  if (m) {
    if (m.counts.total !== items.length) fail('meta count');
    if (!m.cautions.some(c => c.startsWith('0 件は'))) fail('meta caution');
    if (!ISO.test(m.retrieved_utc ?? '')) fail('meta retrieved_utc');
  }
  return true;
}

async function readJson(path, fallback) {
  try { return JSON.parse(await readFile(path, 'utf8')); } catch { return fallback; }
}
async function writeAtomic(path, value) {
  await writeFile(`${path}.tmp`, JSON.stringify(value, null, 1) + '\n');
  await rename(`${path}.tmp`, path);
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const args = process.argv.slice(2);
  const nowAt = args.indexOf('--now');
  const now = nowAt >= 0 ? Date.parse(args.splice(nowAt, 2)[1]) : Date.now();
  const [outDir, input] = args;
  const root = join(dirname(fileURLToPath(import.meta.url)), '..');
  try {
    let data, snapshot;
    if (input) {
      data = JSON.parse(await readFile(input, 'utf8'));
      snapshot = {file: input.split('/').pop(), builtAt: data.buildAt ?? null, via: 'local file'};
    } else {
      const s = await fetchSnapshot();
      data = s.data;
      snapshot = {file: s.file, sha256: s.manifest?.sha256 ?? null, builtAt: s.builtAt, counts: s.manifest?.counts ?? null, via: s.via};
    }
    const notices = await readJson(join(root, 'docs/data/launches/notices.json'), {});
    const events = (await readJson(join(root, 'docs/data/launches/events.json'), {})).events ?? [];
    const items = build(data, {now, snapshot, launches: launchIndex(notices, events)});
    const m = meta({items, now, snapshot, retrievedAt: iso(Date.now())});
    validate(items, m);
    await mkdir(outDir, {recursive: true});
    await writeAtomic(join(outDir, 'military.json'), {generated_utc: m.retrieved_utc, snapshot: snapshot.file ?? null, items});
    await writeAtomic(join(outDir, 'meta.json'), m);
    const c = m.counts;
    console.log(`::notice title=NOTAM military::${c.total} items (${Object.entries(c.by_category).map(([k, v]) => `${k}=${v}`).join(' ')}; launch refs ${c.with_launch_ref}) from ${snapshot.file} (${snapshot.via})`);
  } catch (error) {
    const cause = error?.cause ? ` (cause: ${error.cause.code ?? ''} ${error.cause.message ?? error.cause})` : '';
    console.log(`::error title=NOTAM military::${`${error?.message ?? error}${cause}`.replace(/%/g, '%25').replace(/\r?\n/g, '%0A')}`);
    console.error(error);
    process.exit(1);
  }
}
