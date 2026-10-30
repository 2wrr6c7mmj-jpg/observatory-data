// Read NOTAM Watch's published snapshot and keep the launch-related notices.
//
//   node notam/extract.mjs OUT.json [SNAPSHOT.json]
//
// Without SNAPSHOT.json it downloads https://notam-watch.pages.dev/data/manifest.json
// and the snapshot it names, and checks the snapshot's SHA-256 against the
// manifest. These are static files: reading them does not start a Cloudflare
// Worker, so NOTAM Watch's free-tier request allowance is not used.
import {createHash} from 'node:crypto';
import {readFile, writeFile} from 'node:fs/promises';
import {classifyAviationActivity} from './vendor/aviation-activity.mjs';
import {classifyMaritimeActivity} from './vendor/maritime-activity.mjs';

const ORIGIN = 'https://notam-watch.pages.dev';
const UA = 'observatory-data/1 (+https://github.com/2wrr6c7mmj-jpg/observatory-data)';

async function download(url, maxBytes) {
  const response = await fetch(url, {headers: {'User-Agent': UA}, redirect: 'error', signal: AbortSignal.timeout(60000)});
  if (!response.ok) throw Error(`${url}: HTTP ${response.status}`);
  const bytes = Buffer.from(await response.arrayBuffer());
  if (bytes.length > maxBytes) throw Error(`${url}: larger than ${maxBytes} bytes`);
  return bytes;
}

async function snapshot(path) {
  if (path) return {data: JSON.parse(await readFile(path, 'utf8')), file: path};
  const manifest = JSON.parse(await download(`${ORIGIN}/data/manifest.json`, 64 * 1024));
  if (!/^snapshot-[a-f0-9]{16}\.json$/.test(manifest.file)) throw Error('Unexpected snapshot name in manifest');
  const bytes = await download(`${ORIGIN}/data/${manifest.file}`, 60 * 1024 * 1024);
  const digest = createHash('sha256').update(bytes).digest('hex');
  if (manifest.sha256 && digest !== manifest.sha256) throw Error('Snapshot does not match the manifest digest');
  return {data: JSON.parse(bytes), file: manifest.file, builtAt: manifest.builtAt ?? manifest.generatedAt ?? null};
}

const iso = ms => (Number.isFinite(ms) ? new Date(ms).toISOString().replace(/\.\d{3}Z$/, 'Z') : null);
const shapes = list => (Array.isArray(list) ? list : [])
  .filter(s => s && Array.isArray(s.points) && s.points.length)
  .map(s => ({kind: s.kind, points: s.points.map(([lat, lon]) => [+lat.toFixed(4), +lon.toFixed(4)]), ...(s.radius ? {radius: s.radius} : {})}));

export function launchNotices(data) {
  const out = [];
  for (const [source, classify] of [['aviation', classifyAviationActivity], ['maritime', classifyMaritimeActivity]]) {
    for (const item of data?.[source]?.items ?? []) {
      const activity = classify(item);
      if (!activity.tags.includes('launch')) continue;
      out.push({
        key: `${source}:${item.id}`, source, id: item.id,
        authority: item.location ?? item.source ?? item.sourceId ?? null,
        start_utc: iso(item.start), end_utc: iso(item.end),
        evidence: activity.evidence.filter(e => e.tag === 'launch').map(e => e.matched),
        shapes: shapes(item.shapes),
        text: String(item.text ?? '').slice(0, 1500),
      });
    }
  }
  return out;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const [out, input] = process.argv.slice(2);
  const {data, file, builtAt} = await snapshot(input);
  const notices = launchNotices(data);
  await writeFile(out, JSON.stringify({snapshot: file, snapshot_built: builtAt, retrieved_utc: iso(Date.now()), notices}) + '\n');
  console.log(`::notice::launch notices: ${notices.length} from ${file}`);
}
