// Read NOTAM Watch's published snapshot and keep the launch-related notices.
//
//   node notam/extract.mjs OUT.json [SNAPSHOT.json]
//
// Without SNAPSHOT.json it downloads https://notam-watch.pages.dev/data/manifest.json
// and the snapshot it names. Snapshots over 25 MiB are not published as one JSON
// file: the manifest then lists 8 MiB byte parts (`transport`) and optionally a
// gzip copy (`compressed`). NOTAM Watch's own reader (vendored) checks every
// part and the full snapshot against the manifest's SHA-256. These are static
// files: reading them does not start a Cloudflare Worker, so NOTAM Watch's
// free-tier request allowance is not used.
import {readFile, writeFile} from 'node:fs/promises';
import {classifyAviationActivity} from './vendor/aviation-activity.mjs';
import {classifyMaritimeActivity} from './vendor/maritime-activity.mjs';
import {readSnapshotBytes, SNAPSHOT_LIMIT} from './vendor/snapshot-transport.mjs';

const ORIGIN = 'https://notam-watch.pages.dev';
const UA = 'observatory-data/1';

export async function download(url, maxBytes, fetcher = fetch) {
  const response = await fetcher(url, {headers: {'User-Agent': UA}, redirect: 'error', signal: AbortSignal.timeout(60000)});
  if (!response.ok) throw Error(`${url}: HTTP ${response.status}`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  if (bytes.length > maxBytes) throw Error(`${url}: ${bytes.length} bytes, larger than ${maxBytes}`);
  return bytes;
}

/** Reads the current snapshot named by the manifest (inline, byte parts or gzip). */
export async function fetchSnapshot(fetcher = fetch, origin = ORIGIN) {
  const manifest = JSON.parse(new TextDecoder().decode(await download(`${origin}/data/manifest.json`, 64 * 1024, fetcher)));
  if (!/^snapshot-[a-f0-9]{16}\.json$/.test(manifest.file ?? '')) throw Error(`Unexpected snapshot name in manifest: ${manifest.file}`);
  const fetched = [];
  const reader = (file, max) => { fetched.push(file); return download(`${origin}/data/${file}`, max, fetcher); };
  const bytes = await readSnapshotBytes(manifest, reader, {maxBytes: SNAPSHOT_LIMIT});
  const kinds = fetched.map(f => (/\.gzip-/.test(f) ? 'gzip' : /\.part-/.test(f) ? 'parts' : 'json'));
  const via = `${[...new Set(kinds)].join(' then ')}, ${fetched.length} file(s)`;
  return {data: JSON.parse(new TextDecoder('utf-8', {fatal: true}).decode(bytes)), file: manifest.file,
    builtAt: manifest.buildAt ?? manifest.builtAt ?? manifest.generatedAt ?? null, via};
}

async function snapshot(path) {
  if (path) return {data: JSON.parse(await readFile(path, 'utf8')), file: path, via: 'local file'};
  return fetchSnapshot();
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
  try {
    const {data, file, builtAt, via} = await snapshot(input);
    const notices = launchNotices(data);
    await writeFile(out, JSON.stringify({snapshot: file, snapshot_built: builtAt, retrieved_utc: iso(Date.now()), notices}) + '\n');
    console.log(`::notice::launch notices: ${notices.length} from ${file} (${via})`);
  } catch (error) {
    // Without this the runner only reports "exit code 1" and the reason stays in the log.
    const cause = error?.cause ? ` (cause: ${error.cause.code ?? ''} ${error.cause.message ?? error.cause})` : '';
    console.log(`::error title=NOTAM Watch snapshot::${`${error?.message ?? error}${cause}`.replace(/%/g, '%25').replace(/\r?\n/g, '%0A')}`);
    console.error(error);
    process.exit(1);
  }
}
