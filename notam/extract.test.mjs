import test from 'node:test';
import assert from 'node:assert/strict';
import {launchNotices} from './extract.mjs';

test('keeps launch notices from both sources with the shared rules', () => {
  const data = {
    aviation: {items: [
      {id: 'A1', location: 'ZSHA', start: Date.UTC(2026, 10, 1, 1), end: Date.UTC(2026, 10, 1, 3),
       text: 'A1 NOTAMN E) AEROSPACE FLT ACT WILL TAKE PLACE. ROCKET DEBRIS FALL AREA', shapes: [{kind: 'polygon', points: [[30.1234567, 125.1], [30.2, 125.2], [30.3, 125.1]]}]},
      {id: 'A2', text: 'A2 NOTAMN E) FIREWORKS DISPLAY', shapes: []},
      {id: 'A3', text: 'A3 NOTAMN E) MILITARY EXER', shapes: []},
    ]},
    maritime: {items: [{id: 'M1', text: '火箭残骸坠落 危险区域', start: Date.UTC(2026, 10, 1, 1), shapes: [{kind: 'circle', points: [[31, 126]], radius: 9000}]}]},
  };
  const out = launchNotices(data);
  assert.deepEqual(out.map(n => n.key), ['aviation:A1', 'maritime:M1']);
  assert.equal(out[0].start_utc, '2026-11-01T01:00:00Z');
  assert.deepEqual(out[0].shapes[0].points[0], [30.1235, 125.1]);
  assert.ok(out[0].evidence.length > 0);
});

// The transport cases below build manifests with NOTAM Watch's own (vendored) writer.
import {fetchSnapshot} from './extract.mjs';
import {makeSnapshotTransport, snapshotHash} from './vendor/snapshot-transport.mjs';

const ORIGIN = 'https://example.test';
function site(files) {
  const asked = [];
  const fetcher = async url => {
    asked.push(url.slice(ORIGIN.length + 6));
    const body = files.get(url.slice(ORIGIN.length + 6));
    return body ? new Response(body) : new Response('Not found', {status: 404});
  };
  return {fetcher, asked};
}
async function publish(data, {compression = false, pad = 0} = {}) {
  const json = JSON.stringify({...data, padding: 'x'.repeat(pad)});
  const bytes = new TextEncoder().encode(json);
  const sha256 = await snapshotHash(bytes);
  const file = `snapshot-${sha256.slice(0, 16)}.json`;
  const {transport, parts, compressed} = await makeSnapshotTransport(bytes, {compression});
  const files = new Map(parts);
  if (!transport) files.set(file, bytes);
  const manifest = {schemaVersion: 1, file, sha256, buildAt: '2026-10-30T05:20:00.000Z', counts: {},
    ...(transport ? {transport} : {}), ...(compressed ? {compressed} : {})};
  files.set('manifest.json', JSON.stringify(manifest));
  return {files, manifest};
}
const DATA = {aviation: {items: [{id: 'A1', text: 'E) ROCKET DEBRIS FALL AREA', shapes: []}]}, maritime: {items: []}};

test('reads an inline snapshot', async () => {
  const {files} = await publish(DATA);
  const {fetcher} = site(files);
  const got = await fetchSnapshot(fetcher, ORIGIN);
  assert.equal(got.data.aviation.items[0].id, 'A1');
  assert.equal(got.builtAt, '2026-10-30T05:20:00.000Z');
});

test('reads a snapshot over 25 MiB from its byte parts (no single JSON file is published)', async () => {
  const {files, manifest} = await publish(DATA, {pad: 26 * 1024 * 1024});
  assert.equal(manifest.transport.parts.length, 4);
  assert.ok(!files.has(manifest.file));
  const {fetcher, asked} = site(files);
  const got = await fetchSnapshot(fetcher, ORIGIN);
  assert.equal(got.data.aviation.items[0].id, 'A1');
  assert.ok(!asked.includes(manifest.file));
});

test('prefers the gzip copy and falls back to the parts when it is broken', async () => {
  const {files, manifest} = await publish(DATA, {pad: 26 * 1024 * 1024, compression: true});
  assert.ok(manifest.compressed);
  let {fetcher, asked} = site(files);
  assert.equal((await fetchSnapshot(fetcher, ORIGIN)).data.aviation.items[0].id, 'A1');
  assert.deepEqual(asked, ['manifest.json', manifest.compressed.file]);
  files.delete(manifest.compressed.file);
  ({fetcher, asked} = site(files));
  assert.equal((await fetchSnapshot(fetcher, ORIGIN)).data.aviation.items[0].id, 'A1');
  assert.equal(asked.length, 2 + manifest.transport.parts.length);
});

test('rejects a part that does not match the manifest', async () => {
  const {files, manifest} = await publish(DATA, {pad: 26 * 1024 * 1024});
  const first = manifest.transport.parts[0].file;
  const broken = new Uint8Array(files.get(first)); broken[0] ^= 1;
  files.set(first, broken);
  await assert.rejects(fetchSnapshot(site(files).fetcher, ORIGIN), /part hash or length mismatch/);
});

test('reports a missing file with its URL and status', async () => {
  const {files, manifest} = await publish(DATA);
  files.delete(manifest.file);
  await assert.rejects(fetchSnapshot(site(files).fetcher, ORIGIN), /snapshot-[a-f0-9]{16}\.json: HTTP 404/);
});
