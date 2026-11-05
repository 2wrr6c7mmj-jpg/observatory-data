import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {build, classify, counts, launchIndex, meta, qLine, timeStatus, validate, SITE} from './military.mjs';

const fixture = JSON.parse(readFileSync(new URL('./fixtures/military-snapshot.json', import.meta.url), 'utf8'));
const NOW = Date.UTC(2026, 10, 4, 0, 0);
const launches = launchIndex(
  {notices: {'aviation:B5829/26': {id: 'B5829/26'}}},
  [{notices: ['aviation:B5829/26', 'maritime:X'], site: '文昌', method: 'aligned', window_start_utc: '2026-11-20T11:00Z', window_end_utc: '2026-11-23T14:00Z'}],
);
const items = build(fixture, {now: NOW, snapshot: {file: 'snapshot-0123456789abcdef.json'}, launches});
const byKey = Object.fromEntries(items.map(i => [i.key, i]));

test('keeps military and launch notices in East Asia, now or within 30 days', () => {
  assert.deepEqual(Object.keys(byKey).sort(), [
    'aviation:A1234/26', 'aviation:B5829/26', 'aviation:P4290/26',
    'maritime:NAVAREA XI 9999/26', 'maritime:琼航警0950/26', 'maritime:辽航警0901/26', 'maritime:闽航警0612/26',
  ]);
});

test('drops civil, expired, far-future, cancelled, out-of-region and unrelated notices', () => {
  for (const id of ['A2000/26', 'A3000/26', 'A4000/26', 'A5000/26', 'A6000/26', 'V3389/26']) assert.ok(!byKey[`aviation:${id}`], id);
  assert.ok(!byKey['maritime:渝航通〔2026〕0194号']);
  assert.ok(!byKey['maritime:HN177/2026'], 'every compiled period has ended');
});

test('classifies by keywords first and Q-codes second', () => {
  assert.equal(byKey['aviation:B5829/26'].category, 'launch');
  assert.deepEqual(byKey['aviation:B5829/26'].categories, ['launch', 'firing']);
  assert.equal(byKey['aviation:P4290/26'].category, 'exercise');
  const danger = byKey['aviation:A1234/26'];
  assert.equal(danger.category, 'danger_area');
  assert.deepEqual(danger.basis, [{by: 'q-code', category: 'danger_area', rule: 'Q-RT', matched: 'QRTCA'}]);
  assert.equal(danger.area.basis, 'q-line');
  assert.deepEqual(danger.area.shapes[0].points, [[37, 126]]);
  assert.equal(byKey['maritime:闽航警0612/26'].category, 'firing');
  // Danger area by Q-code is not added on top of a keyword category.
  assert.ok(!classify({text: 'Q)RJJJ/QRDCA/IV/BO/W/000/100/3500N13900E010\nE)AIR EXER'}, 'aviation').categories.includes('danger_area'));
});

test('time window and status', () => {
  assert.equal(byKey['aviation:B5829/26'].valid.status, 'upcoming');
  assert.equal(byKey['aviation:P4290/26'].valid.status, 'active');
  assert.equal(byKey['maritime:NAVAREA XI 9999/26'].valid.status, 'unknown');
  assert.equal(byKey['aviation:B5829/26'].valid.start_utc, '2026-11-20T11:00:00Z');
  assert.equal(byKey['aviation:B5829/26'].valid.schedule, '1100-1400');
  const v = byKey['maritime:辽航警0901/26'].valid;
  assert.deepEqual([v.status, v.basis, v.start_utc, v.end_utc], ['active', 'validity-period', '2026-11-02T16:00:00Z', '2026-11-08T16:00:00Z']);
  const u = byKey['maritime:琼航警0950/26'].valid;
  assert.deepEqual([u.status, u.basis, u.periods.length], ['upcoming', 'period', 1]);
  assert.equal(byKey['maritime:NAVAREA XI 9999/26'].valid.basis, 'none');
  assert.equal(timeStatus({start: NOW - 1, end: null, permanent: true}, NOW), 'active');
  assert.equal(timeStatus({start: NOW + 31 * 86400e3, end: null}, NOW), null);
  assert.equal(timeStatus({start: NOW + 29 * 86400e3, end: null}, NOW), 'upcoming');
});

test('carries original text, source, NOTAM Watch reference and the launch match', () => {
  const b = byKey['aviation:B5829/26'];
  assert.match(b.original.text, /AEROSPACE FLT ACT/);
  assert.equal(b.source.via, 'NOTAM Watch');
  assert.equal(b.issuer.code, 'RPHI');
  assert.equal(b.q_code, 'QWMLW');
  assert.deepEqual(b.notam_watch, {url: SITE, dataset: 'aviation', id: 'B5829/26', snapshot: 'snapshot-0123456789abcdef.json'});
  assert.equal(b.launch.notice_key, 'aviation:B5829/26');
  assert.equal(b.launch.event_index, 0);
  assert.equal(b.launch.site, '文昌');
  assert.equal(byKey['aviation:P4290/26'].launch, null);
  const m = byKey['maritime:闽航警0612/26'];
  assert.equal(m.source.name, '福建海事局');
  assert.equal(m.issuer.country, 'CN');
  assert.equal(byKey['maritime:NAVAREA XI 9999/26'].area.basis, 'source-country');
  assert.equal(byKey['aviation:P4290/26'].area.shapes[0].radius_m, 55560);
});

test('meta counts, cautions and validation', () => {
  const m = meta({items, now: NOW, snapshot: {file: 'x'}, retrievedAt: '2026-11-04T00:50:00Z'});
  assert.equal(m.counts.total, 7);
  assert.deepEqual(m.counts.by_category, {launch: 1, firing: 3, exercise: 2, military: 0, gnss: 0, danger_area: 1});
  assert.equal(m.counts.with_launch_ref, 1);
  assert.ok(m.cautions[0].startsWith('0 件は'));
  assert.ok(validate(items, m));
  assert.throws(() => validate([{...items[0], category: 'other'}], null));
  assert.throws(() => validate([items[0], items[0]], null));
  assert.throws(() => validate(items, {...m, counts: counts([])}));
  const outside = {...items[0], area: {basis: 'shape', shapes: [{kind: 'point', points: [[10, 170]]}]}};
  assert.throws(() => validate([outside], null));
});

test('Q-line parsing', () => {
  assert.deepEqual(qLine('Q) RPHI/QWMLW/IV/BO /W /000/999/1245N11610E038'),
    {fir: 'RPHI', code: 'QWMLW', subject: 'WM', condition: 'LW', lat: 12.75, lon: 116.1667});
  assert.equal(qLine('no q line'), null);
});

test('the same id twice in one dataset becomes one item (the later start)', () => {
  const av = fixture.aviation.items.find((x) => build({aviation: {items: [x]}}, {now: NOW}).length === 1);
  assert.ok(av, 'fixture has an aviation item that is kept');
  const later = {...av, start: (Number.isFinite(av.start) ? av.start : NOW) + 3600e3};
  const out = build({aviation: {items: [av, later]}}, {now: NOW});
  assert.equal(out.length, 1);
  assert.equal(build.lastDuplicates, 1);
  assert.equal(out[0].valid.start_utc, new Date(later.start).toISOString().replace(/\.\d{3}Z$/, 'Z'));
  const m = meta({items: out, now: NOW, snapshot: {}, retrievedAt: '2026-11-05T00:00:00Z', duplicates: 1});
  validate(out, m);
  assert.equal(m.duplicates_merged, 1);
});
