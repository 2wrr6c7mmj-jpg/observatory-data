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
