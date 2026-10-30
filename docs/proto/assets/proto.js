// 分析の試作：共通の描画（外部ライブラリなし）
'use strict';
const P = (() => {
  const NS = 'http://www.w3.org/2000/svg';
  const DAY = 86400000;
  const el = (tag, attrs = {}, parent) => {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) if (v !== undefined && v !== null) e.setAttribute(k, v);
    if (parent) parent.appendChild(e);
    return e;
  };
  const h = (tag, attrs = {}, html) => {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    if (html !== undefined) e.innerHTML = html;
    return e;
  };
  const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const t = s => Date.parse(s + 'T00:00:00Z');
  const iso = ms => new Date(ms).toISOString().slice(0, 10);
  const fmt = (v, nd = 1) => v === null || v === undefined ? '—' : (Number.isInteger(v) ? String(v) : v.toFixed(nd));
  const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  async function load(name) {
    const r = await fetch('../data/' + name, { cache: 'no-cache' });
    if (!r.ok) throw new Error(name + ': ' + r.status);
    return r.json();
  }

  function niceTicks(lo, hi, n = 5) {
    if (hi <= lo) hi = lo + 1;
    const step0 = (hi - lo) / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= step0) || step0;
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }

  function dateTicks(x0, x1) {
    const out = [];
    const d0 = new Date(x0), span = (x1 - x0) / DAY;
    if (span > 1800) { // 5年を超える期間は年の目盛り（第2弾の長い系列）
      const yStep = span > 3650 ? 2 : 1;
      for (let y = d0.getUTCFullYear() + 1; Date.UTC(y, 0, 1) <= x1; y++) if (y % yStep === 0) out.push({ v: Date.UTC(y, 0, 1), label: String(y) });
      return out;
    }
    const stepM = span > 900 ? 6 : span > 400 ? 3 : span > 120 ? 1 : 0;
    if (stepM) {
      let y = d0.getUTCFullYear(), m = d0.getUTCMonth() + 1;
      while (true) {
        if (m > 11) { y++; m -= 12; }
        const v = Date.UTC(y, m, 1);
        if (v > x1) break;
        if (m % stepM === 0) out.push({ v, label: m === 0 ? String(y) : (y + '-' + String(m + 1).padStart(2, '0')) });
        m++;
      }
    } else {
      for (let v = x0; v <= x1; v += 7 * DAY) out.push({ v, label: iso(v).slice(5) });
    }
    return out;
  }

  // 汎用の時系列図。series: {kind:'line'|'band'|'dots'|'bars'|'step', color, data:[{x,y,y0,y1}], r, width, dash}
  function chart(container, o) {
    const W = o.width || 960, H = o.height || 260, m = { l: 44, r: 12, t: o.ylabel ? 24 : 10, b: 24, ...(o.margin || {}) };
    const fig = h('div', { class: 'fig' });
    if (o.legend) {
      const lg = h('div', { class: 'legend' });
      for (const it of o.legend) lg.appendChild(h('span', {}, `<i class="${it.line ? 'ln' : ''}" style="background:${it.color}"></i>${esc(it.label)}`));
      fig.appendChild(lg);
    }
    const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': o.title || '' });
    fig.appendChild(svg);
    const tip = h('div', { class: 'tip' });
    fig.appendChild(tip);
    if (o.caption) fig.appendChild(h('div', { class: 'cap' }, o.caption));
    container.appendChild(fig);
    const xs = [], ys = [];
    for (const s of o.series) for (const p of s.data) {
      xs.push(p.x);
      for (const k of ['y', 'y0', 'y1']) if (p[k] !== null && p[k] !== undefined && isFinite(p[k])) ys.push(p[k]);
    }
    const x0 = o.xmin ?? Math.min(...xs), x1 = o.xmax ?? Math.max(...xs);
    let y0 = o.ymin ?? Math.min(0, ...ys), y1 = o.ymax ?? Math.max(...ys, 1);
    if (o.yclip) y1 = Math.min(y1, o.yclip);
    const yt = niceTicks(y0, y1, o.yticks || 5);
    y1 = Math.max(y1, yt[yt.length - 1]);
    const pad = o.bars ? (o.barWidth || 1) / 2 : 0;
    const X = v => m.l + (v - x0 + pad) / (x1 - x0 + 2 * pad || 1) * (W - m.l - m.r);
    const Y = v => H - m.b - (Math.min(v, y1) - y0) / (y1 - y0 || 1) * (H - m.t - m.b);
    const g = el('g', {}, svg);
    for (const v of yt) {
      el('line', { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), stroke: css('--grid'), 'stroke-width': 1 }, g);
      el('text', { x: m.l - 6, y: Y(v) + 4, 'text-anchor': 'end' }, g).textContent = o.yfmt ? o.yfmt(v) : v;
    }
    const ticks = o.xticks ? o.xticks(x0, x1) : dateTicks(x0, x1);
    for (const tk of ticks) {
      el('line', { x1: X(tk.v), x2: X(tk.v), y1: H - m.b, y2: H - m.b + 4, stroke: css('--axis') }, g);
      el('text', { x: X(tk.v), y: H - 6, 'text-anchor': 'middle' }, g).textContent = tk.label;
    }
    el('line', { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, stroke: css('--axis') }, g);
    if (o.ylabel) el('text', { x: m.l, y: 12, 'text-anchor': 'start', dx: -40 }, g).textContent = o.ylabel;
    for (const sp of o.spans || []) {
      el('rect', { x: X(sp.x0), width: Math.max(1, X(sp.x1) - X(sp.x0)), y: m.t, height: H - m.t - m.b, fill: sp.color, opacity: sp.opacity ?? 0.18 }, g);
    }
    for (const s of o.series) {
      const pts = s.data.filter(p => p.y !== null && p.y !== undefined || s.kind === 'band');
      if (s.kind === 'band') {
        let seg = [];
        const flush = () => {
          if (seg.length > 1) {
            const up = seg.map(p => `${X(p.x)},${Y(p.y1)}`), dn = seg.slice().reverse().map(p => `${X(p.x)},${Y(p.y0)}`);
            el('polygon', { points: up.concat(dn).join(' '), fill: s.color, stroke: 'none' }, g);
          }
          seg = [];
        };
        for (const p of s.data) { if (p.y0 === null || p.y0 === undefined || p.y1 === null || p.y1 === undefined) flush(); else seg.push(p); }
        flush();
      } else if (s.kind === 'line' || s.kind === 'step') {
        let d = '', pen = false, prev = null;
        for (const p of s.data) {
          if (p.y === null || p.y === undefined || (prev !== null && s.gap && p.x - prev > s.gap)) { pen = false; }
          if (p.y === null || p.y === undefined) { prev = null; continue; }
          if (s.kind === 'step' && pen) d += `H${X(p.x)}`;
          d += (pen ? 'L' : 'M') + X(p.x) + ',' + Y(p.y);
          pen = true; prev = p.x;
        }
        el('path', { d, fill: 'none', stroke: s.color, 'stroke-width': s.width || 2, 'stroke-dasharray': s.dash, 'stroke-linejoin': 'round' }, g);
      } else if (s.kind === 'dots') {
        for (const p of pts) el('circle', { cx: X(p.x), cy: Y(p.y), r: s.r || 3, fill: p.color || s.color, stroke: css('--surface'), 'stroke-width': 1.5 }, g);
      } else if (s.kind === 'bars') {
        const bw = Math.max(1, (X(x0 + (o.barWidth || 1)) - X(x0)) - 2);
        for (const p of pts) {
          const yb = Y(Math.max(0, y0)), yy = Y(p.y);
          el('rect', { x: X(p.x) - bw / 2 + (s.offset || 0) * bw / 2, y: Math.min(yb, yy), width: s.offset !== undefined ? bw / 2 : bw, height: Math.abs(yb - yy), fill: p.color || s.color, rx: Math.min(4, bw / 4) }, g);
        }
      }
    }
    for (const vl of o.vlines || []) {
      el('line', { x1: X(vl.x), x2: X(vl.x), y1: m.t, y2: H - m.b, stroke: vl.color || css('--ink2'), 'stroke-width': vl.width || 1, 'stroke-dasharray': vl.dash || '4 3' }, g);
      if (vl.label) el('text', { x: X(vl.x) + 3, y: m.t + 10 + (vl.row || 0) * 12, 'text-anchor': 'start' }, g).textContent = vl.label;
    }
    if (o.yclip) el('text', { x: W - m.r, y: 12, 'text-anchor': 'end' }, g).textContent = `縦軸は ${o.yclip} で切っています`;
    // ホバー：最も近い x の説明を出す
    if (o.tip) {
      const cross = el('line', { y1: m.t, y2: H - m.b, stroke: css('--ink2'), 'stroke-width': 1, opacity: 0 }, svg);
      const keys = Array.from(new Set(xs)).sort((a, b) => a - b);
      const hit = el('rect', { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: 'transparent' }, svg);
      const move = ev => {
        const box = svg.getBoundingClientRect();
        const px = (ev.clientX - box.left) / box.width * W;
        const xv = x0 - pad + (px - m.l) / (W - m.l - m.r) * (x1 - x0 + 2 * pad);
        let lo = 0, hi = keys.length - 1;
        while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (keys[mid] < xv) lo = mid; else hi = mid; }
        const k = Math.abs(keys[lo] - xv) < Math.abs(keys[hi] - xv) ? keys[lo] : keys[hi];
        const html = o.tip(k);
        if (!html) { tip.style.display = 'none'; cross.setAttribute('opacity', 0); return; }
        cross.setAttribute('x1', X(k)); cross.setAttribute('x2', X(k)); cross.setAttribute('opacity', 0.5);
        tip.innerHTML = html; tip.style.display = 'block';
        const fb = fig.getBoundingClientRect();
        let left = ev.clientX - fb.left + 14;
        if (left + tip.offsetWidth > fb.width - 4) left = ev.clientX - fb.left - tip.offsetWidth - 14;
        tip.style.left = Math.max(0, left) + 'px';
        tip.style.top = (ev.clientY - fb.top + 10) + 'px';
      };
      hit.addEventListener('mousemove', move);
      hit.addEventListener('mouseleave', () => { tip.style.display = 'none'; cross.setAttribute('opacity', 0); });
    }
    return { fig, svg, X, Y };
  }

  // 暦（年ごとの段、週を列、曜日を行）
  function calendar(container, days, colorOf, tipOf, legend) {
    const fig = h('div', { class: 'fig' });
    if (legend) {
      const lg = h('div', { class: 'legend' });
      for (const it of legend) lg.appendChild(h('span', {}, `<i style="background:${it.color}"></i>${esc(it.label)}`));
      fig.appendChild(lg);
    }
    const years = {};
    for (const d of days) (years[d.date.slice(0, 4)] ||= []).push(d);
    const cs = 12, gap = 2, left = 40, rowH = 7 * (cs + gap) + 18;
    const ys = Object.keys(years).sort();
    const W = left + 54 * (cs + gap) + 4, H = ys.length * rowH + 4;
    const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img' });
    fig.appendChild(svg);
    const tip = h('div', { class: 'tip' });
    fig.appendChild(tip);
    container.appendChild(fig);
    ys.forEach((y, yi) => {
      const top = yi * rowH + 14;
      el('text', { x: 0, y: top + 3 * (cs + gap) + 8 }, svg).textContent = y;
      const jan1 = Date.UTC(+y, 0, 1), off = (new Date(jan1).getUTCDay() + 6) % 7;
      for (let mo = 0; mo < 12; mo++) {
        const dd = (Date.UTC(+y, mo, 1) - jan1) / DAY;
        el('text', { x: left + Math.floor((dd + off) / 7) * (cs + gap), y: top - 3 }, svg).textContent = (mo + 1) + '月';
      }
      for (const d of years[y]) {
        const doy = (t(d.date) - jan1) / DAY;
        const wk = Math.floor((doy + off) / 7), wd = (new Date(t(d.date)).getUTCDay() + 6) % 7;
        const c = el('rect', { x: left + wk * (cs + gap), y: top + wd * (cs + gap), width: cs, height: cs, rx: 2, fill: colorOf(d) }, svg);
        c.addEventListener('mousemove', ev => {
          tip.innerHTML = tipOf(d); tip.style.display = 'block';
          const fb = fig.getBoundingClientRect();
          let lft = ev.clientX - fb.left + 12;
          if (lft + tip.offsetWidth > fb.width) lft = ev.clientX - fb.left - tip.offsetWidth - 12;
          tip.style.left = lft + 'px'; tip.style.top = (ev.clientY - fb.top + 10) + 'px';
        });
        c.addEventListener('mouseleave', () => { tip.style.display = 'none'; });
      }
    });
    return fig;
  }

  function table(container, head, rows, cls = []) {
    const tb = h('table');
    tb.innerHTML = '<thead><tr>' + head.map((x, i) => `<th class="${cls[i] || ''}">${esc(x)}</th>`).join('') + '</tr></thead><tbody>' +
      rows.map(r => '<tr>' + r.map((x, i) => `<td class="${cls[i] || ''}">${x}</td>`).join('') + '</tr>').join('') + '</tbody>';
    container.appendChild(tb);
    return tb;
  }

  // 「データの版と出典」
  function provenance(container, data) {
    const v = data.data_version || {}, s = data.source || {};
    const defs = data.definition_changes || [];
    const dl = h('dl', { class: 'meta' });
    dl.innerHTML = `
      <dt>データの版</dt><dd>台湾国防部 日報 daily.json 生成 ${esc(v.daily_generated_utc)}（UTC）／status.json 生成 ${esc(v.status_generated_utc)}／読み取り処理 第${esc(v.parser_version)}版／${esc(v.first_day)}〜${esc(v.last_day)}（${esc(v.days)}日分）</dd>
      <dt>この図の計算</dt><dd>${esc(data.computed_utc)}（UTC）に計算（計算の版 ${esc(v.algo_version)}）。元データ：<a href="../../data/taiwan-mnd/daily.csv">daily.csv</a>・<a href="../../data/taiwan-mnd/meta.json">meta.json</a>・<a href="../../data/taiwan-mnd/status.json">status.json</a></dd>
      <dt>出典</dt><dd>${esc(s.name)}「中共解放軍臺海周邊海、空域動態」日報（${(s.urls || []).map(u => `<a href="${esc(u)}">${esc(u)}</a>`).join('・')}）。${esc(s.rating)}</dd>
      ${defs.length ? `<dt>日報の書式・定義の変更（meta.json）</dt><dd><ul>${defs.map(x => `<li>${esc(x.date)}：${esc(x.change)}</li>`).join('')}</ul></dd>` : ''}`;
    container.appendChild(dl);
  }

  function defDate(s) { // "2024-08" / "2024-09-02" / "2022-11頃" → ms（月だけなら1日）
    const m = String(s).match(/(\d{4})-(\d{2})(?:-(\d{2}))?/);
    return m ? Date.UTC(+m[1], +m[2] - 1, m[3] ? +m[3] : 1) : null;
  }

  function oneIn(p) {
    if (p === null || p === undefined) return '—';
    if (p <= 0) return '1万日超に1回';
    const n = 1 / p;
    if (n >= 10000) return '1万日超に1回';
    if (n < 1.5) return 'ほぼ毎日';
    const r = n < 10 ? Math.round(n) : n < 100 ? Math.round(n / 5) * 5 : n < 1000 ? Math.round(n / 50) * 50 : Math.round(n / 500) * 500;
    return `およそ ${r.toLocaleString()} 日に1回`;
  }

  function pfmt(p) { // 生の極小値（1e-14 など）は出さず、下限で丸める
    if (p === null || p === undefined) return '—';
    if (p < 0.0001) return '<0.0001';
    return p < 0.01 ? p.toFixed(4) : p.toFixed(2);
  }


  // 格子（ヒートマップ）。rows/cols: [{key,label}]、value(r,c) は数値か null（null は対象外の色）
  function heat(container, o) {
    const fig = h('div', { class: 'fig' });
    if (o.legend) {
      const lg = h('div', { class: 'legend' });
      for (const it of o.legend) lg.appendChild(h('span', {}, `<i style="background:${it.color}"></i>${esc(it.label)}`));
      fig.appendChild(lg);
    }
    const left = o.left || 110, top = o.top || 18, ch = o.cellH || 18, W = o.width || 960;
    const cw = (W - left - 8) / o.cols.length, H = top + o.rows.length * ch + 4;
    const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': o.title || '' });
    fig.appendChild(svg);
    const tip = h('div', { class: 'tip' });
    fig.appendChild(tip);
    if (o.caption) fig.appendChild(h('div', { class: 'cap' }, o.caption));
    container.appendChild(fig);
    o.cols.forEach((c, j) => {
      if (c.tick) el('text', { x: left + j * cw, y: top - 5, 'text-anchor': 'start' }, svg).textContent = c.tick;
    });
    o.rows.forEach((rw, i) => {
      el('text', { x: left - 6, y: top + i * ch + ch * 0.68, 'text-anchor': 'end' }, svg).textContent = rw.label;
      o.cols.forEach((c, j) => {
        const v = o.value(rw, c);
        const rc = el('rect', { x: left + j * cw, y: top + i * ch, width: Math.max(0.5, cw - (cw > 4 ? 1 : 0)), height: ch - 2, fill: o.color(v, rw, c) }, svg);
        if (o.tip) {
          rc.addEventListener('mousemove', ev => {
            tip.innerHTML = o.tip(rw, c, v); tip.style.display = 'block';
            const fb = fig.getBoundingClientRect();
            let lft = ev.clientX - fb.left + 12;
            if (lft + tip.offsetWidth > fb.width) lft = ev.clientX - fb.left - tip.offsetWidth - 12;
            tip.style.left = Math.max(0, lft) + 'px'; tip.style.top = (ev.clientY - fb.top + 10) + 'px';
          });
          rc.addEventListener('mouseleave', () => { tip.style.display = 'none'; });
        }
      });
    });
    return { fig, svg };
  }

  // 0〜1 を薄い色から濃い色へ（--s1 を濃さで）
  function ramp(x) {
    if (x === null || x === undefined) return css('--na');
    const a = Math.max(0, Math.min(1, x));
    return a === 0 ? css('--grid') : `color-mix(in srgb, ${css('--s1')} ${Math.round(18 + a * 82)}%, ${css('--surface')})`;
  }

  // 複数の出典（第2弾）：data.sources = [{name, urls, rating, version, files, credit, license, processing}]
  function sources(container, data) {
    const dl = h('dl', { class: 'meta' });
    let html = '';
    for (const s of data.sources || []) {
      html += `<dt>${esc(s.name)}</dt><dd>データの版：${esc(s.version)}<br>元データ：${(s.files || []).map(f => `<a href="../../data/${esc(f)}">${esc(f)}</a>`).join('・')}<br>出典：${(s.urls || []).map(u => `<a href="${esc(u)}">${esc(u)}</a>`).join('・')}${s.rating ? '<br>' + esc(s.rating) : ''}${s.credit ? '<br>' + esc(s.credit) : ''}${s.license ? '<br>利用規約：' + esc(s.license) : ''}${s.processing ? '<br>' + esc(s.processing) : ''}</dd>`;
    }
    html += `<dt>この図の計算</dt><dd>${esc(data.computed_utc)}（UTC）に計算（計算の版 ${esc(data.algo_version)}、<a href="https://github.com/2wrr6c7mmj-jpg/observatory-data/blob/main/analysis/proto2.py">analysis/proto2.py</a>）</dd>`;
    dl.innerHTML = html;
    container.appendChild(dl);
  }

  return { load, chart, calendar, table, provenance, heat, ramp, sources, defDate, oneIn, pfmt, t, iso, fmt, esc, css, h, DAY };
})();
