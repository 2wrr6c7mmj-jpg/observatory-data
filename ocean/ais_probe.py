"""AIS 受信試験（probe のみ）：aisstream.io の WebSocket を湛江周辺で一定時間購読し、集計だけを書く。

生のメッセージは保存しない（メモリ上で数えるだけ）。API キーは環境変数 AISSTREAM_API_KEY から読み、
ログ・報告には出さない（例外文にキーが含まれても伏せる）。

    python -m ocean.ais_probe --minutes 20 --report DIR

区画：内側 = 区画 v2（110.425–110.455E, 21.195–21.265N）、外側 = 110.3–110.6E, 21.0–21.4N。
対照（受信が届いているかの確認、件数のみ）：海口（瓊州海峡）と香港。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

URL = "wss://stream.aisstream.io/v0/stream"
INNER = (21.195, 21.265, 110.425, 110.455)  # lat0, lat1, lon0, lon1
OUTER = (21.0, 21.4, 110.3, 110.6)
CONTROLS = {
    "haikou": (19.95, 20.25, 110.0, 110.6),
    "hongkong": (22.20, 22.40, 114.05, 114.30),
}
# 受信範囲の確認用の広域（0.5° の升目ごとのユニーク MMSI 数だけを数える）
REGION = (18.0, 23.5, 108.0, 115.0)
CELL = 0.5
STILL_KN = 0.5
POS_TYPES = {"PositionReport", "StandardClassBPositionReport", "ExtendedClassBPositionReport"}


def inside(box, lat, lon) -> bool:
    return box[0] <= lat <= box[1] and box[2] <= lon <= box[3]


def bbox(box):
    return [[box[0], box[2]], [box[1], box[3]]]


class Tally:
    def __init__(self):
        self.msgs = 0
        self.by_type = Counter()
        self.area_msgs = Counter()
        self.area_mmsi = defaultdict(set)
        self.ship_type = {}          # mmsi -> type code (outer only)
        self.last = {}               # mmsi -> (lat, lon, sog, t) of last position (outer only)
        self.pos_times = defaultdict(list)  # mmsi -> receive times of position reports (outer)
        self.mid = Counter()         # MID of unique outer MMSI
        self.cls = {}                # mmsi -> 'A' / 'B'
        self.errors = []
        self.first_msg_s = None
        self.cells = defaultdict(set)

    def feed(self, m: dict, t: float, t0: float):
        if "error" in m:
            self.errors.append(str(m.get("error"))[:200])
            return
        mt = m.get("MessageType", "?")
        self.by_type[mt] += 1
        if mt == "SubscriptionConfirmation":
            return
        md = m.get("MetaData") or {}
        try:
            mmsi = int(md.get("MMSI"))
            lat = float(md.get("latitude"))
            lon = float(md.get("longitude"))
        except (TypeError, ValueError):
            return
        self.msgs += 1
        if self.first_msg_s is None:
            self.first_msg_s = t - t0
        if inside(REGION, lat, lon):
            self.cells[(int(lat // CELL), int(lon // CELL))].add(mmsi)
        areas = []
        if inside(OUTER, lat, lon):
            areas.append("outer")
            if inside(INNER, lat, lon):
                areas.append("inner")
        for k, b in CONTROLS.items():
            if inside(b, lat, lon):
                areas.append(k)
        for a in areas:
            self.area_msgs[a] += 1
            self.area_mmsi[a].add(mmsi)
        if "outer" not in areas:
            return
        body = (m.get("Message") or {}).get(mt) or {}
        if mt in POS_TYPES:
            sog = body.get("Sog")
            self.last[mmsi] = (lat, lon, sog, t)
            self.pos_times[mmsi].append(t)
            self.cls[mmsi] = "A" if mt == "PositionReport" else "B"
        if mt == "ShipStaticData":
            self.ship_type[mmsi] = body.get("Type")
            self.cls.setdefault(mmsi, "A")
        elif mt == "StaticDataReport":
            rb = body.get("ReportB") or {}
            if rb.get("Valid") or rb.get("ShipType"):
                self.ship_type[mmsi] = rb.get("ShipType")
            self.cls.setdefault(mmsi, "B")

    def report(self, meta: dict) -> dict:
        outer = self.area_mmsi["outer"]
        inner = self.area_mmsi["inner"]
        mids = Counter(str(x)[:3] for x in outer)
        types = Counter(str(self.ship_type.get(x, "不明")) for x in outer)
        types_inner = Counter(str(self.ship_type.get(x, "不明")) for x in inner)
        intervals = []
        per_ship_med = []
        for x, ts in self.pos_times.items():
            d = [b - a for a, b in zip(ts, ts[1:]) if b > a]
            intervals += d
            if d:
                per_ship_med.append(statistics.median(d))
        still = []
        for x in sorted(inner):
            lt = self.last.get(x)
            if not lt or lt[2] is None or lt[2] > STILL_KN or not inside(INNER, lt[0], lt[1]):
                continue
            still.append({"mid": str(x)[:3], "type": self.ship_type.get(x), "cls": self.cls.get(x),
                          "lat": round(lt[0], 3), "lon": round(lt[1], 3), "sog": lt[2],
                          "n_pos": len(self.pos_times.get(x, []))})

        def q(v, p):
            if not v:
                return None
            v = sorted(v)
            return round(v[min(len(v) - 1, int(p * len(v)))], 1)
        return {
            **meta,
            "messages_with_position": self.msgs,
            "first_message_after_s": None if self.first_msg_s is None else round(self.first_msg_s, 1),
            "by_message_type": dict(self.by_type.most_common()),
            "area_messages": dict(self.area_msgs),
            "area_unique_mmsi": {k: len(v) for k, v in self.area_mmsi.items()},
            "outer_class": dict(Counter(self.cls.get(x, "?") for x in outer)),
            "outer_mid": dict(mids.most_common(15)),
            "outer_ship_type": dict(types.most_common()),
            "inner_ship_type": dict(types_inner.most_common()),
            "inner_still": still,
            "interval_s_all": {"n": len(intervals), "p10": q(intervals, .1), "p50": q(intervals, .5),
                               "p90": q(intervals, .9)},
            "interval_s_per_ship_median": {"ships": len(per_ship_med), "p10": q(per_ship_med, .1),
                                           "p50": q(per_ship_med, .5), "p90": q(per_ship_med, .9)},
            "errors": self.errors[:5],
            "cells": {f"{a*CELL:.1f}N{b*CELL:.1f}E": len(v) for (a, b), v in sorted(self.cells.items())},
        }


def scrub(s: str, key: str) -> str:
    return s.replace(key, "***") if key else s


async def run(minutes: float, key: str, tally: Tally, meta: dict):
    import websockets
    t0 = time.time()
    end = t0 + minutes * 60
    sub = {"APIKey": key,
           "BoundingBoxes": [bbox(REGION)]}
    connects = 0
    while time.time() < end:
        connects += 1
        try:
            async with websockets.connect(URL, open_timeout=20, max_size=2 ** 22) as ws:
                await ws.send(json.dumps(sub))
                while time.time() < end:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=max(1, min(30, end - time.time())))
                    except asyncio.TimeoutError:
                        continue
                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8", "replace")
                    try:
                        m = json.loads(raw)
                    except ValueError:
                        tally.errors.append("非 JSON の受信")
                        continue
                    if m.get("MessageType") == "SubscriptionConfirmation":
                        meta["auth"] = "成功（SubscriptionConfirmation を受信）"
                        meta["compression"] = (m.get("Message") or {}).get("CompressionEnabled")
                    tally.feed(m, time.time(), t0)
                    if "error" in m:
                        meta["auth"] = "失敗またはエラー：" + scrub(str(m["error"]), key)[:200]
        except Exception as e:  # noqa: BLE001
            msg = scrub(f"{type(e).__name__}: {e}", key)[:300]
            tally.errors.append(msg)
            print("接続エラー:", msg, file=sys.stderr)
            if connects >= 6:
                break
            await asyncio.sleep(min(30, 5 * connects))
    meta["connects"] = connects
    meta["elapsed_min"] = round((time.time() - t0) / 60, 1)


def text_reports(r: dict) -> dict:
    s = []
    s.append(f"開始 {r['start_utc']} UTC、受信 {r.get('elapsed_min')} 分、接続 {r.get('connects')} 回")
    s.append(f"キー認証：{r.get('auth', '不明（確認メッセージなし）')}、圧縮 {r.get('compression')}")
    s.append(f"位置付きメッセージ {r['messages_with_position']}（最初の受信まで {r['first_message_after_s']} 秒）")
    s.append("種類別：" + json.dumps(r["by_message_type"], ensure_ascii=False))
    s.append("区画別メッセージ数：" + json.dumps(r["area_messages"], ensure_ascii=False))
    s.append("区画別ユニーク MMSI：" + json.dumps(r["area_unique_mmsi"], ensure_ascii=False))
    s.append("外側の Class：" + json.dumps(r["outer_class"], ensure_ascii=False))
    s.append("外側の MID（国籍上位）：" + json.dumps(r["outer_mid"], ensure_ascii=False))
    s.append("送信間隔（全位置報告の隣接差、秒）：" + json.dumps(r["interval_s_all"], ensure_ascii=False))
    s.append("送信間隔（船ごとの中央値の分布、秒）：" + json.dumps(r["interval_s_per_ship_median"], ensure_ascii=False))
    if r["errors"]:
        s.append("エラー：" + " / ".join(r["errors"]))
    t = ["船種コード（外側、ユニーク MMSI 数。不明 = 静的情報を受けていない）："
         + json.dumps(r["outer_ship_type"], ensure_ascii=False),
         "船種コード（内側）：" + json.dumps(r["inner_ship_type"], ensure_ascii=False)]
    st = [f"内側で速度 ≤{STILL_KN} kn の船 {len(r['inner_still'])} 隻（MID・船種・Class・位置は 0.001° に丸め）："]
    for x in r["inner_still"][:60]:
        st.append(f"MID{x['mid']} 種{x['type']} {x['cls']} {x['lat']},{x['lon']} sog{x['sog']} 報告{x['n_pos']}")
    c = [f"広域 {REGION} の 0.5° 升目ごとのユニーク MMSI（升目の南西端、0 の升目は省略）："]
    c += [f"{k}: {v}" for k, v in r.get("cells", {}).items()]
    return {"summary": "\n".join(s), "types": "\n".join(t), "inner-still": "\n".join(st),
            "cells": "\n".join(c)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=20)
    ap.add_argument("--report", required=True)
    a = ap.parse_args(argv)
    key = os.environ.get("AISSTREAM_API_KEY", "").strip()
    meta = {"start_utc": time.strftime("%Y-%m-%d %H:%M", time.gmtime())}
    out = Path(a.report)
    out.mkdir(parents=True, exist_ok=True)
    tally = Tally()
    if not key:
        meta["auth"] = "キーなし（secret AISSTREAM_API_KEY が空、または未登録）"
    else:
        asyncio.run(run(a.minutes, key, tally, meta))
    r = tally.report(meta)
    (out / "result.json").write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    for k, v in text_reports(r).items():
        (out / f"{k}.txt").write_text(v, encoding="utf-8")
    print(text_reports(r)["summary"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
