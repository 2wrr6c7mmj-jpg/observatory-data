# sat-raw/2026-11-10（raw-1）

毎日の Sentinel-1 処理（sat-ships）が数えた撮影の切り出しを、取得したままの形で残したもの。main には入れない。14 日より古い日のブランチは毎日の実行で消す。

- `<地点>/<撮影>.png`：CDSE Process API の応答そのもの（16 bit RGBA：VV・VH・局所入射角・データマスク。dB = 値/800 − 50、0 は値なし）
- `<地点>/<撮影>.json`：範囲・画素の大きさ・軌道・PU・陸マスクの版と、cfar-v1 の結果（再実行の照合用）
- `masks/<地点>-land-v<N>.png`：検出に渡した陸マスク（0 水、1 陸、2 構造物）

読み方：`python -m sat.raw_store fetch --dest DIR` → `sat.raw_store.load(DIR/<日>/<地点>/<撮影>.json)`、`python -m sat.raw_store check DIR/<日>` で cfar-v1 を回し直して照合。

Contains modified Copernicus Sentinel data 2026. 陸マスク：(c) OpenStreetMap contributors, ODbL 1.0
