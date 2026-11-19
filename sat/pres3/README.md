# sat/pres3 — pres-3（試行・研究）

海軍の岸壁区間 × 日の在/不在を、Sentinel-2 L2A の公開 COG（5 帯）＋区間の過去の答え（見本・履歴）＋像の差分＋現行 S1 の判定で
求める研究コード。設計と結果は `docs/data/sat/eval/pres3-design.md`、数字は `docs/data/sat/eval/pres3.json`、升ごとの標本外の確率は
`docs/data/sat/eval/pres3_oos.csv`。

**毎日の収集の流れ（`sat/run.py`・ワークフロー）には接続していない。公開の数・画面・既存の評価票は変えない。**
**目標の正答率 95% には全升では届いていない**（LOPO 0.871、前進 0.893）。

## 依存

- Python 3.12 以上、numpy
- rasterio（取得 `fetch` とパッチ `build` だけ）
- scikit-learn（融合 `fuse` だけ。`HistGradientBoostingClassifier`）

確かめた版：Python 3.13.18、numpy 2.5.3、rasterio 1.5.2、scikit-learn 1.9.1（版が違うと勾配ブースティングの結果が小数第 3 位で動くことがある）。
CI（`.github/workflows/test.yml`）は numpy だけを入れる。`tests/test_sat_pres3.py` は numpy だけで動く部分（特徴・差分・見本と履歴・評価の関数）を試す。

## 再現の手順

作業の置き場（数 GB。リポジトリの `docs/` の外）を `--work` か環境変数 `PRES3_WORK` で指定する。全部リポジトリの根から実行する。

```sh
export PRES3_WORK=/path/to/work

# 1. 取得：inputs/scenes.csv の 2,517 撮影（9 港、2023-06〜2026-11）の窓を <work>/data/<港>/*.npz に置く（約 4.6 GB）
python -m sat.pres3.fetch
# 2. 区間×日の整流パッチ（24×41×5 帯）と升の一覧（答え・評価票の升かどうか）
python -m sat.pres3.build
# 3. 答えを使わない特徴 545 個（港ごとに独立。並べて走らせてよい）
python -m sat.pres3.dataset --ports ZJG &
python -m sat.pres3.dataset --ports LSN,SYL,XSN &
python -m sat.pres3.dataset --ports QDH,SLP,ZDH &
python -m sat.pres3.dataset --ports QJG,QGZ & wait
# 4. 像の差分の特徴（52 個）。pathfeat は「試したが採用していない」特徴で、試した構成の表を出すのに要る
python -m sat.pres3.diffeat
python -m sat.pres3.pathfeat
# 5. 融合と標本外の評価（LOPO・前進）。試した構成の表を出す
python -m sat.pres3.fuse
# 6. 採用した構成 ('opt+diff+S1', 1) の数字：<work>/pres3_results.json・pres3_oos.csv
python -m sat.pres3.final            # --publish を付けると docs/data/sat/eval/pres3.json・pres3_oos.csv を書く
```

所要時間の目安（4 コア・15 GB、3部の実測）：取得は回線しだい（実験のときの記録では 9 港で 40 分前後）、パッチ 約 3 分、特徴 約 5 分（4 並列。湛江が最長）、
差分・追跡 約 2 分、融合 約 38 分（汎用の判定の学習が大半。`<work>/gen_cache.pkl` に残るので 2 回目は 約 2 分）、最終の数字 約 1 分。

乱数の種は固定（勾配ブースティング `random_state=0`、ブートストラップ `seed=0`）。同じ窓・同じ版なら数字は一致する。

### 取得について

- 出どころ：Element84 Earth Search（`https://earth-search.aws.element84.com/v1`、STAC API）→ AWS Open Data の Sentinel-2 L2A の COG。認証なし。
  帯は blue・green・red・nir（10 m）と swir16・scl（20 m）、港の範囲（`sat/sites.json` の bbox）＋0.004 度の窓だけを読む。CDSE（Sentinel Hub）の枠は使わない。
- 既定では `inputs/scenes.csv` に並べた撮影をそのまま取る（公開の数字の元になった撮影の一覧）。一覧は最初、日付の検索
  （`--start 2024-10-01 --end 2026-11-19` と `--start 2023-06-01 --end 2024-09-30`、`--max-cloud 85`。撮影の多い MGRS タイル 1 つ・1 日 1 撮影）で作った。
  青島（QDH）には最初の試しで取った別のタイルの撮影も入っている（同じ日に複数あれば `build` が nodata の少ない方を使う）。
  日付の検索からやり直すと、タイルの選び方が期間で変わるので同じ一覧にはならない。
- 一覧のうち、カタログから消えた撮影があれば `fetch` が「not in the catalogue」に出す（その分は再現できない）。

### 再現に使う、リポジトリの中の入力

| 入力 | 使い方 |
|---|---|
| `docs/data/sat/labels/<港>/review.json`（8 港）、`labels/ZJG-s2-sections.geojson` | 区間の線と 120 m 沖向きの枠 |
| `docs/data/sat/labels/<港>-s2-answers.csv`、`labels/ZJG-s2-labels.csv`（verified） | 所有者の答え（正解。見本・履歴にも使う） |
| `docs/data/sat/eval/presence.json`（eval-presence-6） | 評価票の升かどうかと、現行 S1 の判定 p |
| `sat/pres3/inputs/s1_cells.csv` | 評価票の升 737 の、対の S1 撮影の帯の VH 平均 `xh`（dB）と時刻差 `dt_h`。2部の S1 の調査（pres-2）の出力から写した。リポジトリのコードでは作り直せない |
| `sat/pres3/inputs/scenes.csv` | 取得する撮影の一覧 |

`presence.json` と答えの CSV は日々増える。数字は 2026-11-19 の時点のもの（評価票の升 737、答えのある升 1,531）で、後の版で実行すると升が増えて数字が変わる。

## ファイル

| ファイル | 中身 |
|---|---|
| `fetch.py` | 取得（rasterio） |
| `labels.py` | 区間の枠・答え・評価票の升（リポジトリを読むだけ） |
| `build.py` | 整流パッチ（rasterio） |
| `feats.py` | 答えを使わない特徴。区間の参照（画素ごとの分位）は判定日より前の晴れの日だけから作る |
| `dataset.py` | 特徴の計算と保存、成績の 1 行表示 |
| `diffeat.py` | 像の差分の特徴（直前の答えの日・過去の在/不在の日・過去の晴れの日の空/満の像との増減） |
| `pathfeat.py` | 変化の追跡（試したが採用していない） |
| `model.py` | 汎用の判定（港抜き）と、見本・履歴の特徴（判定日より前の答えだけ） |
| `evalcv.py` | LOPO・前進検証・撮影単位のブートストラップ・変わった升 |
| `fuse.py` | 融合と評価、試した構成の表 |
| `final.py` | 採用した構成の数字、公開用の JSON・CSV |

## 出典

- Contains modified Copernicus Sentinel data 2023–2026（Element84 Earth Search／AWS Open Data 経由）
- 岸壁の線 © OpenStreetMap contributors
