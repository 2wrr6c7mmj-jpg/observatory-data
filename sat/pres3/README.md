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
CI（`.github/workflows/test.yml`）は numpy だけを入れる。`tests/test_sat_pres3.py` は numpy だけで動く部分（特徴・差分・見本と履歴・評価の関数）を、
`tests/test_sat_pres3_daily.py` は 1 日ずつの判定（合成データ。scikit-learn の要る 2 件は CI では飛ばす）を試す。

## 再現の手順

作業の置き場（数 GB。**リポジトリの外**。中を指定すると止まる）を `--work` か環境変数 `PRES3_WORK` で指定する。全部リポジトリの根から実行する。

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
                                     # --publish-cells は pres3_oos.csv だけを書く（pres3.json とその生成時刻は変えない）
# 7. confirm-1 の仕分けの表（docs/data/sat/eval/confirm-design.md の §2）。pres3_oos.csv だけを読む。作業の置き場は要らない
python -m sat.pres3.triage           # --json で値だけ、--no-ci で CI を省く
```

`pres3_oos.csv` の列：`p_pres3_*` が採用した構成、`p_hist_*` が履歴だけ、`p_gen_lopo` が答えを使わない汎用の判定、
`p_opt_lopo`・`p_opt_fwd`（末尾の 2 列、小数 6 桁）が **S1 を使わない構成 pres-3o**（`('opt', 1)`）。confirm-1 は pres-3o を使う。

所要時間の目安（4 コア・15 GB、3部の実測）：取得は回線しだい（実験のときの記録では 9 港で 40 分前後）、パッチ 約 3 分、特徴 約 5 分（4 並列。湛江が最長）、
差分・追跡 約 2 分、融合 約 38 分（大半は融合の側の学習：重みつきの学習 130 回。汎用の判定は `<work>/gen_cache.pkl` に残るが、2 回目も 30 分以上かかる）、最終の数字 約 1 分。

**`build` をやり直したら、後の段のファイルは作り直す。** 特徴・差分・汎用の判定・融合のファイル（`feat_*.npz`・`diff_*.npz`・`path_*.npz`・
`gen_cache.pkl`・`fuse_out.pkl`）は升を「升の一覧（`cells_<港>.json`）の何行目か」で引く。答えや撮影が増えた後に `build` をやり直すと行がずれるので、
各ファイルに升の一覧の指紋（SHA-256）を入れてあり、合わないファイルは使わない（特徴と汎用の判定は計算し直し、差分・融合は止まって作り直しを求める）。
迷ったら作業の置き場の `data/` 以外を消して 2 からやり直す。

乱数の種は固定（勾配ブースティング `random_state=0`、ブートストラップ `seed=0`）。同じ窓・同じ版なら数字は一致する。

## 1 日ずつ判定する形（confirm-1、`daily.py`。PR-2）

確認回し（`docs/data/sat/eval/confirm-design.md`）で毎日の判定に使う形。**まだどのワークフローからも呼ばれない**（つなぐのは後の PR）。
判定器は pres-3o＝S1 を使わない構成 `('opt', 1)`（答えを使わない汎用の判定＋同じ区間の判定日より前の答えの見本・履歴）。

ある日 d の点数が見てよいもの：**像は d 以前**（区間の参照は d より前の晴れの日）、**答えは d より前の所有者の答えだけ**（自動確定は答えではない）、
**係数は締め日より前の答えで学んだ固定のもの**。`tests/test_sat_pres3_daily.py` が合成データで、後の日の像・答えを変えても点数が変わらないことを確かめる。

置き場は 3 つとも（港・区間・日）のキーで引く（升の一覧の何行目か、では引かない）。**どれもリポジトリの外に置く**（中を指定すると止まる）。

| 置き場 | 形 |
|---|---|
| パッチの保管 `<store>/<港>/<YYYY-MM>.npz` | `patch` [n, 24, 41, 5] float16、`cells`（升ごとの記録の JSON：区間・日・雲・影・欠測・撮影の ID と時刻。**答えは入れない**）、`offs`。1 升 約 7.6 KB（圧縮後） |
| 係数 `<model>.npz` | 2 つの勾配ブースティングの木を配列にしたもの（節の値・特徴・閾値・欠測の行き先・子）、学習の升の港抜きの汎用の点数、特徴の名前、見出しの JSON（版 `pres3o-YYYYMM`・締め日・学習の升の数・scikit-learn と numpy の版）。pickle は使わない。**予測は numpy だけ**（scikit-learn が要るのは学習のときだけ。scikit-learn の予測と 1e-12 以内で一致） |
| 点数の表（CSV） | `site_id, section_id, date, acquired_utc, p, model_version, cloud_shadow, nodata, status, answers_through`（`sat/confirm.py` の入力の列と同じ名前・単位） |

点数の表の読み方：`p`＝判定器の点数（0〜1、小数 6 桁。在が高い）。**空**＝像なし、または区間の参照の晴れの日が 6 日未満（判定器の入力が足りない。受け取る側が `no_reference` として回す）。
`cloud_shadow`・`nodata`＝区間の画素のうち雲＋影（SCL 3・8・9・10）・欠測の割合（0〜1、小数 4 桁）。`status`＝`no_image`（雲＋影が 0.2 超か欠測あり）か空。
`answers_through`＝この点数に使った、その区間の最後の答えの日（判定日より前。無ければ空）。`--diagnostics` を付けると `n_answers`（使った答えの数）・`n_reference_days`・`scene_id` も出す。
答えは「その区間・その日のパッチが保管にあるもの」だけを使う（像の無い日の答えは見本にできない）。
自動確定か回すかの仕分け（閾値・答えの数・日数・港の on/off）はこの表の外（`sat/confirm.py`）。

```sh
# 研究の作業の置き場のパッチを、キーで引く保管に写す（履歴の種）
python -m sat.pres3.daily import --work "$PRES3_WORK" --store /path/to/store
# 係数：締め日より前の答えで学ぶ（scikit-learn。港抜きの汎用の判定 9 回＋全体 1 回＋融合 1 回）
python -m sat.pres3.daily fit --work "$PRES3_WORK" --cutoff 2026-11-01 --model /path/to/pres3o-202611.npz
# 新しい撮影：直近 N 日を検索し、保管に無い日の窓を取ってパッチにする（rasterio・ネットワーク。CDSE は使わない）
python -m sat.pres3.daily fetch --store /path/to/store --windows /path/to/windows --days 10
# 判定：保管の最後の日（--since で範囲）。答えは labels/<港>-s2-answers.csv と labels/<港>-confirm-answers.csv（あれば）
python -m sat.pres3.daily score --store /path/to/store --model /path/to/pres3o-202611.npz --out /path/to/scores.csv
# 確かめ：研究の前進検証を 1 日ずつやり直し、pres3_oos.csv の p_opt_fwd と比べる（境界ごとに学び直す）
python -m sat.pres3.daily check --work "$PRES3_WORK" --store /path/to/store --models /path/to/check-models
```

- 取得するタイルは港ごとに 1 つ：`inputs/scenes.csv` の最後の 365 日でいちばん多いタイル（青島港内 51STA、古鎮口 51STV。古鎮口は 2024 年までは 50SQE だった）。タイルの雲 85% 以下、1 日 1 撮影（研究の一覧と同じ選び方）。
- 水の色の補正と「晴れの日」の印は、その日のその港の**全部の区間**のパッチから作る。保管から区間を間引くと点数が変わる（間引くなら港×日の水の色を別に残す作りに直す）。
- 学習（`fit`）と確かめ（`check`）は、いまは研究の作業の置き場の升と特徴から学ぶ。保管と答えのファイルだけから学ぶ形（月 1 回の再学習）は後の PR。

### 取得について

- 出どころ：Element84 Earth Search（`https://earth-search.aws.element84.com/v1`、STAC API）→ AWS Open Data の Sentinel-2 L2A の COG。認証なし。
  帯は blue・green・red・nir（10 m）と swir16・scl（20 m）、港の範囲（`sat/sites.json` の bbox）＋0.004 度の窓だけを読む。CDSE（Sentinel Hub）の枠は使わない。
- 既定では `inputs/scenes.csv` に並べた撮影をそのまま取る（公開の数字の元になった撮影の一覧）。一覧は最初、日付の検索
  （`--start 2024-10-01 --end 2026-11-19` と `--start 2023-06-01 --end 2024-09-30`、`--max-cloud 85`。撮影の多い MGRS タイル 1 つ・1 日 1 撮影）で作った。
  青島（QDH）には最初の試しで取った別のタイルの撮影も入っている（同じ日に複数あれば `build` が nodata の少ない方を使う）。
  日付の検索からやり直すと、タイルの選び方が期間で変わるので同じ一覧にはならない。
- 一覧のうち、カタログから消えた撮影があれば `fetch` が「not in the catalogue」に出す（その分は再現できない）。
- **拒否（HTTP 4xx。403・429 を含む）には送り直さない**。そのまま失敗として出す（`daily fetch` はその港を止めて `::error` に出す）。
  送り直すのは一時の失敗（つながらない・途中で切れた・時間切れ・5xx）だけ：STAC の検索は最大 3 回（5・10・15 秒あけて）、
  `fetch` の撮影の窓は最大 2 回（4・8 秒）。COG を読む GDAL は `GDAL_HTTP_RETRY_CODES=500,502,503,504`（GDAL 3.10 以上。GDAL の既定の
  429 を外す）で最大 3 回（3 秒から倍々）。GDAL 3.10 より古いと 429 を外せないので、送り直さない（`GDAL_HTTP_MAX_RETRY=0`）。
  `tests/test_sat_pres3_fetch.py` が偽の応答（と rasterio のある所では 127.0.0.1 の偽のサーバー）で確かめる。外への要求では試さない。

### 再現に使う、リポジトリの中の入力

| 入力 | 使い方 |
|---|---|
| `docs/data/sat/labels/<港>/review.json`（8 港）、`labels/ZJG-s2-sections.geojson` | 区間の線と 120 m 沖向きの枠 |
| `docs/data/sat/labels/<港>-s2-answers.csv`、`labels/ZJG-s2-labels.csv`（verified） | 所有者の答え（正解。見本・履歴にも使う） |
| `docs/data/sat/eval/presence.json`（eval-presence-6） | 評価票の升かどうかと、現行 S1 の判定 p |
| `sat/pres3/inputs/s1_cells.csv` | 評価票の升 737 の、対の S1 撮影の帯の VH 平均 `xh`（dB）と時刻差 `dt_h`。2部の S1 の調査（pres-2）の出力から写した。リポジトリのコードでは作り直せない |
| `sat/pres3/inputs/scenes.csv` | 取得する撮影の一覧 |

### S1 の 3 列について（数字を読むとき）

融合に渡す S1 は、現行の判定 p・対の S1 撮影の帯の VH 平均 xh・時刻差 dt_h の 3 列。

- 対の S1 撮影が **S2 の日より後**（dt_h > 0、最大 +79.4 時間）の升が、LOPO で 365/737、前進で 302/516 ある（旅順は 125/143・107/107）。
  **運用では判定日にその S1 は手に入らない。** S1 を使わない構成（`opt`）は LOPO 0.870・前進 0.888 で、採用した構成（0.871・0.893）とほぼ同じ。
  毎日の流れにつなぐ段では、判定日以前の S1 だけで測り直す。
- **湛江の現行 S1 の p は較正の標本内**（評価票で 44 升とも `oos: false`）。湛江の pres-3 の数字は標本外でない（設計書 §6 (g)）。

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
| `daily.py` | confirm-1：1 日ずつ判定する形。パッチの保管・係数のファイル・点数の表（上の節） |
| `triage.py` | confirm-1：確信のある升（自動確定）と回す升の仕分けの表。規則 v1。`pres3_oos.csv` だけを読む（numpy） |

## 出典

- Contains modified Copernicus Sentinel data 2023–2026（Element84 Earth Search／AWS Open Data 経由）
- 岸壁の線 © OpenStreetMap contributors
