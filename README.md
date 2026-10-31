# observatory-data

[世界情勢観測](https://observatory-watch.pages.dev/) の公開データの収集と分析を行うリポジトリです。東アジア、とくに中国の軍事動静を対象にしています。

## このリポジトリが担うこと

NOTAM Watch と共有している Cloudflare の無料枠（Pages デプロイ、Workers リクエスト、D1、cron）を使わないように、重い処理はすべてここで行います。

| 処理 | 置き場所 |
| --- | --- |
| 収集・解析 | このリポジトリの GitHub Actions（公開リポジトリのため無料） |
| データの配信 | このリポジトリの GitHub Pages（`docs/`）。世界情勢観測の画面が直接読み込む |

データを更新しても、世界情勢観測や NOTAM Watch の Pages は再公開されません。

## データ

| データ | 内容 | ファイル |
| --- | --- | --- |
| `taiwan-mnd` | 台湾国防部の日報（中国軍機・中間線越え・艦艇・公務船・気球） | [daily.csv](docs/data/taiwan-mnd/daily.csv)・[daily.json](docs/data/taiwan-mnd/daily.json)・[meta.json](docs/data/taiwan-mnd/meta.json)・[status.json](docs/data/taiwan-mnd/status.json) |
| `jcg-senkaku` | 海上保安庁：尖閣諸島周辺の中国海警局に所属する船舶等（接続水域入域・領海侵入の隻数、日ごと、2012年9月から） | [daily.csv](docs/data/jcg-senkaku/daily.csv)・[daily.json](docs/data/jcg-senkaku/daily.json)・[meta.json](docs/data/jcg-senkaku/meta.json)・[status.json](docs/data/jcg-senkaku/status.json) |
| `jso` | 統合幕僚監部の公表資料（中国・ロシア艦艇の海峡通過、往復、演習・政治イベントとの対応） | [passages.csv](docs/data/jso/passages.csv)・[round_trips.csv](docs/data/jso/round_trips.csv)・[context.csv](docs/data/jso/context.csv)・[summary.json](docs/data/jso/summary.json)・[meta.json](docs/data/jso/meta.json) |
| `launches` | ロケット・ミサイル発射に関わる NOTAM・航行警報（NOTAM Watch のデータから）、発射場・方位の推定、Launch Library 2 の打上げ記録との照合 | [events.json](docs/data/launches/events.json)・[notices.json](docs/data/launches/notices.json)・[ll2.json](docs/data/launches/ll2.json)・[meta.json](docs/data/launches/meta.json) |
| `pla-tracker` | PLA Activity Tracker（個人による集約、CC BY 4.0）：台湾国防部の日報を写した日ごとの中国軍機・中間線越え・艦艇の数（2026-01 から）。国防部の日報の自動取得を止めたため、2026-10-31 以降はこれで補う。国防部の日報との突き合わせつき | [daily.csv](docs/data/pla-tracker/daily.csv)・[daily.json](docs/data/pla-tracker/daily.json)・[compare.json](docs/data/pla-tracker/compare.json)・[meta.json](docs/data/pla-tracker/meta.json)・[status.json](docs/data/pla-tracker/status.json) |
| `sat`（試行） | Sentinel-1 による港内の艦船候補数（湛江 ZJG の 1 地点）。撮影ごとの港内水域・岸壁帯の候補数、候補の位置と長さの目安、切り出し画像（VV の灰 PNG と、疑似カラーの WebP `color/`）。岸壁に横付けされた船の占有数（最小〜最大、試行・精度未測定、相対軌道ごとの基準との変化）と、艦種の長さの分布・桟橋別の事前分布で狭めた隻数の推定（est-1：モデル上の最頻値とモデル内の 80% 区間〔横抱きの確率の誤り・誤検出・数えられない区間・OSM に無い岸壁・艦種表の誤り・区間の独立の仮定・上限 16 隻は区間に入っていない〕、試行・精度未測定、前提は推定を含む。艦種表は meta.json の quay.estimate.class_table）、その横に軍艦の可能性が高い船の隻数（est-2：横抱きの列数と区画・長さ・S2 の色から。較正 1 回目（所有者の目視ラベル 81 件、一部は仮置き）、モデル内の 80% 区間、AIS は使わない）、変化カラー `change/` | [ship_counts.csv](docs/data/sat/ship_counts.csv)・`detections/YYYY-MM.geojson`・[quay_counts.csv](docs/data/sat/quay_counts.csv)・`quay/YYYY-MM.geojson`・[quay_est2.csv](docs/data/sat/quay_est2.csv)・[meta.json](docs/data/sat/meta.json)・[status.json](docs/data/sat/status.json)・[usage.csv](docs/data/sat/usage.csv)・[CHANGES.md](docs/data/sat/CHANGES.md) |
| `health.json` | 収集の健全性（起動シーケンスと監視で共用する小さな集約）：源ごとの心拍（最後に成功した実行）・データの新しさ（最新の撮影日・公表日）・状態（ok／warn／alert／halt＝意図した停止）と根拠、今日の要点の数（湛江は `trial` の印付き）、Actions の定期実行の結論、Sentinel-1 の撮影の抜け・PU の 2 つの記録。基準は `analysis/health.py` の `THRESHOLDS`（暫定、2026-12 に見直し）。health.yml が収集の後と 4 時間ごとに作る（中身が同じなら書き直さないが 6 h ごとには書き直す）。**`generated_utc` が 12 h より古ければ監視（health.yml）自体が止まっている** | [health.json](docs/data/health.json) |
| `ocean`（実測のみ） | 外洋（台湾周辺 T・東シナ海 E・南西諸島 R）の Sentinel-1 の撮影頻度（0.25° 格子ごとの IW 通過数、相対軌道・向き・時刻、IW 以外のモード）と公開の遅れ。CDSE の OData カタログを probe（`probe/ocean-coverage` への push、ワークフロー ocean-probe.yml）で 1 回読んだ結果で、定期更新はしない | [coverage_probe.json](docs/data/ocean/coverage_probe.json) |

各データの定義、情報源の評価、定義変更の履歴、限界は `meta.json` に書いています（`ocean` は 1 回だけの実測で meta.json が無く、出典・方法・注記は coverage_probe.json の中の `source`・`method`・`attribution`・`notes` にあります）。

`ocean` の公開の遅れは、窓の終わり近くに撮影されてまだ公開されていない製品が数に入らない（右側の打ち切り）ため、実際より短めに出ている可能性があります。

`jcg-senkaku` の出典：海上保安庁ホームページ（https://www.kaiho.mlit.go.jp/mission/senkaku/senkaku.html）。公共データ利用規約 第1.0版（PDL1.0）に基づき、このリポジトリが PDF から数値を抜き出して日ごとの表に加工しています。加工したデータは海上保安庁が作成したものではありません。

`pla-tracker` の出典：Source: ROC Ministry of National Defense; compilation & charts: PLA Activity Tracker (https://pla-tracker.skyfaring.net)（CC BY 4.0）。records.csv を 1 日 1 回取り込み、数値の列を整数にして取得時刻を付けています（値は変えていない）。同じ日に国防部の日報（`taiwan-mnd`）があればそちらを優先します。艦艇（ships_total）は 2026-03 下旬から公務船を含み、中間線越え（median_line_cross）は国防部の median_line_or_adiz より狭い量なので、`taiwan-mnd` の列にそのままつなげません（meta.json の precedence・fields）。ファイルは main での最初の実行で作られます。

`sat` の出典：Contains modified Copernicus Sentinel data 2026（年は撮影年。meta.json の credit は自動で年が入るが、この README と docs/index.html・docs/sat/index.html の表記は手書きなので、撮影年が変わったら（2027 年の撮影が入ったら「2026-2027」に）直す）。Copernicus Data Space Ecosystem の Sentinel Hub Process API で σ0 の切り出しを取得し、このリポジトリが自動で数えています。陸マスクと岸壁の数に使う桟橋・岸壁・海岸線の線（`sat/masks/`、`quay/*.geojson` の区間の線）は © OpenStreetMap contributors（ODbL 1.0）。数は「候補数」で在泊艦数ではなく、艦種・国籍は分かりません。試行ページは [docs/sat/](docs/sat/index.html)（Pages では `sat/`）。

`sat/cfar2.py`（cfar-v2 PR1、**実験中**）は外洋向けの検出器の統計部分（ln σ0 の二回通しリング統計で標準化した z、2.56 km のタイルごとの分布の当てはめ・AIC と上側部分 Q-Q によるモデル選択、モデル別の閾値 z_t）です。合成クラッタの単体試験（`tests/test_sat_cfar2.py`）だけで使い、毎日の実行・湛江の数（cfar-v1、`sat/detect.py`）・公開データには使っていません。限界：K 分布（テクスチャ形状 ν=1）の尖ったクラッタでは実測誤警報率が名目の 1.56 倍（1e-3）・2.13 倍（1e-4）で目標の 1/2〜2 倍を外れる（1/3〜3 倍には入る）。タイルの当てはめは z1≥5 の画素（1 回目の検出）を除くため、クラッタ自身の上側の裾も切っている。ガンマ（loggamma）は設計書のモーメント法（ψ₁(L)=var y）ではなく最尤推定。平滑化と借用は、そのタイルで当てはめに成功した（params の有る）モデルだけから選び、借用は 3 タイル以内に限る（それより遠いタイルは判定しない）。

`sat/detect2.py`（cfar-v2 PR2、**実験中**）は cfar2 の z から候補を作る後段です：ヒステリシス（種 PFA 1e-7・伸長 1e-3）、3×3 closing・穴埋め・面積、白 top-hat（記録のみ）、VH による信頼度 A/B、形状（長さ・幅・向き・面積・長さ/幅・充填率）、サイドローブ（十字帯）と方位アンビギュイティ（Δ=λRPRF/2V）の除外（記録は残す。方針は「見逃しより疑い付きで残す」：サイドローブは親より 20 dB 以上弱く・5 px 未満・反対側の腕の同じ距離に同じ強さ（±3 dB）のものがある場合だけ除外し、それ以外の十字帯の候補は `sidelobe_suspect` を付けて数に残す。腕の 300〜600 m では 20 dB 以上弱いものだけを候補にする。アンビギュイティは ±Δ の周りの楕円（方位 ±400 m・距離 ±100 m）の中で、親と向き ±20°・長さ ±50% が似ているものだけ除外し、似ていないものは `ambiguity_suspect`。VH の背景が 0 のときは A にしない）、航跡の暗さ（記録のみ）、large の目印（80 m 等は仮置き）。副帯の PRF（IW1 1717.13・IW2 1451.37・IW3 1685.82 Hz）と副帯の入射角の境界は**仮置き**（ESA の公開文書で確かめられたのは軌道高度 約 698 km、PRF の範囲 1000〜3000 Hz、副帯ごとに PRF が 1 つであることまで。製品の注記 XML と実画像のアンビギュイティで確かめるまで暫定）。合成画像の試験（`tests/test_sat_detect2.py`）だけで使い、毎日の実行・湛江・公開データには使っていません。限界：(1) 尖ったクラッタでは見逃しが v1 より多い。1部の審査では、合成 K3 で長さ 100 m 以上の検出率が v2 74%（23/31）、v1 94%（29/31）だった。見逃したのは SCR 3.4〜9.6 dB の船で、217 m・9.6 dB の船は閾値にわずかに届かなかった。今の試験画像（随伴艦・混雑を加えたもの）の K3 では、随伴艦などを除いた長さ 100 m 以上の船が v1 8/14、v2 4/14（種 PFA 1e-7）、6/14（1e-6）。種の PFA は設定（`pfa_seed`）で変えられる。実画像の評価で、種の PFA（1e-7 / 1e-6）と、成分の面積で積分した検定（1 画素の z でなく成分全体の z で判定するもの）を比べる。(2) 長さは合成画像で +10〜20 m 長めに出る（ヒステリシスと closing による。実画像では分解能のにじみも足される）。AIS 照合で補正係数を出す予定（PR5）。このため `large`（測った長さ 80 m 以上）は実長 約 62 m の船から立ちうる。(3) 合成画像のサイドローブは離れた 3 画素の片として作っており、親の船体に続く連続した腕は試していない。アンビギュイティの位置に、親と向き・長さの似た本物の船がいると除外される（区別できない）。

`sat/detect2.py` の限界（VH と低風、cfar2-pr2.1）：(4) 雑音を引いた後の VH σ0 が 0 以下の画素（符号 0、dB では NaN）は「データなし」にしない。データなしは dataMask（`datamask` 引数）と VV の無データだけで決め、3 px の膨張除外はその境界だけにかける。VH 0 以下の画素は VH の対数統計（リング背景・タイルの当てはめ・z_vh）からだけ除き、VH の線形の平均（A の +3 dB 判定）では候補側・背景側とも 0 として数える。VV の当てはめ・検出には影響しない。(5) 雑音を引いた VH では VH の閾値が保守的すぎる：合成画像で VH の 1e-3 の実測超過率は公称の約 0.4〜0.5 倍、z_vh は雑音なしの約 6 割。このため A の判定は主に +3 dB の比で決まり、その比も背景の 0 以下を 0 に切り上げる分だけ小さめに出る（A が付きにくい側）。(6) 実画像（probe-raw/ocean 2026-10-30、9 撮影×108 タイル）では、船を除いても VV の 1e-4 の実測超過率が公称の 2〜3.5 倍で、設計の許容（1/3〜3 倍）を外れる場面がある（1e-3 は約 1.6 倍）。(7) VV σ0 が 0 以下で符号 0 になった画素は、データなしとして膨張除外される（非常な低風での限界）。注：`info` の `nodata_band_fraction` は画像の全画素が分母、`vh_zero_fraction` は海（データあり・陸以外）の画素が分母。

`sat/eval_detect.py`（eval-detect-1、**実験**）は検出手法を比べる評価の土台です（2部の比較実験の計画 §3 の段1・段2。PU は使いません）。保存済みのチップ（外洋の raw probe `probe/ocean-cfar-raw-ops3`、湛江の S1・S2 組 `probe/sat-s2-labels-ops3`）を読み、予測と正解をハンガリアン法（自前の実装。Tests に scipy は入れない）で 200 m・100 m 以内に 1 対 1 で照合し、適合率・再現率・F1、誤警報／1,000 km²、岸からの距離帯（0–1／1–2／2–5／5 km 超、帯ごとに予測・正解とも自分の距離で絞ってから照合）、xView3 の公式（reference/metric.py）と同じ岸近く F1（正解は岸から 2 km 以内、予測は 2 km＋照合距離以内、岸近くの正解と予測の両方がある撮影だけを集計）、長さ帯（<25、25–50、50–100、100–150、≥150 m）の検出率、長さの性能（xView3 の compute_length_performance と同じ：長さを 500 m で頭打ちにし、TP の |誤差|/真の長さ の平均、スコアは 1−min(平均, 1)）を出します。割合はすべて Wilson の 95% 区間（F1 は TP+(FP+FN)/2 を試行数とみなした近似、誤警報率は Poisson のスコア区間、長さ誤差の平均はブートストラップ）。正解の「たぶん船」「不明」に当たった予測は正解にも誤警報にも数えません（2部の計画 §3.3 の方針。xView3 は LOW だけを除き MEDIUM は数えるので、ここは xView3 と違う）。全手法の候補を 200 m でまとめた目視用の CSV（`vid, scene, lon, lat, sources, label, note`、sources は空）も作ります。段2 の実行：`python -m sat.eval_detect --ocean-raw DIR --zjg-raw DIR --out 判定者用DIR --key-out 鍵DIR [--verified CSV]`。目視の手順：(1) `--out` には目視用 CSV・表・summary.json だけが出る。(2) 鍵（vid→手法、`pool-*-key.json`）と手法別の候補（`candidates-*.geojson`）は `--key-out` に出る（`--out` と同じ場所・入れ子は拒否）。(3) 判定者には `--out` だけを渡し、判定が済むまで鍵と手法別の候補を公開ブランチ（probe ブランチを含む）に置かない（手元か司令部に置く）。(4) 判定済み CSV を `read_judged` で正解に戻してから鍵と突き合わせ、その後で公開する。cfar-v2 を main の扱い（cfar2-pr2.1）と PR #33 前の扱い（VH の 0 コードをデータなし。VV を VH の NaN の所で NaN にして再現。旧コードと候補・当てはめが一致することを確認済み）の両方で通します。湛江の保存チップは岸壁の帯（約 19% の画素）しか無いので、cfar-v2 の設定のままでは当てはめができず、`harbour_ref`（保存外を陸とみなし窓を縮めた参考設定、本番の設定ではない）も並べます。毎日の処理・公開データには使っていません。出典：Contains modified Copernicus Sentinel data [年]。

`ocean/cfar_probe.py`（cfar-v2 PR3、**実験中**）は外洋の Sentinel-1 を取り、cfar-v1 と cfar-v2 を同じ配列で比べる probe です（`--mode compare` は 5 範囲の 20 m、`raw` は原因調べ用の生のタイル、`wide` は 40 m の広域スナップショット）。`probe/ocean-cfar*` への push（ocean-cfar-probe.yml）でだけ動き、結果と使った PU（`docs/data/ocean/usage.csv`）は probe ブランチにだけ commit します。毎日の実行・湛江の数・公開データには使いません。上限はコードに書かず、ワークフローの env（`BUDGET_MONTH`・`OCEAN_CAP`・`GLOBAL_CAP`・`GLOBAL_MARGIN`）から`--budget-month`・`--ocean-cap`・`--global-cap`・`--global-margin` として必ず渡します（既定値なし）。実行月（UTC）が `--budget-month` と違えば CDSE に何も送らず（カタログも）終了コード 2、実行中に月が変わったら次の要求の前に止め、使った分は usage に書いて終了コード 2 です。使える PU は港のアーカイブと同じ guard（`sat.port_archive.guard`）で、min（外洋の上限 − 外洋の今月分、合計の上限 − 余裕 − main の sat usage − 港のアーカイブの今月分 − その他の usage − 外洋の今月分、`--max-pu`）です。使用量は `python -m sat.port_archive gather` で origin の全ブランチの写しを集め、同じ記録は 1 回だけ数えます（港・外洋・湛江の毎日分を合わせた和で守る）。ports-archive.yml の `OCEAN_CAP` は港側が外洋の未使用分を予約する値なので、外洋の `OCEAN_CAP` と同じ値にします（2026-10 は両方 4,600 PU。10 月の外洋は既に 4,993 PU 使っているため、10 月中の外洋の残りは 0）。二つの yml の `BUDGET_MONTH`・`OCEAN_CAP`・`GLOBAL_CAP`・`GLOBAL_MARGIN`（港側は `GUARD_ARGS` の中）が違うと `tests/test_ocean_cfar_probe.py` が落ちます。現在の env は 2026-10 の値のままで、11 月以降は司令部がその月の枠を決めて書き直すまで取得しません。使った PU とリクエスト数は、カタログの後と要求ごとに usage の 1 行（実行ごとに 1 行、書き直し）へ書き足し、どこで終わっても（URLError・TimeoutError・IncompleteRead・PNG や検出の例外・認証の失敗・枠切れの 403、ジョブの timeout や取り消しの SIGTERM）例外を出し直す前に書きます。終了コードは 0＝誤りなし、1＝個々の要求の失敗（または想定外の例外）、2＝月違い、3＝認証の失敗（403 を含む）です。データなしの扱いは `sat/detect2.py`（cfar2-pr2.1）と同じで、dataMask を `detect_v2` の `datamask` に渡し、VH の符号 0（σ0 ≤ 0）はデータありとして数えます（raw の `tile_diagnostics` の `nodata_band` も同じ。VH の中央値・5% 点は VH の符号が 0 でない画素だけで求める）。JSON には NaN・Infinity を書かず null にします。出典：Contains modified Copernicus Sentinel data [撮影の年。年をまたぐときは 2025–2026 のような幅]、固定物と陸は © OpenStreetMap contributors（ODbL）。compare の summary.json・report.txt・範囲ごとの GeoJSON、wide・raw の出力に入れています。限界：(1) `cdse.Client._post_sh` の内部の送り直し（429・5xx、最大 3 回、各最大 120 秒待つ）は月の確認（`MonthClient`）とペーサーを通らないため、月の終わりの直前に始まった呼び出しが月の境目を数分越えて送ることがあり、1 回の呼び出しあたり最大 3 件、毎分の速さとリクエスト数を超えることがあります（港側と同じ限界）。(2) 外洋のペーサーは 200 PU・200 リクエスト/分ですが、港のアーカイブ（80 PU/分×最大 3 ジョブ）と同時に走ると合わせて CDSE の 300 PU/分を超えることがあります（429 になり (1) の送り直しになる）。(3) 同時に走るジョブどうしは開始時に互いの usage を読むだけなので、リクエスト数の月の上限は同時実行では保証しません。

## 取得の作法

- 取得の前に robots.txt を確かめる（`collectors/polite.py`）。拒否されている、または確かめられない場合は、その回はその源を飛ばして警告を出す
- ボット対策のチャレンジページが返ったら、回り込まずにその回を止める
- 1件ずつ間隔をあけて取得し、User-Agent にこのリポジトリの URL を入れる。取得済みのものは取り直さない

## 分析の試作

[docs/proto/](docs/proto/index.html)（Pages では `proto/`）に、台湾国防部の日報を使った分析の試作を置いています（収集の穴、平常帯、変化点、艦艇の持続度、大規模演習の前後）。計算は `analysis/proto.py`（標準ライブラリのみ）で、`proto.yml` が収集の後に `docs/proto/data/` だけを書き換えます。演習の日付と出典は `analysis/exercises.json` にあります。

```sh
python -m analysis.proto   # docs/proto/data/*.json を作り直す
```

## 記録の原則

- 時刻は UTC で書き、公表・対象期間・取得の時刻を区別する
- 原文に記載のない数値は不明（null）とし、0 とはみなさない
- 自動で読み取れなかった原文は推測で埋めず、`status.json` に残して確認する
- 数え方の変更（定義変更）は日付つきで記録し、分析で行動の変化と混同しない
- 情報源には立場と評価を付け、どれか一つを事実の基準にしない

## 統幕データの更新

**台湾国防部の取得は停止中です。** 2026-10-30 に GitHub Actions から収集器と同じ User-Agent で読んだ https://www.mnd.gov.tw/robots.txt（Last-Modified 2026-10-13）は、Googlebot 以外のすべて（`User-agent: *`）に `Disallow: /` を指定していたため、`collectors/taiwan_mnd.py` の `SUSPENDED` で取得を止めています（毎日の実行は警告だけで、台湾国防部へは一切リクエストしない。保存済みの日報 `docs/data/taiwan-mnd/` はそのまま残り、`status.json` の `collection` に停止の理由を書く）。収集器は一覧・個別ページの取得前に robots.txt を確かめ（`collectors/polite.py`）、拒否または確認できないときは取得しません。再開は、robots.txt が `/en/news/plaactlist/` と `/en/News/PLAAct/` を許すようになったのを `python -m collectors.taiwan_mnd --check-robots`（robots.txt の1回だけ取得）で確かめてから、`SUSPENDED` を外す変更で行います。

統幕のサイト（mod.go.jp）はボット対策があり自動取得できないため、元データの更新は手作業です（手順は [jso/HANDOFF.md](jso/HANDOFF.md)）。解析済みのデータベース `jso/source/jso.sqlite` を差し替えると、毎日の実行で `docs/data/jso/` が作り直され、台湾国防部の日報との突き合わせも更新されます。

## 実行

```sh
python -m unittest discover -s tests -t .
python -m collectors.taiwan_mnd            # 毎日の取得（新しいものだけ）
python -m collectors.taiwan_mnd --pages 300  # 過去分の遡り（途中から再開できる）
python -m collectors.taiwan_mnd --probe    # 取得先のページ構造の確認だけ
pip install pdfplumber==0.11.10            # jcg-senkaku の PDF の読み取りに使う
python -m collectors.jcg_senkaku           # 海上保安庁 尖閣周辺（新しい PDF と当月分だけ）
python -m collectors.jcg_senkaku --probe /tmp/jcg  # ページと見本の PDF を取得して表示するだけ
python -m jso.build                        # 統幕データの公開用ファイルを作る
python -m collectors.pla_tracker           # PLA Activity Tracker の records.csv（1 本だけ）
pip install numpy==2.2.6 pillow==12.3.0    # sat に使う（Pillow はカラー画像の WebP と凡例）
python -m sat.run                          # Sentinel-1 の新しい撮影を数える（CDSE_SH_CLIENT_ID / CDSE_SH_CLIENT_SECRET が要る）
python -m sat.run --probe /tmp/sat         # /tmp/sat の下にだけ書く確認用
python -m sat.osm_mask ZJG --version 2     # 陸マスクを OSM から作り直す（版を上げ、CHANGES.md に書く）
python -m sat.quay_lines ZJG --out DIR     # OSM の桟橋・岸壁・海岸線（岸壁の数の線。probe で取り、確かめて sat/masks/ に commit）
python -m sat.baseline ZJG --out DIR       # 相対軌道ごとの「船のいない岸壁」の基準（過去約 6 か月をまとめて 1 回取得。probe で作り sat/baselines/ に commit）
python -m sat.port_archive plan --out plan.json --budget-month YYYY-MM --cap PU --global-cap PU --ocean-cap PU   # 湛江以外の主要な港（sat/ports.json）のアーカイブの PU の見込み（カタログだけ、PU を使わない。上限に既定値はない）
python -m ocean.cfar_probe --out DIR --budget-month YYYY-MM --ocean-cap PU --global-cap PU --global-margin PU [--mode compare|raw|wide] [--dry-run]   # 外洋の cfar-v1/v2 比較（実験中、probe/ocean-cfar* でだけ動く。上限に既定値はなく、月が違えば何も送らず終了コード 2、認証の失敗は 3）
```

GitHub Actions の「Collect public data」は毎日 10:23（日本時間）に、「Collect JCG Senkaku」は毎日 12:43（日本時間）に、「Sentinel-1 ship candidates」は毎日 14:17（日本時間）に、「Collect PLA Activity Tracker」は毎日 15:37（日本時間）に動きます（別々の concurrency group で、互いに止め合わない）。Sentinel-1 は無料枠を守るため、1 回の実行で最大 12 撮影、月 1000 処理単位（PU）までに自分で制限し、使った PU を usage.csv に残します。確認用の probe（`--probe`、`probe/sat-ships` への push）は同じ CDSE アカウントを使いますが、結果は一時ディレクトリにしか書かないため、その PU は公開の usage.csv に残らず、毎日の実行の月上限にも数えられません（probe 側は公開の usage.csv の今月分を自分の上限に数えます）。probe の使った PU とリクエスト数は、その実行の注釈（`probe-pu`）に出ます。probe の結果はどこにも保存されないため、毎日の実行の月上限に probe 分を含める手段はありません。probe を多く回した月は、注釈の合計か CDSE の利用状況画面で使用量を確かめてください。ただし岸壁の数の基準を作る probe（`probe/sat-baseline` への push、sat-baseline.yml。約 50 シーン・約 55 PU）は使用量が大きいため、基準を commit するときにその PU を usage.csv に 1 行足し、月上限に数えます。湛江以外の中国海軍の主要な港（`sat/ports.json`）の過去 12 か月の Sentinel-1・2 のアーカイブ（`sat/port_archive.py`）は、`probe/ports-archive-*` への push（ports-archive.yml）でだけ動き、結果と使った PU（`docs/data/sat/ports_usage.csv`）は probe ブランチにだけ commit します。この PU は毎日の実行の月 1000 PU の上限に数えません。代わりに別の安全ガードを持ちます：2026 年 10 月の値は港の取得全体で 4,750 PU、sat・外洋・港の合計で 9,995 PU（余裕 50 PU を残す。外洋の上限 4,600 PU のうち未使用分は予約として差し引く）で、10 月限りの値です。11 月以降は既定値がなく、その月の自主上限を決めて ports-archive.yml の `BUDGET_MONTH`・`GUARD_ARGS`・`OCEAN_CAP` に書き、実行時に `--budget-month` と上限（`--cap`・`--global-cap`・`--ocean-cap`）を明示して渡すまで取得しません（plan・locate・run・baseline のどれも、月が違えば要求の前に止まり、実行中に UTC の月が変わっても次の要求の前に止まります）。使用量は main の usage.csv と origin の全ブランチの usage CSV（港の `docs/data/sat/ports/*/usage.csv` と `docs/data/sat/ports_usage.csv` を含む）から集め、全部の写しの行を合わせて、同じ記録（run_utc・site_id・用途（archive_version）・リクエスト数・PU が同じ行）は 1 回だけ数えます（別々のブランチが足した行はどれも数え、複数のブランチや ports_usage.csv と港ごとの usage.csv の両方に写った行は二重に数えません）。次の港の probe は、アーカイブのデータがある最新の probe ブランチ（現時点では `probe/ports-s2ext-ops3`）から切ってください。取得済みの判定は作業ツリーの `docs/data/sat/ports/` しか見ないため、main から切ると全港を取り直そうとします。陸マスクは、アーカイブのマスクの置き場（`docs/data/sat/ports/masks/`）に無い版があれば、まず `sat/masks/` の同じ版（毎日の処理用で格子が同じもの）を写し、それも無いときだけ Overpass から作ります（毎日の処理とアーカイブで同じマスクになる）。SLP は範囲 v2（陸マスク SLP-land-v2）に変わったため、次の実行で過去分を v2 で取り直します（見込み約 140 PU）。SLP v2 の取り直しは 11 月の枠で行います（10 月にはもう港の取得をしません）。v1 の行と画像は残り、同じ月に v1・v2 の 2 系列が並びます。concurrency group は全ブランチで 1 つ（`probe-ports-archive`）です。限界：cdse.Client が内部で行う 429・5xx の再送（1 回の呼び出しで最大 4 送信）は送る前の確認を通らず、送った後に 60 秒窓とリクエスト数に足すため、リクエストの上限と速さを 1 回の呼び出しあたり最大 3 件超えることがあります。`sat/ports.json` は `sat/sites.json`（唯一の地点の設定）とは別の設定ファイルですが、過去画像の一括取得用で毎日の処理の sites.json とは用途が違うため、司令部が例外として了承しています（重複する SYL・SLP の範囲・中心・陸マスクの版の整合は tests/test_sat_port_archive.py の test_ports_json で確かめています）。台湾国防部の過去分の遡りは、手動実行で `pages` を大きくします。
