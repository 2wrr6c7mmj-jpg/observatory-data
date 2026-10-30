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
| `sat`（試行） | Sentinel-1 による港内の艦船候補数（湛江 ZJG の 1 地点）。撮影ごとの港内水域・岸壁帯の候補数、候補の位置と長さの目安、切り出し画像（VV の灰 PNG と、疑似カラーの WebP `color/`）。岸壁に横付けされた船の占有数（最小〜最大、試行・精度未測定、相対軌道ごとの基準との変化）と、艦種の長さの分布・桟橋別の事前分布で狭めた隻数の推定（est-1：モデル上の最頻値とモデル内の 80% 区間〔横抱きの確率の誤り・誤検出・数えられない区間・OSM に無い岸壁・艦種表の誤り・区間の独立の仮定・上限 16 隻は区間に入っていない〕、試行・精度未測定、前提は推定を含む。艦種表は meta.json の quay.estimate.class_table）、その横に軍艦の可能性が高い船の隻数（est-2：横抱きの列数と区画・長さ・S2 の色から。仮置き・較正前、モデル内の 80% 区間、AIS は使わない）、変化カラー `change/` | [ship_counts.csv](docs/data/sat/ship_counts.csv)・`detections/YYYY-MM.geojson`・[quay_counts.csv](docs/data/sat/quay_counts.csv)・`quay/YYYY-MM.geojson`・[quay_est2.csv](docs/data/sat/quay_est2.csv)・[meta.json](docs/data/sat/meta.json)・[status.json](docs/data/sat/status.json)・[usage.csv](docs/data/sat/usage.csv)・[CHANGES.md](docs/data/sat/CHANGES.md) |
| `ocean`（実測のみ） | 外洋（台湾周辺 T・東シナ海 E・南西諸島 R）の Sentinel-1 の撮影頻度（0.25° 格子ごとの IW 通過数、相対軌道・向き・時刻、IW 以外のモード）と公開の遅れ。CDSE の OData カタログを probe（`probe/ocean-coverage` への push、ワークフロー ocean-probe.yml）で 1 回読んだ結果で、定期更新はしない | [coverage_probe.json](docs/data/ocean/coverage_probe.json) |

各データの定義、情報源の評価、定義変更の履歴、限界は `meta.json` に書いています（`ocean` は 1 回だけの実測で meta.json が無く、出典・方法・注記は coverage_probe.json の中の `source`・`method`・`attribution`・`notes` にあります）。

`ocean` の公開の遅れは、窓の終わり近くに撮影されてまだ公開されていない製品が数に入らない（右側の打ち切り）ため、実際より短めに出ている可能性があります。

`jcg-senkaku` の出典：海上保安庁ホームページ（https://www.kaiho.mlit.go.jp/mission/senkaku/senkaku.html）。公共データ利用規約 第1.0版（PDL1.0）に基づき、このリポジトリが PDF から数値を抜き出して日ごとの表に加工しています。加工したデータは海上保安庁が作成したものではありません。

`pla-tracker` の出典：Source: ROC Ministry of National Defense; compilation & charts: PLA Activity Tracker (https://pla-tracker.skyfaring.net)（CC BY 4.0）。records.csv を 1 日 1 回取り込み、数値の列を整数にして取得時刻を付けています（値は変えていない）。同じ日に国防部の日報（`taiwan-mnd`）があればそちらを優先します。艦艇（ships_total）は 2026-03 下旬から公務船を含み、中間線越え（median_line_cross）は国防部の median_line_or_adiz より狭い量なので、`taiwan-mnd` の列にそのままつなげません（meta.json の precedence・fields）。ファイルは main での最初の実行で作られます。

`sat` の出典：Contains modified Copernicus Sentinel data 2026（年は撮影年。meta.json の credit は自動で年が入るが、この README と docs/index.html・docs/sat/index.html の表記は手書きなので、撮影年が変わったら（2027 年の撮影が入ったら「2026-2027」に）直す）。Copernicus Data Space Ecosystem の Sentinel Hub Process API で σ0 の切り出しを取得し、このリポジトリが自動で数えています。陸マスクと岸壁の数に使う桟橋・岸壁・海岸線の線（`sat/masks/`、`quay/*.geojson` の区間の線）は © OpenStreetMap contributors（ODbL 1.0）。数は「候補数」で在泊艦数ではなく、艦種・国籍は分かりません。試行ページは [docs/sat/](docs/sat/index.html)（Pages では `sat/`）。

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
```

GitHub Actions の「Collect public data」は毎日 10:23（日本時間）に、「Collect JCG Senkaku」は毎日 12:43（日本時間）に、「Sentinel-1 ship candidates」は毎日 14:17（日本時間）に、「Collect PLA Activity Tracker」は毎日 15:37（日本時間）に動きます（別々の concurrency group で、互いに止め合わない）。Sentinel-1 は無料枠を守るため、1 回の実行で最大 12 撮影、月 1000 処理単位（PU）までに自分で制限し、使った PU を usage.csv に残します。確認用の probe（`--probe`、`probe/sat-ships` への push）は同じ CDSE アカウントを使いますが、結果は一時ディレクトリにしか書かないため、その PU は公開の usage.csv に残らず、毎日の実行の月上限にも数えられません（probe 側は公開の usage.csv の今月分を自分の上限に数えます）。probe の使った PU とリクエスト数は、その実行の注釈（`probe-pu`）に出ます。probe の結果はどこにも保存されないため、毎日の実行の月上限に probe 分を含める手段はありません。probe を多く回した月は、注釈の合計か CDSE の利用状況画面で使用量を確かめてください。ただし岸壁の数の基準を作る probe（`probe/sat-baseline` への push、sat-baseline.yml。約 50 シーン・約 55 PU）は使用量が大きいため、基準を commit するときにその PU を usage.csv に 1 行足し、月上限に数えます。台湾国防部の過去分の遡りは、手動実行で `pages` を大きくします。
