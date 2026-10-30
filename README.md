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
| `sat`（試行） | Sentinel-1 による港内の艦船候補数（湛江 ZJG の 1 地点）。撮影ごとの港内水域・岸壁帯の候補数、候補の位置と長さの目安、切り出し画像 | [ship_counts.csv](docs/data/sat/ship_counts.csv)・`detections/YYYY-MM.geojson`・[meta.json](docs/data/sat/meta.json)・[status.json](docs/data/sat/status.json)・[usage.csv](docs/data/sat/usage.csv)・[CHANGES.md](docs/data/sat/CHANGES.md) |
| `ocean`（実測のみ） | 外洋（台湾周辺 T・東シナ海 E・南西諸島 R）の Sentinel-1 の撮影頻度（0.25° 格子ごとの IW 通過数、相対軌道・向き・時刻、IW 以外のモード）と公開の遅れ。CDSE の OData カタログを probe（`probe/ocean-coverage` への push、ワークフロー ocean-probe.yml）で 1 回読んだ結果で、定期更新はしない | [coverage_probe.json](docs/data/ocean/coverage_probe.json) |

各データの定義、情報源の評価、定義変更の履歴、限界は `meta.json` に書いています。

`jcg-senkaku` の出典：海上保安庁ホームページ（https://www.kaiho.mlit.go.jp/mission/senkaku/senkaku.html）。公共データ利用規約 第1.0版（PDL1.0）に基づき、このリポジトリが PDF から数値を抜き出して日ごとの表に加工しています。加工したデータは海上保安庁が作成したものではありません。

`sat` の出典：Contains modified Copernicus Sentinel data 2026（年は撮影年。meta.json の credit は自動で年が入るが、この README と docs/index.html・docs/sat/index.html の表記は手書きなので、撮影年が変わったら（2027 年の撮影が入ったら「2026-2027」に）直す）。Copernicus Data Space Ecosystem の Sentinel Hub Process API で σ0 の切り出しを取得し、このリポジトリが自動で数えています。陸マスクは © OpenStreetMap contributors（ODbL）。数は「候補数」で在泊艦数ではなく、艦種・国籍は分かりません。試行ページは [docs/sat/](docs/sat/index.html)（Pages では `sat/`）。

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
pip install numpy==2.2.6                   # sat に使う
python -m sat.run                          # Sentinel-1 の新しい撮影を数える（CDSE_SH_CLIENT_ID / CDSE_SH_CLIENT_SECRET が要る）
python -m sat.run --probe /tmp/sat         # /tmp/sat の下にだけ書く確認用
python -m sat.osm_mask ZJG --version 2     # 陸マスクを OSM から作り直す（版を上げ、CHANGES.md に書く）
```

GitHub Actions の「Collect public data」は毎日 10:23（日本時間）に、「Collect JCG Senkaku」は毎日 12:43（日本時間）に、「Sentinel-1 ship candidates」は毎日 14:17（日本時間）に動きます（別々の concurrency group で、互いに止め合わない）。Sentinel-1 は無料枠を守るため、1 回の実行で最大 12 撮影、月 1000 処理単位（PU）までに自分で制限し、使った PU を usage.csv に残します。確認用の probe（`--probe`、`probe/sat-ships` への push）は同じ CDSE アカウントを使いますが、結果は一時ディレクトリにしか書かないため、その PU は公開の usage.csv に残らず、毎日の実行の月上限にも数えられません（probe 側は公開の usage.csv の今月分を自分の上限に数えます）。probe の使った PU とリクエスト数は、その実行の注釈（`probe-pu`）に出ます。probe の結果はどこにも保存されないため、毎日の実行の月上限に probe 分を含める手段はありません。probe を多く回した月は、注釈の合計か CDSE の利用状況画面で使用量を確かめてください。台湾国防部の過去分の遡りは、手動実行で `pages` を大きくします。
