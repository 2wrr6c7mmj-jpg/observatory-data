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
| `jso` | 統合幕僚監部の公表資料（中国・ロシア艦艇の海峡通過、往復、演習・政治イベントとの対応） | [passages.csv](docs/data/jso/passages.csv)・[round_trips.csv](docs/data/jso/round_trips.csv)・[context.csv](docs/data/jso/context.csv)・[summary.json](docs/data/jso/summary.json)・[meta.json](docs/data/jso/meta.json) |

各データの定義、情報源の評価、定義変更の履歴、限界は `meta.json` に書いています。

## 記録の原則

- 時刻は UTC で書き、公表・対象期間・取得の時刻を区別する
- 原文に記載のない数値は不明（null）とし、0 とはみなさない
- 自動で読み取れなかった原文は推測で埋めず、`status.json` に残して確認する
- 数え方の変更（定義変更）は日付つきで記録し、分析で行動の変化と混同しない
- 情報源には立場と評価を付け、どれか一つを事実の基準にしない

## 統幕データの更新

統幕のサイト（mod.go.jp）はボット対策があり自動取得できないため、元データの更新は手作業です（手順は [jso/HANDOFF.md](jso/HANDOFF.md)）。解析済みのデータベース `jso/source/jso.sqlite` を差し替えると、毎日の実行で `docs/data/jso/` が作り直され、台湾国防部の日報との突き合わせも更新されます。

## 実行

```sh
python -m unittest discover -s tests -t .
python -m collectors.taiwan_mnd            # 毎日の取得（新しいものだけ）
python -m collectors.taiwan_mnd --pages 300  # 過去分の遡り（途中から再開できる）
python -m collectors.taiwan_mnd --probe    # 取得先のページ構造の確認だけ
python -m jso.build                        # 統幕データの公開用ファイルを作る
```

GitHub Actions の「Collect public data」は毎日 10:23（日本時間）に動きます。過去分の遡りは、手動実行で `pages` を大きくします。
