# pres3-patches

confirm-1（確認回し）の判定器 pres-3o のパッチの保管と係数。`.github/workflows/sat-confirm.yml` だけが書く（main・docs/ には入らない）。

- `<港>/<YYYY>/<YYYY-MM-DD>.c.npz` その日の晴れの升、`.r.npz` それ以外、`.a.npz` 間引きで残した答えのある升（`sat.pres3.daily.Store`、版 pres3-store-2）。
  1 日のファイルは書いたら書き直さない。直近 18 か月より前の日は `.r` を消す（晴れの升と、点数に要る日ごとの水の色・晴れの印は残る）。
- `models/pres3o-YYYYMM.npz` 係数（締め日＝その月の 1 日より前の所有者の答えで学んだ。`sat.pres3.daily.load_model`）。

最初のコミット：研究の作業の置き場（9 港、2023-06-01〜2026-11-19）を取り込み、2026-11-22 の時点で間引いたもの（303 MB）と pres3o-202611。
Contains modified Copernicus Sentinel data 2023–2026 (Element84 Earth Search / AWS Open Data).
