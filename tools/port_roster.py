"""The ships each watched port is home to, from our roster of the PLA Navy (docs/data/ref/plan-fleet.csv).

    python -m tools.port_roster build     # writes docs/data/ref/port-roster.json
    python -m tools.port_roster check     # fails if the file is not what build would write

The satellite count of a port (sat/quay_naval50) is the ships of 50 m or more seen on its naval piers and quays.
The site shows it beside the number of ships of 50 m or more assigned to the units based there, so that a reader
sees "17 of 25". Which units count for which port is the table SITES below, written by hand:

- in:    units whose base is inside the port's watched range (sat/sites.json bbox).
- other: units the roster gives the same home port, but whose quays are not known to be in the range.
         They are listed under the number, not added to it.

The roster is an estimate from open sources (revision 3 of our roster of the PLA Navy, 2026-11-12;
the src column names the sources, SRC_LEGEND below). Home port is not where a ship is today.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROSTER = ROOT / 'docs/data/ref/plan-fleet.csv'
OUT = ROOT / 'docs/data/ref/port-roster.json'
VERSION = 'roster-3'
MIN_LENGTH_M = 50

# Ships under 50 m: the satellite count leaves them out, so the roster does too.
# Type 022 missile boat 42.6 m; Type 082I minehunter about 45 m (the rows that hold only 082I).
SHORT_CLASSES = {'Type 022', 'Type 082I'}

# The src column of the CSV.
SRC_LEGEND = {
    'CM': '中国国営メディア（新華社・央視・環球時報・環球網・中新社・解放軍報・人民海軍・中国軍号）',
    'ZW': 'Wikipedia 中国語版（2026-08〜09 の版、検索結果の抜粋で確認）', 'ZH': '知乎の編制まとめ（2023〜2026、検索結果の抜粋）',
    'TX': '腾訊新聞・網易・新浪の軍事記事', 'WP': 'Wikipedia 英語版（2026-11 取得）', 'GS': 'GlobalSecurity 編成表',
    'NN': 'Naval News', 'USNI': 'USNI News', 'AR': 'Army Recognition', 'DSA': 'Defence Security Asia', 'TWZ': 'The War Zone',
    'ORF': 'ORF Special Report 234（2024-10）', 'CDB': 'china-defense blog', 'CK': '学習知識（未検証。ほかの出典と併記の行だけ）',
}
TYPE_JA = {
    'CV': '空母', 'LHD': '強襲揚陸艦', 'LPD': '揚陸艦（ドック型）', 'DDG': '駆逐艦', 'FFG': 'フリゲート', 'FFL': 'コルベット',
    'LST': '戦車揚陸艦', 'LSM': '中型揚陸艦', 'SSBN': '弾道ミサイル原潜', 'SSN': '攻撃型原潜', 'SSK': '通常動力潜水艦',
    'AOR': '補給艦', 'AKS': '島嶼補給艦', 'AGI': '情報収集艦', 'MCM': '掃海艦艇', 'PTG': 'ミサイル艇',
}


def pick(home: str | set[str], unit: str | None = None, types: set[str] | None = None, pennants: set[str] | None = None):
    homes = {home} if isinstance(home, str) else home
    def match(r: dict) -> bool:
        return (r['home_port'] in homes and (unit is None or r['unit'] == unit)
                and (types is None or r['type'] in types) and (pennants is None or r['pennant'] in pennants))
    return match


# site id -> groups. Each group: (label, matcher, why). `why` is shown for the "other" groups and kept for "in".
SITES: dict[str, dict] = {
    'ZJG': {
        'in': [('駆逐艦第2支隊', pick('Zhanjiang', '2nd Destroyer Flotilla'), '麻斜（中国語版 Wikipedia・知乎）')],
        'other': [
            ('登陸艦第6支隊', pick('Zhanjiang', '6th Landing Ship Flotilla'), '湛江。岸壁の場所は未確認'),
            ('作戦支援艦第3支隊', pick('Zhanjiang', '3rd Combat Support Ship Flotilla'), '湛江の赤坎。範囲の外の可能性'),
        ],
    },
    'SYL': {
        'in': [
            ('潜水艦第32支隊', pick('Sanya-Yulin', '32nd Submarine Flotilla'), '楡林'),
            ('護衛艦第19支隊', pick('Sanya', '19th Frigate Flotilla'), '楡林'),
        ],
        'other': [],
    },
    'SLP': {
        'in': [
            ('潜水艦第2基地', pick('Sanya-Longpo', '2nd Submarine Base'), '龍坡'),
            ('駆逐艦第9支隊', pick('Sanya', '9th Destroyer Flotilla'), '亜龍湾'),
            ('空母（山東・福建）', pick('Sanya', types={'CV'}), '母港は三亜（亜龍湾の空母の桟橋）。福建は楡池に停泊した時期がある'),
        ],
        'other': [],
    },
    'QGZ': {
        'in': [
            ('駆逐艦第1支隊', pick('Qingdao', '1st Destroyer Flotilla'), '古鎮口'),
            ('空母（遼寧）', pick('Qingdao', types={'CV'}), '楡池（古鎮口）'),
        ],
        'other': [
            ('作戦支援艦第1支隊', pick('Qingdao', '1st Combat Support Ship Flotilla'), '青島。岸壁の場所は未確認'),
            ('登陸艦第1大隊', pick('Qingdao', '1st Landing Ship Squadron'), '青島。岸壁の場所は未確認'),
        ],
    },
    'QDH': {'in': [('潜水艦第2支隊', pick('Qingdao', '2nd Submarine Flotilla'), '青島の市北')], 'other': []},
    'QJG': {'in': [('潜水艦第1基地', pick('Qingdao-Jianggezhuang', '1st Submarine Base'), '姜各庄')], 'other': []},
    'LSN': {
        'in': [
            ('駆逐艦第10支隊', pick('Lushun', '10th Destroyer Flotilla'), '旅順口東港'),
            ('054AG型（鄂爾多斯・臨汾）', pick('Lushun', '', {'FFG'}), '大連某軍港・北部戦区。支隊は不明'),
            ('潜水艦第12支隊', pick('Lushun', '12th Submarine Flotilla'), '旅順口'),
        ],
        'other': [],
    },
    'ZDH': {
        'in': [
            ('駆逐艦第3支隊', pick('Zhoushan', '3rd Destroyer Flotilla'), '舟山の定海'),
            ('駆逐艦第6支隊', pick('Zhoushan', '6th Destroyer Flotilla'), '舟山の定海'),
        ],
        'other': [('作戦支援艦第2支隊', pick('Zhoushan', '2nd Combat Support Ship Flotilla'), '舟山。岸壁の場所は未確認')],
    },
    'XSN': {'in': [('潜水艦第42支隊', pick('Ningbo-Xiangshan', '42nd Submarine Flotilla'), '象山')], 'other': []},
    'SWS': {'in': [('護衛艦第14支隊', pick('Shanghai', '14th Frigate Flotilla'), '呉淞')], 'other': []},
    'XMN': {'in': [('護衛艦第16支隊', pick('Xiamen', '16th Frigate Flotilla'), '廈門')], 'other': []},
    'STU': {'in': [('導弾艇第34大隊', pick('Shantou', '34th Missile Boat Squadron'), '汕頭の礐石。022型（42.6 m）は 50 m 未満')], 'other': []},
}


def load_rows(path: Path = ROSTER) -> list[dict]:
    with open(path, encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r['count'] = int(r['count'])
    return rows


def long_enough(r: dict) -> bool:
    return r['class'] not in SHORT_CLASSES


def group(label: str, match, why: str, rows: list[dict], used: set[int]) -> dict | None:
    mine = [r for r in rows if match(r)]
    if not mine:
        raise ValueError(f'{label}: no roster row matches')
    for r in mine:
        if id(r) in used:
            raise ValueError(f'{label}: {r["class"]} {r["pennant"]} is in two groups')
        used.add(id(r))
    big = [r for r in mine if long_enough(r)]
    types = Counter()
    for r in big:
        types[r['type']] += r['count']
    out = {'name': label, 'n': sum(r['count'] for r in big), 'types': {t: types[t] for t in sorted(types)}, 'where': why}
    short = sum(r['count'] for r in mine if not long_enough(r))
    if short:
        out['short'] = short
    if any(r['port_conf'] == 'low' for r in mine):
        out['low'] = sum(r['count'] for r in mine if r['port_conf'] == 'low')
    return out


def build(rows: list[dict]) -> dict:
    used: set[int] = set()
    sites = {}
    for site, spec in SITES.items():
        ins = [group(*g, rows, used) for g in spec['in']]
        other = [group(*g, rows, used) for g in spec['other']]
        sites[site] = {'assigned': sum(g['n'] for g in ins), 'units': ins, 'other': other}
    return {
        'dataset': 'ref/port-roster',
        'version': VERSION,
        'kind': '参照データ（公開情報からの推定。うちの観測ではない）',
        'basis': '中国海軍の港別艦艇配置（改訂3、2026-11-12）。docs/data/ref/plan-fleet.csv',
        'min_length_m': MIN_LENGTH_M,
        'rule': ('assigned は、その港の監視範囲に駐屯地がある部隊の、全長 50 m 以上の艦の数（潜水艦を含む）。'
                 'other は母港が同じで岸壁が範囲にあるか確かめていない部隊（assigned に足さない）。'
                 '所属と在泊は別で、出港中・整備中・他港への寄港でずれる'),
        'src_legend': SRC_LEGEND,
        'types_ja': TYPE_JA,
        'sites': sites,
    }


def render(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False, indent=1) + '\n'


def main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else 'build'
    text = render(build(load_rows()))
    if cmd == 'build':
        OUT.write_text(text, encoding='utf-8')
        print(f'wrote {OUT.relative_to(ROOT)}')
        return 0
    if cmd == 'check':
        if not OUT.exists() or OUT.read_text(encoding='utf-8') != text:
            print('::error title=port-roster::docs/data/ref/port-roster.json が plan-fleet.csv と合わない。python -m tools.port_roster build で作り直す')
            return 1
        return 0
    print(__doc__)
    return 2


if __name__ == '__main__':
    sys.exit(main(sys.argv))
