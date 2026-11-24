"""配信元ごとの国（発行元の所在）と系列（同じ統制・同じ会社・同じ通信社を 1 つに数える）の表。

独立した出典の数は「系列」の数、出典の国の数は系列の国の重複を除いた数。
- 国営メディアは 1 国 1 系列（TASS・RT・スプートニク・露大統領府は ru-state、CGTN・環球時報・中新社は cn-state）。
- 同じ発行元の別の面（BBC 中国と BBC 国際、SCMP の各面、聯合の英語版と韓国語版、中央社の各面）は 1 系列。
- 米海軍の発表と DVIDS（国防総省の素材置き場）は同じ発表の再掲が多いので 1 系列（us-dod）。
- Sightline Media（Defense News、Military Times、Task & Purpose）、GlobalData（Air Force Technology、Naval Technology）も 1 系列。
- 国は発行元の本拠（資本ではなく所在）。判断が分かれるもの（NK News、メドゥーザ、モスクワ・タイムズ、アルジャジーラ、
  ミドル・イースト・アイ、アフリカニュース、ジ・アビエーショニスト）は下に印を付けた。表に無い配信元は、系列＝配信元 id、国＝不明
  （国の数には入れない）。通信社の転電（時事・共同 → ジャパンタイムズ等）は見分けられず、独立と数える（限界）。
表を直したら tests/test_briefing_cluster.py の表の確認も合わせる。
"""

# 配信元 id -> (系列, 国)。国は ISO 3166-1 alpha-2（国際機関は 'UN'）。
SOURCES = {
    # 中国（国営・党系）
    'ecns': ('cn-state', 'CN'), 'cgtn': ('cn-state', 'CN'), 'cgtn-world': ('cn-state', 'CN'),
    'globaltimes': ('cn-state', 'CN'), 'xinhua': ('cn-state', 'CN'), 'chinamil': ('cn-state', 'CN'),
    # ロシア（国営）
    'tass': ('ru-state', 'RU'), 'rt': ('ru-state', 'RU'), 'sputnik': ('ru-state', 'RU'), 'kremlin': ('ru-state', 'RU'),
    # 北朝鮮（国営）
    'kcna': ('kp-state', 'KP'), 'rodong': ('kp-state', 'KP'),
    # 米国
    'navy': ('us-dod', 'US'), 'dvids': ('us-dod', 'US'),
    'usni': ('usni', 'US'), 'defensenews': ('sightline', 'US'), 'militarytimes': ('sightline', 'US'),
    'taskpurpose': ('sightline', 'US'), 'breakingdefense': ('breakingdefense', 'US'), 'twz': ('twz', 'US'),
    'defenseone': ('defenseone', 'US'), 'defensescoop': ('defensescoop', 'US'), 'afmag': ('afmag', 'US'),
    'warontherocks': ('warontherocks', 'US'), 'cimsec': ('cimsec', 'US'), 'nationalinterest': ('nationalinterest', 'US'),
    '19fortyfive': ('19fortyfive', 'US'), 'gcaptain': ('gcaptain', 'US'), 'maritimeexec': ('maritimeexec', 'US'),
    'spacenews': ('spacenews', 'US'), 'nsf': ('nsf', 'US'), 'rfa': ('rfa', 'US'), 'almonitor': ('almonitor', 'US'),
    'amti': ('csis', 'US'), 'chinapower': ('csis', 'US'), 'jamestown': ('jamestown', 'US'),
    # 英国・欧州ほか
    'bbc': ('bbc', 'GB'), 'bbc-china': ('bbc', 'GB'), 'bbc-africa': ('bbc', 'GB'),
    'guardian-china': ('guardian', 'GB'), 'guardian-russia': ('guardian', 'GB'),
    'fcdo': ('gb-gov', 'GB'), 'ukdj': ('ukdj', 'GB'), 'airforcetech': ('globaldata', 'GB'), 'navaltech': ('globaldata', 'GB'),
    'mee': ('mee', 'GB'),                       # 判断が分かれる（ロンドン拠点）
    'dw': ('dw', 'DE'), 'f24': ('france24', 'FR'), 'navalnews': ('navalnews', 'FR'),
    'africanews': ('africanews', 'FR'),         # 判断が分かれる（ユーロニュース系、リヨン）
    'theaviationist': ('theaviationist', 'IT'),  # 判断が分かれる
    'un': ('un', 'UN'),
    # 中東
    'aj': ('aj', 'QA'),                          # 判断が分かれる（カタール政府系の出資）
    # ロシア語圏の亡命・独立系
    'meduza': ('meduza', 'LV'),                  # 判断が分かれる（リガ拠点）
    'moscowtimes': ('moscowtimes', 'NL'),        # 判断が分かれる（オランダ登記）
    # ウクライナ
    'ukrinform': ('ua-state', 'UA'), 'kyivindependent': ('kyivindependent', 'UA'),
    # 台湾
    'cna': ('cna-tw', 'TW'), 'cna-mainland': ('cna-tw', 'TW'), 'cna-intl': ('cna-tw', 'TW'),
    'focustaiwan': ('cna-tw', 'TW'), 'rti': ('rti', 'TW'), 'pts': ('pts', 'TW'), 'newtalk': ('newtalk', 'TW'),
    # 韓国
    'yonhap': ('yonhap', 'KR'), 'yonhap-nk': ('yonhap', 'KR'), 'yna-politics': ('yonhap', 'KR'),
    'yna-nk-ko': ('yonhap', 'KR'), 'chosun': ('chosun', 'KR'), 'koreaherald': ('koreaherald', 'KR'),
    'korea-net': ('kr-gov', 'KR'), 'dailynk': ('dailynk', 'KR'),
    'nknews': ('nknews', 'KR'),                  # 判断が分かれる（ソウル本社、運営会社は韓国）
    # 日本
    'kyodo': ('kyodo', 'JP'), 'jiji': ('jiji', 'JP'), 'nhk': ('nhk', 'JP'), 'asahi': ('asahi', 'JP'),
    'mainichi': ('mainichi', 'JP'), 'japanforward': ('sankei', 'JP'), 'japantimes': ('japantimes', 'JP'),
    'kantei': ('jp-gov', 'JP'), 'mofa': ('jp-gov', 'JP'),
    # 香港・シンガポール・インド・東南アジア・豪州
    'hkfp': ('hkfp', 'HK'), 'scmp-china': ('scmp', 'HK'), 'scmp-military': ('scmp', 'HK'), 'scmp-asia': ('scmp', 'HK'),
    'asiatimes': ('asiatimes', 'HK'), 'straitstimes': ('straitstimes', 'SG'),
    'thehindu': ('thehindu', 'IN'), 'eurasiantimes': ('eurasiantimes', 'IN'), 'geopolmonitor': ('geopolmonitor', 'CA'),
    'inquirer': ('inquirer', 'PH'), 'philstar': ('philstar', 'PH'), 'manilatimes': ('manilatimes', 'PH'),
    'rappler': ('rappler', 'PH'), 'gma': ('gma', 'PH'), 'vnexpress': ('vnexpress', 'VN'),
    'thediplomat': ('thediplomat', 'US'),
    'abc-au': ('abc-au', 'AU'), 'aspi': ('aspi', 'AU'), 'lowy': ('lowy', 'AU'),
}

# 配信元 id -> (名前, 種別)。observatory-watch の lib/intelligence.ts の sources（97 源）から id・name・kind だけを写した
# （RSS の URL は写さない）。種別が政府発表・国営メディア・国営・党系メディアのものは「一次発表・国営」。
META = {
    "usni": ("USNI News（海軍専門報道）", "専門報道"),
    "amti": ("CSIS AMTI（南シナ海の衛星解析）", "専門報道"),
    "yonhap": ("聯合ニュース 英語版（韓国）", "通信社"),
    "cna": ("中央通訊社（台湾）政治", "通信社"),
    "navalnews": ("Naval News（海軍専門報道）", "専門報道"),
    "kyodo": ("共同通信（Japan Wire）", "通信社"),
    "jiji": ("時事通信", "通信社"),
    "navy": ("米海軍 公開発表", "政府発表"),
    "dvids": ("DVIDS 米国防総省 公開発表", "政府発表"),
    "bbc-china": ("BBC 中国", "報道"),
    "guardian-china": ("ガーディアン 中国", "報道"),
    "hkfp": ("香港フリープレス", "報道"),
    "globaltimes": ("環球時報 英語版", "国営・党系メディア"),
    "tass": ("タス通信", "国営メディア"),
    "meduza": ("メドゥーザ 英語版", "報道"),
    "guardian-russia": ("ガーディアン ロシア", "報道"),
    "yonhap-nk": ("聯合ニュース 北朝鮮", "報道"),
    "dailynk": ("デイリーNK 英語版", "報道"),
    "mee": ("ミドル・イースト・アイ", "報道"),
    "almonitor": ("アルモニター", "報道"),
    "africanews": ("アフリカニュース", "報道"),
    "bbc-africa": ("BBC アフリカ", "報道"),
    "cgtn": ("中国国際テレビ（CGTN）", "国営メディア"),
    "scmp-china": ("サウスチャイナ・モーニング・ポスト 中国", "報道"),
    "scmp-military": ("サウスチャイナ・モーニング・ポスト 軍事", "報道"),
    "focustaiwan": ("中央通訊社（台湾）英語版", "通信社"),
    "thediplomat": ("ザ・ディプロマット（アジア安全保障）", "専門報道"),
    "rt": ("RT（ロシア国営）", "国営メディア"),
    "kremlin": ("ロシア大統領府 発表", "政府発表"),
    "moscowtimes": ("モスクワ・タイムズ", "報道"),
    "kyivindependent": ("キーウ・インディペンデント", "報道"),
    "ukrinform": ("ウクルインフォルム", "国営メディア"),
    "defensenews": ("ディフェンス・ニュース", "専門報道"),
    "twz": ("ザ・ウォー・ゾーン", "専門報道"),
    "breakingdefense": ("ブレイキング・ディフェンス", "専門報道"),
    "thehindu": ("ザ・ヒンドゥー 国際", "報道"),
    "japantimes": ("ジャパンタイムズ", "報道"),
    "sputnik": ("スプートニク（ロシア国営）", "国営メディア"),
    "ecns": ("中国新聞社 英語版", "国営メディア"),
    "mofa": ("外務省 海外安全情報", "政府発表"),
    "fcdo": ("英国外務省 渡航情報", "政府発表"),
    "un": ("国連ニュース 平和・安全保障", "国際機関"),
    "bbc": ("英国公共放送 国際", "報道"),
    "nhk": ("日本放送協会 国際", "報道"),
    "dw": ("ドイチェ・ヴェレ", "報道"),
    "aj": ("アルジャジーラ", "報道"),
    "f24": ("フランス24", "報道"),
    "cna-mainland": ("中央通訊社（台湾）両岸", "通信社"),
    "cna-intl": ("中央通訊社（台湾）国際", "通信社"),
    "rti": ("中央広播電台（Rti、台湾）", "公共放送"),
    "pts": ("公共電視（PTS、台湾）", "公共放送"),
    "newtalk": ("新頭殼（台湾）", "報道"),
    "yna-politics": ("聯合ニュース 政治（韓国語）", "通信社"),
    "yna-nk-ko": ("聯合ニュース 北朝鮮（韓国語）", "通信社"),
    "chosun": ("朝鮮日報 政治", "報道"),
    "koreaherald": ("コリア・ヘラルド", "報道"),
    "korea-net": ("韓国政府 Korea.net", "政府発表"),
    "nknews": ("NK News", "専門報道"),
    "cgtn-world": ("中国国際テレビ（CGTN）国際", "国営メディア"),
    "kantei": ("首相官邸 新着", "政府発表"),
    "mainichi": ("毎日新聞 英語版", "報道"),
    "asahi": ("朝日新聞 速報", "報道"),
    "japanforward": ("JAPAN Forward（産経）", "報道"),
    "scmp-asia": ("サウスチャイナ・モーニング・ポスト アジア", "報道"),
    "straitstimes": ("ストレーツ・タイムズ アジア", "報道"),
    "inquirer": ("インクワイアラー 国際（フィリピン）", "報道"),
    "philstar": ("フィルスター（フィリピン）", "報道"),
    "manilatimes": ("マニラ・タイムズ", "報道"),
    "rappler": ("ラップラー（フィリピン）", "報道"),
    "gma": ("GMA ニュース（フィリピン）", "報道"),
    "vnexpress": ("VnExpress 英語版（ベトナム）", "報道"),
    "abc-au": ("ABC（豪州公共放送）", "報道"),
    "rfa": ("ラジオ・フリー・アジア", "報道"),
    "asiatimes": ("アジア・タイムズ", "報道"),
    "chinapower": ("CSIS ChinaPower", "研究機関"),
    "jamestown": ("ジェームズタウン財団", "研究機関"),
    "aspi": ("ASPI ストラテジスト（豪）", "研究機関"),
    "lowy": ("ローウィー研究所 インタープリター（豪）", "研究機関"),
    "warontherocks": ("ウォー・オン・ザ・ロックス", "専門報道"),
    "cimsec": ("CIMSEC（海洋安全保障）", "専門報道"),
    "geopolmonitor": ("ジオポリティカル・モニター", "専門報道"),
    "defenseone": ("ディフェンス・ワン", "専門報道"),
    "defensescoop": ("ディフェンススクープ", "専門報道"),
    "militarytimes": ("ミリタリー・タイムズ", "専門報道"),
    "taskpurpose": ("タスク・アンド・パーパス", "専門報道"),
    "afmag": ("エア・アンド・スペース・フォーシズ", "専門報道"),
    "airforcetech": ("エアフォース・テクノロジー", "専門報道"),
    "theaviationist": ("ジ・アビエーショニスト", "専門報道"),
    "navaltech": ("ネーバル・テクノロジー", "専門報道"),
    "nationalinterest": ("ナショナル・インタレスト", "専門報道"),
    "19fortyfive": ("19FortyFive", "専門報道"),
    "ukdj": ("UK ディフェンス・ジャーナル", "専門報道"),
    "eurasiantimes": ("ユーラシアン・タイムズ", "報道"),
    "gcaptain": ("gCaptain（海事）", "専門報道"),
    "maritimeexec": ("マリタイム・エグゼクティブ", "専門報道"),
    "spacenews": ("スペースニュース", "専門報道"),
    "nsf": ("NASASpaceflight", "専門報道"),
}
STATE_KINDS = ('政府発表', '国営メディア', '国営・党系メディア')

# 国名（台本の「出典は…」の文に使う）
COUNTRY_JA = {
    'CN': '中国', 'RU': 'ロシア', 'KP': '北朝鮮', 'US': 'アメリカ', 'GB': 'イギリス', 'DE': 'ドイツ', 'FR': 'フランス',
    'IT': 'イタリア', 'UN': '国連', 'QA': 'カタール', 'LV': 'ラトビア', 'NL': 'オランダ', 'UA': 'ウクライナ',
    'TW': '台湾', 'KR': '韓国', 'JP': '日本', 'HK': '香港', 'SG': 'シンガポール', 'IN': 'インド', 'CA': 'カナダ',
    'PH': 'フィリピン', 'VN': 'ベトナム', 'AU': 'オーストラリア',
    'IR': 'イラン', 'IL': 'イスラエル', 'PS': 'パレスチナ', 'LB': 'レバノン', 'SY': 'シリア', 'IQ': 'イラク', 'SD': 'スーダン',
    'YE': 'イエメン', 'PK': 'パキスタン', 'LY': 'リビア', 'VE': 'ベネズエラ',
}


def family(source):
    """系列。表に無い配信元は、配信元 id そのもの（他と束ねない）。"""
    return SOURCES.get(source, (source, None))[0]


def country(source):
    """国（ISO コード）。表に無ければ None。"""
    return SOURCES.get(source, (source, None))[1]


def name(source):
    return META.get(source, (source, '不明'))[0]


def kind(source):
    """種別（サイトの定義）。表に無い配信元は '不明'（落とさない）。"""
    return META.get(source, (source, '不明'))[1]


def is_primary_or_state(source):
    return kind(source) in STATE_KINDS
