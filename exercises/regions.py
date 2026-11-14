"""Named areas for the map. Events carry one area id, not coordinates: the site
draws one label per area at the point below (the user asked for the area,
not exact positions)."""

# id, Japanese name, latitude, longitude
REGIONS = [
    ("sea-of-japan", "日本海", 40.0, 134.5),
    ("okhotsk", "オホーツク海", 50.5, 148.0),
    ("hokkaido", "北海道周辺", 43.3, 143.5),
    ("tohoku-pacific", "東北・三陸沖", 39.0, 143.5),
    ("honshu-south", "関東南方・本州南方", 33.6, 139.6),
    ("shikoku-kii", "四国・紀伊半島沖", 32.6, 134.2),
    ("kyushu", "九州周辺", 32.2, 131.6),
    ("japan", "日本（各地）", 36.0, 138.5),
    ("nansei", "沖縄・南西諸島", 26.3, 127.6),
    ("east-china-sea", "東シナ海", 30.0, 125.5),
    ("yellow-sea", "黄海", 35.5, 123.6),
    ("bohai", "渤海", 38.8, 120.2),
    ("korea", "朝鮮半島", 37.2, 127.6),
    ("taiwan-strait", "台湾海峡", 24.4, 119.6),
    ("taiwan", "台湾周辺", 23.7, 121.0),
    ("taiwan-east", "台湾東方", 23.5, 123.4),
    ("bashi", "バシー海峡・ルソン海峡", 20.6, 121.0),
    ("south-china-sea", "南シナ海", 14.5, 114.5),
    ("spratly", "南沙諸島", 10.0, 114.5),
    ("philippines", "フィリピン", 13.0, 122.5),
    ("philippine-sea", "フィリピン海・西太平洋", 21.5, 132.0),
    ("guam", "グアム・マリアナ", 14.5, 145.0),
    ("hawaii", "ハワイ", 21.0, -157.8),
    ("australia-north", "豪州北部", -13.0, 132.0),
    ("australia", "豪州", -27.0, 147.0),
    ("indonesia", "インドネシア周辺", -3.0, 115.0),
    ("southeast-asia", "東南アジア", 12.0, 102.0),
    ("bay-of-bengal", "ベンガル湾・アンダマン海", 14.0, 89.0),
    ("india", "インド", 22.0, 79.0),
    ("indian-ocean", "インド洋", -5.0, 75.0),
    ("china", "中国本土", 34.0, 110.0),
    ("russia-far-east", "極東ロシア", 48.0, 135.0),
    ("russia", "ロシア（その他）", 55.0, 45.0),
    ("north-pacific", "北太平洋・アラスカ", 57.0, -155.0),
    ("us-mainland", "米本土", 37.0, -100.0),
    ("other", "その他", None, None),
]
BY_ID = {r[0]: r for r in REGIONS}
IDS = [r[0] for r in REGIONS]


def name(region_id):
    r = BY_ID.get(region_id)
    return r[1] if r else BY_ID["other"][1]


def prompt_list():
    return "\n".join(f"- {r[0]}: {r[1]}" for r in REGIONS)


# The place words decide the area when they name one (the model's choice is only the fallback).
# Seas first: "沖縄本島北西の東シナ海" is the East China Sea, not Okinawa.
PLACE_WORDS = [
    ("taiwan-strait", r"台湾海峡|taiwan strait|台灣海峽|台湾海峡"),
    ("bashi", r"バシー海峡|ルソン海峡|bashi|luzon strait"),
    ("east-china-sea", r"東シナ海|東海|east china sea|东海"),
    ("south-china-sea", r"南シナ海|south china sea|南海|西沙|paracel|スカボロー|scarborough|黄岩"),
    ("spratly", r"南沙|spratly|スプラトリー|セカンド・トーマス|second thomas|仁愛"),
    ("sea-of-japan", r"日本海|sea of japan|east sea|日本海"),
    ("yellow-sea", r"黄海|yellow sea"),
    ("bohai", r"渤海|bohai"),
    ("okhotsk", r"オホーツク|okhotsk"),
    ("philippine-sea", r"フィリピン海|philippine sea|西太平洋|western pacific|太平洋上"),
    ("taiwan-east", r"台湾東|台湾以東|east of taiwan|台岛以东"),
    ("taiwan", r"台湾周辺|台湾本島|around taiwan|台岛"),
    ("guam", r"グアム|guam|マリアナ|mariana|テニアン|tinian"),
    ("hawaii", r"ハワイ|hawaii|真珠湾|pearl harbor"),
    ("nansei", r"沖縄|南西諸島|宮古|石垣|与那国|奄美|okinawa|ishigaki|yonaguni|miyako"),
    ("kyushu", r"九州|日向灘|佐世保|鹿児島|大分|長崎|熊本|宮崎|kyushu|sasebo"),
    ("shikoku-kii", r"四国|紀伊|土佐|shikoku"),
    ("honshu-south", r"関東南方|本州南方|房総|伊豆|相模|東京湾|横須賀|yokosuka"),
    ("tohoku-pacific", r"三陸|東北|八戸|tohoku"),
    ("hokkaido", r"北海道|hokkaido|津軽"),
    ("korea", r"朝鮮半島|韓国|korean peninsula|south korea|黄海道"),
    ("philippines", r"フィリピン|philippines|ルソン島|luzon|パラワン|palawan"),
    ("australia-north", r"ダーウィン|darwin|豪州北部|northern australia|northern territory"),
    ("australia", r"豪州|オーストラリア|australia|クイーンズランド|queensland"),
    ("indonesia", r"インドネシア|indonesia"),
    ("bay-of-bengal", r"ベンガル湾|アンダマン|bay of bengal|andaman"),
    ("india", r"インド本土|india\b"),
    ("russia-far-east", r"極東|沿海|ウラジオストク|カムチャツカ|vladivostok|kamchatka|primorye"),
    ("us-mainland", r"カリフォルニア|テキサス|california|texas|san diego"),
]


def from_place(place):
    """The first area (in the order above) that the place words name, or None."""
    import re
    if not place:
        return None
    for rid, pattern in PLACE_WORDS:
        if re.search(pattern, place, re.I):
            return rid
    return None
