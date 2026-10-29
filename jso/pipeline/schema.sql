-- jso-naval-transits  D1 schema  (SQLite dialect)
-- 1 release  ->  N sightings (one per observed event/paragraph)
-- 1 sighting ->  N sighting_units (one per vessel / aircraft)

CREATE TABLE IF NOT EXISTS releases (
  release_id    TEXT PRIMARY KEY,          -- e.g. 2026/p20261027_03
  pub_date      TEXT NOT NULL,             -- YYYY-MM-DD
  nation        TEXT NOT NULL,             -- CN | RU | CNRU | OTHER
  domain        TEXT NOT NULL,             -- ship | air | correction | incident
  title         TEXT NOT NULL,             -- index-page title (may carry class/strait summary)
  url           TEXT NOT NULL,
  pages         INTEGER,
  has_track_map INTEGER DEFAULT 0,         -- 行動概要 map page present
  corrects      TEXT,                      -- release_id this one corrects
  raw_text      TEXT,                      -- full extracted text
  parsed_at     TEXT,
  parse_status  TEXT                       -- ok | partial | failed
);

CREATE TABLE IF NOT EXISTS sightings (
  sighting_id     INTEGER PRIMARY KEY AUTOINCREMENT,
  release_id      TEXT NOT NULL REFERENCES releases(release_id),
  seq             INTEGER NOT NULL,          -- order inside the release
  observed_date   TEXT,                      -- YYYY-MM-DD (start)
  observed_date_end TEXT,                    -- for multi-day activity
  observed_time   TEXT,                      -- HH:MM local (approx)
  ref_point       TEXT,                      -- 基準地 (宮古島 / 大隅半島 ...)
  bearing_jp      TEXT,                      -- 東北東 etc.
  distance_km     REAL,
  lat             REAL,                      -- derived from ref_point + bearing + distance
  lon             REAL,
  heading_jp      TEXT,                      -- 南東進 / 北西進 / 遊弋 ...
  heading_deg     INTEGER,
  strait          TEXT,                      -- normalized strait key (miyako / osumi / tsushima / ...)
  strait_dir      TEXT,                      -- E / W / N / S / NE / NW / SE / SW
  waters_from     TEXT,                      -- 東シナ海 / 太平洋 / 日本海 / オホーツク海 ...
  waters_to       TEXT,
  activity        TEXT,                      -- 発着艦 / 陣形運動 / 遊弋 / 射撃訓練 / 中露共同航行 ...
  activity_count  INTEGER,                   -- e.g. 210 (launch/recovery count)
  same_as_release TEXT,                      -- "...と同一である" -> linked release_id (resolved later)
  same_as_text    TEXT,                      -- raw phrase for the link
  unit_count      INTEGER,                   -- 計N隻 / N機
  sentence        TEXT                       -- source sentence(s)
);
CREATE INDEX IF NOT EXISTS ix_sightings_date   ON sightings(observed_date);
CREATE INDEX IF NOT EXISTS ix_sightings_strait ON sightings(strait, observed_date);
CREATE INDEX IF NOT EXISTS ix_sightings_rel    ON sightings(release_id);

CREATE TABLE IF NOT EXISTS sighting_units (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  sighting_id   INTEGER NOT NULL REFERENCES sightings(sighting_id),
  nation        TEXT NOT NULL,               -- CN | RU
  kind          TEXT NOT NULL,               -- ship | sub | air | uav
  class_jp      TEXT,                        -- ルーヤンⅢ級ミサイル駆逐艦 / Y-9 ...
  class_key     TEXT,                        -- normalized key (luyang3 / jiangkai2 / y9 ...)
  hull_no       TEXT,                        -- 艦番号 (may be NULL before ~2013)
  name          TEXT,                        -- 遼寧 / 山東 ...
  count         INTEGER DEFAULT 1
);
CREATE INDEX IF NOT EXISTS ix_su_hull  ON sighting_units(hull_no);
CREATE INDEX IF NOT EXISTS ix_su_class ON sighting_units(class_key);

CREATE TABLE IF NOT EXISTS vessel_classes (
  class_key   TEXT PRIMARY KEY,
  nation      TEXT NOT NULL,
  class_jp    TEXT NOT NULL,                 -- canonical JSO spelling
  class_nato  TEXT,                          -- Luyang III
  class_native TEXT,                         -- 052D
  type_cat    TEXT NOT NULL,                 -- CV | DDG | FFG | FFL | AOR | AGI | AGS | SS | SSN | SSBN | LST | AS | PGG | TUG | OTHER
  role        TEXT,                          -- surface_combatant | support | intel | amphib | submarine
  aliases     TEXT                           -- JSON array of alternative spellings seen in releases
);

CREATE TABLE IF NOT EXISTS vessels (
  nation      TEXT NOT NULL,
  hull_no     TEXT NOT NULL,
  class_key   TEXT,
  name        TEXT,
  fleet       TEXT,                          -- CN: north | east | south  (inferred from hull_no / open sources)
  first_seen  TEXT,
  last_seen   TEXT,
  sightings_n INTEGER DEFAULT 0,
  PRIMARY KEY (nation, hull_no)
);

CREATE TABLE IF NOT EXISTS straits (
  strait_key  TEXT PRIMARY KEY,
  name_jp     TEXT NOT NULL,
  name_en     TEXT,
  lat         REAL, lon REAL,                -- representative mid-point
  side_a      TEXT, side_b TEXT,             -- waters on each side (for direction -> from/to)
  axis        TEXT,                          -- EW | NS | NESW | NWSE  (which headings count as a passage)
  patterns    TEXT                           -- JSON array of regex fragments used by the parser
);

CREATE TABLE IF NOT EXISTS ref_points (
  name_jp  TEXT PRIMARY KEY,                 -- 宮古島
  pref     TEXT,                             -- 沖縄県
  lat      REAL, lon REAL,
  aliases  TEXT
);

CREATE TABLE IF NOT EXISTS jp_units (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  release_id   TEXT NOT NULL REFERENCES releases(release_id),
  unit_kind    TEXT,                         -- ship | air
  unit_name    TEXT,                         -- まきなみ / P-3C / P-1 / E-2C ...
  parent_unit  TEXT,                         -- 第3護衛隊 / 第5航空群 ...
  base         TEXT                          -- 大湊 / 那覇 ...
);

CREATE TABLE IF NOT EXISTS transit_pairs (
  out_sighting_id INTEGER REFERENCES sightings(sighting_id),
  in_sighting_id  INTEGER REFERENCES sightings(sighting_id),
  method          TEXT,                      -- explicit (同一である) | hull_match | class_match
  days_away       INTEGER,
  PRIMARY KEY (out_sighting_id, in_sighting_id)
);

CREATE TABLE IF NOT EXISTS context_events (
  event_id    INTEGER PRIMARY KEY AUTOINCREMENT,
  date_start  TEXT NOT NULL,
  date_end    TEXT,
  actor       TEXT NOT NULL,                 -- CN | US | JP | TW | RU | KR | PH | AU | MULTI ...
  category    TEXT NOT NULL,                 -- exercise | political | incident | holiday | deployment | statement
  name        TEXT NOT NULL,                 -- 聯合利剣-2024A / Keen Sword 25 / 台湾総統就任 ...
  region      TEXT,                          -- taiwan_strait | ecs | scs | sea_of_japan | wpac ...
  lat REAL, lon REAL,
  notes       TEXT,
  source_url  TEXT
);
CREATE INDEX IF NOT EXISTS ix_ctx_date ON context_events(date_start, date_end);

-- One row per release ready for analysis (joins resolved later by views)
CREATE VIEW IF NOT EXISTS v_transits AS
SELECT s.sighting_id, r.release_id, r.pub_date, r.nation, r.domain,
       s.observed_date, s.observed_time, s.ref_point, s.bearing_jp, s.distance_km, s.lat, s.lon,
       s.heading_jp, s.strait, s.strait_dir, s.waters_from, s.waters_to,
       s.activity, s.activity_count, s.unit_count, s.same_as_release,
       (SELECT group_concat(COALESCE(u.class_jp,'') || CASE WHEN u.hull_no IS NULL THEN '' ELSE '(' || u.hull_no || ')' END, ' / ')
          FROM sighting_units u WHERE u.sighting_id = s.sighting_id) AS units,
       r.title, r.url
FROM sightings s JOIN releases r ON r.release_id = s.release_id;
