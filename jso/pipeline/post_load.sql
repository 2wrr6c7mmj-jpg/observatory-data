-- 1. transit_pairs: explicit links from "〜と同一である" (same_as_release)
--    in = the sighting carrying the link, out = the last strait sighting of the referenced release
DELETE FROM transit_pairs;
INSERT OR IGNORE INTO transit_pairs(out_sighting_id, in_sighting_id, method, days_away)
SELECT o.sighting_id, i.sighting_id, 'explicit',
       CAST(julianday(COALESCE(i.observed_date, r2.pub_date)) - julianday(COALESCE(o.observed_date_end, o.observed_date, r1.pub_date)) AS INTEGER)
FROM sightings i
JOIN releases r2 ON r2.release_id = i.release_id
JOIN releases r1 ON r1.release_id = i.same_as_release
JOIN sightings o ON o.release_id = r1.release_id
WHERE i.same_as_release IS NOT NULL
  AND o.seq = (SELECT max(seq) FROM sightings x WHERE x.release_id = r1.release_id AND x.strait IS NOT NULL)
  AND i.seq = (SELECT max(seq) FROM sightings y WHERE y.release_id = i.release_id AND y.same_as_release IS NOT NULL);

-- 2. vessels registry from hull numbers
DELETE FROM vessels;
INSERT INTO vessels(nation, hull_no, class_key, name, fleet, first_seen, last_seen, sightings_n)
SELECT u.nation, u.hull_no,
       (SELECT class_key FROM sighting_units z WHERE z.nation=u.nation AND z.hull_no=u.hull_no GROUP BY class_key ORDER BY count(*) DESC LIMIT 1),
       (SELECT name FROM sighting_units z WHERE z.nation=u.nation AND z.hull_no=u.hull_no AND name IS NOT NULL ORDER BY rowid DESC LIMIT 1),
       NULL,
       min(COALESCE(s.observed_date, r.pub_date)), max(COALESCE(s.observed_date, r.pub_date)), count(DISTINCT s.release_id)
FROM sighting_units u JOIN sightings s ON s.sighting_id = u.sighting_id JOIN releases r ON r.release_id = s.release_id
WHERE u.hull_no IS NOT NULL
GROUP BY u.nation, u.hull_no;

-- 3. analysis view: one row per strait passage with context counts in ±14 days
CREATE VIEW IF NOT EXISTS v_passages AS
SELECT s.sighting_id, r.release_id, r.nation, r.pub_date, s.observed_date, s.observed_date_end, s.observed_time,
       s.strait, s.strait_dir, s.waters_from, s.waters_to, s.heading_jp, s.ref_point, s.bearing_jp, s.distance_km, s.lat, s.lon,
       s.activity, s.activity_count, s.unit_count,
       (SELECT group_concat(COALESCE(u.class_jp,'') || CASE WHEN u.hull_no IS NULL THEN '' ELSE '(' || u.hull_no || ')' END, ' / ') FROM sighting_units u WHERE u.sighting_id = s.sighting_id) AS units,
       (SELECT group_concat(DISTINCT vc.type_cat) FROM sighting_units u JOIN vessel_classes vc ON vc.class_key = u.class_key WHERE u.sighting_id = s.sighting_id) AS type_cats,
       (SELECT count(*) FROM sighting_units u JOIN vessel_classes vc ON vc.class_key=u.class_key WHERE u.sighting_id=s.sighting_id AND vc.type_cat='CV') AS has_carrier,
       (SELECT count(*) FROM sighting_units u WHERE u.sighting_id=s.sighting_id AND u.kind='sub') AS has_sub,
       (SELECT count(*) FROM sighting_units u JOIN vessel_classes vc ON vc.class_key=u.class_key WHERE u.sighting_id=s.sighting_id AND vc.type_cat IN ('AOR','AKS')) AS has_oiler,
       (SELECT count(*) FROM sighting_units u JOIN vessel_classes vc ON vc.class_key=u.class_key WHERE u.sighting_id=s.sighting_id AND vc.type_cat IN ('AGI','AGS','AGM')) AS has_intel,
       s.same_as_release,
       (SELECT p.days_away FROM transit_pairs p WHERE p.in_sighting_id = s.sighting_id LIMIT 1) AS days_away_from_linked,
       (SELECT group_concat(unit_name, '/') FROM jp_units j WHERE j.release_id = r.release_id) AS jp_units,
       (SELECT count(*) FROM context_events c WHERE c.actor IN ('CN','CNRU') AND c.category='exercise' AND date(c.date_start,'-14 days') <= s.observed_date AND date(COALESCE(c.date_end,c.date_start),'+14 days') >= s.observed_date) AS cn_exercise_14d,
       (SELECT group_concat(c.name, ' | ') FROM context_events c WHERE c.actor IN ('CN','CNRU') AND date(c.date_start,'-14 days') <= s.observed_date AND date(COALESCE(c.date_end,c.date_start),'+14 days') >= s.observed_date) AS cn_ru_events_14d,
       (SELECT count(*) FROM context_events c WHERE c.actor IN ('JP','US','USJP','AU','PH','KR','TW','UK','FR','IN','MULTI','DE') AND c.category='exercise' AND date(c.date_start,'-14 days') <= s.observed_date AND date(COALESCE(c.date_end,c.date_start),'+14 days') >= s.observed_date) AS allied_exercise_14d,
       (SELECT group_concat(c.name, ' | ') FROM context_events c WHERE c.actor IN ('JP','US','USJP','AU','PH','KR','TW','UK','FR','IN','MULTI','DE') AND date(c.date_start,'-14 days') <= s.observed_date AND date(COALESCE(c.date_end,c.date_start),'+14 days') >= s.observed_date) AS allied_events_14d,
       (SELECT group_concat(c.name, ' | ') FROM context_events c WHERE c.category IN ('election','inauguration','summit','visit','statement','policy','anniversary','incident','sanction','legislation') AND date(c.date_start,'-14 days') <= s.observed_date AND date(COALESCE(c.date_end,c.date_start),'+14 days') >= s.observed_date) AS political_events_14d,
       r.title, r.url
FROM sightings s JOIN releases r ON r.release_id = s.release_id
WHERE s.strait IS NOT NULL;
