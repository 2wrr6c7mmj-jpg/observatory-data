# context_allied_exercises.csv — verification notes

Built 2026-10-28. 188 rows, 2008-06 to 2026-10. Every row's `source_url` was actually retrieved
(WebFetch) during the build; dates were taken from that page. The web-search budget ran out
part-way through, so later discovery relied on fetching known URL patterns (JSO/JMSDF PDFs,
JMSDF Self Defense Fleet monthly news pages, USNI tag/tracker pages, DVIDS feature pages).

## Date-quality conventions used in `notes`
- `date = JMSDF news date` / `(JMSDF news date)` in name: the JMSDF Self Defense Fleet English
  news page only gives the publication date; the activity happened on/shortly before it.
  Treat `date_start` as +/- 1 week. Rows: Freedom Edge 24-1, 24-2, 25; several MCAs (2024-06,
  2024-09, 2024-12, 2025-09); Pacific Vanguard 22/24; Pacific Dragon 22/24; Noble Typhoon,
  Noble Chinook, Noble Raven 24, 24-3; Trident 23; Kakadu 2022; JP-US-ROK 2022-09 and 2023-08.
- `(month-level)`: MOD "Major Exercises" page gives month only; `date_start`/`date_end` are the
  first/last day of that month. Rows: JP-US-AU trilateral SCS Oct 2020; multilateral with UK
  CSG21 in SCS Oct 2021; Cope North 22; Kamandag/Sama Sama 2021.
- `dates = DVIDS media/coverage dates`: official start/end not found; range is the span of DVIDS
  stories. Rows: Northern Viper 2020, Iron Fist 23/24, Orient Shield 22/23, Resolute Dragon 25.
- `Date = USNI report date`: Taiwan Strait transits where the 7th Fleet day was not captured
  (Barry 2020, Curtis Wilbur 2021, Ralph Johnson 2022-02, Sampson, cruiser 2022-05, Benfold).
- `end date not in source`: date_end left empty (Talisman Sabre 2019, Malabar 2016, ANNUALEX
  2013/2019/2023, Iron Fist 26, Orient Shield 21-2 and 25).

## Requested items NOT included (could not verify a date from a retrieved source)
- Keen Sword FY2008 (Nov 2008) — globalsecurity lists "Keen Sword 2008" with no dates; the
  avionews "11-19 November" item turned out to be 2022.
- Keen Sword 2008/2010 Japanese JSO PDFs not located (JSO press index fetch truncates at 2025-11).
- Keen Edge 2008 — no source retrieved.
- Resolute Dragon 2021, 2022, 2023, 2024, 2026 — DVIDS feature pages robots-blocked; no dates.
- Orient Shield 2008-2020, 2024, 2026 — Wikipedia page has no dates.
- Yama Sakura (all years) — Wikipedia page does not exist; DVIDS feature has no dates.
- Iron Fist 2008-2022 (California-based) and Iron Fist 25 exact dates.
- Valiant Shield 2018, 2020, 2022 — no dated source retrieved (2022 only via USNI tracker
  mention, included as a carrier-presence row on 2022-06-13).
- Northern Viper 2017, 2023, 2024, 2025 — not retrieved.
- Noble Fusion 2022/2023 — DVIDS blocked; no source.
- Balikatan 2008-2021, 2023, 2025 — Wikipedia gives no dates (2025: Japan first full participant,
  April 2025, exact dates not captured). Balikatan 2026 and 2022/2024 included.
- Cobra Gold 2008-2019, 2021-2024 — Wikipedia gives no exact dates (2013 Japan participation
  noted but undated).
- Pitch Black 2020 (cancelled, COVID) and 2026 — not on Wikipedia page.
- Kakadu 2008-2020, 2022 exact dates — RAN sources not retrieved.
- Pacific Partnership — skipped (optional).
- Pacific Vanguard 2019, 2020, 2021, 2023 — no dated source; 2026 planned (IPD26) but undated.
- Malabar 2020 Phase 2 (mid-Nov 2020, western Indian Ocean) — exact dates not in sources.
- ANNUALEX 2008, 2010, 2012, 2014, 2018 — no dated sources retrieved.
- Freedom Edge 24-1/24-2/25 exact dates (INDOPACOM: 2024-06-27..29, 2024-11-13..15,
  2025-09-15..19 are believed correct but were NOT retrieved) — rows carry JMSDF news dates.
- JP-US-PH-AU MCA 2025 (Jan/Feb/Apr 2025) — not retrieved; 2025-11-14 and 2026 MCAs included.
- UK CSG21 Japan port call (Yokosuka, ~2021-09-04..08) — no retrieved page gives dates; only the
  2021-07-26 SCS PASSEX (Wikipedia) and the Oct 2021 MOD month-level row are included.
- German frigate Bayern Tokyo port call (2021-11-05) — not in the Bundeswehr or Wikipedia
  pages retrieved; ANNUALEX participation (Nov 20-30 2021) is included instead.
- Italian CSG Cavour Yokosuka port call (Aug 2024) — only the 2024-08-09 sailing with Lincoln and
  the Noble Raven 24-3 JMSDF news item are included.
- UK CSG25 Japan (Tokyo) visit Aug 2025 — Wikipedia page lacks it; the JMSDF Sept 2 - Oct 21 2025
  escort/exercise row and the 2025-06-24 Singapore call are included.
- French CSG Clemenceau 2019 and "2023" Japan visits — Charles de Gaulle did not visit Japan in
  the sources retrieved; 2019 only Singapore (28 May 2019). 2025: Pacific Steller row included.
- German task group (Baden-Württemberg/Frankfurt am Main) Taiwan Strait transit (Sept 2024) —
  USNI year-end piece mentions it undated; omitted.
- US 7th Fleet Taiwan Strait transits: no per-year official counts retrieved, so no
  "count per year" summary rows. Individual rows included where a dated USNI article was
  fetched (2020: 3, 2021: 3, 2022: 6, 2023: 1, 2024: 4, 2026: 1). USNI tag archive pages 1-6 were
  read but not every article was opened; many 2019-2021 transits exist in that archive.
- Dual-carrier 2020-06 (Nimitz + Theodore Roosevelt, Philippine Sea, ~Jun 21-23): included as a
  2020-06-23 row from a photo caption; location not confirmed in source.
- Dual-carrier 2021-02 (Theodore Roosevelt + Nimitz, SCS, Feb 9 2021): the USNI tracker snapshot
  (Feb 8) places them separately; row reflects that snapshot, not the joint op.
- Dual-carrier 2022-06 (Lincoln + Reagan + Tripoli): included only as a tracker snapshot row.

## Source caveats
- Wikipedia used where official pages could not be retrieved: RIMPAC 2020 end date, Pitch Black
  (all), Malabar 2024, Balikatan 2022/2024, Cobra Gold 2020, ANNUALEX 2016, Orient Shield 25,
  JS Sazanami Taiwan Strait transit (2024_in_Japan page), HMS Queen Elizabeth/Prince of Wales,
  Charles de Gaulle.
- Keen Edge 2026: JSO PDF gives Jan 29 - Feb 5 2026; a USAF 21st AF article gives Jan 27 - Feb 4.
  JSO official dates used.
- Talisman Sabre 2023: JMSDF PDF gives Jul 20 - Aug 4; Wikipedia gives Jul 22 - Aug 4.
- Malabar 2025: INDOPACOM gives Nov 10-18; JMSDF PDF gives Nov 11-18 (at-sea). INDOPACOM used.
- RIMPAC 2014: sources differ 22 vs 23 nations.
- Region coding: events in Hawaii, Australia, India, Thailand, Indonesia, Singapore and the US
  West Coast are coded `multi` with lat/lon of the venue; Guam/Philippine Sea = `wpac`.
