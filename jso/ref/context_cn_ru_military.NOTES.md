# context_cn_ru_military.csv — verification notes

Built 2026-10-28. 121 rows, 2012-04 to 2026-10. Every row's `source_url` was fetched and the
date/detail checked against the page text. Web search budget was exhausted before all
candidate events could be verified; the following were omitted or are flagged.

## Omitted (could not verify a date from a retrieved source)

- **Liaoning Apr 2018 Pacific transit** (requested) — search results only returned 2020/2022 items; no page fetched confirming an April 2018 Miyako transit. Omitted.
- **Any named PLA large-scale exercise around Taiwan in 2026** — ASPI tracker (fetched) lists nothing after Strait Thunder-2025A; USNI/China Daily pages fetched list Justice Mission-2025 (Dec 2025) as the latest. An SCMP Jan 2026 "PLA holds drill near Taiwan" page could not be fetched (client error). No 2026 named exercise row.
- **Shandong 2026 deployments** (requested) — EurasianTimes 18 Jul 2026 page failed to fetch; USNI June/Aug 2026 pulses fetched mention no Shandong activity. Omitted.
- **Joint Sea-2015 I (Mediterranean)** — mentioned in USNI 2015 article but undated. Omitted.
- **China-Russia 5th joint naval patrol after Northern/Interaction-2024 (Sep-Oct 2024)** — announced in Global Times (fetched) but no patrol dates/route retrieved. Note: USNI calls the Aug 2025 patrol the "fifth annual"; numbering in the CSV follows the source for each row.
- **Jan 2017 PLAAF 8-aircraft Tsushima Strait flight** — SCMP page fetched gives "Monday" without a calendar date. Omitted.
- **Oct 2008 PLAN Tsugaru first transit, Apr 2010 PLAN Miyako/helicopter buzz, Jul 2013 PLAN Soya first transit, Jun 2022 CN/RU ships circling Japan** — not searched before budget ran out. Omitted (would be useful additions).
- **2026-10-20 Japan report of surge in CN/RU airspace violations** — seen only as a USNI sidebar headline, no article fetched. Omitted.
- **Nov 2025 China statements re PM Takaichi Taiwan remarks** — not retrieved; no statement rows for that episode.

## Rows with inferred or partial dates (flagged)

- `2014-05-20..2014-05-26` Joint Sea-2014: USNI gives start 20 May and "week-long"; PLA Daily says sea phase concluded 24 May. End date is approximate.
- `2016-09-12..2016-09-19` Joint Sea-2016: end date inferred from "eight-day exercise" (USNI) / island-seizing drill 18 Sep + closing ceremony Monday (PLA Daily).
- `2021-10-18..2021-10-23` 1st joint patrol: only the Tsugaru transit (18 Oct) is verified from the fetched page; end date is from the later Global Times note that the first patrol lasted 7 days.
- `2022-05-02` Liaoning May 2022: no verified end date (date_end empty). Ops confirmed through 7 May.
- `2022-09-04` CN/RU Soya transit: "Sunday" in a 6 Sep 2022 article, inferred 4 Sep. (USNI Vostok article independently places the La Perouse passage "early Sunday".)
- `2022-09-26..27` Izu Islands transit: weekdays from 28 Sep 2022 Jiji/nippon.com dateline.
- `2022-11-02` Shupang survey ship: "Wednesday" in USNI 4 Nov 2022 article.
- `2023-07-20..23` Northern/Interaction-2023: USNI says Thu-Sun; JSO sighted ships exercising from 18 Jul. Start may be 18 or 20 Jul.
- `2023-07-27..2023-08-27` 3rd joint patrol: end date is the Al Jazeera report date of Russian ships' return ("more than three weeks").
- `2024-07-04..2024-07-16` 4th joint patrol: Russian MoD "complete" release dated 16 Jul; 15-day duration.
- `2024-10-31` dual-carrier SCS drill: date is the PLAN video release date; drill was "late October".
- `2024-12-09` unnamed Dec 2024 drills: ASPI gives 9-11 Dec; USNI only says "began Monday" (9 Dec).
- `2025-05-03` CCG helicopter airspace violation: "yesterday" in 4 May 2025 article.
- `2025-12-06` Liaoning Dec 2025: JSO release confirms 5-6 Dec; 12 Dec return and 6 Dec radar incident from Jiji/MOFA.
- `2026-06-22` Fujian Taiwan Strait: photo dated 22 Jun; ROC MND announcement 23 Jun ("this week").
- `2026-07-13..29` 6th joint patrol: Maritime Executive gives departure 13 Jul / 16 days; USNI pulse gives Vladivostok arrival 29 Jul.
- `2026-08-04` Russian Pacific Fleet main drill: no end date ("several weeks").

## Source notes

- JSO (mod.go.jp/js) PDFs used directly: 2025-12-10 (Liaoning), 2026-07-01 (joint bomber flight). Other JSO data is cited via USNI/Jiji/nippon.com reporting of JSO releases.
- Janes URL for 2021-11-19 cites China MND, not Japan MoD.
- lat/lon are rough centres of activity only; multi-region patrols use a mid-route point.
