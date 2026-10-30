# context_political.csv — verification notes

Built 2026-10-28. 256 rows, 2008-01-12 .. 2026-08-13. Every row carries a source_url that was
actually fetched (WebFetch) or returned by WebSearch and then fetched. Dates taken from the source
text; where the source only implied a date, the row is flagged below.

## Rows with inferred / report-date dates (keep, but treat with caution)
- 2020-10-26 Harpoon Coastal Defense System $2.37B: naval-technology article (dated 2020-10-27) does
  not state the DSCA notification day; 10-26 is the generally reported date.
- 2024-10-25 NASAMS/radar $1.99B: USTBC release dated 10-28; PBS says Taiwan thanked US on Sat 10-26.
  DSCA notification day (10-25) inferred.
- 2024-11-29 $385M F-16/AESA + IMSE: article dated 11-30 says "a day before" Lai's Sat departure.
- 2023-08-23 $500M F-16 IRST: article dated 08-24 says "Wednesday".
- 2023-03-27 Astellas detention: Wikipedia gives only "March 2023"; 03-27 is the BBC report date
  (Japan demanded release). Actual detention was earlier in March.
- 2025-12-22 Hagiuda LDP delegation meets Lai: Wikipedia gives "December 2025"; 12-22 is the date in
  the cited Reuters URL.
- 2026-04-10 Xi–Cheng Li-wun meeting: Wikipedia gives "April 2026"; 04-10 is the cited report date.
- 2023-08-06 CCG water cannon on Sierra Madre resupply: Wikipedia Sierra Madre page says Aug 6;
  most press dated the incident Aug 5 (PCG disclosed Aug 6).
- 2020-06-19 Senkaku "67 consecutive days": date is the "as of" date in the source, not an event day.
- 2018-01-19 US NDS 2018: Wikipedia cites Mattis's Jan 19 2018 unveiling event (no release line).
- 2023-06-18/19 Blinken Beijing: NPR text only says Sunday/Monday; dates from URL (2023/06/19).
- NPC 2013 closing: Wikipedia "First session of the 12th NPC" says 16 March; premier press
  conference list says 17 March. Used 17 March (closing day = press conference day).

## Requested items omitted (could not verify within this run — web search budget exhausted)
- Chinese buoy placed in Japan's EEZ (July 2023): only indirectly referenced (Kishida–Xi 2023-11-16
  MOFA readout demands removal; Senkaku page mentions 2024 removal with no date). No exact placement date.
- Japanese Diet member visits to Taiwan 2022–2024 that drew PRC protest (Furuya Aug 2022, Hagiuda Dec
  2022, Aso Aug 2023 Ketagalan Forum): pages fetched did not give dates. Only Hagiuda Dec 2025 and
  Furuya's 2026-03-30 PRC sanction are included.
- Austin–Dong video call (Apr 2024) and Hegseth–Dong call (Sept 2025): no exact day in sources.
- Xue Jian (Osaka consul general) X post date (Nov 8 2025): Wikipedia timeline implies but does not
  date it; the Japanese protest/summons rows (Nov 10, 14) are included instead via MOFA/crisis page.
- August 2025 extension of the US–China tariff truce and Nov 2025 Busan deal implementation date:
  not confirmed; only Busan meeting 2025-10-30 included.
- Russia's 2025-09 SCO Tianjin summit: not fetched; only the 2025-09-03 parade.
- Putin reported 2026 China visit (China–Russia relations page mentions 16 May 2026): page text looked
  conflated with 2024; omitted.
- North Korea April/July 2023 Hwasong-18 tests: list page gave month only; omitted.
- Second Thomas Shoal Dec 2023, Mar 2024 incidents and July 2024 provisional arrangement: no exact
  dates in fetched pages; omitted.
- 2015 Beijing V-Day parade (Sep 3 2015): not verified; omitted.
- China's reaction sanctions on Japanese individuals in 2025 crisis beyond seafood/travel: not found.
- Taiwan Legislative Yuan 2010-01-09 (by-election) and 2014 local elections: not added (not requested).

## Source caveats
- Several annual rows (PLA Day, PRC National Day, ROC National Day, Chinese New Year) cite one
  reference page for all years; CNY dates are from calendarific.com (consistent with standard tables).
- Wikipedia pages are used where government/wire pages could not be fetched; each was read this run.
- Named PLA exercises around Taiwan were included as 'incident' rows for context even though they
  may duplicate a separate military-activity table.
