#!/usr/bin/env bash
# One long run of the site clock: steps the Observatory site's upkeep (UPKEEP_STEPS requests at most) every
# UPKEEP_EVERY seconds and translates headlines every TRANSLATE_EVERY seconds,
# until RUN_SECONDS have passed. GitHub drops most short scheduled runs of this
# repository (11-17: upkeep ran 4 times instead of 72), so the work runs in one
# job that hands over to the next run when it ends (see site-clock.yml).
#
# PUSH_TOKEN (push of the translations) and UPKEEP_TOKEN (x-upkeep-token for
# /api/maintenance) are taken out of the environment here and given only to
# tools/push_retry.sh and tools/site_upkeep.sh, not to the translation itself.
#
# Exit 1 (a red run, and site-clock.yml then does not hand over; the schedule
# restarts the chain) when a step failed MAX_STREAK times in a row, or more
# than MAX_FAILS times in all. A failed publish never leaves a rebase half done
# or the commit behind (push_retry.sh --reset-on-fail): the next hour translates
# again from the newest main.
set -u
RUN_SECONDS=${RUN_SECONDS:-19800}          # 5 h 30 min (the job limit is 6 h)
# Upkeep every 3 hours, 60 steps: every 20 minutes exhausted D1's free daily
# row reads (5 million) by 12:00 JST on 2026-11-18, which took the site's API down.
UPKEEP_EVERY=${UPKEEP_EVERY:-10800}        # 3 h
UPKEEP_STEPS=${UPKEEP_STEPS:-60}
TRANSLATE_EVERY=${TRANSLATE_EVERY:-3600}   # 60 min
TRANSLATE_MINUTES=${TRANSLATE_MINUTES:-15}
MAX_STREAK=${MAX_STREAK:-3}
MAX_FAILS=${MAX_FAILS:-10}
here=$(cd "$(dirname "$0")" && pwd)
push_token=${PUSH_TOKEN:-}; upkeep_token=${UPKEEP_TOKEN:-}
unset PUSH_TOKEN UPKEEP_TOKEN
start=$(date +%s); next_upkeep=$start; next_translate=$start
upkeeps=0; translations=0; fails=0
upkeep_streak=0; translate_streak=0; worst_upkeep=0; worst_translate=0

upkeep() {
  UPKEEP_TOKEN=$upkeep_token STEPS=$UPKEEP_STEPS bash "$here/site_upkeep.sh"
}

translate() {
  mkdir -p docs/data/translations
  python -m translate.headlines --model "$MODEL" --minutes "$TRANSLATE_MINUTES" || return 1
  git add docs/data/translations
  git diff --cached --quiet && return 0
  git commit -q -m "データ更新: 見出しの翻訳 ($(date -u +%Y-%m-%dT%H:%MZ))"
  PUSH_TOKEN=$push_token bash "$here/push_retry.sh" --tries 3 --reset-on-fail --title "site clock"
}

while :; do
  now=$(date +%s)
  [ $((now - start)) -ge "$RUN_SECONDS" ] && break
  if [ "$now" -ge "$next_upkeep" ]; then
    if upkeep; then upkeeps=$((upkeeps+1)); upkeep_streak=0
    else fails=$((fails+1)); upkeep_streak=$((upkeep_streak+1)); fi
    [ $upkeep_streak -gt $worst_upkeep ] && worst_upkeep=$upkeep_streak
    next_upkeep=$((now + UPKEEP_EVERY))
  fi
  now=$(date +%s)
  if [ "$now" -ge "$next_translate" ] && [ $((now - start + TRANSLATE_MINUTES*60 + 120)) -lt "$RUN_SECONDS" ]; then
    if translate; then translations=$((translations+1)); translate_streak=0
    else
      fails=$((fails+1)); translate_streak=$((translate_streak+1))
      echo "::warning title=site clock::翻訳または公開に失敗（$translate_streak 回続けて）"
    fi
    [ $translate_streak -gt $worst_translate ] && worst_translate=$translate_streak
    next_translate=$((now + TRANSLATE_EVERY))
  fi
  sleep "${TICK:-30}"
done
echo "::notice title=site clock::upkeep $upkeeps, translation $translations, failures $fails (longest runs of failures: upkeep $worst_upkeep, translation $worst_translate)"
ok=true
[ "$worst_translate" -ge "$MAX_STREAK" ] && { echo "::error title=site clock::翻訳が $worst_translate 回続けて失敗"; ok=false; }
[ "$worst_upkeep" -ge "$MAX_STREAK" ] && { echo "::error title=site clock::背景更新が $worst_upkeep 回続けて失敗"; ok=false; }
[ "$fails" -gt "$MAX_FAILS" ] && { echo "::error title=site clock::失敗が $fails 回"; ok=false; }
[ "$ok" = true ]
