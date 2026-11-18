#!/usr/bin/env bash
# One long run of the site clock: steps the Observatory site's upkeep (UPKEEP_STEPS requests at most) every
# UPKEEP_EVERY seconds and translates headlines every TRANSLATE_EVERY seconds,
# until RUN_SECONDS have passed. GitHub drops most short scheduled runs of this
# repository (11-17: upkeep ran 4 times instead of 72), so the work runs in one
# job that hands over to the next run when it ends (see site-clock.yml).
#
# The tokens are not in this script's environment (site-clock.yml): PUSH_TOKEN_FILE and
# UPKEEP_TOKEN_FILE name files (mode 600, in RUNNER_TEMP) that only tools/push_retry.sh and
# tools/site_upkeep.sh read, when they need them. The translation runs as the same user and could read
# the files too; what this avoids is handing the tokens to every child process through the environment
# (the model server itself runs in a container that sees only the model, tools/llama_server.sh).
#
# Exit 1 (a red run, and site-clock.yml then does not hand over; the schedule restarts the chain) when
# - the site refused the upkeep token (401/403) even once (no more upkeep in this run then),
# - more than half of this run's upkeep rounds failed (about 2 rounds in a run, every 3 h),
# - the translation failed MAX_STREAK times in a row, or there were more than MAX_FAILS failures in all.
# A failed publish never leaves a rebase half done or the commit behind (push_retry.sh --reset-on-fail):
# the next hour translates again from the newest main.
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
start=$(date +%s); next_upkeep=$start; next_translate=$start
upkeeps=0; translations=0; fails=0
upkeep_fails=0; upkeep_denied=false; translate_streak=0; worst_translate=0

upkeep() {
  STEPS=$UPKEEP_STEPS bash "$here/site_upkeep.sh"
}

translate() {
  mkdir -p docs/data/translations
  python -m translate.headlines --model "$MODEL" --minutes "$TRANSLATE_MINUTES" || return 1
  git add docs/data/translations
  git diff --cached --quiet && return 0
  git commit -q -m "データ更新: 見出しの翻訳 ($(date -u +%Y-%m-%dT%H:%MZ))"
  bash "$here/push_retry.sh" --tries 3 --reset-on-fail --title "site clock"
}

# Exercises (exercises.yml) are due every 3 hours, but GitHub drops most of its scheduled runs: start it
# (force: false, so it still checks that it is due) when the newest main's collection is 150 min old or more.
EXERCISES_DUE=${EXERCISES_DUE:-9000}
exercises_kick() {
  [ -n "${PUSH_TOKEN_FILE:-}" ] && [ -s "$PUSH_TOKEN_FILE" ] && [ -n "${GITHUB_REPOSITORY:-}" ] || return 0
  git fetch -q origin main 2>/dev/null || return 0
  local last age hfile code
  last=$(git show origin/main:docs/data/exercises/status.json 2>/dev/null \
         | python3 -c 'import json,sys; print(int(json.load(sys.stdin).get("generatedAt", 0)) // 1000)' 2>/dev/null) || last=0
  age=$(( $(date +%s) - ${last:-0} ))
  [ "$age" -ge "$EXERCISES_DUE" ] || return 0
  hfile=$(mktemp)
  { printf 'Authorization: Bearer '; cat -- "$PUSH_TOKEN_FILE"; printf '\n'; } > "$hfile"
  code=$(curl -sS -m 20 -o /dev/null -w '%{http_code}' -X POST -H "@$hfile" -H 'Accept: application/vnd.github+json' \
         "${GITHUB_API_URL:-https://api.github.com}/repos/$GITHUB_REPOSITORY/actions/workflows/exercises.yml/dispatches" \
         -d '{"ref":"main","inputs":{"force":"false"}}') || code=000
  rm -f -- "$hfile"
  if [ "$code" = 204 ]; then echo "exercises: last collection $((age/60)) min ago, started exercises.yml"
  else echo "::warning title=site clock::exercises.yml を起動できなかった（HTTP $code）"; fi
}

while :; do
  now=$(date +%s)
  [ $((now - start)) -ge "$RUN_SECONDS" ] && break
  if [ "$now" -ge "$next_upkeep" ] && [ "$upkeep_denied" = false ]; then
    upkeep; rc=$?
    if [ $rc = 0 ]; then upkeeps=$((upkeeps+1))
    else
      fails=$((fails+1)); upkeep_fails=$((upkeep_fails+1))
      [ $rc = 3 ] && upkeep_denied=true
    fi
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
    exercises_kick
    next_translate=$((now + TRANSLATE_EVERY))
  fi
  sleep "${TICK:-30}"
done
echo "::notice title=site clock::upkeep $upkeeps ok / $upkeep_fails failed, translation $translations, failures $fails (longest run of translation failures: $worst_translate)"
ok=true
[ "$upkeep_denied" = true ] && { echo "::error title=site clock::/api/maintenance がトークンを拒否した（Secret UPKEEP_TOKEN をサイト側の値と合わせる）"; ok=false; }
[ $((upkeep_fails * 2)) -gt $((upkeeps + upkeep_fails)) ] && { echo "::error title=site clock::背景更新が $((upkeeps + upkeep_fails)) 回中 $upkeep_fails 回失敗"; ok=false; }
[ "$worst_translate" -ge "$MAX_STREAK" ] && { echo "::error title=site clock::翻訳が $worst_translate 回続けて失敗"; ok=false; }
[ "$fails" -gt "$MAX_FAILS" ] && { echo "::error title=site clock::失敗が $fails 回"; ok=false; }
[ "$ok" = true ]
