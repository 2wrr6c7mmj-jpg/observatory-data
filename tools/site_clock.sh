#!/usr/bin/env bash
# One long run of the site clock: steps the Observatory site's upkeep every
# UPKEEP_EVERY seconds and translates headlines every TRANSLATE_EVERY seconds,
# until RUN_SECONDS have passed. GitHub drops most short scheduled runs of this
# repository (11-17: upkeep ran 4 times instead of 72), so the work runs in one
# job that hands over to the next run when it ends (see site-clock.yml).
set -u
RUN_SECONDS=${RUN_SECONDS:-19800}          # 5 h 30 min (the job limit is 6 h)
UPKEEP_EVERY=${UPKEEP_EVERY:-1200}         # 20 min
TRANSLATE_EVERY=${TRANSLATE_EVERY:-3600}   # 60 min
TRANSLATE_MINUTES=${TRANSLATE_MINUTES:-15}
SITE=https://observatory-watch.pages.dev
start=$(date +%s); next_upkeep=$start; next_translate=$start; upkeeps=0; translations=0; fails=0

upkeep() {
  local tasks="" bad=0 body task
  for i in $(seq 1 100); do
    body=$(curl -sS -m 25 -H 'Accept: application/json' "$SITE/api/maintenance") || { bad=$((bad+1)); [ $bad -ge 3 ] && break; sleep 5; continue; }
    task=$(printf '%s' "$body" | sed -n 's/.*"task":"\([^"]*\)".*/\1/p')
    tasks="$tasks ${task:-?}"
    [ "$task" = idle ] && break
    sleep 2
  done
  echo "upkeep $(date -u +%H:%MZ): $(echo $tasks | tr ' ' '\n' | sed 's/:.*//' | sort | uniq -c | tr '\n' ' ')"
  [ $bad -lt 3 ]
}

translate() {
  mkdir -p docs/data/translations
  python -m translate.headlines --model "$MODEL" --minutes "$TRANSLATE_MINUTES" || return 1
  git add docs/data/translations
  git diff --cached --quiet && return 0
  git commit -q -m "データ更新: 見出しの翻訳 ($(date -u +%Y-%m-%dT%H:%MZ))"
  for i in 1 2 3; do git pull -q --rebase && git push -q && return 0; sleep 20; done
  return 1
}

while :; do
  now=$(date +%s)
  [ $((now - start)) -ge "$RUN_SECONDS" ] && break
  if [ "$now" -ge "$next_upkeep" ]; then
    upkeep && upkeeps=$((upkeeps+1)) || fails=$((fails+1))
    next_upkeep=$((now + UPKEEP_EVERY))
  fi
  now=$(date +%s)
  if [ "$now" -ge "$next_translate" ] && [ $((now - start + TRANSLATE_MINUTES*60 + 120)) -lt "$RUN_SECONDS" ]; then
    translate && translations=$((translations+1)) || fails=$((fails+1))
    next_translate=$((now + TRANSLATE_EVERY))
  fi
  sleep 30
done
echo "::notice title=site clock::upkeep $upkeeps, translation $translations, failures $fails"
# Many failures in one run: stop the chain (the scheduled restarts pick it up later).
[ "$fails" -le 10 ]
