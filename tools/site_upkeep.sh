#!/usr/bin/env bash
# One round of the Observatory site's background upkeep: calls /api/maintenance (one bounded step per
# request, Workers Free CPU limit) until it reports idle or STEPS requests were made. Prints the tasks
# done. Exit 0: done; 1: the site did not answer (3 failed requests); 3: the site refused the token
# (401/403: the secret UPKEEP_TOKEN does not match the site's value).
#
# Used by tools/site_clock.sh (every 3 h, 60 steps, inside the long Site clock run) and by site-upkeep.yml
# (the fallback while no Site clock run is going).
#
# The token (the repository secret UPKEEP_TOKEN, the same value as on the site) is read from the file
# UPKEEP_TOKEN_FILE (site-clock.yml), or else from UPKEEP_TOKEN (site-upkeep.yml), and sent as the
# x-upkeep-token header from a header file, so that it is on no command line; never printed. Without a
# token the requests are made as before. UPKEEP_ANNOTATE=1 prints the summary as an annotation.
set -u
SITE=${SITE:-https://observatory-watch.pages.dev}
STEPS=${STEPS:-100}
token=${UPKEEP_TOKEN:-}
if [ -n "${UPKEEP_TOKEN_FILE:-}" ] && [ -s "$UPKEEP_TOKEN_FILE" ]; then token=$(cat -- "$UPKEEP_TOKEN_FILE"); fi
hdr=()
if [ -n "$token" ]; then
  hfile=$(mktemp)
  trap 'rm -f "$hfile"' EXIT
  printf 'x-upkeep-token: %s\n' "$token" > "$hfile"
  hdr=(-H "@$hfile")
fi
unset token
tasks="" bad=0
for _ in $(seq 1 "$STEPS"); do
  body=$(curl -sS -m 25 -w '\n%{http_code}' -H 'Accept: application/json' "${hdr[@]}" "$SITE/api/maintenance") || { bad=$((bad+1)); [ $bad -ge 3 ] && break; sleep 5; continue; }
  code=${body##*$'\n'}; body=${body%$'\n'*}
  case "$code" in
    401|403) echo "::error title=upkeep::/api/maintenance が $code を返した（Secret UPKEEP_TOKEN がサイト側の値と合っているか確かめる）"; exit 3 ;;
  esac
  task=$(printf '%s' "$body" | sed -n 's/.*"task":"\([^"]*\)".*/\1/p')
  tasks="$tasks ${task:-?}"
  [ "$task" = idle ] && break
  sleep 2
done
# shellcheck disable=SC2086  # word splitting of the task list is intended
echo "${UPKEEP_ANNOTATE:+::notice title=upkeep::}upkeep $(date -u +%H:%MZ): $(echo $tasks | tr ' ' '\n' | sed 's/:.*//' | sort | uniq -c | tr '\n' ' ')"
[ $bad -lt 3 ]
