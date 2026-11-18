#!/usr/bin/env bash
# One round of the Observatory site's background upkeep: calls /api/maintenance (one bounded step per
# request, Workers Free CPU limit) until it reports idle or STEPS requests were made. Prints the tasks
# done and exits 1 when the site did not answer (3 failed requests).
#
# Used by tools/site_clock.sh (every 3 h, 60 steps, inside the long Site clock run) and by site-upkeep.yml
# (the fallback while no Site clock run is going).
#
# UPKEEP_TOKEN (the repository secret of the same name, the same value as on the site): when set it is
# sent as the x-upkeep-token header, from a file so that it is not on a command line; never printed.
# Without it the requests are made as before. UPKEEP_ANNOTATE=1 prints the summary as an annotation.
set -u
SITE=${SITE:-https://observatory-watch.pages.dev}
STEPS=${STEPS:-100}
hdr=()
if [ -n "${UPKEEP_TOKEN:-}" ]; then
  hfile=$(mktemp)
  trap 'rm -f "$hfile"' EXIT
  chmod 600 "$hfile"
  printf 'x-upkeep-token: %s\n' "$UPKEEP_TOKEN" > "$hfile"
  hdr=(-H "@$hfile")
fi
tasks="" bad=0
for _ in $(seq 1 "$STEPS"); do
  body=$(curl -sS -m 25 -w '\n%{http_code}' -H 'Accept: application/json' "${hdr[@]}" "$SITE/api/maintenance") || { bad=$((bad+1)); [ $bad -ge 3 ] && break; sleep 5; continue; }
  code=${body##*$'\n'}; body=${body%$'\n'*}
  case "$code" in
    401|403) echo "::error title=upkeep::/api/maintenance が $code を返した（Secret UPKEEP_TOKEN がサイト側の値と合っているか確かめる）"; exit 1 ;;
  esac
  task=$(printf '%s' "$body" | sed -n 's/.*"task":"\([^"]*\)".*/\1/p')
  tasks="$tasks ${task:-?}"
  [ "$task" = idle ] && break
  sleep 2
done
# shellcheck disable=SC2086  # word splitting of the task list is intended
echo "${UPKEEP_ANNOTATE:+::notice title=upkeep::}upkeep $(date -u +%H:%MZ): $(echo $tasks | tr ' ' '\n' | sed 's/:.*//' | sort | uniq -c | tr '\n' ' ')"
[ $bad -lt 3 ]
