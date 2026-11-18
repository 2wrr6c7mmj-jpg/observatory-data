#!/usr/bin/env bash
# Pushes the commits on HEAD to origin main, putting them on top of the newest main first.
#
#   bash tools/push_retry.sh [--tries N] [--sleep SECONDS] [--reset-on-fail] [--title NAME]
#
# Every workflow commits only its own directories, so the rebase normally goes through. When it does
# not (a conflict, or main moved again between fetch and push), the rebase is aborted (never left half
# done, so the next try and the next git command start from a clean state), main is fetched again and
# the rebase is tried again after SECONDS. After N failed tries it exits 1; with --reset-on-fail the
# local commits are dropped too (HEAD = origin/main), for a long run that makes them again later
# (tools/site_clock.sh).
#
# Authentication: with PUSH_TOKEN set (checkout with persist-credentials: false), the token is given to
# git for fetch and push only, as a header on the command line of those two commands; it is not
# written to .git/config and not printed. Without PUSH_TOKEN, git uses what the checkout left.
set -u
tries=3
pause=${PUSH_RETRY_SLEEP:-20}
reset_on_fail=false
title=publish
while [ $# -gt 0 ]; do
  case "$1" in
    --tries) tries=$2; shift 2 ;;
    --sleep) pause=$2; shift 2 ;;
    --reset-on-fail) reset_on_fail=true; shift ;;
    --title) title=$2; shift 2 ;;
    *) echo "push_retry.sh: unknown argument $1" >&2; exit 2 ;;
  esac
done

auth=()
if [ -n "${PUSH_TOKEN:-}" ]; then
  basic=$(printf 'x-access-token:%s' "$PUSH_TOKEN" | base64 | tr -d '\n')
  [ -n "${GITHUB_ACTIONS:-}" ] && echo "::add-mask::$basic"
  auth=(-c "http.https://github.com/.extraheader=AUTHORIZATION: basic $basic")
fi

rebasing() {
  [ -d "$(git rev-parse --git-path rebase-merge)" ] || [ -d "$(git rev-parse --git-path rebase-apply)" ]
}

# a rebase left over from an earlier, interrupted publish
if rebasing; then git rebase --abort 2>/dev/null || true; fi

for i in $(seq 1 "$tries"); do
  [ "$i" = 1 ] || sleep "$pause"
  if ! git "${auth[@]}" fetch -q origin main; then
    echo "::warning title=$title::fetch に失敗（$i/$tries 回目）"; continue
  fi
  if ! git rebase -q origin/main >/dev/null 2>&1; then
    why=$(git diff --name-only --diff-filter=U | head -5 | tr '\n' ' ')
    [ -n "$why" ] && why="衝突: $why" || why="変更が残っている: $(git status --porcelain --untracked-files=no | head -5 | tr '\n' ' ')"
    if rebasing; then git rebase --abort 2>/dev/null || true; fi
    echo "::warning title=$title::main への rebase に失敗（$i/$tries 回目。中止して取り直す）$why"
    continue
  fi
  if [ -z "$(git rev-list origin/main..HEAD)" ]; then echo "No changes"; exit 0; fi
  if git "${auth[@]}" push -q origin HEAD:main; then exit 0; fi
  echo "::warning title=$title::push に失敗（$i/$tries 回目）"
done
echo "::error title=$title::$tries 回試みたが push できなかった"
if [ "$reset_on_fail" = true ] && git rev-parse -q --verify origin/main >/dev/null; then
  git reset -q --hard origin/main
fi
exit 1
