#!/usr/bin/env bash
# Publishes docs/data/sat from sat-ships onto main (run from the repository root, on a checkout of main).
#
#   sat/publish.sh "<commit message>"
#
# chips/index.json is shared with ocean-monitor (it writes the "ocean" list, this run the ports' "sites"), and is
# one line of JSON: a plain `git pull --rebase` conflicts whenever ocean-monitor pushed in between, and the day's
# port data is lost. So the index is kept out of the data commit: the run's version is saved aside, and before each
# push the ports' part is copied onto the latest main's index (python -m sat.chips splice) in a commit of its own,
# which is dropped and made again if the push is refused. A failed rebase is aborted and tried again.
set -u
msg="$1"
idx=docs/data/sat/chips/index.json
saved="$(mktemp)"
has_idx=false
git add -A -- docs/data/sat
if ! git diff --cached --quiet -- "$idx" && git show ":$idx" > "$saved" 2>/dev/null; then has_idx=true; fi
git reset -q -- "$idx"
if git cat-file -e "HEAD:$idx" 2>/dev/null; then git checkout -q -- "$idx"; else rm -f -- "$idx"; fi
git diff --cached --quiet || git commit -q -m "$msg"
pushed=false
for i in 1 2 3 4; do
  [ "$i" = 1 ] || sleep "${PUBLISH_SLEEP:-20}"
  git pull -q --rebase origin main || { git rebase --abort 2>/dev/null; echo "::warning title=sat-ships::rebase に失敗（$i 回目）"; continue; }
  spliced=false
  if [ "$has_idx" = true ]; then
    python -m sat.chips splice --sat docs/data/sat --sites "$saved" || { echo "::error title=sat-ships::索引の写し直しに失敗"; exit 1; }
    git add -- "$idx"
    if ! git diff --cached --quiet; then git commit -q -m "データ更新: 衛星画像の索引（港） [sat-ships]"; spliced=true; fi
  fi
  if [ -z "$(git rev-list origin/main..HEAD)" ]; then echo "No changes"; pushed=true; break; fi
  if git push -q origin HEAD:main; then pushed=true; break; fi
  if [ "$spliced" = true ]; then git reset -q --hard HEAD~1; fi
done
rm -f -- "$saved"
[ "$pushed" = true ] || { echo "::error title=sat-ships::push できなかった"; exit 1; }
