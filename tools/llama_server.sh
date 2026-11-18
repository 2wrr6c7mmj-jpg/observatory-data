#!/usr/bin/env bash
# Starts the local model server for translate.yml, exercises.yml and site-clock.yml, from pinned and
# verified parts only, and waits until it answers on http://127.0.0.1:8080.
#
#   bash tools/llama_server.sh [extra llama-server arguments, e.g. -np 1]
#
# - llama.cpp: the official CPU server image (ghcr.io/ggml-org/llama.cpp, tag server-b12464, built
#   2026-11-17 from commit d3c3e242c16b8cbf51400fad410cd69b22d64f35), pulled by its digest, so docker
#   checks every byte. It runs in a container that sees only the model directory (read-only): no
#   checkout, no .git/config, no token.
# - Model: unsloth/Qwen3-8B-GGUF (Apache-2.0) at a fixed commit, the file checked against its SHA-256
#   (the LFS oid on Hugging Face) after every download and every cache restore.
# To move to newer versions, change the values below in one commit (and the cache key in the
# workflows, which carries MODEL_REV).
#
# The server log goes to $RUNNER_TEMP/server.log (LLAMA_LOG), as before.
set -euo pipefail
IMAGE=ghcr.io/ggml-org/llama.cpp@sha256:5b43086e3ba7ea5794d72904654f6cd41224a9b1c6ac0398f4940ee380d30f23
MODEL_REPO=unsloth/Qwen3-8B-GGUF
MODEL_REV=a6adef130ffb23ddaf1a62fec9dced968c9bc482
MODEL_FILE=Qwen3-8B-Q4_K_M.gguf
MODEL_SHA256=120307ba529eb2439d6c430d94104dabd578497bc7bfe7e322b5d9933b449bd4

cache=${LLAMA_CACHE:-$HOME/.cache/llama}
dir="$cache/$MODEL_REV"
path="$dir/$MODEL_FILE"
log=${LLAMA_LOG:-${RUNNER_TEMP:-/tmp}/server.log}
mkdir -p "$dir"

verified() { [ -f "$path" ] && echo "$MODEL_SHA256  $path" | sha256sum -c --status; }

if ! verified; then
  [ -f "$path" ] && echo "::warning title=llama::cached model does not match its SHA-256, downloading again"
  rm -f "$path" "$path.part"
  curl -fsSL --retry 3 --retry-delay 10 -o "$path.part" \
    "https://huggingface.co/$MODEL_REPO/resolve/$MODEL_REV/$MODEL_FILE"
  mv "$path.part" "$path"
  verified || { rm -f "$path"; echo "::error title=llama::$MODEL_FILE does not match the pinned SHA-256"; exit 1; }
fi
echo "model $MODEL_REPO@$MODEL_REV/$MODEL_FILE (sha256 verified)"

docker pull -q "$IMAGE"
docker rm -f llama >/dev/null 2>&1 || true
docker run -d --name llama --cap-drop ALL --security-opt no-new-privileges \
  -p 127.0.0.1:8080:8080 -v "$dir:/models:ro" "$IMAGE" \
  -m "/models/$MODEL_FILE" --jinja -c 8192 -t "$(nproc)" --host 0.0.0.0 --port 8080 "$@" >/dev/null
nohup docker logs -f llama > "$log" 2>&1 &

for i in $(seq 1 120); do
  curl -sf http://127.0.0.1:8080/health >/dev/null && { echo "ready after $((i*5)) s"; exit 0; }
  docker ps -q --filter name=^llama$ --filter status=running | grep -q . || break
  sleep 5
done
tail -40 "$log"; echo "::error title=llama::model server did not start"; exit 1
