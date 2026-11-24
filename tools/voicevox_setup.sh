#!/usr/bin/env bash
# Fetches the VOICEVOX parts for briefing/audio.py from pinned URLs and checks every file against its SHA-256
# before it is used; installs the voicevox_core wheel (no dependencies, no index) into the current Python.
#
#   bash tools/voicevox_setup.sh DIR      # DIR gets onnxruntime/, dict/, vvm/ (what briefing --vv-root takes)
#
# - voicevox_core 0.17.1 (Python binding, GitHub release; not on PyPI), manylinux_2_34 x86_64, cp310-abi3
# - voicevox_onnxruntime 1.23.2 (the ONNX Runtime build VOICEVOX ships)
# - Open JTalk dictionary 1.11 (utf-8)
# - ONE voice model only: the file named in briefing/voice.json, from voicevox_vvm at a fixed commit (15.vvm = もち子さん,
#   style 20). Changing the voice means changing VVM_FILE and VVM_SHA256 here together with voice.json.
# Terms of use of the voice (credit text etc.) are in briefing/voice.json and in voicevox_vvm's README/TERMS.txt.
set -euo pipefail
DIR=${1:?usage: voicevox_setup.sh DIR}
CORE_URL=https://github.com/VOICEVOX/voicevox_core/releases/download/0.17.1/voicevox_core-0.17.1-cp310-abi3-manylinux_2_34_x86_64.whl
CORE_SHA256=82f50527485c938381fb42c5959b0ae0decb8f6d6885ca8cf408ca0762f47013
ORT_URL=https://github.com/VOICEVOX/onnxruntime-builder/releases/download/voicevox_onnxruntime-1.23.2/voicevox_onnxruntime-linux-x64-1.23.2.tgz
ORT_SHA256=0860cfbd5a80201b1bf95cae13ca2db8dc01f32f5674cc8193146047a9212fdb
DICT_URL=https://github.com/r9y9/open_jtalk/releases/download/v1.11.1/open_jtalk_dic_utf_8-1.11.tar.gz
DICT_SHA256=fe6ba0e43542cef98339abdffd903e062008ea170b04e7e2a35da805902f382a
VVM_COMMIT=2c47eb0db00acf5a6dbcc3a6e887d726f9b04d5d   # voicevox_vvm 0.17.1
VVM_FILE=15.vvm
VVM_SHA256=c29b0c0f8e332fcedc2a5de4b4426feb75fc9f587c314da99af8dbc964287221
VVM_URL=https://raw.githubusercontent.com/VOICEVOX/voicevox_vvm/$VVM_COMMIT/vvms/$VVM_FILE

mkdir -p "$DIR/dl" "$DIR/onnxruntime" "$DIR/dict" "$DIR/vvm"
get() {  # url sha256 path
  [ -f "$3" ] && echo "$2  $3" | sha256sum -c --status && return 0
  curl -fsSL --retry 3 --retry-delay 10 -o "$3.part" "$1"
  mv "$3.part" "$3"
  echo "$2  $3" | sha256sum -c --status || { rm -f "$3"; echo "::error title=voicevox::$(basename "$3") does not match the pinned SHA-256"; exit 1; }
}
core="$DIR/dl/$(basename "$CORE_URL")"
get "$CORE_URL" "$CORE_SHA256" "$core"
get "$ORT_URL" "$ORT_SHA256" "$DIR/dl/ort.tgz"
get "$DICT_URL" "$DICT_SHA256" "$DIR/dl/dict.tar.gz"
get "$VVM_URL" "$VVM_SHA256" "$DIR/vvm/$VVM_FILE"
tar -xzf "$DIR/dl/ort.tgz" -C "$DIR/onnxruntime"
tar -xzf "$DIR/dl/dict.tar.gz" -C "$DIR/dict"
python -m pip install --disable-pip-version-check -q --no-deps --no-index "$core"
echo "voicevox parts ready in $DIR (sha256 verified)"
