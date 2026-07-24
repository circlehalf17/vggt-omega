#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CHECKPOINT_DIR="${REPO_ROOT}/checkpoints/tapnet"
CHECKPOINT_PATH="${CHECKPOINT_DIR}/bootstapir_checkpoint_v2.pt"

mkdir -p "${CHECKPOINT_DIR}"

python -m pip install "tapnet[torch] @ git+https://github.com/google-deepmind/tapnet.git"

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  if command -v wget >/dev/null 2>&1; then
    wget -O "${CHECKPOINT_PATH}" "https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.pt"
  else
    curl -L -o "${CHECKPOINT_PATH}" "https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.pt"
  fi
fi

python - <<'PY'
from tapnet.torch import tapir_model
print("BootsTAPIR import OK:", tapir_model.TAPIR)
PY

echo "Checkpoint: ${CHECKPOINT_PATH}"
