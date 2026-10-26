#!/usr/bin/env bash
# Run ON Thor (or via ssh) to create ~/kal layout and install the package.
set -euo pipefail

KAL_ROOT="${KAL_ROOT:-$HOME/kal}"
REPO_SRC="${REPO_SRC:-$KAL_ROOT/src/kernel-agent-loop}"
export PATH="/usr/local/cuda/bin:${PATH}"

mkdir -p "$KAL_ROOT"/{jobs,inbox,src}
mkdir -p "$KAL_ROOT/trusted"

if [[ ! -d "$REPO_SRC/.git" && ! -f "$REPO_SRC/pyproject.toml" ]]; then
  echo "ERROR: repo not found at $REPO_SRC — rsync/checkout the tree first" >&2
  exit 1
fi

# Trusted tree: clean copy without .git noise (ops + package for ops_root).
rsync -a --delete \
  --exclude '.git' --exclude '.venv' --exclude '__pycache__' --exclude '.kal' \
  --exclude '.pytest_cache' --exclude '*.egg-info' \
  "$REPO_SRC/" "$KAL_ROOT/trusted/"

cp -f "$REPO_SRC/deploy/thor/evald.config.json" "$KAL_ROOT/evald.config.json"
# Hidden shapes live only on Thor; seed from the example once, then edit in place.
if [[ ! -f "$KAL_ROOT/hidden_shapes.json" ]]; then
  cp "$REPO_SRC/deploy/thor/hidden_shapes.example.json" "$KAL_ROOT/hidden_shapes.json"
  echo "WARN: seeded $KAL_ROOT/hidden_shapes.json from example — replace with private shapes" >&2
fi

python3 -m venv "$KAL_ROOT/venv"
# shellcheck disable=SC1091
source "$KAL_ROOT/venv/bin/activate"
pip install -U pip setuptools wheel
pip install -e "$REPO_SRC[cuda,test]"

# Lock clocks for subsequent eval sessions (best-effort).
sudo -n nvpmodel -m 0 || true
sudo -n jetson_clocks || true

python - <<'PY'
from kernel_agent_loop.evald.env_guard import read_gpu_env
from kernel_agent_loop.evald.thor_env import prepare_thor_session
print("prepare:", prepare_thor_session())
print("env:", read_gpu_env())
PY

echo "bootstrap ok: config=$KAL_ROOT/evald.config.json venv=$KAL_ROOT/venv"
