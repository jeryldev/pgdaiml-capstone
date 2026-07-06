#!/usr/bin/env bash
# One-time environment bootstrap.
# Prefers uv (fast); falls back to venv + pip. Reuses an existing system
# Python (>= 3.11) when present; only provisions one if none is found.
set -euo pipefail
cd "$(dirname "$0")"

# True if $1 is a Python interpreter meeting the >= 3.11 floor.
py_ok() { "$1" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,11) else 1)' 2>/dev/null; }

# Find the first suitable system Python (may stay empty).
PYBIN=""
for cand in python3.13 python3.12 python3.11 python3 python; do
  if command -v "$cand" >/dev/null 2>&1 && py_ok "$cand"; then
    PYBIN="$(command -v "$cand")"
    break
  fi
done

if command -v uv >/dev/null 2>&1; then
  if [ -n "$PYBIN" ]; then
    echo "uv found. Using existing Python: $PYBIN ($("$PYBIN" -V 2>&1))"
    uv venv --python "$PYBIN"
  else
    echo "uv found. No system Python >= 3.11 — provisioning a uv-managed 3.12."
    uv venv --python 3.12
  fi
  uv pip install -r requirements.txt
else
  echo "uv not found — falling back to venv + pip."
  [ -n "$PYBIN" ] || {
    echo "Error: need Python >= 3.11 in PATH (or install uv: curl -LsSf https://astral.sh/uv/install.sh | sh)."
    exit 1
  }
  echo "Using existing Python: $PYBIN ($("$PYBIN" -V 2>&1))"
  "$PYBIN" -m venv .venv
  ./.venv/bin/pip install --upgrade pip
  ./.venv/bin/pip install -r requirements.txt
fi

# Fetch the dataset using the freshly built venv's Python for verification.
# The standalone data/download_data.sh still works on its own; this just reuses it.
echo
echo "Fetching dataset..."
SETUP_PYBIN="$(pwd)/.venv/bin/python" bash data/download_data.sh

echo
echo "Setup complete. Activate the environment once in your shell:"
echo "    source .venv/bin/activate"
echo "Then launch:  jupyter lab"
