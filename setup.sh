#!/usr/bin/env bash
# One-time environment bootstrap.
#
# uv is the intended path. It reads .python-version, so the interpreter is pinned
# (3.13) the same way requirements.txt pins the packages. A venv + pip fallback is
# kept so a grader without uv can still build the project.
#
# Why the interpreter is pinned and not just the packages: every number in the
# report came out of Python 3.13 with the exact versions in requirements.txt.
# Building on a different interpreter can resolve different wheels, and the
# notebooks would then run to completion and produce slightly different results
# with nothing to tell you they had.
set -euo pipefail
cd "$(dirname "$0")"

PYVER="$(cat .python-version 2>/dev/null || echo 3.13)"

# True if $1 is a Python interpreter meeting the >= 3.11 floor.
py_ok() { "$1" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,11) else 1)' 2>/dev/null; }

if command -v uv >/dev/null 2>&1; then
  # uv picks up .python-version on its own, and downloads the interpreter if the
  # machine does not have it. Nothing else to arrange.
  echo "uv found. Building on Python $PYVER (pinned in .python-version)."
  uv venv
  uv pip install -r requirements.txt
else
  echo "uv not found — falling back to venv + pip."
  echo "(uv is the intended path: curl -LsSf https://astral.sh/uv/install.sh | sh)"

  # Prefer the pinned version, then anything at or above the floor.
  PYBIN=""
  for cand in "python${PYVER}" python3.13 python3.12 python3.11 python3 python; do
    if command -v "$cand" >/dev/null 2>&1 && py_ok "$cand"; then
      PYBIN="$(command -v "$cand")"
      break
    fi
  done

  [ -n "$PYBIN" ] || {
    echo "Error: need Python >= 3.11 in PATH, or install uv (link above)."
    exit 1
  }

  echo "Using: $PYBIN ($("$PYBIN" -V 2>&1))"
  if ! "$PYBIN" -c "import sys; raise SystemExit(0 if sys.version_info[:2] == tuple(int(x) for x in '${PYVER}'.split('.')) else 1)" 2>/dev/null; then
    echo
    echo "  NOTE: this is not Python $PYVER, which is what produced the committed results."
    echo "  The notebooks will still run. Numbers may differ in the last decimal or two."
    echo
  fi

  "$PYBIN" -m venv .venv
  ./.venv/bin/pip install --upgrade pip
  ./.venv/bin/pip install -r requirements.txt
fi

# Fetch the dataset, reusing the freshly built interpreter for its verification step.
echo
echo "Fetching dataset..."
SETUP_PYBIN="$(pwd)/.venv/bin/python" bash data/download_data.sh

echo
echo "Setup complete."
echo
echo "  source .venv/bin/activate     # once per shell"
echo "  bash run_notebooks.sh         # reproduce every result"
echo "  jupyter lab                   # or just read the notebooks, outputs are committed"
