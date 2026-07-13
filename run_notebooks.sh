#!/usr/bin/env bash
#
# I wrote this script because I kept getting the manual version wrong.
#
# Five notebooks have to run in a specific order, each on a clean kernel, and each one
# has to be saved afterwards or its outputs are lost. Doing that by hand in Jupyter,
# I managed to break it in three different ways.
#
# I ran them out of order once. Notebooks 04 and 05 read files that 03 and 04 write,
# so running 05 first does not throw an error. It quietly reads a stale model.joblib
# from an earlier run and produces numbers that look completely reasonable and are
# completely wrong. That is the worst kind of failure, because nothing tells you.
#
# I ran cells out of order inside a single notebook, which meant a cell passed against
# a variable left behind by an earlier experiment, and would have failed on a fresh
# start. Committing that is committing a notebook that does not actually run.
#
# And I committed all five with their outputs stripped, which meant anyone opening them
# on GitHub saw the code and the writing and no results at all. No tables, no charts,
# no SHAP plot. Half the project was invisible.
#
# One command now. Correct interpreter, correct order, fresh kernel each time, outputs
# written back into the files, and a check at the end that they actually landed.
#
# Usage:
#   bash run_notebooks.sh

set -euo pipefail

cd "$(dirname "$0")"
ROOT="$(pwd)"

# ---------------------------------------------------------------------------
# Pin the interpreter before anything else runs.
#
# This script used to call a bare `python` and a bare `jupyter`, which meant that if
# you ran it without activating .venv first, it executed against whatever interpreter
# happened to be on PATH. Best case, an import error. Worst case — and this is the
# whole reason the guard is here — it finds a *different* pandas or scikit-learn,
# runs all five notebooks without complaint, and writes out numbers that quietly
# disagree with the report.
#
# That is the same silent-wrong-answer failure this project spends three notebooks
# learning to hate. It should not be sitting in the script that reproduces them.
#
# Resolution order: an already-activated venv, then the project's own .venv, then
# hard failure. No silent fallback to the system interpreter.
# ---------------------------------------------------------------------------
if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
  PYBIN="$VIRTUAL_ENV/bin/python"
elif [ -x "$ROOT/.venv/bin/python" ]; then
  PYBIN="$ROOT/.venv/bin/python"
else
  echo "ERROR: no project environment found."
  echo
  echo "  Expected .venv/bin/python, or an already-activated virtualenv."
  echo "  Build it first:  ./setup.sh      (uses uv, pins Python 3.13)"
  echo
  echo "  I am refusing to fall back to the system interpreter. It may hold different"
  echo "  package versions than requirements.txt pins, and these notebooks would then"
  echo "  run to completion and produce numbers that disagree with the report, with no"
  echo "  error anywhere to tell you."
  exit 1
fi

echo "interpreter: $PYBIN"
"$PYBIN" - <<'PY'
# Fail loudly here, not four notebooks deep. Every one of these is imported by the
# notebooks, and a missing or mismatched one changes results rather than crashing.
import sys
missing = []
for mod in ("pandas", "numpy", "sklearn", "scipy", "matplotlib",
            "shap", "fairlearn", "xgboost", "joblib", "nbconvert", "ipykernel"):
    try:
        __import__(mod)
    except ImportError:
        missing.append(mod)
if missing:
    sys.exit(f"ERROR: this interpreter is missing {', '.join(missing)}.\n"
             f"       Run: uv pip install -r requirements.txt")

import pandas, numpy, sklearn, shap, fairlearn, xgboost
print(f"            python {sys.version.split()[0]}")
print(f"            pandas {pandas.__version__} | numpy {numpy.__version__} | "
      f"scikit-learn {sklearn.__version__}")
print(f"            shap {shap.__version__} | fairlearn {fairlearn.__version__} | "
      f"xgboost {xgboost.__version__}")

# The committed outputs came out of exactly these versions. If the environment has
# drifted off the pins, say so HERE.
#
# I learned this one the hard way. requirements.txt was briefly rewritten against
# versions from a different machine, which pinned five packages *behind* the env that
# had produced every number in the report. Installing it would have downgraded pandas,
# numpy, scikit-learn, scipy and matplotlib, and the notebooks would have re-run to
# completion, written slightly different numbers into the report, and raised nothing.
EXPECTED = {"pandas": "3.0.3", "numpy": "2.4.6", "scikit-learn": "1.9.0"}
actual = {"pandas": pandas.__version__, "numpy": numpy.__version__,
          "scikit-learn": sklearn.__version__}
drift = {k: (want, actual[k]) for k, want in EXPECTED.items() if actual[k] != want}
if drift:
    print()
    print("  NOTE: this environment has drifted from the pins behind the committed outputs.")
    for k, (want, got) in drift.items():
        print(f"        {k}: pinned {want}, installed {got}")
    print()
    print("        The notebooks will still run. But if the numbers move, the report needs")
    print("        re-checking, not just re-running. To align:")
    print("            uv pip install -r requirements.txt")
PY

if [ ! -f "data/raw/dropout.csv" ]; then
  echo "ERROR: data/raw/dropout.csv is missing. Run: bash data/download_data.sh"
  exit 1
fi

# The order is the whole reason this script exists. 03 writes the split and the
# preprocessor, 04 writes the model and the operating threshold, 05 reads both.
NOTEBOOKS=(
  "01_problem_framing"
  "02_data_understanding"
  "03_eda_feature_engineering"
  "04_modeling"
  "05_ethics_bias_audit"
)

mkdir -p reports/figures models data/processed

for nb in "${NOTEBOOKS[@]}"; do
  echo ""
  echo "=== running $nb"
  # "$PYBIN" -m nbconvert, never a bare `jupyter`. Invoking it as a module binds it
  # to the interpreter resolved above, so there is no PATH lookup that could hand the
  # work to a different environment than the one just verified.
  #
  # --inplace is the flag that matters. It writes the outputs back into the same file
  # instead of printing them and throwing them away.
  #
  # 2400 seconds because notebook 04 tunes an SVM with probability=True inside a grid
  # search, and notebook 05 fits an ExponentiatedGradient. Both are slow. The nbconvert
  # default is 30 seconds and would kill 04 before it finished its first model.
  "$PYBIN" -m nbconvert \
    --to notebook \
    --execute \
    --inplace \
    --ExecutePreprocessor.timeout=2400 \
    "notebooks/${nb}.ipynb"
done

echo ""
echo "=== checking the outputs actually saved"
"$PYBIN" - <<'PY'
import json, glob, os

for path in sorted(glob.glob("notebooks/*.ipynb")):
    nb = json.load(open(path))
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    filled = sum(1 for c in code if c.get("outputs"))
    charts = sum(1 for c in code for o in c.get("outputs", [])
                 if "image/png" in o.get("data", {}))
    size_kb = os.path.getsize(path) / 1024
    warn = "   <-- too small, outputs did not save" if size_kb < 25 else ""
    print(f"  {os.path.basename(path):34} {filled:2}/{len(code):2} cells with output"
          f" | {charts} charts | {size_kb:6.0f} KB{warn}")

print()
print("Size is the check I trust. Stripped, these files are 11 to 21 KB. With outputs")
print("they run to hundreds of KB, because the charts embed as base64 PNGs. Anything")
print("still tiny did not save, whatever the run said.")
PY

echo ""
echo "=== done. Commit with the outputs in:"
echo ""
echo "    git add -A"
echo "    git commit -m 'run all notebooks, save outputs'"
echo "    git push"
echo ""
echo "Then open the repo in an incognito window. If the charts render there, the grader"
echo "can see them. That render is the deliverable."
