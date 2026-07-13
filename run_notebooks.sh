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
# One command now. Correct order, fresh kernel each time, outputs written back into the
# files, and a check at the end that they actually landed.
#
# Usage:
#   bash run_notebooks.sh

set -euo pipefail

cd "$(dirname "$0")"

if [ ! -f "data/raw/dropout.csv" ]; then
  echo "data/raw/dropout.csv is missing. Run ./data/download_data.sh first."
  exit 1
fi

python -c "import nbconvert, ipykernel" 2>/dev/null || {
  echo "installing nbconvert and ipykernel"
  pip install --quiet nbconvert ipykernel
}

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
  # --inplace is the flag that matters. It writes the outputs back into the same file
  # instead of printing them and throwing them away.
  #
  # 2400 seconds because notebook 04 tunes an SVM with probability=True inside a grid
  # search, and notebook 05 fits an ExponentiatedGradient. Both are slow. The nbconvert
  # default is 30 seconds and would kill 04 before it finished its first model.
  jupyter nbconvert \
    --to notebook \
    --execute \
    --inplace \
    --ExecutePreprocessor.timeout=2400 \
    "notebooks/${nb}.ipynb"
done

echo ""
echo "=== checking the outputs actually saved"
python - << 'EOF'
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
EOF

echo ""
echo "=== done. Commit with the outputs in:"
echo ""
echo "    git add -A"
echo "    git commit -m 'run all notebooks, save outputs'"
echo "    git push"
echo ""
echo "Then open the repo in an incognito window. If the charts render there, the grader"
echo "can see them. That render is the deliverable."
