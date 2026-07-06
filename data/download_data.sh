#!/usr/bin/env bash
# Fetch the UCI dataset. Data-only concern; env setup lives in ../setup.sh.
set -euo pipefail
cd "$(dirname "$0")"                 # -> data/
ROOT="$(cd .. && pwd)"              # repo root (where .venv lives)
mkdir -p raw

URL="https://archive.ics.uci.edu/static/public/697/predict+students+dropout+and+academic+success.zip"
echo "Downloading from UCI..."
curl -sL "$URL" -o raw/dropout.zip
unzip -o raw/dropout.zip -d raw/ >/dev/null
rm -f raw/dropout.zip
[ -f raw/data.csv ] && mv -f raw/data.csv raw/dropout.csv
echo "Done -> data/raw/dropout.csv"

# Verification — prefers a caller-provided python ($SETUP_PYBIN), then the
# project venv, then system python3. Never fatal.
PYBIN="${SETUP_PYBIN:-python3}"
[ -x "$ROOT/.venv/bin/python" ] && [ "$PYBIN" = "python3" ] && PYBIN="$ROOT/.venv/bin/python"
if "$PYBIN" -c "import pandas" 2>/dev/null; then
  "$PYBIN" - <<'PY'
import pandas as pd
df = pd.read_csv('raw/dropout.csv', sep=';')
print(f"verified: rows={len(df)}, cols={df.shape[1]}")
PY
else
  echo "(skipped pandas check — run ./setup.sh and activate the venv to enable it)"
fi
