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

# The zip ships its table as data.csv. If UCI ever renames it, the old unconditional
# `[ -f raw/data.csv ] && mv ...` would quietly do nothing, dropout.csv would never
# appear, and the failure would surface two notebooks later as a confusing missing-file
# error a long way from its cause. Fail here instead, where the cause is obvious.
if [ -f raw/data.csv ]; then
  mv -f raw/data.csv raw/dropout.csv
elif [ ! -f raw/dropout.csv ]; then
  echo "ERROR: expected data.csv inside the UCI zip. Got:" >&2
  ls -1 raw/ | sed 's/^/  /' >&2
  echo "The archive layout has changed. Rename the table to raw/dropout.csv by hand," >&2
  echo "or update this script." >&2
  exit 1
fi

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
