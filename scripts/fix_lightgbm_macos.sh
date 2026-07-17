#!/usr/bin/env bash
# macOS: lightgbm wheels need an OpenMP runtime (libomp.dylib).
# Either `brew install libomp`, or run this script to reuse the libomp that
# scikit-learn's wheel already bundles inside the project venv. No system
# files are modified.
set -euo pipefail

VENV="${1:-.venv}"
SITE=$("$VENV/bin/python" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
LGB="$SITE/lightgbm/lib/lib_lightgbm.dylib"

if [ ! -f "$LGB" ]; then
  echo "lightgbm dylib not found at $LGB" >&2
  exit 1
fi

install_name_tool -add_rpath "@loader_path/../../sklearn/.dylibs" "$LGB" 2>/dev/null || true
codesign -f -s - "$LGB"
"$VENV/bin/python" -c "import lightgbm; print('lightgbm OK', lightgbm.__version__)"
