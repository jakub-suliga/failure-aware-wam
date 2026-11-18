#!/bin/bash
# scrontab entry: rebuild the variant report (fawam/report.py; incremental, one cached record per episode).
# Output: results/report -> ~/dev/fawam-report (index.html, data.js, media/), the folder the artifact is published from.
source "$(dirname "$0")/env.sh"
cd "$FAWAM_ROOT"
"$PY_RT" -m fawam.report >> pipeline/report.log 2>&1
