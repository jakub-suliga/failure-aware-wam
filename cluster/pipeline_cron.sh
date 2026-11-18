#!/bin/bash
# scrontab entry (install with: scrontab cluster/pipeline.scrontab). One pipeline tick on a CPU node.
source "$(dirname "$0")/env.sh"
cd "$FAWAM_ROOT"
"$PY_RT" cluster/pipeline.py tick >> pipeline/cron.log 2>&1
