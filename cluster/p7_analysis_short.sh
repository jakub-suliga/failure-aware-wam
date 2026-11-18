#!/bin/bash
# Short twin of the pipeline's fawam_p7_analysis job (OTHER=<its job id>): whichever starts first does the work.
# The analysis skips episodes that already have analysis.json, so a second run is harmless.
source cluster/env.sh
state() { squeue -h -j "$OTHER" -o %T 2>/dev/null; }
if [ "$(state)" = PENDING ]; then
  scancel "$OTHER" && echo "cancelled waiting twin $OTHER"
else
  while [ "$(state)" = RUNNING ]; do echo "twin $OTHER is running, waiting"; sleep 30; done
fi
exec $PY_RT -m fawam.failure_analysis results/sweeps/p7_meta_data/base/table_h_rand
