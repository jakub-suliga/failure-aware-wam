#!/bin/bash
# Unpack the RoboTwin background textures (needed for demo_randomized) and run the pipeline on the hard setting.
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT/RoboTwin/assets"
if [ -f background_texture.zip ]; then
    echo "[$(date)] unzip textures"
    unzip -q -o background_texture.zip && rm background_texture.zip || { echo "unzip failed"; exit 1; }
fi
CONFIG=demo_randomized SAVE_ROOT="$ROOT/results/eval_hard" "$ROOT/scripts/run_all.sh"
