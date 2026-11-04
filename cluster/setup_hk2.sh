#!/bin/bash
# One-time setup of failure-aware-wam on HoreKa 2 (or another SLURM cluster with H100/H200/B200). Run on a LOGIN node
# from the repo root, step by step (each step is idempotent):
#
#   cluster/setup_hk2.sh repos        # lingbot-va + RoboTwin at the pinned commits, local patch, task configs
#   cluster/setup_hk2.sh envs         # conda envs `lingbot` and `robotwin` (pip, from cluster/envs/*_pip_freeze.txt)
#   cluster/setup_hk2.sh curobo       # build cuRobo for the cluster GPUs (needs a GPU node: run via srun, see below)
#   cluster/setup_hk2.sh assets       # LingBot-VA RoboTwin checkpoint (23 GB) + RoboTwin assets
#   cluster/setup_hk2.sh check        # on a GPU node: Vulkan rendering + a 1-episode smoke eval
#
#   srun -p dev-gpu-h100 --gres=gpu:1 -t 00:50:00 --pty cluster/setup_hk2.sh curobo
#   srun -p dev-gpu-h100 --gres=gpu:1 -t 00:50:00 --pty cluster/setup_hk2.sh check
#
# Put the repo in a workspace (ws_allocate), not in $HOME: checkpoints, envs and results are large.
# NOT TESTED ON THE CLUSTER YET (written 2026-11-03 against the local RTX 5090 setup). Known differences:
#   * H100 = sm_90 (B200 = sm_100): cuRobo must be rebuilt with TORCH_CUDA_ARCH_LIST="9.0;10.0"; the torch cu128/cu129
#     wheels run on both.
#   * SAPIEN renders with Vulkan. Compute nodes need libvulkan + the NVIDIA ICD; if `check` fails with
#     "cannot create buffer"/"vk" errors, run the client inside an apptainer image with Vulkan (as the tactile project
#     does for IsaacGym) and keep the server on the host. The simulator must only see ONE GPU (one GPU per job).
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)
CONDA=${CONDA:-$HOME/miniconda3}
step=${1:?usage: setup_hk2.sh repos|envs|curobo|assets|check}

case "$step" in
repos)
    ./setup.sh
    ;;
envs)
    source "$CONDA/etc/profile.d/conda.sh"
    if ! conda env list | grep -q "^lingbot "; then
        conda create -y -n lingbot python=3.10.16
        conda run -n lingbot pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128
        grep -vE "^(torch|torchvision|torchaudio|nvidia-|triton)==" cluster/envs/lingbot_pip_freeze.txt > /tmp/lb_req.txt
        conda run -n lingbot pip install -r /tmp/lb_req.txt
    fi
    if ! conda env list | grep -q "^robotwin "; then
        conda create -y -n robotwin python=3.10
        conda run -n robotwin pip install torch==2.8.0 torchvision --index-url https://download.pytorch.org/whl/cu129
        grep -vE "^(torch|torchvision|torchaudio|nvidia-|triton|nvidia_curobo|curobo)==" \
            cluster/envs/robotwin_pip_freeze.txt > /tmp/rt_req.txt
        conda run -n robotwin pip install -r /tmp/rt_req.txt
        conda install -y -n robotwin -c nvidia cuda-nvcc=12.9 cuda-cudart-dev=12.9  # nvcc for the cuRobo build
    fi
    ;;
curobo)
    source "$CONDA/etc/profile.d/conda.sh"
    conda activate robotwin
    cd RoboTwin/envs
    [ -d curobo ] || git clone https://github.com/NVlabs/curobo.git && (cd curobo && git checkout v0.7.8 2>/dev/null || true)
    cd curobo
    ENV=$CONDA_PREFIX
    export CPATH=$ENV/targets/x86_64-linux/include${CPATH:+:$CPATH}
    export LIBRARY_PATH=$ENV/targets/x86_64-linux/lib${LIBRARY_PATH:+:$LIBRARY_PATH}
    export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-"9.0;10.0"}
    pip install -e . --no-build-isolation
    ;;
assets)
    source "$CONDA/etc/profile.d/conda.sh"
    conda activate lingbot
    huggingface-cli download robbyant/lingbot-va-posttrain-robotwin \
        --local-dir lingbot-va/checkpoints/lingbot-va-posttrain-robotwin
    # the server needs attn_mode torch (inference), training loads with flex itself
    sed -i 's/"attn_mode": *"flex"/"attn_mode": "torch"/' lingbot-va/checkpoints/lingbot-va-posttrain-robotwin/transformer/config.json
    conda activate robotwin
    (cd RoboTwin && bash script/_download_assets.sh)
    ;;
check)
    source cluster/env.sh
    (cd RoboTwin && "$PY_RT" script/test_render.py) && echo "render ok"
    "$PY_RT" -m fawam.probe gap-stats results 2>/dev/null || echo "(no local results for gap stats; artifacts/gap_stats.json from git is used)"
    TASK=pick_diverse_bottles CONFIG=table_h_z00 METHOD=base NUM=1 N_PAR=1 OUT=$ROOT/results/_cluster_smoke \
        bash cluster/eval_job.sh
    ;;
*)
    echo "unknown step $step"; exit 1 ;;
esac
