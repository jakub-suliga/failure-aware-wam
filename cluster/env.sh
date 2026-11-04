# shellcheck shell=bash
# Sourced by every cluster script. Paths come from the repo location; interpreters and partitions can be overridden
# by exporting them first (PY_LB, PY_RT, PY_VLLM, PARTITION, ...).
#   PY_LB    python of the `lingbot` env (server, training)    default: $CONDA_ENVS/lingbot/bin/python
#   PY_RT    python of the `robotwin` env (simulator client)   default: $CONDA_ENVS/robotwin/bin/python
#   PY_VLLM  python of an env with vllm (only for VLM methods) default: $CONDA_ENVS/vllm/bin/python
FAWAM_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export FAWAM_ROOT
CONDA_ENVS=${CONDA_ENVS:-$HOME/miniconda3/envs}
export PY_LB=${PY_LB:-$CONDA_ENVS/lingbot/bin/python}
export PY_RT=${PY_RT:-$CONDA_ENVS/robotwin/bin/python}
export PY_VLLM=${PY_VLLM:-$CONDA_ENVS/vllm/bin/python}
export PYTHONPATH=$FAWAM_ROOT${PYTHONPATH:+:$PYTHONPATH}
_cache=$FAWAM_ROOT/.cache
export HF_HOME=${HF_HOME:-$_cache/huggingface}
export HF_TOKEN_PATH=${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$_cache/torch_extensions}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$_cache/triton}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$_cache/inductor}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
CLUSTER=${CLUSTER:-$(scontrol show config 2>/dev/null | awk '$1 == "ClusterName" {print $3}')}
_profile=$FAWAM_ROOT/cluster/clusters/${CLUSTER}.sh
if [ -n "$CLUSTER" ] && [ -f "$_profile" ]; then
    # shellcheck source=/dev/null
    source "$_profile"
fi
unset _profile _cache
