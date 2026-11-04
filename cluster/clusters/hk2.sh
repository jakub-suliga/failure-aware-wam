# shellcheck shell=bash
# HoreKa 2 (KIT NHR), SLURM ClusterName `hk2`. Sourced by scripts/cluster_env.sh, never executed.
# Docs: https://docs.nhr.kit.edu/clusters/horeka-2/
#
# GPU partitions (all 2-day max walltime):
#   gpu-h100      4x H100-94GB/node,  128 cores, 768 GiB  <- default
#   gpu-h200      4x H200-141GB/node, 192 cores, 768 GiB  -- more headroom than gpu-h100
#   gpu-h200-8    8x H200-141GB, a single node
#   gpu-b200      4x B200-186GB/node (Grace CPUs)
#   dev-gpu-h100  ONE 4x H100 node, 1h max -- probes run one after another, not in parallel
# CPU: cpu (3-day max), dev-cpu (1h max).

GPU_PARTITION=${GPU_PARTITION:-gpu-h100}
DEV_GPU_PARTITION=${DEV_GPU_PARTITION:-dev-gpu-h100}
CPU_PARTITION=${CPU_PARTITION:-cpu}
