#!/bin/bash
#SBATCH --job-name=pcb_rtdetr_smoke
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err

set -uo pipefail

export PROJECT_ROOT=/home/agipml/samarth.sirsat/DDP/pcb
export SCRATCH_ROOT="$PROJECT_ROOT/scratch"          # no writable /scratch — SITE_AMENDMENTS A10
export PCB_EPOCHS=2                                  # SMOKE TEST ONLY
export PCB_BATCH=8

# Compute nodes have NO egress (SITE_AMENDMENTS A2-CORRECTED) — everything must resolve locally.
export YOLO_CONFIG_DIR="$PROJECT_ROOT/offline/ultralytics_cfg"
export MPLCONFIGDIR="$SCRATCH_ROOT/mplcache"
export TORCH_HOME="$PROJECT_ROOT/offline/torch"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
mkdir -p "$MPLCONFIGDIR" "$TORCH_HOME" "$YOLO_CONFIG_DIR"

source /lustre-flash/apps/spack/share/spack/setup-env.sh
spack load anaconda3
source "$HOME/envs/pcb/bin/activate"

# Run from a dir holding the staged .pt files so Ultralytics' internal bare-name lookups
# (notably yolo11n.pt for the AMP allclose check) resolve without a download.
RUNDIR="$SCRATCH_ROOT/jobs/$SLURM_JOB_ID"
mkdir -p "$RUNDIR"; cd "$RUNDIR"
ln -sf "$PROJECT_ROOT"/offline/weights/*.pt .
ln -sf "$PROJECT_ROOT"/hpc_env.py .

echo "=== node=$(hostname) job=$SLURM_JOB_ID cwd=$PWD epochs=$PCB_EPOCHS ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

srun python "$PROJECT_ROOT/rtdetr.py"
echo "=== exit=$? ==="
