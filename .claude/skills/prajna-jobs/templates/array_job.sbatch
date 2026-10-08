#!/bin/bash
# Job-array template — for the 5-fold cross-validation (or any embarrassingly parallel sweep).
# Each array task gets its own $SLURM_ARRAY_TASK_ID, its own run dir, and runs independently,
# so all 5 folds run concurrently across nodes instead of sequentially in one job (roughly
# 10h wall -> ~2h wall for the CV study). See
# pcb-port-to-batch/references/job-decomposition.md for the fold-selection logic to add to the
# ported script.

#SBATCH --job-name=pcb_<TAG>_cv
#SBATCH --partition=l40
#SBATCH --qos=l40                       # INVARIANT: must equal --partition
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00                 # EDIT: per-fold time, not total across folds
#SBATCH --array=0-4                     # 5 folds; use %N suffix (e.g. 0-4%2) to cap concurrency
#SBATCH --output=%x.%A_%a.out
#SBATCH --error=%x.%A_%a.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?set PROJECT_ROOT}"
SCRATCH_ROOT="${SCRATCH_ROOT:?set SCRATCH_ROOT}"

source /lustre-flash/apps/spack/share/spack/setup-env.sh
spack load anaconda3                    # Python 3.11.5 — Spack here has NO `python`
                                        # package; see ../../SITE_AMENDMENTS.md A1
source "${PCB_VENV:-$HOME/envs/pcb}/bin/activate"   # venv lives outside PROJECT_ROOT

export YOLO_CONFIG_DIR="$PROJECT_ROOT/offline/ultralytics_cfg"
export MPLCONFIGDIR="$SCRATCH_ROOT/mplcache"
export TORCH_HOME="$PROJECT_ROOT/offline/torch"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

# FOLD is read by the ported script (e.g. via hpc_env or an env var it checks) to select which
# of the K folds this task trains. Keep each task's run dir separate so results don't collide.
export FOLD="$SLURM_ARRAY_TASK_ID"
RUNDIR="$SCRATCH_ROOT/jobs/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$RUNDIR"
cd "$RUNDIR"
ln -sf "$PROJECT_ROOT"/offline/weights/*.pt .

nvidia-smi

srun python "$PROJECT_ROOT/code/<SCRIPT>.py"
