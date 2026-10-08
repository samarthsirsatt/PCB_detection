#!/bin/bash
# Base template for a single GPU training job. Copy this per script/experiment and edit the
# marked fields. See PRAJNA_CONFIG.md §1/§3 for the values to fill in, and
# ../references/partitions-qos.md for when to deviate from the defaults below.

#SBATCH --job-name=pcb_<TAG>            # EDIT: e.g. pcb_rtdetr_smoke
#SBATCH --partition=l40                 # DEFAULT — see partitions-qos.md; must equal --qos below
#SBATCH --qos=l40                       # INVARIANT: must equal --partition, not "normal" or "standard"
#SBATCH --gres=gpu:1                    # DEFAULT — single GPU; see partitions-qos.md before raising
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8               # EDIT if sinfo shows a different core/GPU ratio
#SBATCH --mem=64G                       # EDIT: raise on oom-kill (host RAM), not CUDA OOM (VRAM)
#SBATCH --time=04:00:00                 # EDIT: be honest — backfill rewards accurate estimates
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

set -euo pipefail

# --- fill from PRAJNA_CONFIG.md §1 ---
PROJECT_ROOT="${PROJECT_ROOT:?set PROJECT_ROOT, e.g. /home/<group>/<user>/pcb}"
SCRATCH_ROOT="${SCRATCH_ROOT:?set SCRATCH_ROOT, e.g. /scratch/<user>/pcb}"

source /lustre-flash/apps/spack/share/spack/setup-env.sh
spack load anaconda3                    # Python 3.11.5 — Spack here has NO `python`
                                        # package; see ../../SITE_AMENDMENTS.md A1
source "${PCB_VENV:-$HOME/envs/pcb}/bin/activate"   # venv lives outside PROJECT_ROOT

# Offline caches — keep Ultralytics/matplotlib/torch off the network and off $HOME dotfiles.
# See prajna-env/references/offline-staging.md for why each of these exists.
export YOLO_CONFIG_DIR="$PROJECT_ROOT/offline/ultralytics_cfg"
export MPLCONFIGDIR="$SCRATCH_ROOT/mplcache"
export TORCH_HOME="$PROJECT_ROOT/offline/torch"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

# Run from a per-job dir that already contains the pretrained .pt files, so any bare
# "yolo11s.pt" / "yolo11n.pt" / "rtdetr-l.pt" resolves locally instead of trying to download.
RUNDIR="$SCRATCH_ROOT/jobs/$SLURM_JOB_ID"
mkdir -p "$RUNDIR"
cd "$RUNDIR"
ln -sf "$PROJECT_ROOT"/offline/weights/*.pt .

nvidia-smi

srun python "$PROJECT_ROOT/code/<SCRIPT>.py"    # EDIT: the ported script to run
