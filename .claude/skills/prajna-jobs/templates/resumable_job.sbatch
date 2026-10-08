#!/bin/bash
# Resumable template — for the longest single run in this project (GMO-DETR, 200 epochs).
# Adds --requeue and a SIGUSR1 trap so a walltime cutoff (or preemption) resumes training from
# last.pt instead of losing the run. Only worth the extra complexity for runs long enough that
# a single --time estimate is genuinely risky; for anything under ~2h, gpu_job.sbatch is simpler.

#SBATCH --job-name=pcb_<TAG>
#SBATCH --partition=l40
#SBATCH --qos=l40                       # INVARIANT: must equal --partition
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00                 # EDIT: your best estimate — --requeue covers the miss
#SBATCH --requeue
#SBATCH --signal=B:USR1@300             # SIGUSR1 300s before the time limit hits
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

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

# Fixed run dir (not per-job-id) so a requeued job finds the same last.pt as its predecessor.
RUNDIR="$SCRATCH_ROOT/jobs/${SLURM_JOB_NAME}_persistent"
mkdir -p "$RUNDIR"
cd "$RUNDIR"
ln -sf "$PROJECT_ROOT"/offline/weights/*.pt .

# On SIGUSR1, just let srun's python process receive it and exit cleanly; Ultralytics writes
# last.pt continuously during training, so nothing extra is needed here beyond not killing the
# job outright. This trap exists mainly as a place to add a graceful-shutdown hook if the
# ported script ever needs one.
trap 'echo "SIGUSR1 received — approaching time limit, requeue will resume from last.pt"' USR1

nvidia-smi

# The ported script should check for an existing $RUNDIR/*/weights/last.pt and pass
# resume=True to Ultralytics' .train() when present — see pcb-port-to-batch/SKILL.md.
srun python "$PROJECT_ROOT/code/<SCRIPT>.py"
