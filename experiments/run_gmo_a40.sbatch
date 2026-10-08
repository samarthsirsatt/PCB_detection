#!/bin/bash
#SBATCH --job-name=gmoexp
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
set -uo pipefail
export PROJECT_ROOT=/home/agipml/samarth.sirsat/DDP/pcb
export SCRATCH_ROOT="$PROJECT_ROOT/scratch"
export YOLO_CONFIG_DIR="$PROJECT_ROOT/offline/ultralytics_cfg"
export MPLCONFIGDIR="$SCRATCH_ROOT/mplcache"
export TORCH_HOME="$PROJECT_ROOT/offline/torch"
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
mkdir -p "$MPLCONFIGDIR" "$TORCH_HOME" "$YOLO_CONFIG_DIR" "$SCRATCH_ROOT/exp_data"
source /lustre-flash/apps/spack/share/spack/setup-env.sh
spack load anaconda3
source "$HOME/envs/pcb/bin/activate"
RUNDIR="$SCRATCH_ROOT/jobs/$SLURM_JOB_ID"; mkdir -p "$RUNDIR"; cd "$RUNDIR"
ln -sf "$PROJECT_ROOT"/offline/weights/*.pt .
echo "=== $EXP_NAME on $(hostname) job=$SLURM_JOB_ID ==="
srun python "$PROJECT_ROOT/experiments/exp_gmodetr.py"
