#!/bin/bash
#SBATCH --job-name=leakcheck
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --exclude=cn22-a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
set -uo pipefail
export PROJECT_ROOT=/home/agipml/samarth.sirsat/DDP/pcb
export YOLO_CONFIG_DIR="$PROJECT_ROOT/offline/ultralytics_cfg" MPLCONFIGDIR="$PROJECT_ROOT/scratch/mplcache"
export TORCH_HOME="$PROJECT_ROOT/offline/torch" HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
source /lustre-flash/apps/spack/share/spack/setup-env.sh; spack load anaconda3; source "$HOME/envs/pcb/bin/activate"
cd "$PROJECT_ROOT"
srun python ddp1_audit/leak_check_eval.py
