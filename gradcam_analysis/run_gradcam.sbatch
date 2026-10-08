#!/bin/bash
#SBATCH --job-name=gradcam
#SBATCH --partition=interactive
#SBATCH --qos=interactive
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
# Usage: sbatch --export=ALL,GC_ARGS="scan" gradcam_analysis/run_gradcam.sbatch
#        several steps in one job: GC_ARGS="cam;compare --n 8"   (";" separates subcommands)
set -uo pipefail
export PROJECT_ROOT=/home/agipml/samarth.sirsat/DDP/pcb
export YOLO_CONFIG_DIR="$PROJECT_ROOT/offline/ultralytics_cfg"
export MPLCONFIGDIR="$PROJECT_ROOT/scratch/mplcache"
export TORCH_HOME="$PROJECT_ROOT/offline/torch"
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
source /lustre-flash/apps/spack/share/spack/setup-env.sh
spack load anaconda3
source "$HOME/envs/pcb/bin/activate"
cd "$PROJECT_ROOT"
IFS=";" read -ra STEPS <<< "$GC_ARGS"
for c in "${STEPS[@]}"; do srun python gradcam_analysis/gradcam_rtdetr.py $c || exit 1; done
