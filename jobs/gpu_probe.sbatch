#!/bin/bash
#SBATCH --job-name=pcb_gpuprobe
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err

# Read-only GPU/environment probe. Trains nothing. Confirms the half of the setup that
# cannot be tested from the login node. See SITE_AMENDMENTS.md.
set -uo pipefail

export PROJECT_ROOT=/home/agipml/samarth.sirsat/DDP/pcb
export SCRATCH_ROOT="$PROJECT_ROOT/scratch"

echo "=== node / slurm ==="
hostname; echo "job=$SLURM_JOB_ID part=$SLURM_JOB_PARTITION gres=${SLURM_GPUS_ON_NODE:-?}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

echo; echo "=== nvidia-smi ==="
nvidia-smi || echo "nvidia-smi FAILED"

echo; echo "=== node-local scratch (config claims TmpDisk=0) ==="
echo "TMPDIR=${TMPDIR:-unset}"; df -h /tmp 2>&1 | tail -1

echo; echo "=== env ==="
source /lustre-flash/apps/spack/share/spack/setup-env.sh
spack load anaconda3
source "$HOME/envs/pcb/bin/activate"
python -V; which python

echo; echo "=== torch / cuda ==="
python - << 'PY'
import torch, time
print("torch        :", torch.__version__)
print("CUDA build   :", torch.version.cuda)
print("is_available :", torch.cuda.is_available())
print("device_count :", torch.cuda.device_count())
if torch.cuda.is_available():
    i = torch.cuda.current_device()
    print("device name  :", torch.cuda.get_device_name(i))
    print("capability   :", torch.cuda.get_device_capability(i))
    print("VRAM total   : %.1f GB" % (torch.cuda.get_device_properties(i).total_memory/1e9))
    t = time.time(); a = torch.randn(4096,4096,device="cuda")
    (a@a).sum().item(); torch.cuda.synchronize()
    print("matmul 4096^2: OK (%.2fs)" % (time.time()-t))
else:
    print("!! NO GPU VISIBLE TO TORCH")
PY

echo; echo "=== ultralytics + hpc_env ==="
cd "$PROJECT_ROOT"
python - << 'PY'
import os
os.environ.setdefault("YOLO_CONFIG_DIR", os.environ["SCRATCH_ROOT"]+"/ultralytics_cfg")
import ultralytics; print("ultralytics  :", ultralytics.__version__)
import hpc_env as E
print("devices()    :", repr(E.devices()))
print("COCO_JSON ok :", E.COCO_JSON.exists())
print("IMG_ROOT  ok :", E.IMG_ROOT.exists())
print("NEG_ROOT  ok :", E.NEG_ROOT.exists())
PY

echo; echo "=== egress FROM COMPUTE NODE (amendment A2 caveat) ==="
for u in https://pypi.org/simple/ https://github.com/ https://objects.githubusercontent.com/; do
  printf '%-45s ' "$u"; curl -sI --max-time 8 "$u" 2>&1 | head -1 || echo UNREACHABLE
done

echo; echo "=== done ==="
