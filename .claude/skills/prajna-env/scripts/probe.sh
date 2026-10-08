#!/bin/bash
# Read-only day-one discovery probe for Prajna. Prints a labelled report; does not modify
# anything. Every command here is also listed individually in ../references/discovery-probe.md
# so you can run any subset by hand instead of this script.
#
# Usage: bash probe.sh
set -uo pipefail

section() { printf '\n=== %s ===\n' "$1"; }

section "Identity"
echo "USER=$(whoami)"
echo "GROUP=$(id -gn)"
echo "HOME=$HOME"

section "Partitions (sinfo)"
sinfo -o "%20P %5a %10l %10G %D %N" 2>&1

section "Per-user SLURM limits (sacctmgr)"
sacctmgr show assoc user="$(whoami)" format=partition,qos,maxjobs,maxwall,maxsubmit 2>&1

section "/home quota"
if command -v lfs >/dev/null 2>&1; then
  lfs quota -h -u "$(whoami)" /home 2>&1
else
  quota -s 2>&1
fi

section "/scratch capacity"
df -h /scratch 2>&1

section "Spack: python"
source /lustre-flash/apps/spack/share/spack/setup-env.sh 2>/dev/null
spack find python 2>&1

section "Spack: py-torch"
spack find py-torch 2>&1

section "Spack: cuda/cudnn"
spack find cuda cudnn 2>&1

section "System python3 fallback"
python3 -c "import sys; print(sys.version)" 2>&1

section "Egress: pypi.org"
curl -sI --max-time 8 https://pypi.org/simple/ 2>&1 | head -1 || echo "UNREACHABLE"

section "Egress: files.pythonhosted.org"
curl -sI --max-time 8 https://files.pythonhosted.org/ 2>&1 | head -1 || echo "UNREACHABLE"

section "Egress: github.com"
curl -sI --max-time 8 https://github.com/ 2>&1 | head -1 || echo "UNREACHABLE"

section "Egress: objects.githubusercontent.com (Ultralytics release assets)"
curl -sI --max-time 8 https://objects.githubusercontent.com/ 2>&1 | head -1 || echo "UNREACHABLE"

section "GPU inventory reminder"
echo "nvidia-smi -L was NOT run here — it must run via srun on a compute node, not the login"
echo "node. Example: srun --partition=<p> --gres=gpu:1 --pty nvidia-smi -L"

section "Done"
echo "Copy the values above into PRAJNA_CONFIG.md, replacing [TODO(day-one)] markers and"
echo "stamping [CONFIRMED $(date +%Y-%m-%d)]. Note anything surprising in config §10."
