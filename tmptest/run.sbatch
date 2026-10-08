#!/bin/bash
# =============================================================================
# General-purpose SLURM template for running ANY Python script on Prajna.
#
# Not project-specific — copy it anywhere, edit the four EDIT lines, submit.
# For the PCB defect-detection scripts specifically, prefer gpu_job.sbatch /
# array_job.sbatch / resumable_job.sbatch in this directory, which add the
# dataset, weights, and runs/ wiring this template deliberately leaves out.
#
# Derived from Prajna_AIML_HPC_User_Manual.md "Parameters used in SLURM job
# script" (the flag table) and "Sample SLURM Scripts for reference". The manual
# has NO Python and NO GPU example — its samples are all Intel-compiler MPI and
# OpenMP jobs — so everything below the #SBATCH block is filled in from what was
# confirmed on the cluster on 2026-08-26. See ../../SITE_AMENDMENTS.md.
#
# !! Do NOT copy the manual's sample scripts verbatim: they annotate directives
# !! with `//` inline comments, which are NOT valid bash. `#SBATCH -N 1 // n`
# !! makes SLURM parse the value as "1 // n" and the job fails to submit. Bash
# !! comments are `#`. This template uses `#` throughout.
#
# Submit:   sbatch python_job.sbatch
# Validate: sbatch --test-only python_job.sbatch     # documented in the manual
# Watch:    squeue -u $USER          Detail: scontrol show job <id>
# After:    sacct -j <id> --format=JobID,State,ExitCode,Elapsed,MaxRSS
# =============================================================================

#SBATCH --job-name=tmpl_test                # EDIT: shows in squeue; keep it short
#SBATCH --partition=l40                  # EDIT: l40 | a40 | dgx | dgx-mpi | interactive | debug
#SBATCH --qos=l40                        # MUST EQUAL --partition (see INVARIANT below)
#SBATCH --gres=gpu:1                     # EDIT: drop this line entirely for a CPU-only job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8                # EDIT: CPU cores; also becomes OMP_NUM_THREADS below
#SBATCH --mem=64G                        # EDIT: HOST RAM, not VRAM. Raise this on oom-kill.
#SBATCH --time=00:15:00                  # EDIT: HH:MM:SS or D-HH:MM:SS. Mandatory — see below.
#SBATCH --output=logs/%x.%j.out          # %x = job-name, %j = job-id
#SBATCH --error=logs/%x.%j.err
# Optional, straight from the manual's flag table — uncomment if wanted:
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=you@iitb.ac.in
##SBATCH --exclusive                     # whole node, no co-tenants
##SBATCH --nodelist=cn40-l40             # pin to a specific node
##SBATCH --dependency=afterok:<jobid>    # run only after <jobid> succeeds

# -----------------------------------------------------------------------------
# INVARIANT: --qos must equal --partition.
#   Every partition here is locked to its same-named QOS (AllowQos=<partition>).
#   Using --qos=normal or --qos=standard fails IMMEDIATELY at submit time with
#   "allocation failure: Invalid qos specification" — the job never queues.
#   (The manual describes this as a stuck-PENDING job instead; that is wrong.)
#
# Partition caps, confirmed 2026-08-26 (the manual says "infinite" — also wrong):
#   l40  2 days, 8x L40S 48G, 6 nodes, DEFAULT, usually starts immediately
#   a40  4 days, 4x A40  48G, 19 nodes
#   dgx  6 days, 8x A100,      9 nodes    dgx-mpi 6 days, 3 nodes (multi-node MPI)
#   interactive 4 h           debug 30 min
# --time is effectively mandatory: backfill rewards an honest estimate, and an
# over-cap value is NOT reliably rejected at submit — keep it under the cap.
# -----------------------------------------------------------------------------

set -euo pipefail            # -e exit on error, -u unset vars are errors, pipefail
                             # Drop -e if your script's non-zero exit is expected.

mkdir -p logs                # --output/--error do NOT create the directory; SLURM
                             # silently discards output if logs/ is missing.

# --- environment -------------------------------------------------------------
# The manual documents `source .../setup-env.sh` then `spack load <pkg>`, but only
# ever loads Intel compilers. For Python, note that Spack on this cluster has NO
# `python` package (`spack find python` errors) — anaconda3 is the only usable
# interpreter. See SITE_AMENDMENTS.md A1.
source /lustre-flash/apps/spack/share/spack/setup-env.sh
spack load anaconda3                                  # Python 3.11.5

# Activate a venv if you have one; harmless to leave pointing at a missing path.
VENV="${VENV:-$HOME/envs/pcb}"                        # EDIT: your venv, or unset VENV
[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"

# --- offline hygiene ---------------------------------------------------------
# COMPUTE NODES HAVE NO INTERNET. The login node does; compute nodes do not
# (pypi, github, all unreachable from inside a job — SITE_AMENDMENTS.md A13).
# Anything your script downloads on first use MUST be pre-staged from the login
# node: model weights, HF checkpoints, torch hub, tokenizers, NLTK data, fonts.
# Never `pip install` from inside a job — install into the venv on the login node.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1                             # stream logs live, don't buffer
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"    # manual's OpenMP guidance
export MPLCONFIGDIR="${MPLCONFIGDIR:-$HOME/.cache/matplotlib}"
mkdir -p "$MPLCONFIGDIR"

# --- run ---------------------------------------------------------------------
cd "$SLURM_SUBMIT_DIR"       # manual's idiom: run from the submitting directory

echo "=== job $SLURM_JOB_ID on $(hostname) ==="
echo "partition=$SLURM_JOB_PARTITION  cpus=$SLURM_CPUS_PER_TASK  submit_dir=$PWD"
echo "python: $(which python)  $(python -V 2>&1)"
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] && nvidia-smi --query-gpu=name,memory.total \
    --format=csv,noheader || echo "no GPU requested"
echo "=== start $(date -Is) ==="

srun python your_script.py                            # EDIT: your script + args

echo "=== end $(date -Is) rc=$? ==="

# =============================================================================
# NOTES
#
# GPU visibility: with --gres=gpu:N, SLURM sets CUDA_VISIBLE_DEVICES and always
#   renumbers from 0. Inside the job your GPUs are 0..N-1 regardless of which
#   physical devices you got — so `device=0` is correct for --gres=gpu:1. Never
#   hardcode a device list longer than what you requested.
#
# CPU vs GPU OOM are different failures:
#   "oom-kill" / "Killed" in the .err    -> host RAM, raise --mem
#   "CUDA out of memory"                 -> VRAM, lower your batch size instead
#
# No node-local scratch worth using: config states TmpDisk=0. /tmp does exist on
#   nodes (~20G) but is node-local and unmanaged. There is also NO user-writable
#   /scratch on this cluster (SITE_AMENDMENTS.md A10) — /scratch is a root-owned
#   SLURM admin mount and /lustre-scratch is admin-provisioned. Write to /home,
#   which currently has no enforced quota, and clean up after yourself.
#
# First `import torch` off Lustre can take minutes with a cold cache. Don't set a
#   tight --time on a job whose first act is a heavy import, and don't mistake a
#   slow import for a hang.
#
# Job arrays: add `#SBATCH --array=0-9` and branch on $SLURM_ARRAY_TASK_ID; logs
#   should then use %A_%a instead of %j. Your per-user caps are MaxJobs=12,
#   MaxSubmit=20, so an array larger than that will queue in batches.
# =============================================================================
