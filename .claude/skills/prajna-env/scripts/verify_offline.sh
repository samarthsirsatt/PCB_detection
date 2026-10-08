#!/bin/bash
# Read-only verification that offline staging + the venv actually work. Run after staging
# per ../references/offline-staging.md, and again as the first step of any triage.
#
# Usage: PROJECT_ROOT=/home/<group>/<user>/pcb bash verify_offline.sh
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?Set PROJECT_ROOT first — see PRAJNA_CONFIG.md §1}"
OFFLINE="$PROJECT_ROOT/offline"
FAIL=0

check() {
  local desc="$1"; shift
  if "$@" >/tmp/verify_offline_out.$$ 2>&1; then
    echo "OK   $desc"
  else
    echo "FAIL $desc"
    sed 's/^/       /' /tmp/verify_offline_out.$$
    FAIL=1
  fi
  rm -f /tmp/verify_offline_out.$$
}

echo "=== Weights ==="
for w in yolo11s.pt yolo11n.pt rtdetr-l.pt; do
  check "weight present: $w" test -f "$OFFLINE/weights/$w"
done

echo
echo "=== Wheels ==="
check "wheels dir non-empty" bash -c "[ -n \"\$(ls -A '$OFFLINE/wheels' 2>/dev/null)\" ]"

echo
echo "=== Venv ==="
VENV="$PROJECT_ROOT/env/pcb"
check "venv exists" test -x "$VENV/bin/python"
if [ -x "$VENV/bin/python" ]; then
  check "ultralytics importable" "$VENV/bin/python" -c "import ultralytics"
  check "torch importable" "$VENV/bin/python" -c "import torch"
  echo
  echo "NOTE: the next check only means something under srun on a GPU node — the login node"
  echo "may have no GPU or a different driver. Run manually if this script isn't inside a job:"
  echo "  srun --partition=<p> --gres=gpu:1 --pty $VENV/bin/python -c \"import torch; print(torch.cuda.is_available())\""
  if [ -n "${SLURM_JOB_ID:-}" ]; then
    check "torch.cuda.is_available() under srun" "$VENV/bin/python" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"
  else
    echo "SKIP torch.cuda.is_available() — not running inside a SLURM job, see NOTE above"
  fi
fi

echo
if [ "$FAIL" -eq 0 ]; then
  echo "All checks passed."
else
  echo "One or more checks failed — see references/offline-staging.md and"
  echo "prajna-env/references/spack-and-python.md."
fi
exit "$FAIL"
