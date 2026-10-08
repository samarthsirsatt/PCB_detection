---
name: prajna-env
description: Set up and verify the Python/CUDA environment on the Prajna AI-ML HPC cluster (Rocky 9, Spack, SLURM). Use when bootstrapping on the cluster for the first time, when running the day-one discovery probe (partitions, QOS names, quota, GPU inventory, internet reachability), when loading Spack packages or creating/activating the project venv, when staging offline pip wheels, pretrained .pt weights, or vendored git repos, or when a job fails with ModuleNotFoundError, a torch/CUDA mismatch, or a failed download of yolo11s.pt / rtdetr-l.pt / Arial.ttf. Also use to fill in or refresh PRAJNA_CONFIG.md.
---

# Prajna environment setup

Read `../../PRAJNA_CONFIG.md` (the project's `.claude/PRAJNA_CONFIG.md`) first. If §1, §3, or §5
still say `[TODO(day-one)]`, you're on day one — go straight to "Day-one discovery" below.
If config and this file disagree, the config wins: it reflects what was actually observed.
Then read `../SITE_AMENDMENTS.md`, which overrides both — as of 2026-08-26 it records that
`spack load python@3.11` **fails on this cluster** (use `spack load anaconda3`, Python 3.11.5)
and that egress works, making the offline-staging path optional rather than required.

## When NOT to use this skill

- Writing or debugging an `.sbatch` file, choosing a partition, or a job stuck PENDING/OOM →
  `prajna-jobs`.
- Editing the training scripts themselves (paths, device args, relative writes) →
  `pcb-port-to-batch`.
- Reading results out of a finished run → `pcb-experiments`.

## INVARIANTS (violating these produces a specific, named failure)

- Compute nodes have `TmpDisk=0` — there is no node-local scratch. Don't design around
  `$TMPDIR`; the staging tier is `/scratch` on Lustre.
- Every one of the four project scripts needs `yolo11n.pt` at train start (Ultralytics' AMP
  allclose check), even the from-scratch GMO-DETR run. Missing it fails all four.
- Never run training directly on the login node — it will be killed. Environment setup
  (installing packages, building the venv) is fine there; training is not.

## Day-one discovery

Run (or ask the agent to run) the read-only commands in `references/discovery-probe.md`, or
the equivalent `scripts/probe.sh`. Neither is mandatory over the other — `probe.sh` is a
convenience wrapper; every command it runs is listed individually in the reference so you can
run half of it, run it by hand, or add commands it doesn't cover.

Write results straight into `PRAJNA_CONFIG.md`, replacing `[TODO(day-one)]` markers in place
and stamping `[CONFIRMED <date>]`. If something surprising turns up (a partition name that
doesn't match the manual, quota far lower than expected, `objects.githubusercontent.com`
blocked while `github.com` is open), append it to config §10 rather than silently overwriting.

## DEFAULTS (sensible starting points — deviate when the reasoning no longer applies)

**Prefer `spack load python@X` + `python -m venv` + pip wheels over `spack install py-torch`.**
*Because:* a Spack-built py-torch is a from-source compile against whatever CUDA Spack picks,
often multi-hour and version-locked to the Spack environment. A pip wheel ships its own CUDA
runtime and installs in seconds. *Deviate when:* `spack find py-torch` already shows a build
matching the driver/CUDA on your target partition — then reusing it beats a 2.5GB wheel
download, especially if egress turns out to be blocked. Details and exact commands:
`references/spack-and-python.md`.

**Stage everything offline before assuming online will work.** *Because:* reachability at
setup time isn't reachability at 3am when an unattended array job starts, and the day-one probe
only tells you what's true right now. *Deviate when:* nothing — stage regardless of what the
probe says; it costs little and removes an entire failure class. Exact staging layout and the
`amp=False` fallback (with its real cost — roughly 2x VRAM, fp32) are in
`references/offline-staging.md`.

## Verifying the environment

`scripts/verify_offline.sh` (read-only) asserts: the three staged weights exist and match the
recorded sha256, `offline/wheels/` is non-empty, the venv imports `ultralytics`, and
`torch.cuda.is_available()` — **run the CUDA check under `srun`, not on the login node**, since
the login node may have no GPU or a different driver than compute nodes.

## Escape hatch

If you need something this skill doesn't cover — a container runtime, a different Python
version, a package Spack doesn't have — that's fine. Check `spack find`/`module avail`
yourself and record what you learn in `PRAJNA_CONFIG.md` §10 so the next session doesn't
re-discover it.
