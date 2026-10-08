---
name: prajna-jobs
description: Author, submit, monitor, and debug SLURM batch jobs on the Prajna AI-ML cluster. Use when writing or editing an .sbatch script, choosing a partition/QOS/--gres/--time/--mem combination, submitting with sbatch or allocating interactively with salloc/srun, checking squeue/sacct/scontrol, cancelling jobs, finding the .out/.err logs for a run, or diagnosing a job that is stuck PENDING, was killed, ran out of memory, hit its walltime, or hit "Job's_QOS_not_permitted_to_use_this_partition". Also use when a process was terminated for running on the login node.
---

# Prajna SLURM jobs

Read `../../PRAJNA_CONFIG.md` §3 for the real partition/QOS/GRES table before picking values —
the table there is kept current with what's actually been confirmed on the cluster — but read
`../SITE_AMENDMENTS.md` first, which corrects that table (node counts, `l40` is capped at 2 days
not UNLIMITED, `dgx`/`dgx-mpi`/`interactive`/`debug` are missing from it) and notes that a
partition/QOS mismatch is rejected at submit time, not left PENDING.

## When NOT to use this skill

- Setting up the venv, Spack packages, or offline weights → `prajna-env`.
- Editing a training script's paths/device args/relative writes → `pcb-port-to-batch`.
- Reading a finished run's metrics → `pcb-experiments`.

## INVARIANTS

- **`--qos` must equal `--partition`.** A mismatch produces exactly this reason string, and
  the job sits PENDING forever, never erroring out on its own:
  ```
  Reason=Job's_QOS_not_permitted_to_use_this_partition_(l40_allows_l40_not_normal)
  ```
- **Never run training on the login node.** It gets killed. All training goes through
  `sbatch` (batch) or `salloc`/`srun` (interactive).
- **`--time` is effectively mandatory.** `l40`'s `DefaultTime=NONE`, and backfill scheduling
  uses your requested time to decide whether you can slot in ahead of others — an inflated or
  absent time limit only delays your own start.

## DEFAULTS (starting points, not mandates)

- **Partition: `l40`** (the cluster default; 7 nodes, 8×L40S 48GB each). *Deviate when:*
  `sinfo -p l40` shows it fully allocated and `a40` (20 nodes, 4×A40 48GB) is free — checking
  is cheap, do it before submitting into a long queue.
- **`--gres=gpu:1`, single GPU.** *Because:* the project's dataset is 441-1,750 training
  images; at batch 16/640px that's roughly 12GB on a 48GB card. Multi-GPU (`device=[0,1]` in
  the original Kaggle scripts) triggers Ultralytics' DDP relaunch path for no throughput gain
  and an extra failure mode (see triage). *Deviate when:* the dataset grows by an order of
  magnitude, or a sweep is specifically about scaling behavior.
- **`--cpus-per-task=8`** for Ultralytics' dataloader workers; adjust to what `sinfo` shows
  available on the target partition.
- **`--mem=64G`** as a starting point — raise it if `sacct` shows `oom-kill`, not
  `CUDA out of memory` (these are different resources, see triage).

Templates to copy and edit, not a wrapper CLI to fight with:
- `templates/gpu_job.sbatch` — single training run, the base case.
- `templates/array_job.sbatch` — for the 5-fold CV (`$SLURM_ARRAY_TASK_ID` selects the fold).
- `templates/resumable_job.sbatch` — adds `--requeue` and a `SIGUSR1` trap for the longest run
  (GMO-DETR, 200 epochs), so a walltime cutoff resumes from `last.pt` instead of losing the run.

## The smoke-test habit

Before committing hours of GPU time to a newly ported script, run it at `EPOCHS=2`,
`--time=00:20:00`. This one habit, stated explicitly, catches path/device/import mistakes
before they cost 5+ GPU-hours. See `pcb-port-to-batch/SKILL.md` for how the scripts expose an
epoch override.

## Monitoring and control

Cheatsheet in `references/slurm-cheatsheet.md`. The essentials:
```bash
sbatch job.sbatch                 # submit
squeue --me                       # your jobs
sacct -j <id> --format=JobID,State,Elapsed,MaxRSS,ReqTRES,ExitCode   # post-mortem
scontrol show job <id>            # full detail, including the Reason= string
scancel <id>                      # cancel
scontrol hold|release <id>        # pause/resume a pending job
```

## When something goes wrong

`references/triage.md` is a symptom → exact string → fix table. Read it before re-running
anything — most failures here have a known signature.

## Escape hatch

Need something not covered — a different partition, a container, an interactive Jupyter
tunnel? That's fine. Check `sinfo`/`sacctmgr`/`man sbatch` yourself and record what you learn
in `PRAJNA_CONFIG.md` §10 so it's not rediscovered next time.
