# Partitions, QOS, and GRES

Full confirmed/unconfirmed table lives in `PRAJNA_CONFIG.md` §3 — that's the source of truth;
this file is the reasoning for how to choose among them.

## The invariant

`--qos` must equal `--partition`, always. A mismatch produces this exact reason and the job
never errors — it just sits PENDING indefinitely:
```
Reason=Job's_QOS_not_permitted_to_use_this_partition_(l40_allows_l40_not_normal)
```
If you ever see a job PENDING with a `Reason=` mentioning `_not_permitted_to_use_this_partition`,
this is it — go fix the mismatched `--qos`/`--partition` pair, don't wait it out.

## Choosing a partition

| partition | when to use it |
|---|---|
| `l40` (default) | Start here for anything in this project. 7 nodes × 8 L40S (48GB) is comfortably enough for 441-1,750 images at batch 8-16. |
| `a40` | `l40` congested (`sinfo -p l40` shows all nodes allocated) — 20 nodes × 4 A40 (48GB) is a same-VRAM alternative with more nodes. |
| L4 partition (name TBD, config §3) | Lower-VRAM (24GB) nodes — fine for inference or CPU-adjacent work, tight for training at these batch sizes. |
| DGX A100 partition (name TBD, config §3) | Only if you deliberately need genuine multi-GPU scaling (large dataset, throughput study) — see the GRES section below on why this project doesn't need it by default. |

Check current load before submitting into a long queue: `sinfo -p <partition>` shows node
state; `squeue -p <partition>` shows what's already queued ahead of you.

## GRES and device= — how they connect

`--gres=gpu:N` is what SLURM grants; Ultralytics' `device=` argument then indexes into
`CUDA_VISIBLE_DEVICES`, which SLURM sets based on that grant. **These two must match.** If a
script requests `device=[0,1]` but the job only got `--gres=gpu:1`, Ultralytics either errors
immediately or hangs trying to initialize a second device that doesn't exist — see
`triage.md`.

**Default for this project is `--gres=gpu:1` + single-device training.** The original Kaggle
scripts hardcode `device=[0,1]` because they were written for a 2×T4 Kaggle notebook — that
was a Kaggle constraint, not a project requirement. At 441 training images and batch 16/640px,
a single 48GB L40S has roughly 4x the headroom actually used; DDP (Ultralytics' multi-GPU path)
adds process-group setup, NCCL, and a re-launch step, none of which buys anything at this
scale. `pcb-port-to-batch`'s `hpc_env.devices()` helper derives the right value from what SLURM
actually granted, so this isn't something you hardcode per script.

Deviate to multi-GPU when the dataset grows by an order of magnitude (the SolDef_AI-scale
6,384-image regime the GMO-DETR paper uses) or a sweep is specifically about scaling behavior.

## Time limits

`l40`'s `DefaultTime=NONE`, `MaxTime=UNLIMITED` — SLURM will not pick a time limit for you, and
there's no cluster-enforced cap stopping an overestimate. Two costs of getting `--time` wrong,
in opposite directions:
- **Too low:** the job is killed mid-run at exactly `DUE TO TIME LIMIT` — see `triage.md`.
  Mitigated by `resumable_job.sbatch` for long runs.
- **Too high:** backfill scheduling (the cluster's scheduling policy) uses your requested time
  to predict when you'll finish and decide whether a shorter job can slot in ahead of a bigger
  pending one. An inflated estimate makes your own job *and* everyone behind it wait longer.

Check `sacctmgr show assoc user=$USER format=partition,qos,maxjobs,maxwall,maxsubmit` for your
actual per-user submission limits — these can be tighter than the partition's advertised
`MaxTime`.
