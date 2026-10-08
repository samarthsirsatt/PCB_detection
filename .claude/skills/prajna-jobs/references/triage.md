# Job failure triage

Symptom → exact signature → fix. Check `sacct -j <id> --format=JobID,State,Elapsed,MaxRSS,ReqTRES,ExitCode`
and `%x.%j.err` first — most failures here have a known signature below.

| symptom | signature | fix |
|---|---|---|
| PENDING forever, never errors | `Reason=Job's_QOS_not_permitted_to_use_this_partition_(l40_allows_l40_not_normal)` (via `scontrol show job <id>`) | `--qos` must equal `--partition` exactly — check both flags in the sbatch script |
| PENDING, normal | `Reason=Resources` or `Reason=Priority` | expected under load; `squeue --start -j <id>` estimates when it'll run; consider `a40` if `l40` is congested |
| PENDING, won't ever run | `Reason=QOSMaxJobsPerUserLimit` or `Reason=AssocMaxWallDurationPerJobLimit` | check `sacctmgr show assoc user=$USER format=partition,qos,maxjobs,maxwall` — you're over your own limit, not a cluster-wide one; lower `--time` or cancel another running job |
| job killed partway through | `.err` ends with `DUE TO TIME LIMIT` | raise `--time`, or switch to `resumable_job.sbatch` so a requeue picks up from `last.pt` |
| exit code 137, job vanished | `slurmstepd: error: ... oom-kill event` in `.out`/`.err`, or `sacct` shows `OUT_OF_MEMORY` | **host RAM**, not GPU — raise `--mem` in the sbatch script |
| `torch.cuda.OutOfMemoryError: CUDA out of memory` | appears in Python traceback, job may still show `COMPLETED`/`FAILED` depending on where it's caught | **GPU VRAM**, different resource from the above — lower `batch` in the training call, do not raise `--mem` (that won't help) |
| process just says `Killed`, no SLURM job ID involved | you're running python directly in an SSH/interactive shell on the login node | training was terminated for running on the login node — always use `sbatch` or `salloc`/`srun` |
| `RuntimeError` or hang referencing `yolo11n.pt` right at train start | happens before any real training output, same failure across all four scripts | Ultralytics' AMP allclose check needs `yolo11n.pt` staged and resolvable — see `prajna-env/references/offline-staging.md`; confirm the CWD symlink in the sbatch template actually ran |
| training hangs at startup with `device=[0,1]` or similar | no crash, just no progress; `nvidia-smi` inside the job shows only 1 GPU visible | `--gres=gpu:N` in the sbatch script doesn't match the script's `device=` list — either raise `--gres` to match, or (preferred, see partitions-qos.md) switch the script to single-GPU via `hpc_env.devices()` |
| `FileNotFoundError: gmo_detr_4a.yaml` or similar right after model construction | only in `gmo_detr_full_pipeline.py`; works when run manually from the repo dir, fails under sbatch | the script writes/reads that file via a relative path — CWD under sbatch is wherever the job dir is, not the repo; needs an absolute path — see `pcb-port-to-batch/references/porting-recipe.md` |
| `IndexError: list index out of range` from something like `labeled_dirs[0]` | only in `pretrain_soldef_yolo.py` | the SolDef_AI auto-discovery glob found nothing — this script is blocked until the SolDef_AI dataset is sourced separately (config §7); this is not a bug to fix, it's a missing input |
| `ModuleNotFoundError` for ultralytics/einops/grad-cam | fails immediately on import | venv isn't activated, or the wheel wasn't staged — see `prajna-env` |
| network-related timeout/`URLError` mid-run | e.g. fetching a font, a checkpoint, or `pip install` still present in the script | either a leftover `!pip install`/download call that porting missed (`pcb-port-to-batch`), or a genuinely offline cluster hitting an unstaged asset (`prajna-env/references/offline-staging.md`) |

## Reading `sacct` output

```
sacct -j <id> --format=JobID,State,Elapsed,MaxRSS,ReqTRES,ExitCode
```
- `MaxRSS` near or above your `--mem` request → host RAM was the problem, not VRAM.
- `ExitCode` of `0:0` with `State=COMPLETED` but bad results → not a SLURM/infra problem at
  all; that's a modeling question for `pcb-experiments`.
- `State=CANCELLED` with no error → check if someone (or a previous debugging session) ran
  `scancel` on it deliberately.

## When nothing here matches

Append the exact error string and what fixed it to `PRAJNA_CONFIG.md` §10 once resolved, so
the next session doesn't rediscover it from scratch.
