# Job decomposition

Recommendation, not a rule — a single job running an entire script end-to-end is fine for a
first smoke test (see `SKILL.md`'s smoke-test habit). Decompose once you're running for real,
because a single multi-hour job loses everything if it dies partway, and ties up one allocation
for the full duration instead of letting independent pieces run in parallel or queue
separately.

## `8classes_kfold.py` → 3 jobs

The script currently runs, in order: a 150-epoch baseline at 640px, a resolution sweep at
960px and 1280px (150 epochs each), then a 5-fold cross-validation at 120 epochs per fold — all
in one process, roughly 17 GPU-hours total.

- **Job 1 — baseline.** Just the 640px training call. ~2h on an L40S.
- **Job 2 — resolution sweep.** The 960px and 1280px calls. ~5h combined; could itself be
  split into 2 jobs if desired, but they're short enough to leave together.
- **Job 3 — cross-validation, as an array job.** The script's CV loop already deletes each
  fold's built dataset between folds "to free disk (Kaggle space had limited space)" — that
  comment no longer applies to `/scratch`, but the cleanup-per-fold pattern maps naturally onto
  `array_job.sbatch`'s `--array=0-4`: each of the 5 folds becomes an independent array task with
  its own run dir, running concurrently across nodes instead of sequentially in one job.
  ~10h sequential → ~2h wall as an array.

To select the fold inside the script, read `FOLD` (set by `array_job.sbatch` from
`$SLURM_ARRAY_TASK_ID`) instead of looping `for fold in range(K_FOLDS)`.

## `pretrain_soldef_yolo.py` → 3 jobs (once SolDef_AI is sourced)

Currently runs 5 trainings back to back: a 100-epoch pretrain, then 4 finetune arms
(150 epochs each) — roughly 12 GPU-hours, and blocked entirely until the SolDef_AI dataset
exists (see `script-inventory.md`).

- **Job 1 — pretrain.** 100 epochs on SolDef_AI. Everything downstream depends on its output
  weight.
- **Job 2 — finetune arms A/B (+ the two no-negatives variants).** Submit with
  `sbatch -d afterok:<pretrain_job_id> finetune.sbatch` so it only starts once the pretrain
  weight exists. All 4 arms could run as their own array job if further parallelism is wanted.
- **Job 3 — EigenCAM visualization.** Needs no GPU and no queue wait for a GPU partition — can
  run under `salloc` on a CPU partition, or even inline on the login node if it's genuinely
  just reading saved weights and producing plots (confirm it doesn't call `.train()` or
  anything GPU-bound before deciding this). Depends only on job 2's finetuned weights, and
  separately on the `YOLO-26-CAM` repo being staged (`prajna-env/references/offline-staging.md`)
  — if that staging hasn't happened yet, jobs 1-2 still run fine without it.

## `rtdetr.py`, `gmo_detr_full_pipeline.py` → single job each

Both already run one coherent training call end-to-end (`rtdetr.py` ~3h, GMO-DETR ~5h at
200 epochs) — no decomposition needed. `gmo_detr_full_pipeline.py`'s 200-epoch run is the best
candidate in this project for `resumable_job.sbatch` if the walltime estimate turns out tight.

## General pattern for job dependencies

```bash
first=$(sbatch --parsable pretrain.sbatch)
sbatch -d afterok:$first finetune.sbatch
```
`sbatch --parsable` prints just the job ID, making it easy to chain in a shell script or by
hand.
