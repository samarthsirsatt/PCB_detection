"""Environment resolution for HPC batch runs of the PCB defect-detection scripts.

Copy this file next to the script being ported and import from it in place of the old Kaggle
path constants (INPUT_ROOT, WORK, YOUR_COCO_JSON, YOUR_IMG_ROOT, NEG_IMG_ROOT, COCO_JSONS,
IMAGE_ROOTS, NEG_IMAGE_DIR, OUT). See PRAJNA_CONFIG.md for what each env var should be set to,
and pcb-port-to-batch/SKILL.md for the porting checklist this supports.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path.home() / "pcb"))
SCRATCH_ROOT = Path(os.environ.get("SCRATCH_ROOT", PROJECT_ROOT / "scratch"))
DATA_ROOT = Path(os.environ.get("PCB_DATA_ROOT", PROJECT_ROOT / "data" / "pcb_all_work"))
WEIGHTS_DIR = Path(os.environ.get("PCB_WEIGHTS_DIR", PROJECT_ROOT / "offline" / "weights"))
RUNS_ROOT = Path(os.environ.get("PCB_RUNS_ROOT", PROJECT_ROOT / "runs"))
WORK = Path(os.environ.get("PCB_WORK", SCRATCH_ROOT / "work"))

COCO_JSON = DATA_ROOT / "coco" / "annotations" / "instances_default.json"
IMG_ROOT = DATA_ROOT / "categories"
NEG_ROOT = DATA_ROOT / "OK PHOTOS_MERGED"

# For scripts that took lists (gmo_detr_full_pipeline.py's COCO_JSONS / IMAGE_ROOTS):
COCO_JSONS = [str(COCO_JSON)]
IMAGE_ROOTS = [str(IMG_ROOT)]
NEG_IMAGE_DIR = str(NEG_ROOT)


def weights(name: str) -> str:
    """Absolute path to a staged pretrained weight, e.g. weights("yolo11s.pt").

    Falls back to the bare name if not staged, so the script still works unmodified if run
    somewhere with real internet access (Ultralytics will then download it as before).
    """
    p = WEIGHTS_DIR / name
    return str(p) if p.exists() else name


def devices():
    """Ultralytics `device=` value derived from what SLURM actually granted.

    Replaces hardcoded device=[0,1] (written for a 2xT4 Kaggle notebook). Honors an explicit
    PCB_DEVICE override (e.g. "0,1" or "cpu") if set, otherwise inspects
    CUDA_VISIBLE_DEVICES as set by SLURM's --gres=gpu:N grant.
    """
    override = os.environ.get("PCB_DEVICE")
    if override:
        if override == "cpu":
            return "cpu"
        if "," in override:
            return [int(x) for x in override.split(",")]
        return int(override)

    visible = [x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x]
    if not visible:
        import torch
        return 0 if torch.cuda.is_available() else "cpu"
    return 0 if len(visible) == 1 else list(range(len(visible)))


def require(*paths):
    """Fail with a clear message instead of an IndexError/FileNotFoundError deep in the script.

    Use this in place of auto-discovery globs that assume Kaggle's /kaggle/input layout —
    e.g. pretrain_soldef_yolo.py's SolDef_AI directory search.
    """
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        raise SystemExit(
            "Missing required input(s) — check PRAJNA_CONFIG.md for the correct path:\n  "
            + "\n  ".join(missing)
        )
