"""One parameterised RT-DETR experiment. Every knob comes from an env var so a
single script + an sbatch array covers the whole matrix.

    EXP_NAME     run name under RUNS_ROOT/experiments/     (required)
    NEG_TOTAL    negatives mixed in                        (default 120 = baseline)
    FLIPUD       vertical-flip probability                 (default 0.5 = baseline)
    NUM_QUERIES  RT-DETR decoder object queries            (default 300 = baseline)
    RECT         1 = rect=True (kill letterbox padding)    (default 0)
    IMGSZ        train/val resolution                      (default 640)
    BATCH        batch size                                (default 8)
    EPOCHS       epochs                                    (default 150)
    FOLD         0-4 for k-fold CV, unset for 80/20 split  (default unset)

Baseline config (all defaults) reproduces rtdetr.py -> published 0.770 / 0.556.
"""
import json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hpc_env import RUNS_ROOT, weights, devices          # noqa: E402
import exp_common as C                                   # noqa: E402

EXP = os.environ["EXP_NAME"]
NEG = int(os.environ.get("NEG_TOTAL", 120))
FLIPUD = float(os.environ.get("FLIPUD", 0.5))
NQ = int(os.environ.get("NUM_QUERIES", 300))
RECT = os.environ.get("RECT", "0") == "1"
IMGSZ = int(os.environ.get("IMGSZ", 640))
BATCH = int(os.environ.get("BATCH", 8))
EPOCHS = int(os.environ.get("EPOCHS", 150))
SEED = int(os.environ.get("SEED", 42))
FOLD = os.environ.get("FOLD")
FOLD = int(FOLD) if FOLD not in (None, "") else None

print(f"=== EXP {EXP} ===")
print(f"    neg={NEG} flipud={FLIPUD} nq={NQ} rect={RECT} imgsz={IMGSZ} "
      f"batch={BATCH} epochs={EPOCHS} fold={FOLD}")

data = C.build_dataset(n_neg=NEG, fold=FOLD)

from ultralytics import RTDETR                            # noqa: E402

if NQ == 300:
    model = RTDETR(weights("rtdetr-l.pt"))                # stock pretrained path
else:
    # Custom query count needs a patched YAML; transfer pretrained weights on top so
    # the comparison stays like-for-like (only the decoder query bank differs).
    ycfg = Path(os.environ.get("SCRATCH_ROOT", ".")) / f"rtdetr-l-nq{NQ}.yaml"
    C.make_nq_yaml(NQ, ycfg)
    model = RTDETR(str(ycfg))
    model.load(weights("rtdetr-l.pt"))
    print(f"[model] nq={NQ} from {ycfg}, pretrained weights transferred")

project = str(Path(RUNS_ROOT) / "experiments")
model.train(
    data=str(data / "data.yaml"),
    epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, patience=30,
    project=project, name=EXP, exist_ok=True,
    device=devices(), rect=RECT,
    # --- augmentation: identical to the rtdetr.py baseline except FLIPUD ---
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    degrees=15, shear=10, scale=0.5,
    fliplr=0.5, flipud=FLIPUD, mosaic=1.0,
    lr0=0.0001, seed=SEED, workers=6,
)

res = model.val(data=str(data / "data.yaml"), split="val", imgsz=IMGSZ)
names = [c["name"] for c in sorted(
    C.load_8class()[0]["categories"], key=lambda c: c["id"])]
out = {
    "exp": EXP,
    "config": {"neg": NEG, "flipud": FLIPUD, "nq": NQ, "rect": RECT,
               "imgsz": IMGSZ, "batch": BATCH, "epochs": EPOCHS, "fold": FOLD, "seed": SEED},
    "map50": float(res.box.map50), "map": float(res.box.map),
    "precision": float(res.box.mp), "recall": float(res.box.mr),
    # box.maps is per-class mAP@0.5:0.95 (it indexes Metric.ap). box.ap50 is per-class
    # AP@0.5 but is ordered by ap_class_index, so it must be scattered back by class id.
    "per_class_map": {n: float(res.box.maps[i]) if i < len(res.box.maps) else None
                      for i, n in enumerate(names)},
    "per_class_ap50": None,
}
try:
    _ap50 = list(res.box.ap50); _idx = list(res.box.ap_class_index)
    _sc = {int(c): float(_ap50[j]) for j, c in enumerate(_idx) if j < len(_ap50)}
    out["per_class_ap50"] = {n: _sc.get(i) for i, n in enumerate(names)}
except Exception as e:
    print("[warn] per-class AP50 unavailable:", e)
dest = Path(project) / EXP / "summary.json"
dest.write_text(json.dumps(out, indent=2))
print("\n=== RESULT " + EXP + " ===")
print(json.dumps({k: out[k] for k in ("map50", "map", "precision", "recall")}, indent=2))
print("summary ->", dest)
