"""One parameterised GMO-DETR experiment — the GMO twin of exp_rtdetr.py.

Same data pipeline (exp_common), same 8-class split, same augmentation defaults,
so every number here is directly comparable to the RT-DETR study's table.

    EXP_NAME     run name under RUNS_ROOT/experiments/     (required)
    NEG_TOTAL    negatives mixed in                        (default 120 = baseline)
    FLIPUD       vertical-flip probability                 (default 0.5 = baseline)
    NUM_QUERIES  RTDETRDecoder object queries              (default 300 = baseline)
    P2           1 = add stride-4 detection head           (default 0)
    IMGSZ        train/val resolution                      (default 640)
    BATCH        batch size                                (default 8)
    EPOCHS       epochs                                    (default 200 = paper)
    SEED         random seed                               (default 42)
    FOLD         0-4 for k-fold CV, unset for 80/20 split  (default unset)

Baseline (all defaults) = gmo_detr_full_pipeline.py's training config on the
8-class subset: 640px, batch 8, AdamW, lr 1e-4, wd 1e-4, 200 epochs, from scratch.
"""
import json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hpc_env import RUNS_ROOT, SCRATCH_ROOT, devices     # noqa: E402
import exp_common as C                                   # noqa: E402

EXP = os.environ["EXP_NAME"]
NEG = int(os.environ.get("NEG_TOTAL", 120))
FLIPUD = float(os.environ.get("FLIPUD", 0.5))
NQ = int(os.environ.get("NUM_QUERIES", 300))
P2 = os.environ.get("P2", "0") == "1"
IMGSZ = int(os.environ.get("IMGSZ", 640))
BATCH = int(os.environ.get("BATCH", 8))
EPOCHS = int(os.environ.get("EPOCHS", 200))
SEED = int(os.environ.get("SEED", 42))
FOLD = os.environ.get("FOLD")
FOLD = int(FOLD) if FOLD not in (None, "") else None

print(f"=== EXP {EXP} (GMO-DETR) ===")
print(f"    neg={NEG} flipud={FLIPUD} nq={NQ} p2={P2} imgsz={IMGSZ} "
      f"batch={BATCH} epochs={EPOCHS} seed={SEED} fold={FOLD}")

data = C.build_dataset(n_neg=NEG, fold=FOLD)

import gmo_arch                                          # noqa: E402
gmo_arch.register()

from ultralytics import RTDETR                           # noqa: E402

ycfg = Path(SCRATCH_ROOT) / f"gmo_detr_{EXP}.yaml"
gmo_arch.make_yaml(ycfg, nc=8, nq=NQ, p2=P2)
model = RTDETR(str(ycfg))
n_par = sum(p.numel() for p in model.model.parameters())
print(f"[model] {ycfg} — {n_par/1e6:.2f} M params (from scratch, no pretrained weights)")

project = str(Path(RUNS_ROOT) / "experiments")
model.train(
    data=str(data / "data.yaml"),
    epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, patience=30,
    project=project, name=EXP, exist_ok=True,
    device=devices(),
    # --- paper's optimiser settings (gmo_detr_full_pipeline.py, Table 2) ---
    optimizer="AdamW", lr0=1e-4, weight_decay=1e-4, warmup_epochs=3,
    # --- augmentation: identical to the RT-DETR experiments except FLIPUD ---
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    degrees=15, shear=10, scale=0.5,
    fliplr=0.5, flipud=FLIPUD, mosaic=1.0,
    seed=SEED, workers=6, val=True, plots=True,
)

res = model.val(data=str(data / "data.yaml"), split="val", imgsz=IMGSZ)
names = [c["name"] for c in sorted(
    C.load_8class()[0]["categories"], key=lambda c: c["id"])]
out = {
    "exp": EXP,
    "model": "GMO-DETR",
    "config": {"neg": NEG, "flipud": FLIPUD, "nq": NQ, "p2": P2,
               "imgsz": IMGSZ, "batch": BATCH, "epochs": EPOCHS,
               "fold": FOLD, "seed": SEED, "params_m": round(n_par / 1e6, 2)},
    "map50": float(res.box.map50), "map": float(res.box.map),
    "precision": float(res.box.mp), "recall": float(res.box.mr),
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
