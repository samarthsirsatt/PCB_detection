"""Classification ceiling: how separable are the 8 defect classes, given perfect boxes?

93% of images contain exactly one object, so the task is nearly single-label
classification. This crops every GT box (with context padding) and trains a plain
classifier on the SAME image-level split as the detectors.

Reading the result:
  cls accuracy >> detection mAP@0.5  -> localisation is the bottleneck
  cls accuracy ~= detection mAP@0.5  -> the classes themselves are not separable;
                                        architecture work on the detector is misdirected
"""
import json, os, shutil, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from PIL import Image                                     # noqa: E402
from hpc_env import RUNS_ROOT, SCRATCH_ROOT, weights, devices   # noqa: E402
import exp_common as C                                    # noqa: E402

PAD = float(os.environ.get("CROP_PAD", 0.25))   # context around the box
EPOCHS = int(os.environ.get("EPOCHS", 100))
IMGSZ = int(os.environ.get("IMGSZ", 224))
MODEL = os.environ.get("CLS_MODEL", "yolo11s-cls.pt")

root = Path(SCRATCH_ROOT) / "exp_data" / f"cls_pad{PAD}"
if not (root / "train").exists():
    print(f"[data] building crop dataset at {root}")
    shutil.rmtree(root, ignore_errors=True)
    coco8, resolve = C.load_8class()
    names = {c["id"]: c["name"] for c in coco8["categories"]}
    tr, va = C.stratified_split(coco8)
    byid = {im["id"]: im for im in coco8["images"]}
    counts = Counter()
    for a in coco8["annotations"]:
        im = byid[a["image_id"]]
        src = resolve(im["file_name"])
        if src is None:
            continue
        split = "train" if a["image_id"] in tr else "val"
        cls = names[a["category_id"]]
        x, y, w, h = a["bbox"]
        px, py = w * PAD, h * PAD
        box = (max(0, x - px), max(0, y - py),
               min(im["width"], x + w + px), min(im["height"], y + h + py))
        d = root / split / cls.replace(" ", "_")
        d.mkdir(parents=True, exist_ok=True)
        try:
            Image.open(src).convert("RGB").crop(box).save(d / f"{a['id']}.jpg", quality=95)
            counts[(split, cls)] += 1
        except Exception as e:
            print("  skip", src, e)
    for split in ("train", "val"):
        tot = sum(v for (s, _), v in counts.items() if s == split)
        print(f"[data] {split}: {tot} crops")
        for (s, c), v in sorted(counts.items()):
            if s == split:
                print(f"         {c:<24} {v}")
else:
    print(f"[data] reusing {root}")

from ultralytics import YOLO                              # noqa: E402

m = YOLO(weights(MODEL))
m.train(data=str(root), epochs=EPOCHS, imgsz=IMGSZ, batch=64,
        project=str(Path(RUNS_ROOT) / "experiments"), name="cls_ceiling",
        exist_ok=True, device=devices(), seed=C.SEED, workers=6,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=15, fliplr=0.5, flipud=0.0)

res = m.val(data=str(root), split="val", imgsz=IMGSZ)
out = {"exp": "cls_ceiling", "model": MODEL, "pad": PAD, "imgsz": IMGSZ,
       "top1": float(res.top1), "top5": float(res.top5)}
dest = Path(RUNS_ROOT) / "experiments/cls_ceiling/summary.json"
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text(json.dumps(out, indent=2))
print("\n=== CLASSIFICATION CEILING ===")
print(f"  top-1 {res.top1:.4f}   top-5 {res.top5:.4f}")
print("summary ->", dest)
