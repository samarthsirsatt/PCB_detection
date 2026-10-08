"""Re-evaluate trained checkpoints on the SAME val split, with and without the images that also appear in train.
Inputs: ddp1_audit/val_exact_leak_free.txt, val_near_leak_free.txt (from leak_scan.py). No training; weights unchanged."""
import json, os, shutil
from pathlib import Path
import yaml
from ultralytics import RTDETR
PCB = Path(os.environ.get("PROJECT_ROOT", "/home/agipml/samarth.sirsat/DDP/pcb"))
SRC = PCB / "scratch/exp_data/neg120"; WORK = PCB / "scratch/ddp1_leak_check"; OUT = PCB / "ddp1_audit"
NAMES = ["Component Liftup", "Component Missing", "Component No Solder", "Component Solder Dry",
         "Polarity Wrong", "RYB Wrong Sequence", "Solder Ball", "Solder Short"]
sets = {"full_val": [p.name for p in (SRC/"images/val").iterdir()],
        "minus_exact_copies": (OUT/"val_exact_leak_free.txt").read_text().split(),
        "minus_near_copies": (OUT/"val_near_leak_free.txt").read_text().split()}
def build(tag, files):
    d = WORK / tag; shutil.rmtree(d, ignore_errors=True)
    (d/"images/val").mkdir(parents=True); (d/"labels/val").mkdir(parents=True)
    nb = 0
    for f in files:
        os.symlink(SRC/"images/val"/f, d/"images/val"/f)
        lab = SRC/"labels/val"/(Path(f).stem + ".txt"); shutil.copy(lab, d/"labels/val"/lab.name)
        nb += sum(1 for l in lab.read_text().split("\n") if l.strip())
    (d/"data.yaml").write_text(yaml.safe_dump({"path": str(d), "train": "images/val", "val": "images/val",
                                              "names": dict(enumerate(NAMES))}, sort_keys=False))
    return d, nb
models = {"nq30": PCB/"runs/experiments/nq30/weights/best.pt", "nq30_s1": PCB/"runs/experiments/nq30_s1/weights/best.pt",
          "base": PCB/"runs/experiments/base/weights/best.pt"}
res = []
for mname, w in models.items():
    m = RTDETR(str(w))
    for tag, files in sets.items():
        d, nb = build(tag, files)
        r = m.val(data=str(d/"data.yaml"), split="val", imgsz=640, device=0, plots=False, verbose=False,
                  project=str(WORK/"runs"), name=f"{mname}_{tag}", exist_ok=True)
        res.append(dict(model=mname, subset=tag, images=len(files), boxes=nb, map50=float(r.box.map50),
                        map5095=float(r.box.map), precision=float(r.box.mp), recall=float(r.box.mr)))
        print(res[-1], flush=True)
(OUT/"leak_check_results.json").write_text(json.dumps(res, indent=1))
