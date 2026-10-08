"""Find train/val leakage in the standard 80/20 split (scratch/exp_data/neg120) used by base, nq30 and most runs.
Hashes every image in images/train and images/val (16x16 difference hash), then for each val image finds the
closest train image. Writes ddp1_audit/val_leak_status.csv and ddp1_audit/val_{exact,near}_leak_free.txt.
Read-only on data; run with the project venv python (needs cv2, numpy)."""
import csv, cv2, numpy as np
from pathlib import Path
D = Path("scratch/exp_data/neg120/images"); OUT = Path("ddp1_audit")
def dh(p):
    im = cv2.resize(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE), (17, 16), interpolation=cv2.INTER_AREA)
    return (im[:, 1:] > im[:, :-1]).flatten()
def cls(n): return n.split("__")[1] if n.startswith("categories__") else "NEGATIVE"
tr = sorted((D/"train").iterdir()); va = sorted((D/"val").iterdir())
T = np.stack([dh(p) for p in tr]); rows = []
for p in va:
    d = (T != dh(p)).sum(1); j = int(d.argmin())
    rows.append(dict(val_image=p.name, val_class=cls(p.name), min_dist=int(d[j]), nearest_train=tr[j].name,
                     train_class=cls(tr[j].name), class_conflict=int(cls(p.name) != cls(tr[j].name) and d[j] <= 10)))
with open(OUT/"val_leak_status.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for tag, thr in (("exact", 0), ("near", 10)):
    keep = [r["val_image"] for r in rows if r["min_dist"] > thr]
    (OUT/f"val_{tag}_leak_free.txt").write_text("\n".join(keep) + "\n")
n = len(rows); ex = [r for r in rows if r["min_dist"] == 0]; ne = [r for r in rows if r["min_dist"] <= 10]
print(f"val images: {n} | train images: {len(tr)}")
print(f"exact copy in train (dist 0):      {len(ex)}  ({len(ex)/n:.0%})")
print(f"near/exact copy in train (<=10):   {len(ne)}  ({len(ne)/n:.0%})")
pos = [r for r in rows if r["val_class"] != "NEGATIVE"]; pn = [r for r in pos if r["min_dist"] <= 10]
print(f"defect val images: {len(pos)} | with copy in train: {len(pn)} ({len(pn)/len(pos):.0%})")
print(f"negative val images with copy in train: {sum(1 for r in rows if r['val_class']=='NEGATIVE' and r['min_dist']<=10)}")
print("by class (defect val images with a copy in train / total):")
from collections import Counter
a = Counter(r["val_class"] for r in pn); b = Counter(r["val_class"] for r in pos)
for c in sorted(b): print(f"   {c:22s} {a[c]:2d}/{b[c]}")
cc = [r for r in rows if r["class_conflict"]]
print(f"\nLABEL CONFLICTS (copy in train filed under a DIFFERENT class): {len(cc)}")
for r in cc: print("   VAL", r["val_class"], "|", r["val_image"][-45:], " <->  TRAIN", r["train_class"], "|", r["nearest_train"][-40:], "dist", r["min_dist"])
