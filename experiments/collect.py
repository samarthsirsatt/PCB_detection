"""Aggregate experiment summaries into the comparison tables used in README.md.

Applies the project's noise floor when judging differences: the published 5-fold CV
std on mAP@0.5 is +/-0.021, so anything smaller than that is not a real effect.
"""
import json, statistics as st, sys
from pathlib import Path

RUNS = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/experiments")
NOISE50, NOISE9550 = 0.021, 0.020

def curve_stats(run_dir):
    """Peak metrics + within-run epoch jitter, read from results.csv.

    best.pt is chosen on Ultralytics' blended *fitness*, so the summary number can sit
    below the run's own mAP@0.5 peak. Reporting peak and jitter alongside it keeps a
    comparison from hinging on checkpoint selection.
    """
    import csv
    p = run_dir / "results.csv"
    if not p.exists():
        return {}
    r = list(csv.DictReader(open(p)))
    if not r:
        return {}
    k50 = next((c for c in r[0] if "mAP50(B)" in c), None)
    k95 = next((c for c in r[0] if "mAP50-95(B)" in c), None)
    if not k50:
        return {}
    v50 = [float(x[k50]) for x in r]
    v95 = [float(x[k95]) for x in r] if k95 else []
    tail = v50[-40:] if len(v50) >= 40 else v50
    return {"epochs": len(r), "peak50": max(v50),
            "peak95": max(v95) if v95 else None,
            "jitter": st.stdev(tail) if len(tail) > 1 else 0.0}


rows, cls = {}, None
for s in sorted(RUNS.glob("*/summary.json")):
    d = json.loads(s.read_text())
    if "top1" in d:
        cls = d
    else:
        d.update(curve_stats(s.parent))
        rows[d["exp"]] = d

base = rows.get("base")


def delta(v, b, noise):
    if b is None:
        return ""
    d = v - b
    sig = "significant" if abs(d) > noise else "within noise"
    return f"{d:+.4f} ({sig})"


print("=" * 96)
print("EXPERIMENT RESULTS — 8-class, val split, RT-DETR-l unless noted")
print("=" * 96)
print(f"{'experiment':<12} {'mAP@0.5':>8} {'mAP@.5:.95':>11} {'P':>7} {'R':>7} "
      f"{'peak50':>7} {'ep':>4} {'jit':>6}  {'vs base (mAP@0.5)':<26}")
print("-" * 96)
order = ["base", "neg300", "neg600", "neg900", "neg1195", "flipud0", "nq30",
         "rect", "res960", "res1280"]
for k in order + [k for k in rows if k not in order and not k.startswith("cv")]:
    if k not in rows:
        continue
    r = rows[k]
    b = base["map50"] if base else None
    pk = f"{r['peak50']:.4f}" if r.get("peak50") is not None else "—"
    ep = r.get("epochs", "—")
    ji = f"{r['jitter']:.4f}" if r.get("jitter") is not None else "—"
    print(f"{k:<12} {r['map50']:>8.4f} {r['map']:>11.4f} {r['precision']:>7.4f} "
          f"{r['recall']:>7.4f} {pk:>7} {ep:>4} {ji:>6}  {delta(r['map50'], b, NOISE50):<26}")

if base and base.get("peak50") is not None:
    print("-" * 96)
    print(f"  peak50 = best mAP@0.5 across epochs (best.pt is picked on blended fitness, "
          f"so it can sit lower)")
    print(f"  jit    = std of mAP@0.5 over the last 40 epochs = this run's OWN noise; "
          f"compare deltas against it")

cvs = [rows[k] for k in sorted(rows) if k.startswith("cv")]
if cvs:
    print("-" * 96)
    m50 = [c["map50"] for c in cvs]
    m = [c["map"] for c in cvs]
    p = [c["precision"] for c in cvs]
    r = [c["recall"] for c in cvs]
    sd = st.stdev(m50) if len(m50) > 1 else 0.0
    sd2 = st.stdev(m) if len(m) > 1 else 0.0
    print(f"{'RT-DETR CV':<12} {st.mean(m50):>8.4f} {st.mean(m):>11.4f} "
          f"{st.mean(p):>7.4f} {st.mean(r):>7.4f}   {len(cvs)}-fold")
    print(f"{'  +/- std':<12} {sd:>8.4f} {sd2:>11.4f}")
    print(f"\n  folds mAP@0.5: {['%.4f' % x for x in m50]}")
    print(f"\n  >>> YOLO11s published CV: 0.761 +/- 0.021")
    lo, hi = st.mean(m50) - sd, st.mean(m50) + sd
    print(f"  >>> RT-DETR CV 1-sigma band: [{lo:.4f}, {hi:.4f}]")
    verdict = ("OVERLAPS YOLO -> difference NOT established"
               if lo <= 0.761 + 0.021 and hi >= 0.761 - 0.021 else "SEPARATED from YOLO")
    print(f"  >>> {verdict}")

if cls:
    print("-" * 96)
    print(f"CLASSIFICATION CEILING ({cls['model']}, pad={cls['pad']}, "
          f"imgsz={cls['imgsz']}): top-1 = {cls['top1']:.4f}")
    if base:
        print(f"  detection mAP@0.5 = {base['map50']:.4f}  ->  gap = "
              f"{cls['top1'] - base['map50']:+.4f}")

if base:
    print("-" * 96)
    print("PER-CLASS mAP@0.5:0.95  (box.maps — NOT mAP@0.5)")
    names = list((base.get('per_class_map') or {}).keys())
    hdr = [k for k in order if k in rows]
    print(f"  {'class':<24}" + "".join(f"{h:>10}" for h in hdr))
    for n in names:
        print(f"  {n:<24}" + "".join(
            f"{(rows[h].get('per_class_map') or {}).get(n) or 0:>10.3f}" for h in hdr))
print("=" * 96)
