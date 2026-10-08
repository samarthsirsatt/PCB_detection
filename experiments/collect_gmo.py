"""Aggregate the GMO-DETR experiment summaries against the GMO baseline.

Same noise floor as the RT-DETR study (+/-0.021 on mAP@0.5, from the published
5-fold CV std), so a delta smaller than that is not a real effect.
"""
import csv, json, statistics as st, sys
from pathlib import Path

RUNS = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/experiments")
NOISE50 = 0.021


def curve_stats(run_dir):
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


rows = {}
for s in sorted(RUNS.glob("gmo_*/summary.json")):
    d = json.loads(s.read_text())
    d.update(curve_stats(s.parent))
    rows[d["exp"]] = d

base = rows.get("gmo_base")
b50 = base["map50"] if base else None

print("=" * 104)
print("GMO-DETR EXPERIMENTS — 8-class dataset, same split/aug as the RT-DETR study")
print("=" * 104)
print(f"{'experiment':<16} {'mAP@0.5':>8} {'mAP@.5:.95':>11} {'P':>7} {'R':>7} "
      f"{'peak50':>7} {'ep':>4} {'jit':>6} {'par':>6}  {'vs gmo_base':<24}")
print("-" * 104)
order = ["gmo_base", "gmo_nq30", "gmo_res1280", "gmo_flipud0", "gmo_neg300", "gmo_p2"]
for k in order + [k for k in rows if k not in order]:
    if k not in rows:
        continue
    r = rows[k]
    d = ""
    if b50 is not None:
        dv = r["map50"] - b50
        d = f"{dv:+.4f} ({'significant' if abs(dv) > NOISE50 else 'within noise'})"
    pk = f"{r['peak50']:.4f}" if r.get("peak50") is not None else "-"
    ji = f"{r['jitter']:.4f}" if r.get("jitter") is not None else "-"
    par = r.get("config", {}).get("params_m", "-")
    print(f"{k:<16} {r['map50']:>8.4f} {r['map']:>11.4f} {r['precision']:>7.4f} "
          f"{r['recall']:>7.4f} {pk:>7} {r.get('epochs','-'):>4} {ji:>6} {par:>6}  {d:<24}")

if rows:
    print("-" * 104)
    print("PER-CLASS AP@0.5")
    hdr = [k for k in order if k in rows] + [k for k in rows if k not in order]
    names = list((rows[hdr[0]].get("per_class_ap50") or {}).keys())
    print(f"  {'class':<24}" + "".join(f"{h.replace('gmo_',''):>12}" for h in hdr))
    for n in names:
        print(f"  {n:<24}" + "".join(
            f"{((rows[h].get('per_class_ap50') or {}).get(n) or 0):>12.3f}" for h in hdr))
print("=" * 104)
