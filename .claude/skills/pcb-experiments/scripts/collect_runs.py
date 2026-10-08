#!/usr/bin/env python3
"""Read-only: walk an Ultralytics runs/ tree and print a markdown comparison table.

Usage:
    python collect_runs.py [RUNS_ROOT]

RUNS_ROOT defaults to $PCB_RUNS_ROOT, then ./runs. Nothing here writes to the runs directory.
cv_fold* subdirectories are aggregated into a single mean +/- std row rather than listed
individually -- see pcb-experiments/references/run-layout.md for why.
"""
import csv
import os
import re
import statistics
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None


def read_args(run_dir: Path) -> dict:
    args_path = run_dir / "args.yaml"
    if not args_path.exists():
        return {}
    if yaml is not None:
        with open(args_path) as f:
            return yaml.safe_load(f) or {}
    # minimal fallback parser if pyyaml isn't in this venv
    out = {}
    for line in args_path.read_text().splitlines():
        if ":" in line and not line.strip().startswith("#"):
            k, _, v = line.partition(":")
            out[k.strip()] = v.strip()
    return out


def read_last_row(run_dir: Path) -> dict:
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return {}
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    return {k.strip(): v.strip() for k, v in rows[-1].items()}


def metric(row: dict, *candidates):
    for c in candidates:
        for k, v in row.items():
            if k == c or k.replace(" ", "") == c:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return None
    return None


def summarize(run_dir: Path) -> dict:
    args = read_args(run_dir)
    row = read_last_row(run_dir)
    return {
        "name": run_dir.name,
        "imgsz": args.get("imgsz"),
        "batch": args.get("batch"),
        "epochs": args.get("epochs"),
        "map50": metric(row, "metrics/mAP50(B)", "metrics/mAP50"),
        "map5095": metric(row, "metrics/mAP50-95(B)", "metrics/mAP50-95"),
        "precision": metric(row, "metrics/precision(B)", "metrics/precision"),
        "recall": metric(row, "metrics/recall(B)", "metrics/recall"),
        "final_epoch": row.get("epoch"),
    }


def fmt(v, nd=3):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def main():
    runs_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        os.environ.get("PCB_RUNS_ROOT", "runs")
    )
    if not runs_root.exists():
        print(f"No such directory: {runs_root}", file=sys.stderr)
        sys.exit(1)

    fold_pat = re.compile(r"^(?P<base>.+?)_?cv_?fold\d+$", re.IGNORECASE)
    singles = []
    fold_groups: dict[str, list[dict]] = {}

    for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        s = summarize(run_dir)
        m = fold_pat.match(run_dir.name)
        if m:
            fold_groups.setdefault(m.group("base"), []).append(s)
        else:
            singles.append(s)

    rows = list(singles)
    for base, folds in fold_groups.items():
        map50s = [f["map50"] for f in folds if f["map50"] is not None]
        map5095s = [f["map5095"] for f in folds if f["map5095"] is not None]
        rows.append({
            "name": f"{base} (n={len(folds)} folds, mean+/-std)",
            "imgsz": folds[0]["imgsz"],
            "batch": folds[0]["batch"],
            "epochs": folds[0]["epochs"],
            "map50": statistics.mean(map50s) if map50s else None,
            "map5095": statistics.mean(map5095s) if map5095s else None,
            "precision": None,
            "recall": None,
            "final_epoch": None,
            "_std50": statistics.pstdev(map50s) if len(map50s) > 1 else 0.0,
            "_std5095": statistics.pstdev(map5095s) if len(map5095s) > 1 else 0.0,
        })

    header = ["run", "imgsz", "batch", "epochs", "mAP50", "mAP50-95", "P", "R", "last_epoch"]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for r in rows:
        map50_str = fmt(r["map50"])
        map5095_str = fmt(r["map5095"])
        if "_std50" in r:
            map50_str = f"{map50_str} +/- {r['_std50']:.3f}"
            map5095_str = f"{map5095_str} +/- {r['_std5095']:.3f}"
        print("| " + " | ".join([
            r["name"], fmt(r["imgsz"], 0), fmt(r["batch"], 0), fmt(r["epochs"], 0),
            map50_str, map5095_str, fmt(r["precision"]), fmt(r["recall"]),
            fmt(r["final_epoch"], 0),
        ]) + " |")

    print("\nBaselines (README): YOLO11s 0.757/0.524 | RT-DETR 0.770/0.556 | "
          "GMO-DETR(12cls) 0.534/0.350")
    print("Noise floor (5-fold CV): mAP50 +/-0.021, mAP50-95 +/-0.020 -- "
          "gaps under ~0.04 are not distinguishable from noise.")


if __name__ == "__main__":
    main()
