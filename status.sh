#!/bin/bash
# Simple progress view for the experiment matrix.
cd /home/agipml/samarth.sirsat/DDP/pcb
echo "==================== RUNNING NOW ===================="
squeue -u "$USER" -o "%.9i %.12j %.6P %.9T %.8M" | sed 1d | while read -r id name part st tm; do
  f=$(ls logs/${name}.${id}.out 2>/dev/null | head -1)
  ep=""
  [ -n "$f" ] && ep=$(tr '\r' '\n' < "$f" | sed -e 's/\x1b\[[0-9;]*[A-Za-z]//g' -e 's/^\[K//' | grep -oE '^ *[0-9]+/[0-9]+ ' | tail -1 | tr -d ' ')
  # latest val line for this run
  mp=""
  [ -n "$f" ] && mp=$(tr '\r' '\n' < "$f" | sed -e 's/\x1b\[[0-9;]*[A-Za-z]//g' -e 's/^\[K//' | grep -E '^ *all +[0-9]' | tail -1 | awk '{print "mAP50="$6"  mAP50-95="$7}')
  printf "  %-12s %-4s %-9s %6s  epoch %-9s %s\n" "$name" "$part" "$st" "$tm" "${ep:-—}" "$mp"
done
echo
echo "==================== FINISHED ======================="
python3 - << 'PY'
import json,glob
rows=[]
for s in sorted(glob.glob("runs/experiments/*/summary.json")):
    d=json.load(open(s))
    if "top1" in d: rows.append((d["exp"],"top1=%.4f"%d["top1"],"",""))
    else: rows.append((d["exp"],"mAP50=%.4f"%d["map50"],"mAP50-95=%.4f"%d["map"],"R=%.4f"%d["recall"]))
if not rows: print("  (none yet)")
for r in rows: print("  %-14s %-16s %-18s %s"%r)
PY
echo
echo "==================== STILL QUEUED ==================="
n=$(wc -l < experiments/queue.txt 2>/dev/null || echo 0)
echo "  $n waiting: $(tr '\n' ' ' < experiments/queue.txt | sed 's/|[^ ]*//g')"
