#!/bin/bash
# Submit queued experiments onto l40 as submit slots free (l40 MaxSubmit=5).
cd /home/agipml/samarth.sirsat/DDP/pcb
Q=experiments/queue.txt; LOG=experiments/drain.log
while [ -s "$Q" ]; do
  n=$(squeue -h -u "$USER" -p l40 | wc -l)
  if [ "$n" -lt 5 ]; then
    line=$(head -1 "$Q")
    name="${line%%|*}"; vars="${line#*|}"
    [ "$vars" = "$name" ] && vars=""
    ex="EXP_NAME=$name"; [ -n "$vars" ] && ex="$ex,$vars"
    jid=$(sbatch --parsable --job-name="$name" --export=ALL,$ex experiments/run_exp.sbatch 2>&1)
    if [[ "$jid" =~ ^[0-9]+$ ]]; then
      echo "$(date +%H:%M:%S) submitted $name -> $jid [$vars]" >> "$LOG"
      echo "$jid $name" >> experiments/submitted.txt
      sed -i 1d "$Q"
    else
      echo "$(date +%H:%M:%S) retry $name: $jid" >> "$LOG"
    fi
  fi
  sleep 60
done
echo "$(date +%H:%M:%S) queue drained" >> "$LOG"
