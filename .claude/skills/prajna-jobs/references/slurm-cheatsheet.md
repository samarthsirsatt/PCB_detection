# SLURM command cheatsheet

Verified against `Prajna_AIML_HPC_User_Manual.md`.

## Submitting

```bash
sbatch job.sbatch                       # submit a batch script
sbatch --array=0-4 array_job.sbatch      # job array (or set --array inside the script)
sbatch --array=0-15%4 job.sbatch         # cap concurrent array tasks at 4
sbatch -d singleton job.sbatch           # wait for any previous job with the same name
sbatch -d afterok:<jobid> job.sbatch     # wait for a specific job to finish successfully
sbatch --test-only job.sbatch            # validate + estimate start time, submits nothing
```

## Interactive allocation

```bash
salloc --partition=l40 --qos=l40 --gres=gpu:1 --time=01:00:00
# then, once granted:
srun --pty bash
```
Never run training directly in a plain SSH shell on the login node — only inside an
`salloc`/`srun` allocation or an `sbatch` job.

## Monitoring

```bash
squeue --me                             # your jobs
squeue -p l40                           # everything queued on a partition
squeue --start -j <id>                  # estimated start time for a pending job
sacct -j <id> --format=JobID,State,Elapsed,MaxRSS,ReqTRES,ExitCode
scontrol show job <id>                  # full detail, including Reason=
sprio -j <id>                           # priority factor breakdown
```

## Controlling

```bash
scancel <id>                            # cancel a job
scancel --me                            # cancel everything you own (careful)
scontrol hold <id>                      # pause a pending job
scontrol release <id>                   # resume it
scontrol update jobid=<id> set TimeLimit=4-00:00:00   # change a submitted job's time limit
```

## Cluster/node info

```bash
sinfo -o "%20P %5a %10l %10G %D %N"     # partitions, availability, time limit, GRES, nodes
scontrol show node <name>               # detailed node info (CPU, mem, GRES, TmpDisk)
scontrol show partition <name>          # AllowQos, MaxTime, DefaultTime for a partition
sacctmgr show assoc user=$USER format=partition,qos,maxjobs,maxwall,maxsubmit
```
