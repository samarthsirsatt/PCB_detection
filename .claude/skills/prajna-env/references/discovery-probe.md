# Day-one discovery probe

All commands below are read-only. Run them individually, or run `../scripts/probe.sh` which
runs all of them and prints a labelled report. Either way, write results into
`PRAJNA_CONFIG.md`, replacing the relevant `[TODO(day-one)]` marker and stamping
`[CONFIRMED <date>]`.

| command | tells you | fills config section |
|---|---|---|
| `whoami` | your username | §1 `USER` |
| `id -gn` | your primary group | §1 `GROUP` |
| `echo $HOME` | your home root | §1 `HOME_ROOT` |
| `sinfo -o "%20P %5a %10l %10G %D %N"` | every partition name, availability, time limit, GRES string, node count | §3 (fills the two `?` rows — DGX A100 and L4 partition names) |
| `scontrol show partition <p>` for each partition found above, `\| grep -E "AllowQos\|MaxTime\|DefaultTime"` | confirms the QOS name matches the partition name (the invariant) for partitions not already confirmed | §3 |
| `sacctmgr show assoc user=$USER format=partition,qos,maxjobs,maxwall,maxsubmit` | your actual per-user limits — these can be tighter than the partition's advertised max | §3 |
| `nvidia-smi -L` | GPU model/count actually present — **run this via `srun --partition=<p> --gres=gpu:1 --pty nvidia-smi -L`, not on the login node**, which may have no GPU or a different one | §3 sanity check |
| `lfs quota -h -u $USER /home` (or `quota -s` if `lfs` unavailable) | your `/home` quota and current usage | §2 |
| `df -h /scratch` | `/scratch` capacity | §2 |
| `spack find python` | available Python versions via Spack | §4, §5 |
| `spack find py-torch` | whether a pre-built torch already exists (see `spack-and-python.md` for the decision this feeds) | §4, §5 |
| `spack find cuda cudnn` | CUDA/cuDNN versions Spack can provide | §4 |
| `curl -sI --max-time 8 https://pypi.org/simple/` | pip index reachability | §6 |
| `curl -sI --max-time 8 https://files.pythonhosted.org/` | pip wheel download reachability | §6 |
| `curl -sI --max-time 8 https://github.com/` | git/GitHub page reachability | §6 |
| `curl -sI --max-time 8 https://objects.githubusercontent.com/` | Ultralytics release-asset reachability — **this can differ from plain `github.com`**; both must be tested separately | §6 |
| `python3 -c "import sys; print(sys.version)"` | system Python fallback if not using Spack's | §5 |

## Reading the egress results

- All four `curl` calls succeed (HTTP response, not a timeout) → cluster has open egress.
  Still stage offline per `references/offline-staging.md` — reachability today doesn't
  guarantee reachability when an unattended job runs later.
- `github.com` succeeds but `objects.githubusercontent.com` times out → GitHub's release CDN is
  blocked even though the site isn't. This specifically breaks bare-name Ultralytics weight
  downloads (`YOLO("yolo11s.pt")`) even though `git clone` of a repo might still work. Record
  both results distinctly in config §6, don't collapse them into one "internet: partial" line.
- Everything times out → fully offline except the Anthropic API (which must work, since Claude
  Code is running). Offline staging is mandatory, not a precaution.

## After the probe

If any discovered partition name, QOS, or limit contradicts what's already in
`PRAJNA_CONFIG.md` §3 (`[FROM-MANUAL]` rows), the probe result wins — update the table and note
the discrepancy in §10.
