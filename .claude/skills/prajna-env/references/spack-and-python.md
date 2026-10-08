# Spack and the Python environment

## Activating Spack

Every shell that needs Spack packages runs this first (also goes at the top of every
`.sbatch` script — see `prajna-jobs/templates/gpu_job.sbatch`):

```bash
source /lustre-flash/apps/spack/share/spack/setup-env.sh
```

## Useful Spack commands (all from the manual, verified against it)

```bash
spack find                          # installed packages
spack find --loaded                 # currently loaded packages/compilers
spack load <name>@<version>         # load a package into the environment
spack compilers                     # available compiler toolchains
spack list <query>                  # search available packages (wildcards auto-added)
spack install <spec>                # e.g. spack install gromacs@2020.5 +cuda~mpi+blas %intel ^intel-mkl
spack uninstall <spec>
spack env create -d ./my_env <name> # named environment for grouping installs
spack env activate -p <name>        # -p shows the active env in the prompt
spack env deactivate
spack env list
```

Operators: `%` selects a compiler, `^` selects a dependency variant/provider, `@` pins a
version, `+`/`~` enable/disable a build variant.

## The Python decision

**Default: `spack load python@X` (or system `python3` if Spack has nothing suitable) + a
project-local venv + pip wheels for torch/ultralytics/einops/etc.**

Why this over `spack install py-torch`:
- A Spack-built `py-torch` compiles from source against whatever CUDA/cuDNN Spack resolves —
  this can take hours and locks you into a specific Spack environment for every future job.
- A pip wheel (`torch==X.Y.Z+cuXXX`) ships its own CUDA runtime statically, installs in under a
  minute, and is exactly what Ultralytics is tested against upstream.
- This project has no lockfile today (confirmed: no requirements.txt, environment.yml, or
  setup.py anywhere in the repo) — building one via `pip freeze` after this setup is more
  useful than a Spack environment spec that only this cluster can reproduce.

**Deviate when `spack find py-torch` already shows a build matching your target GPU's
driver/CUDA version.** In that case, loading it beats downloading a ~2.5GB wheel — especially
if the day-one egress probe found `pypi.org`/`files.pythonhosted.org` unreachable. Record which
path you took and why in `PRAJNA_CONFIG.md` §5.

## Building the venv

```bash
source /lustre-flash/apps/spack/share/spack/setup-env.sh
spack load python@<version>          # from config §4
python -m venv "$PROJECT_ROOT/env/pcb"
source "$PROJECT_ROOT/env/pcb/bin/activate"
```

Install path depends on the offline decision (see `offline-staging.md`):
```bash
# online:
pip install ultralytics einops grad-cam
# offline:
pip install --no-index --find-links "$PROJECT_ROOT/offline/wheels" ultralytics einops grad-cam
```

Either way, freeze it — this project has never had a reproducibility artifact before:
```bash
pip freeze > "$PROJECT_ROOT/offline/requirements.lock"
```

Record the interpreter source, torch version, and CUDA build in `PRAJNA_CONFIG.md` §5.
