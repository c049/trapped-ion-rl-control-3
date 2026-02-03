# Gadi run prep (PBS)

This folder contains scripts to prepare environments and run the trapped-ion
cat example on Gadi. It assumes you will run **server + client on the same GPU
node** (localhost socket). Adjust modules/queue/project to your allocation.

## 1) Create environments on a login node

Run from the repo root:

```sh
cd /scratch/<PROJECT>/<USER>/trapped-ion-rl-control-4

# Edit these for your environment:
export PROJECT=<PROJECT>
export PY_MOD=python3/3.11.5
export CUDA_MOD=cuda/12.2
export CUDNN_MOD=cudnn/8.9.7

# If you want GPU JAX, set these:
export JAX_GPU=1
export JAX_PIP_SPEC="jax[cuda12]"
export JAX_WHL_URL="https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"

bash gadi/setup_envs.sh
```

Notes:
- If you are unsure about module names, run `module avail` on Gadi and update
  `PY_MOD`, `CUDA_MOD`, `CUDNN_MOD`.
- If you do not set `JAX_GPU=1`, the script installs CPU-only JAX.

## 2) Submit the job

Edit `gadi/run_job.pbs` and set:
- `PROJECT` (PBS project code)
- `GPU_QUEUE` (your GPU queue name)
- modules (if needed)

Submit:
```sh
qsub gadi/run_job.pbs
```

Logs go to `logs/` under the repo root.
