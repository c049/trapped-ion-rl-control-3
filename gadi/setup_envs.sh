#!/usr/bin/env bash
set -euo pipefail

# Run from repo root on a Gadi login node.
# Customize these env vars before running:
#   PROJECT, PY_MOD, CUDA_MOD, CUDNN_MOD, VENV_ROOT
#   JAX_GPU=1, JAX_PIP_SPEC, JAX_WHL_URL

PROJECT="${PROJECT:-YOUR_PROJECT}"
SCRATCH_DIR="${SCRATCH:-/scratch/${PROJECT}/${USER}}"
VENV_ROOT="${VENV_ROOT:-${SCRATCH_DIR}/qcrl_envs}"

PY_MOD="${PY_MOD:-python3}"
CUDA_MOD="${CUDA_MOD:-}"
CUDNN_MOD="${CUDNN_MOD:-}"

module purge
module load "${PY_MOD}"
if [[ -n "${CUDA_MOD}" ]]; then
  module load "${CUDA_MOD}"
fi
if [[ -n "${CUDNN_MOD}" ]]; then
  module load "${CUDNN_MOD}"
fi

mkdir -p "${VENV_ROOT}"

echo "==> Creating TF env at ${VENV_ROOT}/venv_tf"
python -m venv "${VENV_ROOT}/venv_tf"
source "${VENV_ROOT}/venv_tf/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install "tensorflow==2.15.1" "tf-agents==0.19.0" h5py
python -m pip install -e . --no-build-isolation --no-deps
deactivate

echo "==> Creating dynamiqs/JAX env at ${VENV_ROOT}/venv_dq"
python -m venv "${VENV_ROOT}/venv_dq"
source "${VENV_ROOT}/venv_dq/bin/activate"
python -m pip install --upgrade pip setuptools wheel

if [[ "${JAX_GPU:-0}" == "1" ]]; then
  : "${JAX_PIP_SPEC:=jax[cuda12]}"
  : "${JAX_WHL_URL:=}"
  if [[ -z "${JAX_WHL_URL}" ]]; then
    echo "Set JAX_WHL_URL to the JAX CUDA wheel index URL from JAX docs." >&2
    exit 1
  fi
  python -m pip install "${JAX_PIP_SPEC}" -f "${JAX_WHL_URL}"
else
  python -m pip install "jax==0.4.38" "jaxlib==0.4.38"
fi

python -m pip install "dynamiqs==0.3.4" matplotlib h5py
python -m pip install -e . --no-build-isolation --no-deps
deactivate

echo "==> Done. Envs at ${VENV_ROOT}"
