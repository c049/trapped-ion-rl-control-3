#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-mu61}"
PROJECT_DIR="${PROJECT_DIR:-/scratch/${PROJECT}/${USER}/quantum_control_rl_server}"
BINOMIAL_DIR="${PROJECT_DIR}/examples/trapped_ion_binomial"
SWEEP_MODE="${SWEEP_MODE:-quasi_static}"  # quasi_static | stochastic
CHAIN_JOBS="${CHAIN_JOBS:-0}"             # 1 = serial dependency chain, 0 = submit in parallel
PENALTIES_CSV="${PENALTIES_CSV:-}"
SWEEP_ID="${SWEEP_ID:-${SWEEP_MODE}_penalty_sweep_$(date +%Y%m%d_%H%M%S)}"
SWEEP_ROOT="${SWEEP_ROOT:-${BINOMIAL_DIR}/penalty_sweep/${SWEEP_ID}}"
BASELINE_NPZ="${BASELINE_NPZ:-${BINOMIAL_DIR}/checkpoint/nonrobust_baseline_pulses.npz}"
DEPHASE_GAMMA="${DEPHASE_GAMMA:-18.0}"

if [[ "${SWEEP_MODE}" != "quasi_static" && "${SWEEP_MODE}" != "stochastic" ]]; then
  echo "Unsupported SWEEP_MODE=${SWEEP_MODE}. Use quasi_static or stochastic." >&2
  exit 1
fi
if [[ -z "${PENALTIES_CSV}" ]]; then
  if [[ "${SWEEP_MODE}" == "stochastic" ]]; then
    PENALTIES_CSV="0.2,1.0"
  else
    PENALTIES_CSV="0,0.5,1,2,4"
  fi
fi

if [[ ! -d "${PROJECT_DIR}" ]]; then
  echo "PROJECT_DIR not found: ${PROJECT_DIR}" >&2
  exit 1
fi
if [[ ! -f "${BASELINE_NPZ}" ]]; then
  echo "BASELINE_NPZ not found: ${BASELINE_NPZ}" >&2
  exit 1
fi

mkdir -p "${SWEEP_ROOT}"

IFS=',' read -r -a PENALTIES <<< "${PENALTIES_CSV}"
if [[ "${#PENALTIES[@]}" -eq 0 ]]; then
  echo "No penalties parsed from PENALTIES_CSV=${PENALTIES_CSV}" >&2
  exit 1
fi

echo "Submitting binomial robust penalty sweep"
echo "  Project dir: ${PROJECT_DIR}"
echo "  Sweep root : ${SWEEP_ROOT}"
echo "  Mode       : ${SWEEP_MODE}"
echo "  Chain jobs : ${CHAIN_JOBS}"
echo "  Penalties  : ${PENALTIES_CSV}"
echo "  Baseline   : ${BASELINE_NPZ}"

prev_job=""
job_ids=()

for raw_p in "${PENALTIES[@]}"; do
  p="$(echo "${raw_p}" | xargs)"
  if [[ "${SWEEP_MODE}" == "stochastic" ]]; then
    tag="stoch_p_${p//./p}"
    dephase_model="stochastic"
    detuning_weighting="uniform"
    force_gaussian_weighting="0"
  else
    tag="p_${p//./p}"
    dephase_model="quasi_static"
    detuning_weighting="gaussian"
    force_gaussian_weighting="1"
  fi
  deps=()
  if [[ "${CHAIN_JOBS}" == "1" && -n "${prev_job}" ]]; then
    deps=(-W "depend=afterok:${prev_job}")
  fi
  jid="$(
    qsub "${deps[@]}" \
      -v "PROJECT=${PROJECT},PROJECT_DIR=${PROJECT_DIR},SWEEP_ROOT=${SWEEP_ROOT},SWEEP_TAG=${tag},ROBUST_TRAINING=1,DEPHASE_MODEL=${dephase_model},DEPHASE_DETUNING_FRAC=0.05,DEPHASE_NOISE_SAMPLES_TRAIN=7,DEPHASE_NOISE_SAMPLES_EVAL=13,DEPHASE_NOISE_SAMPLES_REFINE=17,DEPHASE_INCLUDE_NOMINAL=1,DEPHASE_OBJECTIVE_INCLUDE_NOMINAL=1,DEPHASE_DETUNING_WEIGHTING=${detuning_weighting},DEPHASE_QUASI_SAMPLER=grid,DEPHASE_GAUSSIAN_SIGMA_FRAC=0.50,DEPHASE_STOCHASTIC_STD_MODE=gamma_dt,DEPHASE_GAMMA=${DEPHASE_GAMMA},DEPHASE_STOCHASTIC_CORRELATION=segment,DEPHASE_STOCHASTIC_CLIP_STD=3.0,DEPHASE_STOCHASTIC_SHARED_ACROSS_BATCH=1,OMEGA_RABI_HZ=2000.0,T_STEP=1.0e-5,LEARN_DURATION_SCALE=1,DURATION_INIT_SCALE=1.0,DURATION_ACTION_SCALE=0.25,DURATION_SCALE_PENALTY_LAMBDA=${DURATION_SCALE_PENALTY_LAMBDA:-0.05},DURATION_SCALE_TARGET=1.0,DURATION_SCALE_ONLY_ABOVE_TARGET=1,AMP_OVERSHOOT_LAMBDA=0.01,AMP_OVERSHOOT_BASE=1.0,STRICT_TEACHER_ALIGNMENT=1,FORCE_GAUSSIAN_WEIGHTING=${force_gaussian_weighting},FORCE_GRID_QUASI_SAMPLER=1,FORCE_STOCHASTIC_GAMMA_DT=1,OMEGA_TSTEP_SCALE_REF_HZ=2000.0,OMEGA_TSTEP_SCALE_REF_TSTEP=1.0e-5,OMEGA_TSTEP_SCALE_TOL=0.20,AUTO_ADJUST_QUASI_GRID_ODD_SAMPLES=1,ROBUST_NOMINAL_FID_FLOOR=0.985,ROBUST_FLOOR_PENALTY=${p},ROBUST_COMPARE_BASELINE_NPZ=${BASELINE_NPZ},ROBUST_COMPARE_BASELINE_DURATION_SCALE=${ROBUST_COMPARE_BASELINE_DURATION_SCALE:-},FINAL_VALIDATE_ENABLE=${FINAL_VALIDATE_ENABLE:-1},FINAL_VALIDATE_NUM_SEEDS=${FINAL_VALIDATE_NUM_SEEDS:-5},FINAL_VALIDATE_NOISE_SAMPLES=${FINAL_VALIDATE_NOISE_SAMPLES:-65},FINAL_VALIDATE_SEED_BASE=${FINAL_VALIDATE_SEED_BASE:-7100001},FINAL_VALIDATE_USE_SCORE_GUARD=${FINAL_VALIDATE_USE_SCORE_GUARD:-1}" \
      "${PROJECT_DIR}/gadi/run_job_binomial_penalty_sweep.pbs"
  )"
  echo "  penalty=${p} tag=${tag} -> ${jid}"
  if [[ "${CHAIN_JOBS}" == "1" ]]; then
    prev_job="${jid}"
  fi
  job_ids+=("${jid}")
done

jobs_txt="${SWEEP_ROOT}/submitted_jobs.txt"
{
  echo "SWEEP_ROOT=${SWEEP_ROOT}"
  echo "MODE=${SWEEP_MODE}"
  echo "CHAIN_JOBS=${CHAIN_JOBS}"
  echo "PENALTIES=${PENALTIES_CSV}"
  echo "BASELINE_NPZ=${BASELINE_NPZ}"
  for jid in "${job_ids[@]}"; do
    echo "${jid}"
  done
} > "${jobs_txt}"

cat <<EOF

Submitted ${#job_ids[@]} jobs.
Job list saved to: ${jobs_txt}

When all jobs finish, aggregate with:
  source /scratch/${PROJECT}/${USER}/qcrl_envs/venv_dq/bin/activate
  python ${PROJECT_DIR}/gadi/aggregate_binomial_penalty_sweep.py --sweep-root ${SWEEP_ROOT}

Monitor queue:
  qstat -u ${USER}
EOF
