import logging
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

if os.environ.get("DQ_FORCE_GPU", "0") == "1":
    os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")

from quantum_control_rl_server.remote_env_tools import Client
from trapped_ion_binomial_sim_function import (
    trapped_ion_binomial_sim,
    trapped_ion_binomial_sim_batch,
    characteristic_grid,
    prepare_characteristic_distribution,
    characteristic_norm,
    binomial_target_fock_statistics,
)

logger = logging.getLogger("RL")
logger.propagate = False
logger.handlers = []
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)

client_socket = Client()
(host, port) = (os.environ.get("HOST", "127.0.0.1"), int(os.environ.get("PORT", "5555")))
client_socket.connect((host, port))

FAST_SMOKE = os.environ.get("FAST_SMOKE", "0") == "1"

N_STEPS = int(os.environ.get("N_STEPS", "120"))
N_SEGMENTS = int(os.environ.get("N_SEGMENTS", "60"))
if N_STEPS <= 0 or N_SEGMENTS <= 0:
    raise ValueError("N_STEPS and N_SEGMENTS must both be positive.")
if N_STEPS % N_SEGMENTS != 0:
    raise ValueError(
        f"N_STEPS ({N_STEPS}) must be divisible by N_SEGMENTS ({N_SEGMENTS})."
    )
SEG_LEN = N_STEPS // N_SEGMENTS
T_STEP = float(os.environ.get("T_STEP", "1.0e-5"))
if not np.isfinite(T_STEP) or T_STEP <= 0.0:
    raise ValueError(f"T_STEP must be > 0 and finite, got {T_STEP}")
OMEGA_RABI_HZ = float(os.environ.get("OMEGA_RABI_HZ", "2000.0"))
if not np.isfinite(OMEGA_RABI_HZ) or OMEGA_RABI_HZ <= 0.0:
    raise ValueError(f"OMEGA_RABI_HZ must be > 0 and finite, got {OMEGA_RABI_HZ}")
OMEGA_RABI = 2 * np.pi * OMEGA_RABI_HZ
SAMPLE_EXTENT = 4.0
PLOT_EXTENT = float(os.environ.get("PLOT_EXTENT", "6.0"))
N_BOSON = int(os.environ.get("N_BOSON", "30"))
_binomial_code_env = os.environ.get("BINOMIAL_CODE", "d3_z").strip()
BINOMIAL_CODE = _binomial_code_env if _binomial_code_env else "d3_z"
_binomial_phase_env = os.environ.get("BINOMIAL_REL_PHASE", "").strip()
BINOMIAL_REL_PHASE = None if _binomial_phase_env == "" else float(_binomial_phase_env)
BINOMIAL_TAIL_WARN = float(os.environ.get("BINOMIAL_TARGET_TAIL_WARN", "1.0e-3"))
BINOMIAL_TAIL_ERROR = float(os.environ.get("BINOMIAL_TARGET_TAIL_ERROR", "5.0e-3"))
ALLOW_LOW_N_BOSON = os.environ.get("ALLOW_LOW_N_BOSON", "0") == "1"

TRAIN_POINTS_STAGE1 = int(os.environ.get("TRAIN_POINTS_STAGE1", "120"))
TRAIN_POINTS_STAGE2 = int(os.environ.get("TRAIN_POINTS_STAGE2", "240"))
TRAIN_POINTS_STAGE3 = int(os.environ.get("TRAIN_POINTS_STAGE3", "960"))
TRAIN_STAGE1_EPOCHS = int(os.environ.get("TRAIN_STAGE1_EPOCHS", "120"))
TRAIN_STAGE2_EPOCHS = int(os.environ.get("TRAIN_STAGE2_EPOCHS", "240"))

CHAR_GRID_SIZE = 61
FINAL_GRID_SIZE = 61
PLOT_GRID_SIZE = 121

if FAST_SMOKE:
    N_BOSON = 12
    TRAIN_POINTS_STAGE1 = 20
    TRAIN_POINTS_STAGE2 = 40
    TRAIN_POINTS_STAGE3 = 60
    TRAIN_STAGE1_EPOCHS = 2
    TRAIN_STAGE2_EPOCHS = 4
    CHAR_GRID_SIZE = 21
    FINAL_GRID_SIZE = 31
    PLOT_GRID_SIZE = 61

if N_BOSON <= 2:
    raise ValueError(f"N_BOSON must be > 2, got {N_BOSON}")

BINOMIAL_STATS = binomial_target_fock_statistics(
    BINOMIAL_CODE,
    N_BOSON,
    rel_phase=BINOMIAL_REL_PHASE,
)
logger.info(
    "Binomial target: code=%s rel_phase=%s mean_n=%.4f tail(n>=%d)=%.3e edge_prob=%.3e",
    BINOMIAL_CODE,
    "none" if BINOMIAL_REL_PHASE is None else f"{BINOMIAL_REL_PHASE:.3f}",
    BINOMIAL_STATS["mean_n"],
    BINOMIAL_STATS["tail_start"],
    BINOMIAL_STATS["tail_mass"],
    BINOMIAL_STATS["edge_prob"],
)
if BINOMIAL_STATS["tail_mass"] > BINOMIAL_TAIL_WARN:
    logger.warning(
        "Binomial truncation warning: tail mass %.3e above threshold %.3e",
        BINOMIAL_STATS["tail_mass"],
        BINOMIAL_TAIL_WARN,
    )
if BINOMIAL_STATS["tail_mass"] > BINOMIAL_TAIL_ERROR and not ALLOW_LOW_N_BOSON:
    raise ValueError(
        "N_BOSON appears too small for the selected binomial target "
        f"(tail mass {BINOMIAL_STATS['tail_mass']:.3e} > {BINOMIAL_TAIL_ERROR:.3e}). "
        "Increase N_BOSON or set ALLOW_LOW_N_BOSON=1 to override."
    )

SMOOTH_LAMBDA = 0.0
SMOOTH_PHI_WEIGHT = 1.0
SMOOTH_AMP_WEIGHT = 0.2
AMP_OVERSHOOT_LAMBDA = float(os.environ.get("AMP_OVERSHOOT_LAMBDA", "0.0"))
AMP_OVERSHOOT_BASE = float(os.environ.get("AMP_OVERSHOOT_BASE", "1.0"))
DURATION_SCALE_PENALTY_LAMBDA = float(
    os.environ.get("DURATION_SCALE_PENALTY_LAMBDA", "0.0")
)
DURATION_SCALE_TARGET = float(os.environ.get("DURATION_SCALE_TARGET", "1.0"))
DURATION_SCALE_ONLY_ABOVE_TARGET = (
    os.environ.get("DURATION_SCALE_ONLY_ABOVE_TARGET", "1") == "1"
)
REWARD_SCALE = 1.0
REWARD_CLIP = None
CHAR_REWARD_OBJECTIVE = os.environ.get("CHAR_REWARD_OBJECTIVE", "overlap_real").lower()
CHAR_REWARD_OBJECTIVE_STAGE2 = os.environ.get(
    "CHAR_REWARD_OBJECTIVE_STAGE2", ""
).strip().lower()
CHAR_REWARD_SWITCH_EPOCH = int(os.environ.get("CHAR_REWARD_SWITCH_EPOCH", "-1"))
if CHAR_REWARD_OBJECTIVE_STAGE2 == "":
    CHAR_REWARD_OBJECTIVE_STAGE2 = CHAR_REWARD_OBJECTIVE
_valid_char_objectives = {"overlap_real", "overlap_abs", "nmse", "nmse_exp"}
if CHAR_REWARD_OBJECTIVE not in _valid_char_objectives:
    raise ValueError(
        f"Unsupported CHAR_REWARD_OBJECTIVE={CHAR_REWARD_OBJECTIVE}, "
        f"expected one of {sorted(_valid_char_objectives)}"
    )
if CHAR_REWARD_OBJECTIVE_STAGE2 not in _valid_char_objectives:
    raise ValueError(
        f"Unsupported CHAR_REWARD_OBJECTIVE_STAGE2={CHAR_REWARD_OBJECTIVE_STAGE2}, "
        f"expected one of {sorted(_valid_char_objectives)}"
    )
if CHAR_REWARD_SWITCH_EPOCH < -1:
    raise ValueError(
        f"CHAR_REWARD_SWITCH_EPOCH must be >= -1, got {CHAR_REWARD_SWITCH_EPOCH}"
    )
CHAR_USE_FIXED_REWARD_NORM = os.environ.get("CHAR_USE_FIXED_REWARD_NORM", "0") == "1"
CHAR_REWARD_SWITCH_MIN_BEST_EVAL = float(
    os.environ.get("CHAR_REWARD_SWITCH_MIN_BEST_EVAL", "-1.0")
)
CHAR_REWARD_STAGE2_PATIENCE_EVAL = int(
    os.environ.get("CHAR_REWARD_STAGE2_PATIENCE_EVAL", "12")
)
CHAR_REWARD_STAGE2_MIN_GAIN = float(
    os.environ.get("CHAR_REWARD_STAGE2_MIN_GAIN", "0.01")
)
CHAR_REWARD_STAGE2_ALLOW_REVERT = (
    os.environ.get("CHAR_REWARD_STAGE2_ALLOW_REVERT", "1") == "1"
)
CHAR_REWARD_AUTO_RESCALE = os.environ.get("CHAR_REWARD_AUTO_RESCALE", "1") == "1"
CHAR_REWARD_AUTO_RESCALE_TARGET_P90 = float(
    os.environ.get("CHAR_REWARD_AUTO_RESCALE_TARGET_P90", "1.0")
)
CHAR_REWARD_AUTO_RESCALE_TRIGGER_P90 = float(
    os.environ.get("CHAR_REWARD_AUTO_RESCALE_TRIGGER_P90", "3.0")
)
EVAL_INTERVAL_HINT = int(os.environ.get("EVAL_INTERVAL", "10"))
if EVAL_INTERVAL_HINT <= 0:
    raise ValueError(f"EVAL_INTERVAL must be > 0, got {EVAL_INTERVAL_HINT}")
NUM_EPOCHS_HINT = int(os.environ.get("NUM_EPOCHS", "2000"))
if NUM_EPOCHS_HINT <= 0:
    raise ValueError(f"NUM_EPOCHS must be > 0, got {NUM_EPOCHS_HINT}")
if CHAR_REWARD_STAGE2_PATIENCE_EVAL < 0:
    raise ValueError(
        f"CHAR_REWARD_STAGE2_PATIENCE_EVAL must be >= 0, got {CHAR_REWARD_STAGE2_PATIENCE_EVAL}"
    )
if not np.isfinite(CHAR_REWARD_STAGE2_MIN_GAIN) or CHAR_REWARD_STAGE2_MIN_GAIN < 0.0:
    raise ValueError(
        f"CHAR_REWARD_STAGE2_MIN_GAIN must be finite and >= 0, got {CHAR_REWARD_STAGE2_MIN_GAIN}"
    )
if (
    not np.isfinite(CHAR_REWARD_AUTO_RESCALE_TARGET_P90)
    or CHAR_REWARD_AUTO_RESCALE_TARGET_P90 <= 0.0
):
    raise ValueError(
        "CHAR_REWARD_AUTO_RESCALE_TARGET_P90 must be finite and > 0, "
        f"got {CHAR_REWARD_AUTO_RESCALE_TARGET_P90}"
    )
if (
    not np.isfinite(CHAR_REWARD_AUTO_RESCALE_TRIGGER_P90)
    or CHAR_REWARD_AUTO_RESCALE_TRIGGER_P90 <= 0.0
):
    raise ValueError(
        "CHAR_REWARD_AUTO_RESCALE_TRIGGER_P90 must be finite and > 0, "
        f"got {CHAR_REWARD_AUTO_RESCALE_TRIGGER_P90}"
    )
if not np.isfinite(AMP_OVERSHOOT_LAMBDA) or AMP_OVERSHOOT_LAMBDA < 0.0:
    raise ValueError(
        f"AMP_OVERSHOOT_LAMBDA must be finite and >= 0, got {AMP_OVERSHOOT_LAMBDA}"
    )
if not np.isfinite(AMP_OVERSHOOT_BASE) or AMP_OVERSHOOT_BASE <= 0.0:
    raise ValueError(
        f"AMP_OVERSHOOT_BASE must be finite and > 0, got {AMP_OVERSHOOT_BASE}"
    )
if (
    not np.isfinite(DURATION_SCALE_PENALTY_LAMBDA)
    or DURATION_SCALE_PENALTY_LAMBDA < 0.0
):
    raise ValueError(
        "DURATION_SCALE_PENALTY_LAMBDA must be finite and >= 0, "
        f"got {DURATION_SCALE_PENALTY_LAMBDA}"
    )
if not np.isfinite(DURATION_SCALE_TARGET) or DURATION_SCALE_TARGET <= 0.0:
    raise ValueError(
        f"DURATION_SCALE_TARGET must be finite and > 0, got {DURATION_SCALE_TARGET}"
    )

N_SHOTS_TRAIN = 0
N_SHOTS_EVAL = 0

ACTION_NOISE_PHI = 0.0
ACTION_NOISE_AMP = 0.0
ACTION_NOISE_DURATION = 0.0

CHAR_START_MODE = os.environ.get("CHAR_START_MODE", "radial_topk").lower()
CHAR_RADIAL_EXP = float(os.environ.get("CHAR_RADIAL_EXP", "1.0"))
CHAR_ALPHA_SCALE = float(os.environ.get("CHAR_ALPHA_SCALE", "1.0"))
_alpha_scales_env = os.environ.get("CHAR_ALPHA_SCALES", "").strip()
if _alpha_scales_env:
    CHAR_ALPHA_SCALES = [float(v.strip()) for v in _alpha_scales_env.split(",") if v.strip()]
else:
    CHAR_ALPHA_SCALES = [CHAR_ALPHA_SCALE]
CHAR_SAMPLER_MODE = os.environ.get("CHAR_SAMPLER_MODE", "radial_stratified").lower()
CHAR_RADIAL_BINS = int(os.environ.get("CHAR_RADIAL_BINS", "8"))
CHAR_IMPORTANCE_POWER = float(os.environ.get("CHAR_IMPORTANCE_POWER", "1.0"))
PHASE_CLIP = float(os.environ.get("PHASE_CLIP", str(np.pi)))
AMP_MIN = float(os.environ.get("AMP_MIN", "0.0"))
AMP_MAX = float(os.environ.get("AMP_MAX", "2.0"))
LEARN_DURATION_SCALE = os.environ.get("LEARN_DURATION_SCALE", "1") == "1"
DURATION_MIN_SCALE = float(os.environ.get("DURATION_MIN_SCALE", "0.25"))
DURATION_MAX_SCALE = float(os.environ.get("DURATION_MAX_SCALE", "4.0"))
if not np.isfinite(DURATION_MIN_SCALE) or DURATION_MIN_SCALE <= 0.0:
    raise ValueError(
        f"DURATION_MIN_SCALE must be finite and > 0, got {DURATION_MIN_SCALE}"
    )
if (
    not np.isfinite(DURATION_MAX_SCALE)
    or DURATION_MAX_SCALE <= 0.0
    or DURATION_MAX_SCALE < DURATION_MIN_SCALE
):
    raise ValueError(
        "DURATION_MAX_SCALE must be finite, > 0, and >= DURATION_MIN_SCALE, "
        f"got {DURATION_MAX_SCALE} (min={DURATION_MIN_SCALE})"
    )

CHAR_UNIFORM_MIX = float(os.environ.get("CHAR_UNIFORM_MIX", "0.5"))
FINAL_REFINE_SAMPLES = int(os.environ.get("FINAL_REFINE_SAMPLES", "512"))
FINAL_REFINE_SCALE = float(os.environ.get("FINAL_REFINE_SCALE", "1.0"))
FINAL_REFINE_SEED = int(os.environ.get("FINAL_REFINE_SEED", "1234"))
FINAL_REFINE_ROUNDS = int(os.environ.get("FINAL_REFINE_ROUNDS", "8"))
FINAL_REFINE_TOPK = int(os.environ.get("FINAL_REFINE_TOPK", "24"))
FINAL_REFINE_TOP_EVAL_CENTERS = int(
    os.environ.get("FINAL_REFINE_TOP_EVAL_CENTERS", "3")
)
FINAL_REFINE_DECAY = float(os.environ.get("FINAL_REFINE_DECAY", "0.6"))
FINAL_REFINE_MIN_SIGMA = float(os.environ.get("FINAL_REFINE_MIN_SIGMA", "0.05"))
FINAL_REFINE_ENABLE_AMP = os.environ.get("FINAL_REFINE_ENABLE_AMP", "0") == "1"
FINAL_REFINE_MIN_SIGMA_AMP = float(
    os.environ.get("FINAL_REFINE_MIN_SIGMA_AMP", "0.02")
)
FINAL_REFINE_INIT_SIGMA_AMP = float(
    os.environ.get("FINAL_REFINE_INIT_SIGMA_AMP", "0.15")
)
FINAL_REFINE_AMP_START_ROUND = int(os.environ.get("FINAL_REFINE_AMP_START_ROUND", "4"))
FINAL_REFINE_USE_LOC_CENTER = os.environ.get("FINAL_REFINE_USE_LOC_CENTER", "1") == "1"
FINAL_REFINE_USE_TRAIN_CENTER = os.environ.get("FINAL_REFINE_USE_TRAIN_CENTER", "1") == "1"
FINAL_REFINE_CENTER_SEED_STRIDE = int(
    os.environ.get("FINAL_REFINE_CENTER_SEED_STRIDE", "1000003")
)
FINAL_REFINE_FULL_STEPS = os.environ.get("FINAL_REFINE_FULL_STEPS", "0") == "1"
FINAL_REFINE_FULL_SAMPLES = int(os.environ.get("FINAL_REFINE_FULL_SAMPLES", "2048"))
FINAL_REFINE_FULL_ROUNDS = int(os.environ.get("FINAL_REFINE_FULL_ROUNDS", "6"))
FINAL_REFINE_FULL_TOPK = int(os.environ.get("FINAL_REFINE_FULL_TOPK", "64"))
FINAL_REFINE_FULL_SCALE = float(os.environ.get("FINAL_REFINE_FULL_SCALE", "0.6"))
FINAL_REFINE_FULL_DECAY = float(os.environ.get("FINAL_REFINE_FULL_DECAY", "0.6"))
FINAL_REFINE_FULL_MIN_SIGMA = float(
    os.environ.get("FINAL_REFINE_FULL_MIN_SIGMA", "0.003")
)
FINAL_REFINE_FULL_SIGMA_FACTOR = float(
    os.environ.get("FINAL_REFINE_FULL_SIGMA_FACTOR", "0.5")
)
FINAL_REFINE_FULL_ENABLE_AMP = (
    os.environ.get("FINAL_REFINE_FULL_ENABLE_AMP", "0") == "1"
)
FINAL_REFINE_FULL_MIN_SIGMA_AMP = float(
    os.environ.get("FINAL_REFINE_FULL_MIN_SIGMA_AMP", "0.0015")
)
FINAL_REFINE_FULL_SIGMA_FACTOR_AMP = float(
    os.environ.get(
        "FINAL_REFINE_FULL_SIGMA_FACTOR_AMP",
        str(FINAL_REFINE_FULL_SIGMA_FACTOR),
    )
)
FINAL_VALIDATE_ENABLE = os.environ.get("FINAL_VALIDATE_ENABLE", "1") == "1"
FINAL_VALIDATE_NUM_SEEDS = int(os.environ.get("FINAL_VALIDATE_NUM_SEEDS", "5"))
FINAL_VALIDATE_NOISE_SAMPLES = int(os.environ.get("FINAL_VALIDATE_NOISE_SAMPLES", "65"))
FINAL_VALIDATE_SEED_BASE = int(os.environ.get("FINAL_VALIDATE_SEED_BASE", "7100001"))
FINAL_VALIDATE_USE_SCORE_GUARD = os.environ.get("FINAL_VALIDATE_USE_SCORE_GUARD", "1") == "1"

ROBUST_TRAINING = os.environ.get("ROBUST_TRAINING", "0") == "1"
DEPHASE_MODEL = os.environ.get("DEPHASE_MODEL", "quasi_static").strip().lower()
DEPHASE_DETUNING_FRAC = float(os.environ.get("DEPHASE_DETUNING_FRAC", "0.05"))
DEPHASE_NOISE_SAMPLES_TRAIN = int(os.environ.get("DEPHASE_NOISE_SAMPLES_TRAIN", "6"))
DEPHASE_NOISE_SAMPLES_EVAL = int(os.environ.get("DEPHASE_NOISE_SAMPLES_EVAL", "12"))
DEPHASE_NOISE_SAMPLES_REFINE = int(os.environ.get("DEPHASE_NOISE_SAMPLES_REFINE", "16"))
DEPHASE_INCLUDE_NOMINAL = os.environ.get("DEPHASE_INCLUDE_NOMINAL", "1") == "1"
DEPHASE_OBJECTIVE_INCLUDE_NOMINAL = (
    os.environ.get("DEPHASE_OBJECTIVE_INCLUDE_NOMINAL", "1") == "1"
)
# Teacher guidance on gaussian weighting applies to quasi-static sweeps over
# evenly spaced detuning points. In stochastic mode we already sample detuning
# from a Gaussian process, so uniform objective weights avoid double-biasing
# toward near-zero detuning by default.
_default_detuning_weighting = (
    "gaussian" if DEPHASE_MODEL == "quasi_static" else "uniform"
)
DEPHASE_DETUNING_WEIGHTING = os.environ.get(
    "DEPHASE_DETUNING_WEIGHTING",
    _default_detuning_weighting,
).strip().lower()
DEPHASE_QUASI_SAMPLER = os.environ.get("DEPHASE_QUASI_SAMPLER", "grid").strip().lower()
DEPHASE_GAUSSIAN_SIGMA_FRAC = float(
    os.environ.get("DEPHASE_GAUSSIAN_SIGMA_FRAC", "0.50")
)
DEPHASE_STOCHASTIC_SIGMA_FRAC = float(
    os.environ.get("DEPHASE_STOCHASTIC_SIGMA_FRAC", str(DEPHASE_DETUNING_FRAC))
)
DEPHASE_STOCHASTIC_STD_MODE = os.environ.get(
    "DEPHASE_STOCHASTIC_STD_MODE",
    "gamma_dt",
).strip().lower()
DEPHASE_GAMMA = float(os.environ.get("DEPHASE_GAMMA", "18.0"))
DEPHASE_STOCHASTIC_CORRELATION = os.environ.get(
    "DEPHASE_STOCHASTIC_CORRELATION",
    "segment",
).strip().lower()
DEPHASE_STOCHASTIC_CLIP_STD = float(
    os.environ.get("DEPHASE_STOCHASTIC_CLIP_STD", "3.0")
)
DEPHASE_STOCHASTIC_SHARED_ACROSS_BATCH = (
    os.environ.get("DEPHASE_STOCHASTIC_SHARED_ACROSS_BATCH", "1") == "1"
)
ROBUST_NOMINAL_FID_FLOOR = float(os.environ.get("ROBUST_NOMINAL_FID_FLOOR", "0.985"))
ROBUST_FLOOR_PENALTY = float(os.environ.get("ROBUST_FLOOR_PENALTY", "0.0"))
ROBUST_COMPARE_BASELINE_NPZ = os.environ.get("ROBUST_COMPARE_BASELINE_NPZ", "").strip()
_baseline_duration_scale_override = os.environ.get(
    "ROBUST_COMPARE_BASELINE_DURATION_SCALE",
    "",
).strip()
if _baseline_duration_scale_override:
    ROBUST_COMPARE_BASELINE_DURATION_SCALE = float(_baseline_duration_scale_override)
else:
    ROBUST_COMPARE_BASELINE_DURATION_SCALE = None
DEPHASE_SWEEP_MAX_FRAC = float(os.environ.get("DEPHASE_SWEEP_MAX_FRAC", "0.08"))
DEPHASE_SWEEP_POINTS = int(os.environ.get("DEPHASE_SWEEP_POINTS", "21"))
STRICT_TEACHER_ALIGNMENT = os.environ.get("STRICT_TEACHER_ALIGNMENT", "1") == "1"
FORCE_GAUSSIAN_WEIGHTING = os.environ.get("FORCE_GAUSSIAN_WEIGHTING", "1") == "1"
FORCE_GRID_QUASI_SAMPLER = os.environ.get("FORCE_GRID_QUASI_SAMPLER", "1") == "1"
FORCE_STOCHASTIC_GAMMA_DT = os.environ.get("FORCE_STOCHASTIC_GAMMA_DT", "1") == "1"
OMEGA_TSTEP_SCALE_REF_HZ = float(os.environ.get("OMEGA_TSTEP_SCALE_REF_HZ", "2000.0"))
OMEGA_TSTEP_SCALE_REF_TSTEP = float(os.environ.get("OMEGA_TSTEP_SCALE_REF_TSTEP", "1.0e-5"))
OMEGA_TSTEP_SCALE_TOL = float(os.environ.get("OMEGA_TSTEP_SCALE_TOL", "0.20"))
AUTO_ADJUST_QUASI_GRID_ODD_SAMPLES = (
    os.environ.get("AUTO_ADJUST_QUASI_GRID_ODD_SAMPLES", "1") == "1"
)

if DEPHASE_MODEL not in {"quasi_static", "stochastic"}:
    raise ValueError(
        f"Unsupported DEPHASE_MODEL={DEPHASE_MODEL}, expected quasi_static or stochastic."
    )
if DEPHASE_DETUNING_WEIGHTING not in {"uniform", "gaussian"}:
    raise ValueError(
        f"Unsupported DEPHASE_DETUNING_WEIGHTING={DEPHASE_DETUNING_WEIGHTING}, expected uniform or gaussian."
    )
if DEPHASE_QUASI_SAMPLER not in {"grid", "uniform"}:
    raise ValueError(
        f"Unsupported DEPHASE_QUASI_SAMPLER={DEPHASE_QUASI_SAMPLER}, expected grid or uniform."
    )
if (
    not np.isfinite(DEPHASE_GAUSSIAN_SIGMA_FRAC)
    or DEPHASE_GAUSSIAN_SIGMA_FRAC <= 0.0
):
    raise ValueError(
        "DEPHASE_GAUSSIAN_SIGMA_FRAC must be finite and > 0, "
        f"got {DEPHASE_GAUSSIAN_SIGMA_FRAC}"
    )
if (
    not np.isfinite(DEPHASE_STOCHASTIC_SIGMA_FRAC)
    or DEPHASE_STOCHASTIC_SIGMA_FRAC < 0.0
):
    raise ValueError(
        "DEPHASE_STOCHASTIC_SIGMA_FRAC must be finite and >= 0, "
        f"got {DEPHASE_STOCHASTIC_SIGMA_FRAC}"
    )
if DEPHASE_STOCHASTIC_STD_MODE not in {"gamma_dt", "frac_omega"}:
    raise ValueError(
        "DEPHASE_STOCHASTIC_STD_MODE must be gamma_dt or frac_omega, "
        f"got {DEPHASE_STOCHASTIC_STD_MODE}"
    )
if DEPHASE_STOCHASTIC_CORRELATION not in {"segment", "step"}:
    raise ValueError(
        "DEPHASE_STOCHASTIC_CORRELATION must be segment or step, "
        f"got {DEPHASE_STOCHASTIC_CORRELATION}"
    )
if not np.isfinite(DEPHASE_GAMMA) or DEPHASE_GAMMA < 0.0:
    raise ValueError(f"DEPHASE_GAMMA must be finite and >= 0, got {DEPHASE_GAMMA}")
if not np.isfinite(DEPHASE_STOCHASTIC_CLIP_STD) or DEPHASE_STOCHASTIC_CLIP_STD < 0.0:
    raise ValueError(
        "DEPHASE_STOCHASTIC_CLIP_STD must be finite and >= 0, "
        f"got {DEPHASE_STOCHASTIC_CLIP_STD}"
    )
if not np.isfinite(DEPHASE_DETUNING_FRAC) or DEPHASE_DETUNING_FRAC < 0.0:
    raise ValueError(
        f"DEPHASE_DETUNING_FRAC must be finite and >= 0, got {DEPHASE_DETUNING_FRAC}"
    )
if DEPHASE_NOISE_SAMPLES_TRAIN <= 0:
    raise ValueError(
        f"DEPHASE_NOISE_SAMPLES_TRAIN must be > 0, got {DEPHASE_NOISE_SAMPLES_TRAIN}"
    )
if DEPHASE_NOISE_SAMPLES_EVAL <= 0:
    raise ValueError(
        f"DEPHASE_NOISE_SAMPLES_EVAL must be > 0, got {DEPHASE_NOISE_SAMPLES_EVAL}"
    )
if DEPHASE_NOISE_SAMPLES_REFINE <= 0:
    raise ValueError(
        f"DEPHASE_NOISE_SAMPLES_REFINE must be > 0, got {DEPHASE_NOISE_SAMPLES_REFINE}"
    )
if not np.isfinite(ROBUST_NOMINAL_FID_FLOOR):
    raise ValueError(
        f"ROBUST_NOMINAL_FID_FLOOR must be finite, got {ROBUST_NOMINAL_FID_FLOOR}"
    )
if not np.isfinite(ROBUST_FLOOR_PENALTY) or ROBUST_FLOOR_PENALTY < 0.0:
    raise ValueError(
        f"ROBUST_FLOOR_PENALTY must be finite and >= 0, got {ROBUST_FLOOR_PENALTY}"
    )
if ROBUST_COMPARE_BASELINE_DURATION_SCALE is not None and (
    (not np.isfinite(ROBUST_COMPARE_BASELINE_DURATION_SCALE))
    or ROBUST_COMPARE_BASELINE_DURATION_SCALE <= 0.0
):
    raise ValueError(
        "ROBUST_COMPARE_BASELINE_DURATION_SCALE must be finite and > 0 when set, "
        f"got {ROBUST_COMPARE_BASELINE_DURATION_SCALE}"
    )
if not np.isfinite(DEPHASE_SWEEP_MAX_FRAC) or DEPHASE_SWEEP_MAX_FRAC <= 0.0:
    raise ValueError(
        f"DEPHASE_SWEEP_MAX_FRAC must be finite and > 0, got {DEPHASE_SWEEP_MAX_FRAC}"
    )
if DEPHASE_SWEEP_POINTS < 2:
    raise ValueError(f"DEPHASE_SWEEP_POINTS must be >= 2, got {DEPHASE_SWEEP_POINTS}")
if FINAL_VALIDATE_NUM_SEEDS <= 0:
    raise ValueError(
        f"FINAL_VALIDATE_NUM_SEEDS must be > 0, got {FINAL_VALIDATE_NUM_SEEDS}"
    )
if FINAL_VALIDATE_NOISE_SAMPLES <= 0:
    raise ValueError(
        f"FINAL_VALIDATE_NOISE_SAMPLES must be > 0, got {FINAL_VALIDATE_NOISE_SAMPLES}"
    )
if not np.isfinite(OMEGA_TSTEP_SCALE_REF_HZ) or OMEGA_TSTEP_SCALE_REF_HZ <= 0.0:
    raise ValueError(
        f"OMEGA_TSTEP_SCALE_REF_HZ must be finite and > 0, got {OMEGA_TSTEP_SCALE_REF_HZ}"
    )
if (
    not np.isfinite(OMEGA_TSTEP_SCALE_REF_TSTEP)
    or OMEGA_TSTEP_SCALE_REF_TSTEP <= 0.0
):
    raise ValueError(
        "OMEGA_TSTEP_SCALE_REF_TSTEP must be finite and > 0, "
        f"got {OMEGA_TSTEP_SCALE_REF_TSTEP}"
    )
if not np.isfinite(OMEGA_TSTEP_SCALE_TOL) or OMEGA_TSTEP_SCALE_TOL < 0.0:
    raise ValueError(
        f"OMEGA_TSTEP_SCALE_TOL must be finite and >= 0, got {OMEGA_TSTEP_SCALE_TOL}"
    )
if ROBUST_TRAINING and DEPHASE_INCLUDE_NOMINAL:
    for _name, _value in [
        ("DEPHASE_NOISE_SAMPLES_TRAIN", DEPHASE_NOISE_SAMPLES_TRAIN),
        ("DEPHASE_NOISE_SAMPLES_EVAL", DEPHASE_NOISE_SAMPLES_EVAL),
        ("DEPHASE_NOISE_SAMPLES_REFINE", DEPHASE_NOISE_SAMPLES_REFINE),
    ]:
        if _value <= 1:
            raise ValueError(
                f"{_name} must be > 1 when DEPHASE_INCLUDE_NOMINAL=1 so robust averaging has nonzero-noise samples."
            )

if FAST_SMOKE and ROBUST_TRAINING:
    DEPHASE_NOISE_SAMPLES_TRAIN = min(DEPHASE_NOISE_SAMPLES_TRAIN, 3)
    DEPHASE_NOISE_SAMPLES_EVAL = min(DEPHASE_NOISE_SAMPLES_EVAL, 5)
    DEPHASE_NOISE_SAMPLES_REFINE = min(DEPHASE_NOISE_SAMPLES_REFINE, 5)


def _ensure_odd_quasi_sample_count(name, value):
    value = int(value)
    if value % 2 == 1:
        return value
    if AUTO_ADJUST_QUASI_GRID_ODD_SAMPLES:
        adjusted = value + 1
        logger.warning(
            "%s=%d adjusted to %d to keep quasi-static grid symmetric when "
            "DEPHASE_INCLUDE_NOMINAL=1.",
            name,
            value,
            adjusted,
        )
        return adjusted
    raise ValueError(
        f"{name} must be odd when DEPHASE_MODEL=quasi_static, "
        f"DEPHASE_QUASI_SAMPLER=grid, and DEPHASE_INCLUDE_NOMINAL=1; got {value}."
    )


if (
    ROBUST_TRAINING
    and DEPHASE_MODEL == "quasi_static"
    and DEPHASE_QUASI_SAMPLER == "grid"
    and DEPHASE_INCLUDE_NOMINAL
):
    DEPHASE_NOISE_SAMPLES_TRAIN = _ensure_odd_quasi_sample_count(
        "DEPHASE_NOISE_SAMPLES_TRAIN",
        DEPHASE_NOISE_SAMPLES_TRAIN,
    )
    DEPHASE_NOISE_SAMPLES_EVAL = _ensure_odd_quasi_sample_count(
        "DEPHASE_NOISE_SAMPLES_EVAL",
        DEPHASE_NOISE_SAMPLES_EVAL,
    )
    DEPHASE_NOISE_SAMPLES_REFINE = _ensure_odd_quasi_sample_count(
        "DEPHASE_NOISE_SAMPLES_REFINE",
        DEPHASE_NOISE_SAMPLES_REFINE,
    )


def _validate_teacher_alignment():
    if not ROBUST_TRAINING:
        return

    if (
        DEPHASE_MODEL == "quasi_static"
        and FORCE_GAUSSIAN_WEIGHTING
        and DEPHASE_DETUNING_WEIGHTING != "gaussian"
    ):
        raise ValueError(
            "Teacher alignment requires DEPHASE_DETUNING_WEIGHTING=gaussian "
            "for quasi-static robust mode."
        )
    if (
        DEPHASE_MODEL == "quasi_static"
        and FORCE_GRID_QUASI_SAMPLER
        and DEPHASE_QUASI_SAMPLER != "grid"
    ):
        raise ValueError(
            "Teacher alignment requires DEPHASE_QUASI_SAMPLER=grid for quasi-static robust training."
        )
    if (
        DEPHASE_MODEL == "stochastic"
        and FORCE_STOCHASTIC_GAMMA_DT
        and DEPHASE_STOCHASTIC_STD_MODE != "gamma_dt"
    ):
        raise ValueError(
            "Teacher alignment requires DEPHASE_STOCHASTIC_STD_MODE=gamma_dt for stochastic robust training."
        )

    # Keep Omega and t_step on the same scale when using physical gamma.
    if DEPHASE_MODEL == "stochastic" and DEPHASE_STOCHASTIC_STD_MODE == "gamma_dt":
        omega_scale = OMEGA_RABI_HZ / max(OMEGA_TSTEP_SCALE_REF_HZ, 1.0e-15)
        expected_t_step = OMEGA_TSTEP_SCALE_REF_TSTEP / max(omega_scale, 1.0e-15)
        rel_err = abs(T_STEP - expected_t_step) / max(expected_t_step, 1.0e-15)
        if rel_err > OMEGA_TSTEP_SCALE_TOL:
            msg = (
                "Omega/t_step scale mismatch for stochastic gamma_dt mode: "
                f"OMEGA_RABI_HZ={OMEGA_RABI_HZ:.6g}, T_STEP={T_STEP:.6g}, "
                f"expected T_STEP≈{expected_t_step:.6g} from reference "
                f"(OMEGA_TSTEP_SCALE_REF_HZ={OMEGA_TSTEP_SCALE_REF_HZ:.6g}, "
                f"OMEGA_TSTEP_SCALE_REF_TSTEP={OMEGA_TSTEP_SCALE_REF_TSTEP:.6g}). "
                "Teacher guidance: if Omega changes by k, t_step should change by 1/k."
            )
            if STRICT_TEACHER_ALIGNMENT:
                raise ValueError(msg)
            logger.warning(msg)

    if LEARN_DURATION_SCALE and DURATION_SCALE_PENALTY_LAMBDA <= 0.0:
        logger.warning(
            "LEARN_DURATION_SCALE=1 but DURATION_SCALE_PENALTY_LAMBDA=0. "
            "Teacher guidance suggests adding explicit time regularization."
        )


_validate_teacher_alignment()


def _build_characteristic_distribution(grid_size):
    all_points = []
    all_targets = []
    all_weights = []
    all_areas = []
    n_scales = len(CHAR_ALPHA_SCALES)
    for alpha_scale in CHAR_ALPHA_SCALES:
        points_i, target_i, weights_i, area_i = prepare_characteristic_distribution(
            n_boson=N_BOSON,
            extent=SAMPLE_EXTENT,
            grid_size=grid_size,
            binomial_code=BINOMIAL_CODE,
            mix_uniform=CHAR_UNIFORM_MIX,
            alpha_scale=alpha_scale,
            binomial_phase=BINOMIAL_REL_PHASE,
            importance_power=CHAR_IMPORTANCE_POWER,
        )
        all_points.extend(points_i)
        all_targets.append(target_i)
        all_weights.append(weights_i / float(n_scales))
        all_areas.append(area_i)
    return (
        all_points,
        np.concatenate(all_targets),
        np.concatenate(all_weights),
        float(np.mean(all_areas)),
    )


CHAR_POINTS, CHAR_TARGET, CHAR_WEIGHTS, CHAR_AREA = _build_characteristic_distribution(
    CHAR_GRID_SIZE
)
FINAL_POINTS, FINAL_TARGET, FINAL_WEIGHTS, FINAL_AREA = prepare_characteristic_distribution(
    n_boson=N_BOSON,
    extent=SAMPLE_EXTENT,
    grid_size=FINAL_GRID_SIZE,
    binomial_code=BINOMIAL_CODE,
    mix_uniform=CHAR_UNIFORM_MIX,
    alpha_scale=CHAR_ALPHA_SCALE,
    binomial_phase=BINOMIAL_REL_PHASE,
    importance_power=CHAR_IMPORTANCE_POWER,
)
CHAR_NORM = characteristic_norm(CHAR_TARGET, CHAR_AREA)
FINAL_NORM = characteristic_norm(FINAL_TARGET, FINAL_AREA)
CHAR_RADII = np.abs(np.asarray(CHAR_POINTS))

TOPK_COUNT = min(TRAIN_POINTS_STAGE1, len(CHAR_POINTS))
if CHAR_START_MODE == "topk":
    score = np.abs(CHAR_TARGET)
else:
    radii = np.maximum(CHAR_RADII, 1e-6)
    score = np.abs(CHAR_TARGET) * (radii ** CHAR_RADIAL_EXP)


def _build_stage1_topk_indices(score, count):
    # For stage-1 warmup, keep strongest informative points globally.
    return np.argsort(score)[-count:]


topk_idx = _build_stage1_topk_indices(score, TOPK_COUNT)
TOPK_POINTS = [CHAR_POINTS[i] for i in topk_idx]
TOPK_TARGET = CHAR_TARGET[topk_idx]
TOPK_WEIGHTS = np.full(TOPK_COUNT, 1.0 / TOPK_COUNT, dtype=float)
TOPK_NORM = characteristic_norm(TOPK_TARGET, CHAR_AREA)

logger.info(
    "Characteristic sampling: start_mode=%s alpha_scales=%s radial_exp=%.2f binomial_code=%s binomial_phase=%s",
    CHAR_START_MODE,
    ",".join(f"{v:.3f}" for v in CHAR_ALPHA_SCALES),
    CHAR_RADIAL_EXP,
    BINOMIAL_CODE,
    "none" if BINOMIAL_REL_PHASE is None else f"{BINOMIAL_REL_PHASE:.3f}",
)
logger.info(
    "Characteristic reward objective schedule: base=%s stage2=%s switch_train_epoch=%d | stage epochs: %d -> %d -> end",
    CHAR_REWARD_OBJECTIVE,
    CHAR_REWARD_OBJECTIVE_STAGE2,
    CHAR_REWARD_SWITCH_EPOCH,
    TRAIN_STAGE1_EPOCHS,
    TRAIN_STAGE2_EPOCHS,
)
logger.info(
    "Characteristic reward normalization: fixed_overlap_norm=%s",
    CHAR_USE_FIXED_REWARD_NORM,
)
logger.info(
    "Reward switch guard: min_best_eval=%.3f patience_eval=%d min_gain=%.4f allow_revert=%s",
    CHAR_REWARD_SWITCH_MIN_BEST_EVAL,
    CHAR_REWARD_STAGE2_PATIENCE_EVAL,
    CHAR_REWARD_STAGE2_MIN_GAIN,
    CHAR_REWARD_STAGE2_ALLOW_REVERT,
)
logger.info(
    "Reward auto-rescale: enabled=%s target_p90=%.3f trigger_p90=%.3f",
    CHAR_REWARD_AUTO_RESCALE,
    CHAR_REWARD_AUTO_RESCALE_TARGET_P90,
    CHAR_REWARD_AUTO_RESCALE_TRIGGER_P90,
)
logger.info(
    "Characteristic point sampler: mode=%s radial_bins=%d uniform_mix=%.2f importance_power=%.2f",
    CHAR_SAMPLER_MODE,
    CHAR_RADIAL_BINS,
    CHAR_UNIFORM_MIX,
    CHAR_IMPORTANCE_POWER,
)
logger.info(
    "Action clipping: phase_clip=%.3f amp_range=[%.3f, %.3f] duration_scale_range=[%.3f, %.3f]",
    PHASE_CLIP,
    AMP_MIN,
    AMP_MAX,
    DURATION_MIN_SCALE,
    DURATION_MAX_SCALE,
)
logger.info(
    "Amplitude overshoot penalty: lambda=%.6f base=%.3f",
    AMP_OVERSHOOT_LAMBDA,
    AMP_OVERSHOOT_BASE,
)
logger.info(
    "Duration regularization: lambda=%.6f target=%.3f above_target_only=%s",
    DURATION_SCALE_PENALTY_LAMBDA,
    DURATION_SCALE_TARGET,
    DURATION_SCALE_ONLY_ABOVE_TARGET,
)
logger.info(
    "Pulse timing: n_steps=%d n_segments=%d seg_len=%d t_step=%.6f",
    N_STEPS,
    N_SEGMENTS,
    SEG_LEN,
    T_STEP,
)
logger.info(
    "Drive scale: omega=2pi*%.3f Hz (%.6e rad/s), omega*t_step=%.6e rad",
    OMEGA_RABI_HZ,
    OMEGA_RABI,
    OMEGA_RABI * T_STEP,
)
logger.info(
    "Characteristic plotting extent: %.3f (sampling extent: %.3f)",
    PLOT_EXTENT,
    SAMPLE_EXTENT,
)
logger.info(
    "Final refinement setup: samples=%d rounds=%d topk=%d top_eval_centers=%d scale=%.3f decay=%.3f min_sigma=%.3f amp_opt=%s min_sigma_amp=%.3f init_sigma_amp=%.3f amp_start_round=%d use_loc_center=%s use_train_center=%s",
    FINAL_REFINE_SAMPLES,
    FINAL_REFINE_ROUNDS,
    FINAL_REFINE_TOPK,
    FINAL_REFINE_TOP_EVAL_CENTERS,
    FINAL_REFINE_SCALE,
    FINAL_REFINE_DECAY,
    FINAL_REFINE_MIN_SIGMA,
    FINAL_REFINE_ENABLE_AMP,
    FINAL_REFINE_MIN_SIGMA_AMP,
    FINAL_REFINE_INIT_SIGMA_AMP,
    FINAL_REFINE_AMP_START_ROUND,
    FINAL_REFINE_USE_LOC_CENTER,
    FINAL_REFINE_USE_TRAIN_CENTER,
)
logger.info(
    "Full-step refinement: enabled=%s samples=%d rounds=%d topk=%d scale=%.3f decay=%.3f phase_sigma_factor=%.3f phase_min_sigma=%.4f amp_opt=%s amp_sigma_factor=%.3f amp_min_sigma=%.4f",
    FINAL_REFINE_FULL_STEPS,
    FINAL_REFINE_FULL_SAMPLES,
    FINAL_REFINE_FULL_ROUNDS,
    FINAL_REFINE_FULL_TOPK,
    FINAL_REFINE_FULL_SCALE,
    FINAL_REFINE_FULL_DECAY,
    FINAL_REFINE_FULL_SIGMA_FACTOR,
    FINAL_REFINE_FULL_MIN_SIGMA,
    FINAL_REFINE_FULL_ENABLE_AMP,
    FINAL_REFINE_FULL_SIGMA_FACTOR_AMP,
    FINAL_REFINE_FULL_MIN_SIGMA_AMP,
)
logger.info(
    "Final candidate validation: enabled=%s seeds=%d noise_samples=%d seed_base=%d score_guard=%s",
    FINAL_VALIDATE_ENABLE,
    FINAL_VALIDATE_NUM_SEEDS,
    FINAL_VALIDATE_NOISE_SAMPLES,
    FINAL_VALIDATE_SEED_BASE,
    FINAL_VALIDATE_USE_SCORE_GUARD,
)
logger.info(
    "Robust dephasing: enabled=%s model=%s detuning_frac=%.4f include_nominal=%s objective_include_nominal=%s samples(train/eval/refine)=%d/%d/%d quasi_sampler=%s weighting=%s gauss_sigma_frac=%.3f stochastic_std_mode=%s stochastic_gamma=%.6f stochastic_corr=%s stochastic_sigma_frac=%.3f stochastic_sigma_abs=%.6e stochastic_clip_std=%.2f stochastic_shared_batch=%s floor=%.4f penalty=%.3f",
    ROBUST_TRAINING,
    DEPHASE_MODEL,
    DEPHASE_DETUNING_FRAC,
    DEPHASE_INCLUDE_NOMINAL,
    DEPHASE_OBJECTIVE_INCLUDE_NOMINAL,
    DEPHASE_NOISE_SAMPLES_TRAIN,
    DEPHASE_NOISE_SAMPLES_EVAL,
    DEPHASE_NOISE_SAMPLES_REFINE,
    DEPHASE_QUASI_SAMPLER,
    DEPHASE_DETUNING_WEIGHTING,
    DEPHASE_GAUSSIAN_SIGMA_FRAC,
    DEPHASE_STOCHASTIC_STD_MODE,
    DEPHASE_GAMMA,
    DEPHASE_STOCHASTIC_CORRELATION,
    DEPHASE_STOCHASTIC_SIGMA_FRAC,
    (
        np.sqrt(
            2.0
            * DEPHASE_GAMMA
            / max(
                (max(SEG_LEN, 1) * T_STEP)
                if DEPHASE_STOCHASTIC_CORRELATION == "segment"
                else T_STEP,
                1.0e-15,
            )
        )
        if DEPHASE_STOCHASTIC_STD_MODE == "gamma_dt"
        else DEPHASE_STOCHASTIC_SIGMA_FRAC * OMEGA_RABI
    ),
    DEPHASE_STOCHASTIC_CLIP_STD,
    DEPHASE_STOCHASTIC_SHARED_ACROSS_BATCH,
    ROBUST_NOMINAL_FID_FLOOR,
    ROBUST_FLOOR_PENALTY,
)
logger.info(
    "Teacher alignment flags: strict=%s force_gaussian=%s force_grid_quasi=%s force_stochastic_gamma_dt=%s ref_omega_hz=%.3f ref_t_step=%.3e tol=%.3f auto_adjust_quasi_odd=%s",
    STRICT_TEACHER_ALIGNMENT,
    FORCE_GAUSSIAN_WEIGHTING,
    FORCE_GRID_QUASI_SAMPLER,
    FORCE_STOCHASTIC_GAMMA_DT,
    OMEGA_TSTEP_SCALE_REF_HZ,
    OMEGA_TSTEP_SCALE_REF_TSTEP,
    OMEGA_TSTEP_SCALE_TOL,
    AUTO_ADJUST_QUASI_GRID_ODD_SAMPLES,
)
if (
    ROBUST_TRAINING
    and DEPHASE_MODEL == "quasi_static"
    and DEPHASE_DETUNING_WEIGHTING == "uniform"
    and (not FORCE_GAUSSIAN_WEIGHTING)
):
    logger.warning(
        "Using uniform detuning weighting in quasi-static robust mode. "
        "Teacher guidance recommends gaussian weighting so large-detuning "
        "samples do not dominate."
    )
if ROBUST_TRAINING and DEPHASE_MODEL == "stochastic" and DEPHASE_DETUNING_WEIGHTING == "gaussian":
    logger.warning(
        "Stochastic mode currently uses Gaussian sampling plus Gaussian objective weighting. "
        "This can over-focus near-zero detuning."
    )
if ROBUST_COMPARE_BASELINE_DURATION_SCALE is not None:
    logger.info(
        "Baseline comparison override: duration_scale=%.6f",
        ROBUST_COMPARE_BASELINE_DURATION_SCALE,
    )


def _effective_train_epoch(epoch, epoch_type):
    if epoch_type == "final":
        return NUM_EPOCHS_HINT
    if epoch_type == "evaluation":
        return int(epoch) * max(1, EVAL_INTERVAL_HINT)
    return int(epoch)


_reward_schedule_state = {
    "active": CHAR_REWARD_OBJECTIVE,
    "switched": False,
    "reverted": False,
    "switch_eval_epoch": None,
    "anchor_best_eval": -np.inf,
    "stage2_best_eval": -np.inf,
}


def _update_reward_objective(epoch, epoch_type, best_eval_metric):
    active = _reward_schedule_state["active"]
    if CHAR_REWARD_OBJECTIVE_STAGE2 == CHAR_REWARD_OBJECTIVE:
        return active

    effective_train_epoch = _effective_train_epoch(epoch, epoch_type)

    # Switch only at evaluation boundaries and only once the policy is mature enough.
    can_switch = (
        not _reward_schedule_state["switched"]
        and CHAR_REWARD_SWITCH_EPOCH >= 0
        and epoch_type == "evaluation"
        and effective_train_epoch >= CHAR_REWARD_SWITCH_EPOCH
        and best_eval_metric >= CHAR_REWARD_SWITCH_MIN_BEST_EVAL
    )
    if can_switch:
        _reward_schedule_state["active"] = CHAR_REWARD_OBJECTIVE_STAGE2
        _reward_schedule_state["switched"] = True
        _reward_schedule_state["switch_eval_epoch"] = int(epoch)
        _reward_schedule_state["anchor_best_eval"] = float(best_eval_metric)
        _reward_schedule_state["stage2_best_eval"] = float(best_eval_metric)
        return _reward_schedule_state["active"]

    if (
        _reward_schedule_state["switched"]
        and (not _reward_schedule_state["reverted"])
        and epoch_type == "evaluation"
    ):
        _reward_schedule_state["stage2_best_eval"] = max(
            float(_reward_schedule_state["stage2_best_eval"]),
            float(best_eval_metric),
        )
        waited_eval_epochs = int(epoch) - int(_reward_schedule_state["switch_eval_epoch"])
        if (
            CHAR_REWARD_STAGE2_ALLOW_REVERT
            and waited_eval_epochs >= CHAR_REWARD_STAGE2_PATIENCE_EVAL
            and _reward_schedule_state["stage2_best_eval"]
            < (_reward_schedule_state["anchor_best_eval"] + CHAR_REWARD_STAGE2_MIN_GAIN)
        ):
            _reward_schedule_state["active"] = CHAR_REWARD_OBJECTIVE
            _reward_schedule_state["reverted"] = True

    return _reward_schedule_state["active"]


def _auto_rescale_rewards(reward_data, epoch, epoch_type):
    reward_arr = np.asarray(reward_data, dtype=float)
    if (not CHAR_REWARD_AUTO_RESCALE) or reward_arr.size == 0:
        return reward_data
    p90 = float(np.percentile(np.abs(reward_arr), 90.0))
    if (not np.isfinite(p90)) or p90 <= CHAR_REWARD_AUTO_RESCALE_TRIGGER_P90:
        return reward_data
    factor = CHAR_REWARD_AUTO_RESCALE_TARGET_P90 / p90
    scaled = reward_arr * factor
    if epoch_type == "evaluation" or int(epoch) % 20 == 0:
        logger.info(
            "Reward auto-rescale at %s epoch %d: p90=%.6f factor=%.6f",
            epoch_type,
            int(epoch),
            p90,
            factor,
        )
    return scaled.astype(np.float32)


def _expand_segments(arr):
    return np.repeat(np.asarray(arr, dtype=float), SEG_LEN, axis=1)


def _detuning_abs_max():
    return float(DEPHASE_DETUNING_FRAC * OMEGA_RABI)


def _stochastic_correlation_time():
    if DEPHASE_STOCHASTIC_CORRELATION == "segment":
        return float(max(SEG_LEN, 1) * T_STEP)
    return float(T_STEP)


def _stochastic_sigma_abs():
    if DEPHASE_STOCHASTIC_STD_MODE == "gamma_dt":
        # Teacher's stochastic surrogate for Lindblad dephasing:
        # sigma_delta = sqrt(2 * gamma / tau_c).
        tau_c = max(_stochastic_correlation_time(), 1.0e-15)
        return float(np.sqrt(2.0 * DEPHASE_GAMMA / tau_c))
    return float(DEPHASE_STOCHASTIC_SIGMA_FRAC * OMEGA_RABI)


def _noise_samples_for_epoch_type(epoch_type):
    if epoch_type == "evaluation":
        return max(1, DEPHASE_NOISE_SAMPLES_EVAL)
    if epoch_type == "final":
        return max(1, DEPHASE_NOISE_SAMPLES_REFINE)
    return max(1, DEPHASE_NOISE_SAMPLES_TRAIN)


def _sample_quasi_static_detuning_matrix(
    rng,
    batch_size,
    n_samples,
    include_nominal,
    shared_across_batch=True,
):
    if n_samples <= 0:
        raise ValueError(f"n_samples must be > 0, got {n_samples}")
    delta_max = _detuning_abs_max()
    if DEPHASE_QUASI_SAMPLER == "grid":
        if include_nominal:
            if int(n_samples) == 1:
                base = np.array([0.0], dtype=float)
            else:
                n_non_nominal = int(n_samples) - 1
                if n_non_nominal % 2 != 0:
                    raise ValueError(
                        "Quasi-static grid with DEPHASE_INCLUDE_NOMINAL=1 requires an odd "
                        "total sample count so the non-nominal grid has +/- pairs. "
                        f"Got n_samples={n_samples}."
                    )
                half = n_non_nominal // 2
                if half <= 0:
                    base = np.array([0.0], dtype=float)
                else:
                    magnitudes = np.linspace(
                        delta_max / float(half),
                        delta_max,
                        half,
                        dtype=float,
                    )
                    remainder = np.concatenate((-magnitudes[::-1], magnitudes))
                    base = np.concatenate(([0.0], remainder))
        else:
            base = np.linspace(-delta_max, delta_max, int(n_samples), dtype=float)
        detuning = np.repeat(base[None, :], int(batch_size), axis=0)
    elif shared_across_batch:
        detuning = rng.uniform(-delta_max, delta_max, size=(1, n_samples))
        detuning = np.repeat(detuning, int(batch_size), axis=0)
    else:
        detuning = rng.uniform(-delta_max, delta_max, size=(int(batch_size), n_samples))
    if include_nominal:
        detuning[:, 0] = 0.0
    return detuning.astype(float, copy=False)


def _sample_stochastic_detuning_tensor(
    rng,
    batch_size,
    n_samples,
    n_steps,
    include_nominal,
    shared_across_batch=True,
):
    if n_samples <= 0:
        raise ValueError(f"n_samples must be > 0, got {n_samples}")
    if n_steps <= 0:
        raise ValueError(f"n_steps must be > 0, got {n_steps}")
    sigma = _stochastic_sigma_abs()
    corr_steps = int(n_steps)
    if DEPHASE_STOCHASTIC_CORRELATION == "segment":
        corr_steps = int(N_SEGMENTS)
    base_shape = (
        (1, int(n_samples), corr_steps)
        if shared_across_batch
        else (int(batch_size), int(n_samples), corr_steps)
    )
    if sigma == 0.0:
        detuning = np.zeros(base_shape, dtype=float)
    else:
        detuning = rng.normal(loc=0.0, scale=sigma, size=base_shape)
        if DEPHASE_STOCHASTIC_CLIP_STD > 0.0:
            clip_abs = float(DEPHASE_STOCHASTIC_CLIP_STD * sigma)
            detuning = np.clip(detuning, -clip_abs, clip_abs)
    if shared_across_batch:
        detuning = np.repeat(detuning, int(batch_size), axis=0)

    if DEPHASE_STOCHASTIC_CORRELATION == "segment":
        detuning = np.repeat(detuning, max(SEG_LEN, 1), axis=2)
        detuning = detuning[:, :, : int(n_steps)]

    if include_nominal:
        detuning[:, 0, :] = 0.0
    return detuning.astype(float, copy=False)


def _sample_detuning_samples(
    rng,
    batch_size,
    n_samples,
    include_nominal,
):
    if DEPHASE_MODEL == "quasi_static":
        return _sample_quasi_static_detuning_matrix(
            rng,
            batch_size=batch_size,
            n_samples=n_samples,
            include_nominal=include_nominal,
            shared_across_batch=True,
        )
    return _sample_stochastic_detuning_tensor(
        rng,
        batch_size=batch_size,
        n_samples=n_samples,
        n_steps=N_STEPS,
        include_nominal=include_nominal,
        shared_across_batch=DEPHASE_STOCHASTIC_SHARED_ACROSS_BATCH,
    )


def _apply_duration_scale_to_detuning(detuning_samples, duration_scale):
    detuning_arr = np.asarray(detuning_samples, dtype=float)
    if duration_scale is None:
        return detuning_arr
    scale = np.asarray(duration_scale, dtype=float).reshape(-1)
    if detuning_arr.shape[0] != scale.shape[0]:
        raise ValueError(
            f"duration_scale batch mismatch: detuning batch={detuning_arr.shape[0]} vs scale batch={scale.shape[0]}"
        )
    # Keep dephasing-time scaling consistent with the chosen stochastic model:
    # - quasi_static and frac_omega stochastic: delta scales linearly with duration.
    # - gamma_dt stochastic: delta ~ N(0, 2*gamma/tau_c), and tau_c scales with
    #   duration; after applying global Hamiltonian scaling, the effective
    #   detuning coefficient scales as sqrt(duration).
    if DEPHASE_MODEL == "stochastic" and DEPHASE_STOCHASTIC_STD_MODE == "gamma_dt":
        scale_factor = np.sqrt(np.clip(scale, 0.0, None))
    else:
        scale_factor = scale
    if detuning_arr.ndim == 2:
        return detuning_arr * scale_factor[:, None]
    if detuning_arr.ndim == 3:
        return detuning_arr * scale_factor[:, None, None]
    raise ValueError(f"Unexpected detuning sample shape {detuning_arr.shape}")


def _broadcast_duration_scale(duration_scale, batch_size):
    if duration_scale is None:
        return np.ones(int(batch_size), dtype=float)
    arr = np.asarray(duration_scale, dtype=float).reshape(-1)
    if arr.size == 1:
        arr = np.full(int(batch_size), float(arr[0]), dtype=float)
    if arr.size != int(batch_size):
        raise ValueError(
            f"duration_scale size mismatch: expected {batch_size}, got {arr.size}"
        )
    return np.clip(arr, DURATION_MIN_SCALE, DURATION_MAX_SCALE)


def _expand_controls_for_noise(phi_r, phi_b, amp_r, amp_b, detuning_samples):
    detuning_arr = np.asarray(detuning_samples, dtype=float)
    if detuning_arr.ndim not in (2, 3):
        raise ValueError(
            f"detuning samples must be 2D or 3D, got shape {detuning_arr.shape}"
        )
    batch_size = int(detuning_arr.shape[0])
    n_noise = int(detuning_arr.shape[1])
    phi_r_exp = np.repeat(np.asarray(phi_r, dtype=float), n_noise, axis=0)
    phi_b_exp = np.repeat(np.asarray(phi_b, dtype=float), n_noise, axis=0)
    amp_r_exp = np.repeat(np.asarray(amp_r, dtype=float), n_noise, axis=0)
    amp_b_exp = np.repeat(np.asarray(amp_b, dtype=float), n_noise, axis=0)
    if detuning_arr.ndim == 2:
        detuning_flat = detuning_arr.reshape(batch_size * n_noise)
    else:
        detuning_flat = detuning_arr.reshape(batch_size * n_noise, detuning_arr.shape[2])
    return phi_r_exp, phi_b_exp, amp_r_exp, amp_b_exp, detuning_flat, n_noise


def _noise_indices_for_objective(n_noise, include_nominal):
    if include_nominal and (not DEPHASE_OBJECTIVE_INCLUDE_NOMINAL) and n_noise > 1:
        return np.arange(1, int(n_noise), dtype=int)
    return np.arange(0, int(n_noise), dtype=int)


def _detuning_gaussian_sigma_abs():
    if DEPHASE_MODEL == "stochastic":
        scale = max(_stochastic_sigma_abs(), 1.0e-12)
    else:
        scale = max(_detuning_abs_max(), 1.0e-12)
    return float(DEPHASE_GAUSSIAN_SIGMA_FRAC * scale)


def _compute_noise_weights_for_objective(detuning_samples):
    detuning_arr = np.asarray(detuning_samples, dtype=float)
    if detuning_arr.ndim not in (2, 3):
        raise ValueError(
            f"detuning samples must be 2D or 3D, got shape {detuning_arr.shape}"
        )
    batch_size = int(detuning_arr.shape[0])
    n_noise = int(detuning_arr.shape[1])
    noise_idx = _noise_indices_for_objective(n_noise, DEPHASE_INCLUDE_NOMINAL)
    if noise_idx.size == 0:
        noise_idx = np.array([0], dtype=int)

    detuning_obj = detuning_arr[:, noise_idx, ...]
    if DEPHASE_DETUNING_WEIGHTING == "uniform":
        raw_weights = np.ones((batch_size, noise_idx.size), dtype=float)
    else:
        sigma_abs = _detuning_gaussian_sigma_abs()
        if detuning_obj.ndim == 2:
            detuning_metric = np.abs(detuning_obj)
        else:
            detuning_metric = np.sqrt(np.mean(detuning_obj ** 2, axis=2))
        raw_weights = np.exp(-0.5 * (detuning_metric / sigma_abs) ** 2)
        raw_weights = np.where(np.isfinite(raw_weights), raw_weights, 0.0)

    weight_sum = np.sum(raw_weights, axis=1, keepdims=True)
    valid = np.isfinite(weight_sum) & (weight_sum > 0.0)
    weights = np.divide(
        raw_weights,
        weight_sum,
        out=np.full_like(raw_weights, 1.0 / raw_weights.shape[1]),
        where=valid,
    )
    return noise_idx, weights


def _smoothness_penalty(phi_r, phi_b, amp_r, amp_b):
    axis = 1 if np.ndim(phi_r) > 1 else 0
    dphi_r = np.diff(phi_r, axis=axis)
    dphi_b = np.diff(phi_b, axis=axis)
    damp_r = np.diff(amp_r, axis=axis)
    damp_b = np.diff(amp_b, axis=axis)
    phi_pen = 0.5 * (np.mean(dphi_r ** 2, axis=axis) + np.mean(dphi_b ** 2, axis=axis))
    amp_pen = 0.5 * (np.mean(damp_r ** 2, axis=axis) + np.mean(damp_b ** 2, axis=axis))
    return SMOOTH_PHI_WEIGHT * phi_pen + SMOOTH_AMP_WEIGHT * amp_pen


def _amplitude_overshoot_penalty(amp_r, amp_b):
    axis = 1 if np.ndim(amp_r) > 1 else 0
    excess_r = np.maximum(0.0, np.asarray(amp_r, dtype=float) - AMP_OVERSHOOT_BASE)
    excess_b = np.maximum(0.0, np.asarray(amp_b, dtype=float) - AMP_OVERSHOOT_BASE)
    return 0.5 * (
        np.mean(excess_r ** 2, axis=axis) + np.mean(excess_b ** 2, axis=axis)
    )


def _duration_scale_penalty(duration_scale):
    scale = np.asarray(duration_scale, dtype=float).reshape(-1)
    if DURATION_SCALE_ONLY_ABOVE_TARGET:
        delta = np.maximum(0.0, scale - DURATION_SCALE_TARGET)
    else:
        delta = scale - DURATION_SCALE_TARGET
    return delta ** 2


def _log_action_stats(tag, phi_r, phi_b, amp_r, amp_b):
    def _stats(arr):
        arr = np.asarray(arr)
        return float(np.mean(arr)), float(np.std(arr)), float(np.min(arr)), float(np.max(arr))

    for name, arr in [
        ("phi_r", phi_r),
        ("phi_b", phi_b),
        ("amp_r", amp_r),
        ("amp_b", amp_b),
    ]:
        mean, std, vmin, vmax = _stats(arr)
        logger.info(
            "%s %s stats: mean=%.4f std=%.4f min=%.4f max=%.4f",
            tag,
            name,
            mean,
            std,
            vmin,
            vmax,
        )


def _log_batch_diversity(tag, phi_r, phi_b, amp_r, amp_b):
    def _batch_stats(arr):
        arr = np.asarray(arr)
        if arr.ndim < 2:
            return 0.0, 0.0
        std_over_batch = np.std(arr, axis=0)
        return float(np.mean(std_over_batch)), float(np.max(std_over_batch))

    for name, arr in [
        ("phi_r", phi_r),
        ("phi_b", phi_b),
        ("amp_r", amp_r),
        ("amp_b", amp_b),
    ]:
        mean_std, max_std = _batch_stats(arr)
        logger.info(
            "%s %s batch-std: mean=%.6f max=%.6f",
            tag,
            name,
            mean_std,
            max_std,
        )


def _sample_characteristic_points(rng, n_points, mode=None):
    mode = CHAR_SAMPLER_MODE if mode is None else mode
    if mode == "weighted":
        idx = rng.choice(len(CHAR_POINTS), size=n_points, replace=True, p=CHAR_WEIGHTS)
        samp_probs = CHAR_WEIGHTS[idx]
    elif mode == "uniform":
        idx = rng.choice(len(CHAR_POINTS), size=n_points, replace=True)
        samp_probs = np.full(n_points, 1.0 / len(CHAR_POINTS), dtype=float)
    elif mode == "radial_stratified":
        n_bins = max(1, CHAR_RADIAL_BINS)
        r_max = float(np.max(CHAR_RADII))
        edges = np.linspace(0.0, r_max + 1e-12, n_bins + 1)
        idx_list = []
        bin_candidates = []
        bin_mass = np.zeros(n_bins, dtype=float)
        for bi in range(n_bins):
            lo = edges[bi]
            hi = edges[bi + 1]
            if bi == n_bins - 1:
                mask = (CHAR_RADII >= lo) & (CHAR_RADII <= hi)
            else:
                mask = (CHAR_RADII >= lo) & (CHAR_RADII < hi)
            candidates = np.flatnonzero(mask)
            bin_candidates.append(candidates)
            if candidates.size > 0:
                bin_mass[bi] = float(np.sum(CHAR_WEIGHTS[candidates]))

        mass_total = float(np.sum(bin_mass))
        if mass_total <= 0.0 or not np.isfinite(mass_total):
            quotas = np.full(n_bins, n_points // n_bins, dtype=int)
            quotas[: (n_points % n_bins)] += 1
        else:
            raw = n_points * (bin_mass / mass_total)
            quotas = np.floor(raw).astype(int)
            remaining = int(n_points - int(np.sum(quotas)))
            if remaining > 0:
                frac = raw - quotas
                order = np.argsort(frac)[::-1]
                for bi in order[:remaining]:
                    quotas[bi] += 1
        # q(alpha): actual sampling distribution induced by stratified sampling.
        # This must be used for importance weighting in reward evaluation.
        q = np.zeros(len(CHAR_POINTS), dtype=float)
        for bi in range(n_bins):
            candidates = bin_candidates[bi]
            if candidates.size == 0:
                continue
            take = min(int(quotas[bi]), n_points - len(idx_list))
            if take <= 0:
                break
            local_w = CHAR_WEIGHTS[candidates]
            local_w_sum = float(np.sum(local_w))
            if local_w_sum > 0.0 and np.isfinite(local_w_sum):
                local_w = local_w / local_w_sum
                sampled = rng.choice(candidates, size=take, replace=True, p=local_w)
                q[candidates] += (take / float(n_points)) * local_w
            else:
                sampled = rng.choice(candidates, size=take, replace=True)
                q[candidates] += (take / float(n_points)) / float(candidates.size)
            idx_list.extend(sampled.tolist())
        n_fill = n_points - len(idx_list)
        if n_fill > 0:
            fill = rng.choice(
                len(CHAR_POINTS),
                size=n_fill,
                replace=True,
                p=CHAR_WEIGHTS,
            )
            idx_list.extend(fill.tolist())
            q += (n_fill / float(n_points)) * CHAR_WEIGHTS
        idx = np.asarray(idx_list, dtype=int)
        q = np.maximum(q, 1e-12)
        q = q / float(np.sum(q))
        samp_probs = q[idx]
    else:
        raise ValueError(f"Unknown CHAR_SAMPLER_MODE={mode}")
    points = [CHAR_POINTS[i] for i in idx]
    targets = CHAR_TARGET[idx]
    weights = np.asarray(samp_probs, dtype=float)
    return points, targets, weights


def _select_train_points(epoch, rng):
    if epoch < TRAIN_STAGE1_EPOCHS:
        return TOPK_POINTS, TOPK_TARGET, TOPK_WEIGHTS, TOPK_NORM
    if epoch < TRAIN_STAGE2_EPOCHS:
        points, targets, weights = _sample_characteristic_points(rng, TRAIN_POINTS_STAGE2)
        return points, targets, weights, CHAR_NORM
    points, targets, weights = _sample_characteristic_points(rng, TRAIN_POINTS_STAGE3)
    return points, targets, weights, CHAR_NORM


EVAL_RNG = np.random.default_rng(12345)
EVAL_POINTS, EVAL_TARGET, EVAL_WEIGHTS = _sample_characteristic_points(
    EVAL_RNG, TRAIN_POINTS_STAGE3, mode=CHAR_SAMPLER_MODE
)


def _eval_fidelity_batch(
    phi_r_coeff,
    phi_b_coeff,
    amp_r_coeff,
    amp_b_coeff,
    motional_detuning=0.0,
):
    phi_r_full = _expand_segments(phi_r_coeff)
    phi_b_full = _expand_segments(phi_b_coeff)
    amp_r_full = _expand_segments(amp_r_coeff)
    amp_b_full = _expand_segments(amp_b_coeff)
    return _eval_fidelity_batch_full(
        phi_r_full,
        phi_b_full,
        amp_r_full,
        amp_b_full,
        motional_detuning=motional_detuning,
    )


def _eval_fidelity_batch_full(
    phi_r_full,
    phi_b_full,
    amp_r_full,
    amp_b_full,
    motional_detuning=0.0,
):
    _, fidelity_batch, _, _ = trapped_ion_binomial_sim_batch(
        phi_r_full,
        phi_b_full,
        amp_r=amp_r_full,
        amp_b=amp_b_full,
        n_boson=N_BOSON,
        omega=OMEGA_RABI,
        t_step=T_STEP,
        binomial_code=BINOMIAL_CODE,
        binomial_phase=BINOMIAL_REL_PHASE,
        sample_points=[0.0 + 0.0j],
        target_values=np.array([1.0 + 0.0j], dtype=complex),
        sample_weights=np.array([1.0], dtype=float),
        sample_area=1.0,
        reward_scale=1.0,
        reward_clip=None,
        reward_norm=None,
        n_shots=0,
        return_details=True,
        return_density=False,
        reward_mode="characteristic",
        characteristic_objective=CHAR_REWARD_OBJECTIVE_STAGE2,
        motional_detuning=motional_detuning,
    )
    return np.asarray(fidelity_batch, dtype=float)


def _eval_robust_reward_and_fidelity_batch_full(
    phi_r_full,
    phi_b_full,
    amp_r_full,
    amp_b_full,
    sample_points,
    target_values,
    sample_weights,
    reward_norm,
    n_shots,
    reward_objective,
    rng,
    epoch_type,
    duration_scale=None,
):
    batch_size = int(phi_r_full.shape[0])
    n_noise = _noise_samples_for_epoch_type(epoch_type)
    detuning_samples = _sample_detuning_samples(
        rng,
        batch_size=batch_size,
        n_samples=n_noise,
        include_nominal=DEPHASE_INCLUDE_NOMINAL,
    )
    duration_scale_batch = (
        _broadcast_duration_scale(duration_scale, batch_size)
        if duration_scale is not None
        else None
    )
    detuning_samples = _apply_duration_scale_to_detuning(
        detuning_samples,
        duration_scale_batch,
    )
    (
        phi_r_exp,
        phi_b_exp,
        amp_r_exp,
        amp_b_exp,
        detuning_flat,
        n_noise_expanded,
    ) = _expand_controls_for_noise(
        phi_r_full,
        phi_b_full,
        amp_r_full,
        amp_b_full,
        detuning_samples,
    )
    if n_noise_expanded != n_noise:
        raise RuntimeError(
            f"Noise sample size mismatch: expected {n_noise}, got {n_noise_expanded}"
        )
    reward_flat, fidelity_flat, _, _ = trapped_ion_binomial_sim_batch(
        phi_r_exp,
        phi_b_exp,
        amp_r=amp_r_exp,
        amp_b=amp_b_exp,
        n_boson=N_BOSON,
        omega=OMEGA_RABI,
        t_step=T_STEP,
        binomial_code=BINOMIAL_CODE,
        binomial_phase=BINOMIAL_REL_PHASE,
        sample_points=sample_points,
        target_values=target_values,
        sample_weights=sample_weights,
        sample_area=CHAR_AREA,
        reward_scale=REWARD_SCALE,
        reward_clip=REWARD_CLIP,
        reward_norm=reward_norm,
        n_shots=n_shots,
        return_details=True,
        reward_mode="characteristic",
        characteristic_objective=reward_objective,
        motional_detuning=detuning_flat,
    )
    reward_matrix = np.asarray(reward_flat, dtype=float).reshape(batch_size, n_noise)
    fidelity_matrix = np.asarray(fidelity_flat, dtype=float).reshape(batch_size, n_noise)
    noise_idx, noise_weights = _compute_noise_weights_for_objective(detuning_samples)

    if DEPHASE_INCLUDE_NOMINAL:
        fidelity_nominal = fidelity_matrix[:, 0]
    else:
        fidelity_nominal = _eval_fidelity_batch_full(
            phi_r_full,
            phi_b_full,
            amp_r_full,
            amp_b_full,
            motional_detuning=0.0,
        )

    reward_robust = np.sum(reward_matrix[:, noise_idx] * noise_weights, axis=1)
    fidelity_robust = np.sum(fidelity_matrix[:, noise_idx] * noise_weights, axis=1)
    penalty = ROBUST_FLOOR_PENALTY * np.maximum(
        0.0,
        ROBUST_NOMINAL_FID_FLOOR - fidelity_nominal,
    )
    reward_objective_values = reward_robust - penalty
    robust_score = fidelity_robust - penalty

    return {
        "reward_objective": reward_objective_values,
        "reward_robust": reward_robust,
        "fidelity_nominal": fidelity_nominal,
        "fidelity_robust": fidelity_robust,
        "penalty": penalty,
        "score": robust_score,
        "detuning_samples": detuning_samples,
        "noise_idx": noise_idx,
        "noise_weights": noise_weights,
    }


def _eval_robust_refine_score_batch_full(
    phi_r_full,
    phi_b_full,
    amp_r_full,
    amp_b_full,
    rng,
    duration_scale=None,
    n_noise_override=None,
):
    batch_size = int(phi_r_full.shape[0])
    if n_noise_override is None:
        n_noise = max(1, DEPHASE_NOISE_SAMPLES_REFINE)
    else:
        n_noise = max(1, int(n_noise_override))
    detuning_samples = _sample_detuning_samples(
        rng,
        batch_size=batch_size,
        n_samples=n_noise,
        include_nominal=DEPHASE_INCLUDE_NOMINAL,
    )
    duration_scale_batch = (
        _broadcast_duration_scale(duration_scale, batch_size)
        if duration_scale is not None
        else None
    )
    detuning_samples = _apply_duration_scale_to_detuning(
        detuning_samples,
        duration_scale_batch,
    )
    (
        phi_r_exp,
        phi_b_exp,
        amp_r_exp,
        amp_b_exp,
        detuning_flat,
        n_noise_expanded,
    ) = _expand_controls_for_noise(
        phi_r_full,
        phi_b_full,
        amp_r_full,
        amp_b_full,
        detuning_samples,
    )
    if n_noise_expanded != n_noise:
        raise RuntimeError(
            f"Noise sample size mismatch: expected {n_noise}, got {n_noise_expanded}"
        )
    _, fidelity_flat, _, _ = trapped_ion_binomial_sim_batch(
        phi_r_exp,
        phi_b_exp,
        amp_r=amp_r_exp,
        amp_b=amp_b_exp,
        n_boson=N_BOSON,
        omega=OMEGA_RABI,
        t_step=T_STEP,
        binomial_code=BINOMIAL_CODE,
        binomial_phase=BINOMIAL_REL_PHASE,
        sample_points=[0.0 + 0.0j],
        target_values=np.array([1.0 + 0.0j], dtype=complex),
        sample_weights=np.array([1.0], dtype=float),
        sample_area=1.0,
        reward_scale=1.0,
        reward_clip=None,
        reward_norm=None,
        n_shots=0,
        return_details=True,
        return_density=False,
        reward_mode="characteristic",
        characteristic_objective=CHAR_REWARD_OBJECTIVE_STAGE2,
        motional_detuning=detuning_flat,
    )
    fidelity_matrix = np.asarray(fidelity_flat, dtype=float).reshape(batch_size, n_noise)
    noise_idx, noise_weights = _compute_noise_weights_for_objective(detuning_samples)
    if DEPHASE_INCLUDE_NOMINAL:
        fidelity_nominal = fidelity_matrix[:, 0]
    else:
        fidelity_nominal = _eval_fidelity_batch_full(
            phi_r_full,
            phi_b_full,
            amp_r_full,
            amp_b_full,
            motional_detuning=0.0,
        )
    fidelity_robust = np.sum(fidelity_matrix[:, noise_idx] * noise_weights, axis=1)
    penalty = ROBUST_FLOOR_PENALTY * np.maximum(
        0.0,
        ROBUST_NOMINAL_FID_FLOOR - fidelity_nominal,
    )
    score = fidelity_robust - penalty
    return score, fidelity_nominal, fidelity_robust, penalty


def _eval_refine_objective_batch_full(
    phi_r_full,
    phi_b_full,
    amp_r_full,
    amp_b_full,
    rng,
    duration_scale=None,
):
    batch_size = int(np.asarray(phi_r_full).shape[0])
    duration_scale_batch = _broadcast_duration_scale(duration_scale, batch_size)
    amp_r_drive = np.asarray(amp_r_full, dtype=float) * duration_scale_batch[:, None]
    amp_b_drive = np.asarray(amp_b_full, dtype=float) * duration_scale_batch[:, None]
    if ROBUST_TRAINING:
        score, _, _, _ = _eval_robust_refine_score_batch_full(
            phi_r_full,
            phi_b_full,
            amp_r_drive,
            amp_b_drive,
            rng,
            duration_scale=duration_scale_batch,
        )
        return score
    return _eval_fidelity_batch_full(phi_r_full, phi_b_full, amp_r_drive, amp_b_drive)


def _eval_refine_objective_batch(
    phi_r_coeff,
    phi_b_coeff,
    amp_r_coeff,
    amp_b_coeff,
    rng,
    duration_scale=None,
):
    phi_r_full = _expand_segments(phi_r_coeff)
    phi_b_full = _expand_segments(phi_b_coeff)
    amp_r_full = _expand_segments(amp_r_coeff)
    amp_b_full = _expand_segments(amp_b_coeff)
    return _eval_refine_objective_batch_full(
        phi_r_full,
        phi_b_full,
        amp_r_full,
        amp_b_full,
        rng,
        duration_scale=duration_scale,
    )


def _evaluate_single_pulse_robust_stats(
    phi_r_full,
    phi_b_full,
    amp_r_full,
    amp_b_full,
    duration_scale,
    n_seeds,
    seed_base,
    n_noise,
):
    phi_r_arr = np.asarray(phi_r_full, dtype=float).reshape(1, -1)
    phi_b_arr = np.asarray(phi_b_full, dtype=float).reshape(1, -1)
    amp_r_drive_arr = (
        np.asarray(amp_r_full, dtype=float).reshape(1, -1) * float(duration_scale)
    )
    amp_b_drive_arr = (
        np.asarray(amp_b_full, dtype=float).reshape(1, -1) * float(duration_scale)
    )
    duration_arr = np.array([float(duration_scale)], dtype=float)

    scores = []
    nominals = []
    robust_means = []
    penalties = []
    for i in range(int(n_seeds)):
        seed = int(seed_base) + i * FINAL_REFINE_CENTER_SEED_STRIDE
        score_arr, nom_arr, rob_arr, pen_arr = _eval_robust_refine_score_batch_full(
            phi_r_arr,
            phi_b_arr,
            amp_r_drive_arr,
            amp_b_drive_arr,
            np.random.default_rng(seed),
            duration_scale=duration_arr,
            n_noise_override=n_noise,
        )
        scores.append(float(score_arr[0]))
        nominals.append(float(nom_arr[0]))
        robust_means.append(float(rob_arr[0]))
        penalties.append(float(pen_arr[0]))

    score_np = np.asarray(scores, dtype=float)
    nom_np = np.asarray(nominals, dtype=float)
    rob_np = np.asarray(robust_means, dtype=float)
    pen_np = np.asarray(penalties, dtype=float)
    return {
        "mean_score": float(np.mean(score_np)),
        "std_score": float(np.std(score_np)),
        "min_score": float(np.min(score_np)),
        "max_score": float(np.max(score_np)),
        "mean_f_nom": float(np.mean(nom_np)),
        "std_f_nom": float(np.std(nom_np)),
        "mean_f_rob": float(np.mean(rob_np)),
        "std_f_rob": float(np.std(rob_np)),
        "mean_penalty": float(np.mean(pen_np)),
        "std_penalty": float(np.std(pen_np)),
        "n_seeds": int(n_seeds),
        "n_noise": int(n_noise),
    }


def _write_candidate_validation_csv(output_dir, rows):
    if not rows:
        return ""
    path = os.path.join(output_dir, "final_candidate_validation.csv")
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "label,source_metric,duration_scale,mean_score,std_score,min_score,max_score,"
            "mean_f_nom,std_f_nom,mean_f_rob,std_f_rob,mean_penalty,std_penalty,n_seeds,n_noise\n"
        )
        for row in rows:
            f.write(
                f"{row['label']},{row['source_metric']:.6f},{row['duration_scale']:.6f},"
                f"{row['mean_score']:.6f},{row['std_score']:.6f},{row['min_score']:.6f},{row['max_score']:.6f},"
                f"{row['mean_f_nom']:.6f},{row['std_f_nom']:.6f},{row['mean_f_rob']:.6f},{row['std_f_rob']:.6f},"
                f"{row['mean_penalty']:.6f},{row['std_penalty']:.6f},{row['n_seeds']},{row['n_noise']}\n"
            )
    return path


def _write_stochastic_compare_csv(output_dir, robust_row, baseline_row):
    if robust_row is None or baseline_row is None:
        return ""
    path = os.path.join(output_dir, "stochastic_compare.csv")
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "pulse,mean_score,std_score,mean_f_nom,std_f_nom,mean_f_rob,std_f_rob,"
            "mean_penalty,std_penalty,n_seeds,n_noise\n"
        )
        for pulse_name, row in [("robust", robust_row), ("baseline", baseline_row)]:
            f.write(
                f"{pulse_name},{row['mean_score']:.6f},{row['std_score']:.6f},"
                f"{row['mean_f_nom']:.6f},{row['std_f_nom']:.6f},{row['mean_f_rob']:.6f},{row['std_f_rob']:.6f},"
                f"{row['mean_penalty']:.6f},{row['std_penalty']:.6f},{row['n_seeds']},{row['n_noise']}\n"
            )
    return path


def _refine_full_steps(
    center_phi_r,
    center_phi_b,
    center_amp_r,
    center_amp_b,
    sigma_phi_r,
    sigma_phi_b,
    sigma_amp_r,
    sigma_amp_b,
    rng,
    duration_scale=1.0,
):
    metric_name = "robust_score" if ROBUST_TRAINING else "fidelity"
    if not FINAL_REFINE_FULL_STEPS or FINAL_REFINE_FULL_SAMPLES <= 0:
        phi_r_full = np.repeat(np.asarray(center_phi_r, dtype=float), SEG_LEN)
        phi_b_full = np.repeat(np.asarray(center_phi_b, dtype=float), SEG_LEN)
        amp_r_full = np.repeat(np.asarray(center_amp_r, dtype=float), SEG_LEN)
        amp_b_full = np.repeat(np.asarray(center_amp_b, dtype=float), SEG_LEN)
        objective_val = float(
            _eval_refine_objective_batch_full(
                phi_r_full[None, :],
                phi_b_full[None, :],
                amp_r_full[None, :],
                amp_b_full[None, :],
                rng,
                duration_scale=np.array([float(duration_scale)], dtype=float),
            )[0]
        )
        return phi_r_full, phi_b_full, amp_r_full, amp_b_full, objective_val

    cur_phi_r = np.repeat(np.asarray(center_phi_r, dtype=float), SEG_LEN)
    cur_phi_b = np.repeat(np.asarray(center_phi_b, dtype=float), SEG_LEN)
    cur_amp_r = np.repeat(np.asarray(center_amp_r, dtype=float), SEG_LEN)
    cur_amp_b = np.repeat(np.asarray(center_amp_b, dtype=float), SEG_LEN)

    sigma_phi_r_full = np.maximum(
        np.repeat(np.asarray(sigma_phi_r, dtype=float), SEG_LEN)
        * FINAL_REFINE_FULL_SIGMA_FACTOR,
        FINAL_REFINE_FULL_MIN_SIGMA,
    )
    sigma_phi_b_full = np.maximum(
        np.repeat(np.asarray(sigma_phi_b, dtype=float), SEG_LEN)
        * FINAL_REFINE_FULL_SIGMA_FACTOR,
        FINAL_REFINE_FULL_MIN_SIGMA,
    )
    sigma_amp_r_full = np.maximum(
        np.repeat(np.asarray(sigma_amp_r, dtype=float), SEG_LEN)
        * FINAL_REFINE_FULL_SIGMA_FACTOR_AMP,
        FINAL_REFINE_FULL_MIN_SIGMA_AMP,
    )
    sigma_amp_b_full = np.maximum(
        np.repeat(np.asarray(sigma_amp_b, dtype=float), SEG_LEN)
        * FINAL_REFINE_FULL_SIGMA_FACTOR_AMP,
        FINAL_REFINE_FULL_MIN_SIGMA_AMP,
    )

    best_phi_r = cur_phi_r.copy()
    best_phi_b = cur_phi_b.copy()
    best_amp_r = cur_amp_r.copy()
    best_amp_b = cur_amp_b.copy()
    best_objective = float(
        _eval_refine_objective_batch_full(
            best_phi_r[None, :],
            best_phi_b[None, :],
            best_amp_r[None, :],
            best_amp_b[None, :],
            rng,
            duration_scale=np.array([float(duration_scale)], dtype=float),
        )[0]
    )

    n_rounds = max(1, FINAL_REFINE_FULL_ROUNDS)
    n_samples = max(0, FINAL_REFINE_FULL_SAMPLES)
    topk = max(1, FINAL_REFINE_FULL_TOPK)

    for ridx in range(n_rounds):
        if n_samples <= 0:
            break
        scale = FINAL_REFINE_FULL_SCALE * (FINAL_REFINE_FULL_DECAY ** ridx)
        n_cand = n_samples + 1

        cand_phi_r = np.repeat(cur_phi_r[None, :], n_cand, axis=0)
        cand_phi_b = np.repeat(cur_phi_b[None, :], n_cand, axis=0)
        cand_amp_r = np.repeat(cur_amp_r[None, :], n_cand, axis=0)
        cand_amp_b = np.repeat(cur_amp_b[None, :], n_cand, axis=0)

        cand_phi_r[1:, :] = (
            cur_phi_r[None, :]
            + scale * rng.normal(size=(n_samples, N_STEPS)) * sigma_phi_r_full[None, :]
        )
        cand_phi_b[1:, :] = (
            cur_phi_b[None, :]
            + scale * rng.normal(size=(n_samples, N_STEPS)) * sigma_phi_b_full[None, :]
        )
        if FINAL_REFINE_FULL_ENABLE_AMP:
            cand_amp_r[1:, :] = (
                cur_amp_r[None, :]
                + scale
                * rng.normal(size=(n_samples, N_STEPS))
                * sigma_amp_r_full[None, :]
            )
            cand_amp_b[1:, :] = (
                cur_amp_b[None, :]
                + scale
                * rng.normal(size=(n_samples, N_STEPS))
                * sigma_amp_b_full[None, :]
            )

        cand_phi_r = np.clip(cand_phi_r, -PHASE_CLIP, PHASE_CLIP)
        cand_phi_b = np.clip(cand_phi_b, -PHASE_CLIP, PHASE_CLIP)
        cand_amp_r = np.clip(cand_amp_r, AMP_MIN, AMP_MAX)
        cand_amp_b = np.clip(cand_amp_b, AMP_MIN, AMP_MAX)

        cand_objective = _eval_refine_objective_batch_full(
            cand_phi_r,
            cand_phi_b,
            cand_amp_r,
            cand_amp_b,
            rng,
            duration_scale=np.full(n_cand, float(duration_scale), dtype=float),
        )
        order = np.argsort(cand_objective)[::-1]
        keep = order[: min(topk, len(order))]
        round_best_idx = int(order[0])
        round_best = float(cand_objective[round_best_idx])
        logger.info(
            "Full-step refine round %d/%d | scale=%.4f best_%s=%.6f mean_topk_%s=%.6f",
            ridx + 1,
            n_rounds,
            scale,
            metric_name,
            round_best,
            metric_name,
            float(np.mean(cand_objective[keep])),
        )

        if round_best > best_objective:
            best_objective = round_best
            best_phi_r = cand_phi_r[round_best_idx].copy()
            best_phi_b = cand_phi_b[round_best_idx].copy()
            best_amp_r = cand_amp_r[round_best_idx].copy()
            best_amp_b = cand_amp_b[round_best_idx].copy()

        cur_phi_r = best_phi_r.copy()
        cur_phi_b = best_phi_b.copy()
        cur_amp_r = best_amp_r.copy()
        cur_amp_b = best_amp_b.copy()
        sigma_phi_r_full = np.maximum(np.std(cand_phi_r[keep], axis=0), FINAL_REFINE_FULL_MIN_SIGMA)
        sigma_phi_b_full = np.maximum(np.std(cand_phi_b[keep], axis=0), FINAL_REFINE_FULL_MIN_SIGMA)
        if FINAL_REFINE_FULL_ENABLE_AMP:
            sigma_amp_r_full = np.maximum(
                np.std(cand_amp_r[keep], axis=0), FINAL_REFINE_FULL_MIN_SIGMA_AMP
            )
            sigma_amp_b_full = np.maximum(
                np.std(cand_amp_b[keep], axis=0), FINAL_REFINE_FULL_MIN_SIGMA_AMP
            )

    return best_phi_r, best_phi_b, best_amp_r, best_amp_b, best_objective


def _refine_around_center(
    center_phi_r,
    center_phi_b,
    center_amp_r,
    center_amp_b,
    sigma_phi_r,
    sigma_phi_b,
    sigma_amp_r,
    sigma_amp_b,
    rng,
    duration_scale=1.0,
):
    metric_name = "robust_score" if ROBUST_TRAINING else "fidelity"
    center_phi_r = np.asarray(center_phi_r, dtype=float)
    center_phi_b = np.asarray(center_phi_b, dtype=float)
    center_amp_r = np.asarray(center_amp_r, dtype=float)
    center_amp_b = np.asarray(center_amp_b, dtype=float)
    sigma_phi_r = np.asarray(sigma_phi_r, dtype=float)
    sigma_phi_b = np.asarray(sigma_phi_b, dtype=float)
    sigma_amp_r = np.asarray(sigma_amp_r, dtype=float)
    sigma_amp_b = np.asarray(sigma_amp_b, dtype=float)

    best_phi_r = center_phi_r.copy()
    best_phi_b = center_phi_b.copy()
    best_amp_r = center_amp_r.copy()
    best_amp_b = center_amp_b.copy()
    best_objective = float(
        _eval_refine_objective_batch(
            best_phi_r[None, :],
            best_phi_b[None, :],
            best_amp_r[None, :],
            best_amp_b[None, :],
            rng,
            duration_scale=np.array([float(duration_scale)], dtype=float),
        )[0]
    )

    n_rounds = max(1, FINAL_REFINE_ROUNDS)
    n_samples = max(0, FINAL_REFINE_SAMPLES)
    topk = max(1, FINAL_REFINE_TOPK)

    for ridx in range(n_rounds):
        if n_samples <= 0:
            break
        scale = FINAL_REFINE_SCALE * (FINAL_REFINE_DECAY ** ridx)
        amp_opt_active = FINAL_REFINE_ENABLE_AMP and (ridx >= FINAL_REFINE_AMP_START_ROUND)
        n_cand = n_samples + 1

        cand_phi_r = np.repeat(best_phi_r[None, :], n_cand, axis=0)
        cand_phi_b = np.repeat(best_phi_b[None, :], n_cand, axis=0)
        cand_amp_r = np.repeat(best_amp_r[None, :], n_cand, axis=0)
        cand_amp_b = np.repeat(best_amp_b[None, :], n_cand, axis=0)

        noise_r = rng.normal(size=(n_samples, N_SEGMENTS))
        noise_b = rng.normal(size=(n_samples, N_SEGMENTS))
        cand_phi_r[1:, :] = best_phi_r[None, :] + scale * noise_r * sigma_phi_r[None, :]
        cand_phi_b[1:, :] = best_phi_b[None, :] + scale * noise_b * sigma_phi_b[None, :]
        if amp_opt_active:
            noise_amp_r = rng.normal(size=(n_samples, N_SEGMENTS))
            noise_amp_b = rng.normal(size=(n_samples, N_SEGMENTS))
            cand_amp_r[1:, :] = best_amp_r[None, :] + scale * noise_amp_r * sigma_amp_r[None, :]
            cand_amp_b[1:, :] = best_amp_b[None, :] + scale * noise_amp_b * sigma_amp_b[None, :]
        cand_phi_r = np.clip(cand_phi_r, -PHASE_CLIP, PHASE_CLIP)
        cand_phi_b = np.clip(cand_phi_b, -PHASE_CLIP, PHASE_CLIP)
        cand_amp_r = np.clip(cand_amp_r, AMP_MIN, AMP_MAX)
        cand_amp_b = np.clip(cand_amp_b, AMP_MIN, AMP_MAX)

        cand_objective = _eval_refine_objective_batch(
            cand_phi_r,
            cand_phi_b,
            cand_amp_r,
            cand_amp_b,
            rng,
            duration_scale=np.full(n_cand, float(duration_scale), dtype=float),
        )
        order = np.argsort(cand_objective)[::-1]
        keep = order[: min(topk, len(order))]
        round_best_idx = int(order[0])
        round_best = float(cand_objective[round_best_idx])
        logger.info(
            "Final refine round %d/%d | scale=%.4f best_%s=%.6f mean_topk_%s=%.6f",
            ridx + 1,
            n_rounds,
            scale,
            metric_name,
            round_best,
            metric_name,
            float(np.mean(cand_objective[keep])),
        )

        if round_best > best_objective:
            best_objective = round_best
            best_phi_r = cand_phi_r[round_best_idx].copy()
            best_phi_b = cand_phi_b[round_best_idx].copy()
            best_amp_r = cand_amp_r[round_best_idx].copy()
            best_amp_b = cand_amp_b[round_best_idx].copy()

        sigma_phi_r = np.maximum(np.std(cand_phi_r[keep], axis=0), FINAL_REFINE_MIN_SIGMA)
        sigma_phi_b = np.maximum(np.std(cand_phi_b[keep], axis=0), FINAL_REFINE_MIN_SIGMA)
        if amp_opt_active:
            sigma_amp_r = np.maximum(
                np.std(cand_amp_r[keep], axis=0), FINAL_REFINE_MIN_SIGMA_AMP
            )
            sigma_amp_b = np.maximum(
                np.std(cand_amp_b[keep], axis=0), FINAL_REFINE_MIN_SIGMA_AMP
            )

    return best_phi_r, best_phi_b, best_amp_r, best_amp_b, best_objective


def _update_top_eval_actions(
    records,
    epoch,
    metric,
    phi_r,
    phi_b,
    amp_r,
    amp_b,
    duration_scale,
):
    if FINAL_REFINE_TOP_EVAL_CENTERS <= 0:
        return records
    rec = {
        "epoch": int(epoch),
        "metric": float(metric),
        "phi_r": np.array(phi_r, dtype=float).copy(),
        "phi_b": np.array(phi_b, dtype=float).copy(),
        "amp_r": np.array(amp_r, dtype=float).copy(),
        "amp_b": np.array(amp_b, dtype=float).copy(),
        "duration_scale": float(duration_scale),
    }
    filtered = [r for r in records if int(r["epoch"]) != int(epoch)]
    filtered.append(rec)
    filtered.sort(key=lambda x: float(x["metric"]), reverse=True)
    return filtered[: max(1, FINAL_REFINE_TOP_EVAL_CENTERS)]


def _pulse_to_full_steps(values, key):
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == N_STEPS:
        return arr.copy()
    if arr.size == N_SEGMENTS:
        return np.repeat(arr, SEG_LEN)
    raise ValueError(
        f"Pulse '{key}' has length {arr.size}; expected N_STEPS={N_STEPS} or N_SEGMENTS={N_SEGMENTS}."
    )


def _load_pulses_npz_as_full(npz_path):
    required = ("phi_r", "phi_b", "amp_r", "amp_b")
    with np.load(npz_path) as pulse_data:
        missing = [k for k in required if k not in pulse_data]
        if missing:
            raise ValueError(f"Missing keys {missing} in {npz_path}")
        return tuple(_pulse_to_full_steps(pulse_data[k], k) for k in required)


def _load_duration_scale_from_npz(npz_path):
    if ROBUST_COMPARE_BASELINE_DURATION_SCALE is not None:
        return float(ROBUST_COMPARE_BASELINE_DURATION_SCALE)
    with np.load(npz_path) as pulse_data:
        if "duration_scale" not in pulse_data:
            logger.warning(
                "Baseline pulse file %s has no duration_scale; defaulting to 1.0. "
                "Set ROBUST_COMPARE_BASELINE_DURATION_SCALE to override.",
                npz_path,
            )
            return 1.0
        arr = np.asarray(pulse_data["duration_scale"], dtype=float).reshape(-1)
        if arr.size == 0:
            logger.warning(
                "Baseline pulse file %s has empty duration_scale; defaulting to 1.0. "
                "Set ROBUST_COMPARE_BASELINE_DURATION_SCALE to override.",
                npz_path,
            )
            return 1.0
        return float(arr[0])


def _dephasing_sweep_axis():
    frac_axis = np.linspace(-DEPHASE_SWEEP_MAX_FRAC, DEPHASE_SWEEP_MAX_FRAC, DEPHASE_SWEEP_POINTS)
    detuning_axis = frac_axis * OMEGA_RABI
    return frac_axis, detuning_axis


def _evaluate_dephasing_sweep(phi_r_full, phi_b_full, amp_r_full, amp_b_full, detuning_axis):
    detuning_axis = np.asarray(detuning_axis, dtype=float).reshape(-1)
    n_det = detuning_axis.size
    phi_r_batch = np.repeat(np.asarray(phi_r_full, dtype=float)[None, :], n_det, axis=0)
    phi_b_batch = np.repeat(np.asarray(phi_b_full, dtype=float)[None, :], n_det, axis=0)
    amp_r_batch = np.repeat(np.asarray(amp_r_full, dtype=float)[None, :], n_det, axis=0)
    amp_b_batch = np.repeat(np.asarray(amp_b_full, dtype=float)[None, :], n_det, axis=0)
    return _eval_fidelity_batch_full(
        phi_r_batch,
        phi_b_batch,
        amp_r_batch,
        amp_b_batch,
        motional_detuning=detuning_axis,
    )


def _save_dephasing_sweep_outputs(
    output_dir,
    detuning_frac_axis,
    detuning_axis,
    robust_fidelity,
    baseline_fidelity=None,
):
    robust_csv_path = os.path.join(output_dir, "dephasing_sweep_robust.csv")
    with open(robust_csv_path, "w", encoding="utf-8") as f:
        f.write("detuning_frac,detuning,robust_fidelity\n")
        for frac, delta, fid in zip(detuning_frac_axis, detuning_axis, robust_fidelity):
            f.write(f"{float(frac):.8f},{float(delta):.8e},{float(fid):.8f}\n")

    fig_r, ax_r = plt.subplots(1, 1, figsize=(7, 4))
    ax_r.plot(detuning_frac_axis, robust_fidelity, "o-", label="robust pulse")
    ax_r.set_xlabel("Detuning / Omega")
    ax_r.set_ylabel("Fidelity")
    ax_r.set_title("Robust pulse dephasing sweep")
    ax_r.grid(alpha=0.25)
    ax_r.legend(loc="best")
    fig_r.tight_layout()
    robust_png_path = os.path.join(output_dir, "dephasing_sweep_robust.png")
    fig_r.savefig(robust_png_path, dpi=150)
    plt.close(fig_r)
    logger.info("Saved robust dephasing sweep to %s and %s", robust_csv_path, robust_png_path)

    if baseline_fidelity is None:
        return

    compare_csv_path = os.path.join(output_dir, "dephasing_compare.csv")
    with open(compare_csv_path, "w", encoding="utf-8") as f:
        f.write("detuning_frac,detuning,robust_fidelity,baseline_fidelity\n")
        for frac, delta, fid_r, fid_b in zip(
            detuning_frac_axis,
            detuning_axis,
            robust_fidelity,
            baseline_fidelity,
        ):
            f.write(
                f"{float(frac):.8f},{float(delta):.8e},{float(fid_r):.8f},{float(fid_b):.8f}\n"
            )

    fig_c, ax_c = plt.subplots(1, 1, figsize=(7, 4))
    ax_c.plot(detuning_frac_axis, baseline_fidelity, "o-", label="baseline pulse")
    ax_c.plot(detuning_frac_axis, robust_fidelity, "o-", label="robust pulse")
    ax_c.set_xlabel("Detuning / Omega")
    ax_c.set_ylabel("Fidelity")
    ax_c.set_title("Dephasing robustness comparison")
    ax_c.grid(alpha=0.25)
    ax_c.legend(loc="best")
    fig_c.tight_layout()
    compare_png_path = os.path.join(output_dir, "dephasing_compare.png")
    fig_c.savefig(compare_png_path, dpi=150)
    plt.close(fig_c)
    logger.info("Saved dephasing comparison to %s and %s", compare_csv_path, compare_png_path)


done = False
eval_log_path = os.path.join(os.getcwd(), "eval_fidelity.csv")
if os.environ.get("CLEAR_EVAL_LOG", "1") == "1" and os.path.exists(eval_log_path):
    os.remove(eval_log_path)
eval_robust_log_path = os.path.join(os.getcwd(), "eval_robust_metrics.csv")
if os.environ.get("CLEAR_EVAL_LOG", "1") == "1" and os.path.exists(eval_robust_log_path):
    os.remove(eval_robust_log_path)
best_eval_fidelity = -np.inf
best_eval_score = -np.inf
best_eval_epoch = -1
best_eval_action = None
best_train_reward = -np.inf
best_train_epoch = -1
best_train_action = None
top_eval_actions = []
last_reward_objective = None
reward_switch_block_logged = False
if ROBUST_TRAINING and not ROBUST_COMPARE_BASELINE_NPZ:
    logger.warning(
        "ROBUST_COMPARE_BASELINE_NPZ is not set. Robust-vs-nonrobust sweep comparison will be skipped."
    )
while not done:
    message, done = client_socket.recv_data()
    logger.info("Received message from RL agent server.")
    logger.info("Time stamp: %f", time.time())

    if done:
        logger.info("Training finished.")
        break

    epoch_type = message["epoch_type"]
    fidelity_data = None

    if epoch_type == "final":
        logger.info("Final Epoch")
        baseline_pulses = None
        baseline_duration_scale = 1.0
        baseline_path = ""
        if ROBUST_COMPARE_BASELINE_NPZ:
            baseline_path = (
                ROBUST_COMPARE_BASELINE_NPZ
                if os.path.isabs(ROBUST_COMPARE_BASELINE_NPZ)
                else os.path.join(os.getcwd(), ROBUST_COMPARE_BASELINE_NPZ)
            )
            if not os.path.exists(baseline_path):
                raise FileNotFoundError(f"ROBUST_COMPARE_BASELINE_NPZ not found: {baseline_path}")
            baseline_pulses = _load_pulses_npz_as_full(baseline_path)
            baseline_duration_scale = _load_duration_scale_from_npz(baseline_path)

        locs = message["locs"]
        scales = message["scales"]
        for key in locs.keys():
            logger.info("locs[%s]:", key)
            logger.info(locs[key][0])
            logger.info("scales[%s]:", key)
            logger.info(scales[key][0])
        loc_phi_r = np.array(locs["phi_r"][0], dtype=float)
        loc_phi_b = np.array(locs["phi_b"][0], dtype=float)
        loc_amp_r = np.array(locs.get("amp_r", [np.ones(N_SEGMENTS)])[0], dtype=float)
        loc_amp_b = np.array(locs.get("amp_b", [np.ones(N_SEGMENTS)])[0], dtype=float)
        scale_phi_r = np.array(scales["phi_r"][0], dtype=float)
        scale_phi_b = np.array(scales["phi_b"][0], dtype=float)
        scale_amp_r = np.array(
            scales.get("amp_r", [np.full(N_SEGMENTS, FINAL_REFINE_INIT_SIGMA_AMP)])[0],
            dtype=float,
        )
        scale_amp_b = np.array(
            scales.get("amp_b", [np.full(N_SEGMENTS, FINAL_REFINE_INIT_SIGMA_AMP)])[0],
            dtype=float,
        )

        if best_eval_action is not None:
            if ROBUST_TRAINING:
                logger.info(
                    "Using best evaluation action from epoch %d with robust_score %.6f (nominal_fidelity %.6f)",
                    best_eval_epoch,
                    best_eval_score,
                    best_eval_fidelity,
                )
            else:
                logger.info(
                    "Using best evaluation action from epoch %d with eval fidelity %.6f",
                    best_eval_epoch,
                    best_eval_fidelity,
                )
            base_phi_r = np.array(best_eval_action["phi_r"], dtype=float)
            base_phi_b = np.array(best_eval_action["phi_b"], dtype=float)
            base_amp_r = np.array(best_eval_action["amp_r"], dtype=float)
            base_amp_b = np.array(best_eval_action["amp_b"], dtype=float)
            base_duration_scale = float(best_eval_action.get("duration_scale", 1.0))
        else:
            base_phi_r = loc_phi_r
            base_phi_b = loc_phi_b
            base_amp_r = loc_amp_r
            base_amp_b = loc_amp_b
            duration_vals = np.array(locs.get("duration_scale", [np.array([1.0])])[0])
            base_duration_scale = float(np.asarray(duration_vals, dtype=float).reshape(-1)[0])
        loc_duration_vals = np.array(locs.get("duration_scale", [np.array([1.0])])[0])
        loc_duration_scale = float(np.asarray(loc_duration_vals, dtype=float).reshape(-1)[0])
        base_duration_scale = float(
            np.clip(base_duration_scale, DURATION_MIN_SCALE, DURATION_MAX_SCALE)
        )
        loc_duration_scale = float(
            np.clip(loc_duration_scale, DURATION_MIN_SCALE, DURATION_MAX_SCALE)
        )
        logger.info("Final base duration_scale=%.6f", base_duration_scale)
        seed_phi_r = np.array(base_phi_r, dtype=float).copy()
        seed_phi_b = np.array(base_phi_b, dtype=float).copy()
        seed_amp_r = np.array(base_amp_r, dtype=float).copy()
        seed_amp_b = np.array(base_amp_b, dtype=float).copy()
        seed_duration_scale = float(base_duration_scale)
        seed_source_metric = (
            float(best_eval_score)
            if (ROBUST_TRAINING and np.isfinite(best_eval_score))
            else float(best_eval_fidelity)
        )

        # Multi-round local refinement in phase space. In robust mode this
        # optimizes robust score; otherwise it optimizes nominal fidelity.
        sigma_phi_r = np.maximum(np.asarray(scale_phi_r, dtype=float), FINAL_REFINE_MIN_SIGMA)
        sigma_phi_b = np.maximum(np.asarray(scale_phi_b, dtype=float), FINAL_REFINE_MIN_SIGMA)
        sigma_amp_r = np.maximum(np.asarray(scale_amp_r, dtype=float), FINAL_REFINE_MIN_SIGMA_AMP)
        sigma_amp_b = np.maximum(np.asarray(scale_amp_b, dtype=float), FINAL_REFINE_MIN_SIGMA_AMP)
        centers = [
            (
                base_phi_r,
                base_phi_b,
                base_amp_r,
                base_amp_b,
                base_duration_scale,
                "best_eval",
            )
        ]
        if FINAL_REFINE_USE_LOC_CENTER:
            centers.append(
                (
                    loc_phi_r,
                    loc_phi_b,
                    loc_amp_r,
                    loc_amp_b,
                    loc_duration_scale,
                    "final_loc",
                )
            )
        if FINAL_REFINE_USE_TRAIN_CENTER and best_train_action is not None:
            centers.append(
                (
                    np.array(best_train_action["phi_r"], dtype=float),
                    np.array(best_train_action["phi_b"], dtype=float),
                    np.array(best_train_action["amp_r"], dtype=float),
                    np.array(best_train_action["amp_b"], dtype=float),
                    float(
                        np.clip(
                            float(best_train_action.get("duration_scale", base_duration_scale)),
                            DURATION_MIN_SCALE,
                            DURATION_MAX_SCALE,
                        )
                    ),
                    "best_train_reward",
                )
            )
        if FINAL_REFINE_TOP_EVAL_CENTERS > 0 and top_eval_actions:
            added_top_eval = 0
            for rec in top_eval_actions:
                if best_eval_action is not None and int(rec["epoch"]) == int(best_eval_epoch):
                    continue
                centers.append(
                    (
                        np.array(rec["phi_r"], dtype=float),
                        np.array(rec["phi_b"], dtype=float),
                        np.array(rec["amp_r"], dtype=float),
                        np.array(rec["amp_b"], dtype=float),
                        float(
                            np.clip(
                                float(rec.get("duration_scale", base_duration_scale)),
                                DURATION_MIN_SCALE,
                                DURATION_MAX_SCALE,
                            )
                        ),
                        f"top_eval_epoch_{int(rec['epoch'])}",
                    )
                )
                added_top_eval += 1
            logger.info(
                "Added %d historical top-eval centers for refinement",
                added_top_eval,
            )

        metric_name = "robust_score" if ROBUST_TRAINING else "fidelity"
        global_best = None
        global_best_metric = -np.inf
        global_best_label = "none"
        for cidx, (c_phi_r, c_phi_b, c_amp_r, c_amp_b, c_duration_scale, label) in enumerate(
            centers
        ):
            # Use an independent RNG stream per center to avoid correlated
            # candidate sets when comparing multiple refinement centers.
            center_seed = FINAL_REFINE_SEED + cidx * FINAL_REFINE_CENTER_SEED_STRIDE
            rng_ref = np.random.default_rng(center_seed)
            logger.info("Final refinement center=%s | seed=%d", label, center_seed)
            b_phi_r, b_phi_b, b_amp_r, b_amp_b, b_metric = _refine_around_center(
                c_phi_r,
                c_phi_b,
                c_amp_r,
                c_amp_b,
                sigma_phi_r,
                sigma_phi_b,
                sigma_amp_r,
                sigma_amp_b,
                rng_ref,
                duration_scale=c_duration_scale,
            )
            logger.info(
                "Final refinement center=%s | best %s %.6f",
                label,
                metric_name,
                b_metric,
            )
            if b_metric > global_best_metric:
                global_best_metric = b_metric
                global_best = (
                    b_phi_r,
                    b_phi_b,
                    b_amp_r,
                    b_amp_b,
                    float(np.clip(c_duration_scale, DURATION_MIN_SCALE, DURATION_MAX_SCALE)),
                )
                global_best_label = label

        if global_best is not None:
            (
                base_phi_r,
                base_phi_b,
                base_amp_r,
                base_amp_b,
                base_duration_scale,
            ) = global_best
            logger.info(
                "Final refinement selected center=%s with best sampled %s %.6f (duration_scale=%.6f)",
                global_best_label,
                metric_name,
                global_best_metric,
                base_duration_scale,
            )

        refined_phi_r_full = np.repeat(base_phi_r, SEG_LEN)
        refined_phi_b_full = np.repeat(base_phi_b, SEG_LEN)
        refined_amp_r_full = np.repeat(base_amp_r, SEG_LEN)
        refined_amp_b_full = np.repeat(base_amp_b, SEG_LEN)

        if FINAL_REFINE_FULL_STEPS:
            full_seed = FINAL_REFINE_SEED + 97 * FINAL_REFINE_CENTER_SEED_STRIDE
            logger.info("Starting full-step refinement | seed=%d", full_seed)
            (
                full_phi_r,
                full_phi_b,
                full_amp_r,
                full_amp_b,
                full_best_metric,
            ) = _refine_full_steps(
                base_phi_r,
                base_phi_b,
                base_amp_r,
                base_amp_b,
                sigma_phi_r,
                sigma_phi_b,
                sigma_amp_r,
                sigma_amp_b,
                np.random.default_rng(full_seed),
                duration_scale=base_duration_scale,
            )
            logger.info(
                "Full-step refinement best sampled %s %.6f",
                metric_name,
                full_best_metric,
            )
        else:
            full_phi_r = refined_phi_r_full.copy()
            full_phi_b = refined_phi_b_full.copy()
            full_amp_r = refined_amp_r_full.copy()
            full_amp_b = refined_amp_b_full.copy()
            full_best_metric = float("nan")

        candidate_pool = [
            {
                "label": f"refine_center_{global_best_label}",
                "phi_r": refined_phi_r_full,
                "phi_b": refined_phi_b_full,
                "amp_r": refined_amp_r_full,
                "amp_b": refined_amp_b_full,
                "duration_scale": float(base_duration_scale),
                "source_metric": float(global_best_metric),
            }
        ]
        if FINAL_REFINE_FULL_STEPS:
            candidate_pool.append(
                {
                    "label": "refine_full_step",
                    "phi_r": np.asarray(full_phi_r, dtype=float),
                    "phi_b": np.asarray(full_phi_b, dtype=float),
                    "amp_r": np.asarray(full_amp_r, dtype=float),
                    "amp_b": np.asarray(full_amp_b, dtype=float),
                    "duration_scale": float(base_duration_scale),
                    "source_metric": float(full_best_metric),
                }
            )
        if FINAL_VALIDATE_USE_SCORE_GUARD and best_eval_action is not None:
            candidate_pool.append(
                {
                    "label": "best_eval_seed",
                    "phi_r": np.repeat(seed_phi_r, SEG_LEN),
                    "phi_b": np.repeat(seed_phi_b, SEG_LEN),
                    "amp_r": np.repeat(seed_amp_r, SEG_LEN),
                    "amp_b": np.repeat(seed_amp_b, SEG_LEN),
                    "duration_scale": float(seed_duration_scale),
                    "source_metric": float(seed_source_metric),
                }
            )

        output_dir = os.path.join(os.getcwd(), "outputs")
        os.makedirs(output_dir, exist_ok=True)
        if FINAL_REFINE_FULL_STEPS:
            selected_candidate = next(
                c for c in candidate_pool if c["label"] == "refine_full_step"
            )
        else:
            selected_candidate = candidate_pool[0]
        selected_validation_stats = None
        if ROBUST_TRAINING and FINAL_VALIDATE_ENABLE and len(candidate_pool) > 1:
            scored_pairs = []
            csv_rows = []
            for idx, cand in enumerate(candidate_pool):
                stats = _evaluate_single_pulse_robust_stats(
                    cand["phi_r"],
                    cand["phi_b"],
                    cand["amp_r"],
                    cand["amp_b"],
                    duration_scale=cand["duration_scale"],
                    n_seeds=FINAL_VALIDATE_NUM_SEEDS,
                    seed_base=FINAL_VALIDATE_SEED_BASE + idx * FINAL_REFINE_CENTER_SEED_STRIDE,
                    n_noise=FINAL_VALIDATE_NOISE_SAMPLES,
                )
                row = {
                    "label": cand["label"],
                    "source_metric": float(cand["source_metric"]),
                    "duration_scale": float(cand["duration_scale"]),
                    **stats,
                }
                scored_pairs.append((row, cand))
                csv_rows.append(row)
                logger.info(
                    "Final candidate %s | source_%s=%.6f validated mean_score=%.6f std=%.6f mean_nom=%.6f mean_rob=%.6f",
                    row["label"],
                    metric_name,
                    row["source_metric"],
                    row["mean_score"],
                    row["std_score"],
                    row["mean_f_nom"],
                    row["mean_f_rob"],
                )
            validation_csv = _write_candidate_validation_csv(output_dir, csv_rows)
            if validation_csv:
                logger.info("Saved final candidate validation to %s", validation_csv)
            best_row, selected_candidate = max(scored_pairs, key=lambda x: x[0]["mean_score"])
            selected_validation_stats = best_row
            logger.info(
                "Selected final candidate by validated robust score: %s (mean_score=%.6f std=%.6f n_seeds=%d n_noise=%d)",
                best_row["label"],
                best_row["mean_score"],
                best_row["std_score"],
                best_row["n_seeds"],
                best_row["n_noise"],
            )
        else:
            logger.info(
                "Final candidate validation skipped (robust=%s enabled=%s pool_size=%d)",
                ROBUST_TRAINING,
                FINAL_VALIDATE_ENABLE,
                len(candidate_pool),
            )

        phi_r_final = np.asarray(selected_candidate["phi_r"], dtype=float)
        phi_b_final = np.asarray(selected_candidate["phi_b"], dtype=float)
        amp_r_final = np.asarray(selected_candidate["amp_r"], dtype=float)
        amp_b_final = np.asarray(selected_candidate["amp_b"], dtype=float)
        base_duration_scale = float(selected_candidate["duration_scale"])
        amp_r_drive_final = np.asarray(amp_r_final, dtype=float) * float(base_duration_scale)
        amp_b_drive_final = np.asarray(amp_b_final, dtype=float) * float(base_duration_scale)

        reward_objective_final = _reward_schedule_state["active"]

        _, final_fidelity, rho_final, rho_target = trapped_ion_binomial_sim(
            phi_r_final,
            phi_b_final,
            amp_r=amp_r_drive_final,
            amp_b=amp_b_drive_final,
            n_boson=N_BOSON,
            omega=OMEGA_RABI,
            t_step=T_STEP,
            binomial_code=BINOMIAL_CODE,
            binomial_phase=BINOMIAL_REL_PHASE,
            sample_points=FINAL_POINTS,
            target_values=FINAL_TARGET,
            sample_weights=FINAL_WEIGHTS,
            sample_area=FINAL_AREA,
            reward_scale=REWARD_SCALE,
            reward_clip=REWARD_CLIP,
            reward_norm=FINAL_NORM,
            n_shots=0,
            return_details=True,
            return_density=True,
            reward_mode="characteristic",
            characteristic_objective=reward_objective_final,
        )

        fidelity_path = os.path.join(output_dir, "final_fidelity.txt")
        with open(fidelity_path, "w", encoding="utf-8") as f:
            f.write(f"{final_fidelity:.6f}\n")
        pulse_path = os.path.join(output_dir, "final_pulses.npz")
        np.savez(
            pulse_path,
            phi_r=np.asarray(phi_r_final, dtype=float),
            phi_b=np.asarray(phi_b_final, dtype=float),
            amp_r=np.asarray(amp_r_final, dtype=float),
            amp_b=np.asarray(amp_b_final, dtype=float),
            duration_scale=np.array([float(base_duration_scale)], dtype=float),
        )
        logger.info("Final fidelity %.6f", final_fidelity)
        logger.info("Final duration_scale %.6f", float(base_duration_scale))
        logger.info("Saved final fidelity to %s", fidelity_path)
        logger.info("Saved final pulses to %s", pulse_path)

        final_robust_score = None
        if ROBUST_TRAINING:
            if selected_validation_stats is not None:
                final_robust_score = float(selected_validation_stats["mean_score"])
                final_robust_nominal = float(selected_validation_stats["mean_f_nom"])
                final_robust_mean = float(selected_validation_stats["mean_f_rob"])
                final_robust_penalty = float(selected_validation_stats["mean_penalty"])
                logger.info(
                    "Final robust metrics use validated mean over %d seeds x %d noise samples",
                    int(selected_validation_stats["n_seeds"]),
                    int(selected_validation_stats["n_noise"]),
                )
            else:
                robust_seed = FINAL_REFINE_SEED + 2027 * FINAL_REFINE_CENTER_SEED_STRIDE
                score_arr, nom_arr, rob_arr, pen_arr = _eval_robust_refine_score_batch_full(
                    np.asarray(phi_r_final, dtype=float)[None, :],
                    np.asarray(phi_b_final, dtype=float)[None, :],
                    np.asarray(amp_r_drive_final, dtype=float)[None, :],
                    np.asarray(amp_b_drive_final, dtype=float)[None, :],
                    np.random.default_rng(robust_seed),
                    duration_scale=np.array([float(base_duration_scale)], dtype=float),
                )
                final_robust_score = float(score_arr[0])
                final_robust_nominal = float(nom_arr[0])
                final_robust_mean = float(rob_arr[0])
                final_robust_penalty = float(pen_arr[0])
            robust_txt = os.path.join(output_dir, "final_robust_score.txt")
            with open(robust_txt, "w", encoding="utf-8") as f:
                f.write(
                    f"score={final_robust_score:.6f},f_nom={final_robust_nominal:.6f},"
                    f"f_rob={final_robust_mean:.6f},penalty={final_robust_penalty:.6f}\n"
                )
            logger.info(
                "Final robust metrics: score=%.6f f_nom=%.6f f_rob=%.6f penalty=%.6f",
                final_robust_score,
                final_robust_nominal,
                final_robust_mean,
                final_robust_penalty,
            )
            logger.info("Saved final robust metrics to %s", robust_txt)

            if DEPHASE_MODEL == "stochastic" and baseline_pulses is not None:
                robust_stoch_stats = (
                    selected_validation_stats
                    if selected_validation_stats is not None
                    else _evaluate_single_pulse_robust_stats(
                        phi_r_final,
                        phi_b_final,
                        amp_r_final,
                        amp_b_final,
                        duration_scale=float(base_duration_scale),
                        n_seeds=FINAL_VALIDATE_NUM_SEEDS,
                        seed_base=FINAL_VALIDATE_SEED_BASE,
                        n_noise=FINAL_VALIDATE_NOISE_SAMPLES,
                    )
                )
                phi_r_baseline, phi_b_baseline, amp_r_baseline, amp_b_baseline = baseline_pulses
                baseline_stoch_stats = _evaluate_single_pulse_robust_stats(
                    phi_r_baseline,
                    phi_b_baseline,
                    amp_r_baseline,
                    amp_b_baseline,
                    duration_scale=float(baseline_duration_scale),
                    n_seeds=FINAL_VALIDATE_NUM_SEEDS,
                    seed_base=FINAL_VALIDATE_SEED_BASE + 77 * FINAL_REFINE_CENTER_SEED_STRIDE,
                    n_noise=FINAL_VALIDATE_NOISE_SAMPLES,
                )
                stoch_compare_csv = _write_stochastic_compare_csv(
                    output_dir,
                    robust_stoch_stats,
                    baseline_stoch_stats,
                )
                if stoch_compare_csv:
                    logger.info(
                        "Saved stochastic pulse comparison to %s | robust mean_score=%.6f baseline mean_score=%.6f",
                        stoch_compare_csv,
                        robust_stoch_stats["mean_score"],
                        baseline_stoch_stats["mean_score"],
                    )

        checkpoint_dir = os.path.join(os.getcwd(), "checkpoint")
        os.makedirs(checkpoint_dir, exist_ok=True)
        best_fid_txt = os.path.join(checkpoint_dir, "final_fidelity_best.txt")
        best_pulses_npz = os.path.join(checkpoint_dir, "final_pulses_best.npz")
        prev_best = -np.inf
        if os.path.exists(best_fid_txt):
            try:
                with open(best_fid_txt, "r", encoding="utf-8") as f:
                    prev_best = float(f.read().strip())
            except Exception:
                prev_best = -np.inf
        if final_fidelity > prev_best:
            with open(best_fid_txt, "w", encoding="utf-8") as f:
                f.write(f"{final_fidelity:.6f}\n")
            np.savez(
                best_pulses_npz,
                phi_r=np.asarray(phi_r_final, dtype=float),
                phi_b=np.asarray(phi_b_final, dtype=float),
                amp_r=np.asarray(amp_r_final, dtype=float),
                amp_b=np.asarray(amp_b_final, dtype=float),
                duration_scale=np.array([float(base_duration_scale)], dtype=float),
            )
            logger.info(
                "Updated checkpoint best fidelity from %.6f to %.6f",
                prev_best,
                final_fidelity,
            )
        if ROBUST_TRAINING and final_robust_score is not None:
            best_robust_txt = os.path.join(checkpoint_dir, "final_robust_score_best.txt")
            best_robust_npz = os.path.join(checkpoint_dir, "final_pulses_robust_best.npz")
            prev_robust_best = -np.inf
            if os.path.exists(best_robust_txt):
                try:
                    with open(best_robust_txt, "r", encoding="utf-8") as f:
                        prev_robust_best = float(f.read().strip())
                except Exception:
                    prev_robust_best = -np.inf
            if final_robust_score > prev_robust_best:
                with open(best_robust_txt, "w", encoding="utf-8") as f:
                    f.write(f"{final_robust_score:.6f}\n")
                np.savez(
                    best_robust_npz,
                    phi_r=np.asarray(phi_r_final, dtype=float),
                    phi_b=np.asarray(phi_b_final, dtype=float),
                    amp_r=np.asarray(amp_r_final, dtype=float),
                    amp_b=np.asarray(amp_b_final, dtype=float),
                    duration_scale=np.array([float(base_duration_scale)], dtype=float),
                )
                logger.info(
                    "Updated checkpoint best robust score from %.6f to %.6f",
                    prev_robust_best,
                    final_robust_score,
                )

        grid = np.linspace(-PLOT_EXTENT, PLOT_EXTENT, PLOT_GRID_SIZE)
        chi_target = characteristic_grid(rho_target, grid, grid)
        chi_final = characteristic_grid(rho_final, grid, grid)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        im0 = axes[0].imshow(
            chi_target,
            extent=[-PLOT_EXTENT, PLOT_EXTENT, -PLOT_EXTENT, PLOT_EXTENT],
            origin="lower",
            cmap="RdBu_r",
        )
        axes[0].set_title("Target binomial characteristic")
        axes[0].set_xlabel("Re(alpha)")
        axes[0].set_ylabel("Im(alpha)")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(
            chi_final,
            extent=[-PLOT_EXTENT, PLOT_EXTENT, -PLOT_EXTENT, PLOT_EXTENT],
            origin="lower",
            cmap="RdBu_r",
        )
        axes[1].set_title("Final state characteristic")
        axes[1].set_xlabel("Re(alpha)")
        axes[1].set_ylabel("Im(alpha)")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        fig.tight_layout()
        char_path = os.path.join(output_dir, "char_target_vs_final.png")
        fig.savefig(char_path, dpi=150)
        plt.close(fig)
        logger.info("Saved characteristic plot to %s", char_path)

        detuning_frac_axis, detuning_axis = _dephasing_sweep_axis()
        robust_curve = _evaluate_dephasing_sweep(
            phi_r_final,
            phi_b_final,
            amp_r_drive_final,
            amp_b_drive_final,
            detuning_axis * float(base_duration_scale),
        )
        baseline_curve = None
        if baseline_pulses is not None:
            phi_r_baseline, phi_b_baseline, amp_r_baseline, amp_b_baseline = baseline_pulses
            baseline_curve = _evaluate_dephasing_sweep(
                phi_r_baseline,
                phi_b_baseline,
                np.asarray(amp_r_baseline, dtype=float) * float(baseline_duration_scale),
                np.asarray(amp_b_baseline, dtype=float) * float(baseline_duration_scale),
                detuning_axis * float(baseline_duration_scale),
            )
            logger.info("Loaded baseline pulses for dephasing comparison: %s", baseline_path)
        _save_dephasing_sweep_outputs(
            output_dir,
            detuning_frac_axis,
            detuning_axis,
            robust_curve,
            baseline_curve,
        )

        done = True
        logger.info("Training finished.")
        break

    action_batch = message["action_batch"]
    batch_size = message["batch_size"]
    epoch = message["epoch"]

    phi_r_coeff = action_batch["phi_r"].reshape([batch_size, -1])
    phi_b_coeff = action_batch["phi_b"].reshape([batch_size, -1])
    amp_r_coeff = action_batch["amp_r"].reshape([batch_size, -1])
    amp_b_coeff = action_batch["amp_b"].reshape([batch_size, -1])
    duration_scale_coeff = np.asarray(
        action_batch.get("duration_scale", np.ones((batch_size, 1), dtype=float)),
        dtype=float,
    ).reshape([batch_size, -1])
    best_eval_metric_for_switch = best_eval_score if ROBUST_TRAINING else best_eval_fidelity
    effective_train_epoch = _effective_train_epoch(epoch, epoch_type)
    if (
        epoch_type == "evaluation"
        and CHAR_REWARD_OBJECTIVE_STAGE2 != CHAR_REWARD_OBJECTIVE
        and CHAR_REWARD_SWITCH_EPOCH >= 0
        and effective_train_epoch >= CHAR_REWARD_SWITCH_EPOCH
        and (not _reward_schedule_state["switched"])
        and best_eval_metric_for_switch < CHAR_REWARD_SWITCH_MIN_BEST_EVAL
        and (not reward_switch_block_logged)
    ):
        logger.info(
            "Reward objective switch deferred at eval epoch %d (switch_metric=%.6f < min_best_eval=%.6f)",
            epoch,
            best_eval_metric_for_switch,
            CHAR_REWARD_SWITCH_MIN_BEST_EVAL,
        )
        reward_switch_block_logged = True
    prev_reward_objective = _reward_schedule_state["active"]
    reward_objective = _update_reward_objective(epoch, epoch_type, best_eval_metric_for_switch)
    if reward_objective != last_reward_objective:
        if (
            prev_reward_objective == CHAR_REWARD_OBJECTIVE
            and reward_objective == CHAR_REWARD_OBJECTIVE_STAGE2
        ):
            logger.info(
                "Reward objective switched at %s epoch %d -> %s (anchor_best_eval=%.6f)",
                epoch_type,
                epoch,
                reward_objective,
                _reward_schedule_state["anchor_best_eval"],
            )
        elif (
            prev_reward_objective == CHAR_REWARD_OBJECTIVE_STAGE2
            and reward_objective == CHAR_REWARD_OBJECTIVE
            and _reward_schedule_state["reverted"]
        ):
            logger.info(
                "Reward objective reverted at %s epoch %d -> %s "
                "(stage2_best_eval=%.6f, required>=%.6f)",
                epoch_type,
                epoch,
                reward_objective,
                _reward_schedule_state["stage2_best_eval"],
                _reward_schedule_state["anchor_best_eval"] + CHAR_REWARD_STAGE2_MIN_GAIN,
            )
        else:
            logger.info(
                "Reward objective switched at %s epoch %d -> %s",
                epoch_type,
                epoch,
                reward_objective,
            )
        last_reward_objective = reward_objective

    logger.info("Start %s epoch %d", epoch_type, epoch)
    if epoch_type == "evaluation" or epoch % 20 == 0:
        _log_action_stats(
            f"Epoch {epoch} ({epoch_type})",
            phi_r_coeff,
            phi_b_coeff,
            amp_r_coeff,
            amp_b_coeff,
        )
        _log_batch_diversity(
            f"Epoch {epoch} ({epoch_type})",
            phi_r_coeff,
            phi_b_coeff,
            amp_r_coeff,
            amp_b_coeff,
        )
        logger.info(
            "Epoch %d (%s) duration_scale stats: mean=%.6f std=%.6f min=%.6f max=%.6f",
            epoch,
            epoch_type,
            float(np.mean(duration_scale_coeff)),
            float(np.std(duration_scale_coeff)),
            float(np.min(duration_scale_coeff)),
            float(np.max(duration_scale_coeff)),
        )

    rng = np.random.default_rng(epoch)
    if epoch_type == "training":
        if ACTION_NOISE_PHI > 0.0:
            phi_r_coeff = phi_r_coeff + rng.normal(0.0, ACTION_NOISE_PHI, size=phi_r_coeff.shape)
            phi_b_coeff = phi_b_coeff + rng.normal(0.0, ACTION_NOISE_PHI, size=phi_b_coeff.shape)
        if ACTION_NOISE_AMP > 0.0:
            amp_r_coeff = amp_r_coeff + rng.normal(0.0, ACTION_NOISE_AMP, size=amp_r_coeff.shape)
            amp_b_coeff = amp_b_coeff + rng.normal(0.0, ACTION_NOISE_AMP, size=amp_b_coeff.shape)
        if ACTION_NOISE_DURATION > 0.0:
            duration_scale_coeff = duration_scale_coeff + rng.normal(
                0.0,
                ACTION_NOISE_DURATION,
                size=duration_scale_coeff.shape,
            )

    phi_r_before = phi_r_coeff.copy()
    phi_b_before = phi_b_coeff.copy()
    amp_r_before = amp_r_coeff.copy()
    amp_b_before = amp_b_coeff.copy()
    duration_before = duration_scale_coeff.copy()
    phi_r_coeff = np.clip(phi_r_coeff, -PHASE_CLIP, PHASE_CLIP)
    phi_b_coeff = np.clip(phi_b_coeff, -PHASE_CLIP, PHASE_CLIP)
    amp_r_coeff = np.clip(amp_r_coeff, AMP_MIN, AMP_MAX)
    amp_b_coeff = np.clip(amp_b_coeff, AMP_MIN, AMP_MAX)
    duration_scale_coeff = np.clip(
        duration_scale_coeff,
        DURATION_MIN_SCALE,
        DURATION_MAX_SCALE,
    )
    duration_scale = np.mean(duration_scale_coeff, axis=1)
    if epoch_type == "evaluation" or epoch % 20 == 0:
        clip_phi_r = float(np.mean(phi_r_before != phi_r_coeff))
        clip_phi_b = float(np.mean(phi_b_before != phi_b_coeff))
        clip_amp_r = float(np.mean(amp_r_before != amp_r_coeff))
        clip_amp_b = float(np.mean(amp_b_before != amp_b_coeff))
        clip_duration = float(np.mean(duration_before != duration_scale_coeff))
        logger.info(
            "Clip ratio phi_r=%.3f phi_b=%.3f amp_r=%.3f amp_b=%.3f duration_scale=%.3f",
            clip_phi_r,
            clip_phi_b,
            clip_amp_r,
            clip_amp_b,
            clip_duration,
        )
        logger.info(
            "Duration scale (%s): mean=%.6f std=%.6f min=%.6f max=%.6f",
            epoch_type,
            float(np.mean(duration_scale)),
            float(np.std(duration_scale)),
            float(np.min(duration_scale)),
            float(np.max(duration_scale)),
        )

    phi_r = np.repeat(phi_r_coeff, SEG_LEN, axis=1)
    phi_b = np.repeat(phi_b_coeff, SEG_LEN, axis=1)
    amp_r = np.repeat(amp_r_coeff, SEG_LEN, axis=1) * duration_scale[:, None]
    amp_b = np.repeat(amp_b_coeff, SEG_LEN, axis=1) * duration_scale[:, None]
    if epoch_type == "evaluation":
        n_shots = N_SHOTS_EVAL
        sample_points, target_values, sample_weights = EVAL_POINTS, EVAL_TARGET, EVAL_WEIGHTS
        reward_norm = CHAR_NORM if CHAR_USE_FIXED_REWARD_NORM else None
    else:
        n_shots = N_SHOTS_TRAIN
        sample_points, target_values, sample_weights, reward_norm = _select_train_points(
            epoch, rng
        )
        if not CHAR_USE_FIXED_REWARD_NORM:
            reward_norm = None

    robust_reward_rob = None
    robust_fidelity_nom = None
    robust_fidelity_rob = None
    robust_penalty = None
    robust_score = None
    if ROBUST_TRAINING:
        robust_stats = _eval_robust_reward_and_fidelity_batch_full(
            phi_r,
            phi_b,
            amp_r,
            amp_b,
            sample_points=sample_points,
            target_values=target_values,
            sample_weights=sample_weights,
            reward_norm=reward_norm,
            n_shots=n_shots,
            reward_objective=reward_objective,
            rng=rng,
            epoch_type=epoch_type,
            duration_scale=duration_scale,
        )
        robust_reward_rob = robust_stats["reward_robust"]
        robust_fidelity_nom = robust_stats["fidelity_nominal"]
        robust_fidelity_rob = robust_stats["fidelity_robust"]
        robust_penalty = robust_stats["penalty"]
        robust_score = robust_stats["score"]
        if epoch_type == "evaluation":
            fidelity_data = robust_fidelity_nom

        if epoch_type == "evaluation" or epoch % 20 == 0:
            detuning_samples = np.asarray(robust_stats["detuning_samples"], dtype=float)
            noise_idx = np.asarray(robust_stats["noise_idx"], dtype=int)
            noise_weights = np.asarray(robust_stats["noise_weights"], dtype=float)
            det_eval = np.asarray(detuning_samples[:, noise_idx, ...], dtype=float).reshape(-1)
            weight_eval = noise_weights.reshape(-1)
            eff_samples = 1.0 / np.maximum(np.sum(noise_weights ** 2, axis=1), 1.0e-12)
            scale_ref = (
                _detuning_abs_max()
                if DEPHASE_MODEL == "quasi_static"
                else _stochastic_sigma_abs()
            )
            logger.info(
                "Robust detuning stats (%s): model=%s scale_ref=%.6e mean=%.6e std=%.6e min=%.6e max=%.6e",
                epoch_type,
                DEPHASE_MODEL,
                float(scale_ref),
                float(np.mean(det_eval)),
                float(np.std(det_eval)),
                float(np.min(det_eval)),
                float(np.max(det_eval)),
            )
            logger.info(
                "Robust noise weighting (%s): mode=%s objective_include_nominal=%s min=%.6e max=%.6e mean=%.6e eff_samples_mean=%.3f",
                epoch_type,
                DEPHASE_DETUNING_WEIGHTING,
                DEPHASE_OBJECTIVE_INCLUDE_NOMINAL,
                float(np.min(weight_eval)),
                float(np.max(weight_eval)),
                float(np.mean(weight_eval)),
                float(np.mean(eff_samples)),
            )
    else:
        if epoch_type == "evaluation":
            reward_data, fidelity_data, _, _ = trapped_ion_binomial_sim_batch(
                phi_r,
                phi_b,
                amp_r=amp_r,
                amp_b=amp_b,
                n_boson=N_BOSON,
                omega=OMEGA_RABI,
                t_step=T_STEP,
                binomial_code=BINOMIAL_CODE,
                binomial_phase=BINOMIAL_REL_PHASE,
                sample_points=sample_points,
                target_values=target_values,
                sample_weights=sample_weights,
                sample_area=CHAR_AREA,
                reward_scale=REWARD_SCALE,
                reward_clip=REWARD_CLIP,
                reward_norm=reward_norm,
                n_shots=n_shots,
                return_details=True,
                reward_mode="characteristic",
                characteristic_objective=reward_objective,
            )
        else:
            reward_data = trapped_ion_binomial_sim_batch(
                phi_r,
                phi_b,
                amp_r=amp_r,
                amp_b=amp_b,
                n_boson=N_BOSON,
                omega=OMEGA_RABI,
                t_step=T_STEP,
                binomial_code=BINOMIAL_CODE,
                binomial_phase=BINOMIAL_REL_PHASE,
                sample_points=sample_points,
                target_values=target_values,
                sample_weights=sample_weights,
                sample_area=CHAR_AREA,
                reward_scale=REWARD_SCALE,
                reward_clip=REWARD_CLIP,
                reward_norm=reward_norm,
                n_shots=n_shots,
                reward_mode="characteristic",
                characteristic_objective=reward_objective,
            )
    amp_overshoot_pen = None
    duration_pen = None
    if epoch_type != "evaluation":
        smooth_pen = _smoothness_penalty(phi_r, phi_b, amp_r, amp_b)
        amp_overshoot_pen = _amplitude_overshoot_penalty(amp_r, amp_b)
        duration_pen = _duration_scale_penalty(duration_scale)

    if ROBUST_TRAINING:
        reward_core = np.asarray(robust_reward_rob, dtype=float)
        if epoch_type != "evaluation":
            reward_core = (
                reward_core
                - SMOOTH_LAMBDA * smooth_pen
                - AMP_OVERSHOOT_LAMBDA * amp_overshoot_pen
                - DURATION_SCALE_PENALTY_LAMBDA * duration_pen
            )
        reward_core = _auto_rescale_rewards(reward_core, epoch=epoch, epoch_type=epoch_type)
        # Keep the floor penalty in physical units; do not rescale it.
        reward_data = np.asarray(reward_core, dtype=float) - np.asarray(robust_penalty, dtype=float)
    else:
        if epoch_type != "evaluation":
            reward_data = (
                reward_data
                - SMOOTH_LAMBDA * smooth_pen
                - AMP_OVERSHOOT_LAMBDA * amp_overshoot_pen
                - DURATION_SCALE_PENALTY_LAMBDA * duration_pen
            )
        reward_data = _auto_rescale_rewards(reward_data, epoch=epoch, epoch_type=epoch_type)

    reward_arr = np.asarray(reward_data)
    logger.info("Reward shape %s", reward_arr.shape)
    logger.info("Reward min %.6f max %.6f", float(np.min(reward_arr)), float(np.max(reward_arr)))
    R = np.mean(reward_data)
    std_R = np.std(reward_data)
    logger.info("Average reward %.3f", R)
    logger.info("STDev reward %.3f", std_R)
    if epoch_type != "evaluation" and AMP_OVERSHOOT_LAMBDA > 0.0:
        logger.info(
            "Amplitude overshoot penalty mean=%.6f weighted=%.6f",
            float(np.mean(amp_overshoot_pen)),
            float(AMP_OVERSHOOT_LAMBDA * np.mean(amp_overshoot_pen)),
        )
    if epoch_type != "evaluation" and DURATION_SCALE_PENALTY_LAMBDA > 0.0:
        logger.info(
            "Duration-scale penalty mean=%.6f weighted=%.6f target=%.3f above_target_only=%s",
            float(np.mean(duration_pen)),
            float(DURATION_SCALE_PENALTY_LAMBDA * np.mean(duration_pen)),
            DURATION_SCALE_TARGET,
            DURATION_SCALE_ONLY_ABOVE_TARGET,
        )
    if epoch_type == "training":
        best_idx_train = int(np.argmax(reward_arr))
        batch_best_reward = float(reward_arr[best_idx_train])
        if batch_best_reward > best_train_reward:
            best_train_reward = batch_best_reward
            best_train_epoch = epoch
            best_train_action = {
                "phi_r": phi_r_coeff[best_idx_train].copy(),
                "phi_b": phi_b_coeff[best_idx_train].copy(),
                "amp_r": amp_r_coeff[best_idx_train].copy(),
                "amp_b": amp_b_coeff[best_idx_train].copy(),
                "duration_scale": float(duration_scale[best_idx_train]),
            }
            logger.info(
                "Updated best training action at epoch %d with reward %.6f",
                best_train_epoch,
                best_train_reward,
            )
    if fidelity_data is not None:
        fidelity_arr = np.asarray(fidelity_data, dtype=float)
        mean_fidelity = float(np.mean(fidelity_arr))
        std_fidelity = float(np.std(fidelity_arr))
        if ROBUST_TRAINING and robust_score is not None:
            robust_reward_arr = np.asarray(robust_reward_rob, dtype=float)
            robust_fid_nom_arr = np.asarray(robust_fidelity_nom, dtype=float)
            robust_fid_rob_arr = np.asarray(robust_fidelity_rob, dtype=float)
            robust_pen_arr = np.asarray(robust_penalty, dtype=float)
            robust_score_arr = np.asarray(robust_score, dtype=float)
            best_idx = int(np.argmax(robust_score_arr))
            batch_best_score = float(robust_score_arr[best_idx])
            batch_best_nominal = float(robust_fid_nom_arr[best_idx])
            batch_best_robust = float(robust_fid_rob_arr[best_idx])
            batch_best_penalty = float(robust_pen_arr[best_idx])
            logger.info(
                "Eval fidelity nominal mean %.6f std %.6f",
                mean_fidelity,
                std_fidelity,
            )
            logger.info(
                "Eval robust mean: R_rob=%.6f F_nom=%.6f F_rob=%.6f penalty=%.6f score=%.6f",
                float(np.mean(robust_reward_arr)),
                float(np.mean(robust_fid_nom_arr)),
                float(np.mean(robust_fid_rob_arr)),
                float(np.mean(robust_pen_arr)),
                float(np.mean(robust_score_arr)),
            )
            logger.info(
                "Eval robust batch-best idx=%d: score=%.6f F_nom=%.6f F_rob=%.6f penalty=%.6f",
                best_idx,
                batch_best_score,
                batch_best_nominal,
                batch_best_robust,
                batch_best_penalty,
            )
            if batch_best_score > best_eval_score:
                best_eval_score = batch_best_score
                best_eval_fidelity = batch_best_nominal
                best_eval_epoch = epoch
                best_eval_action = {
                    "phi_r": phi_r_coeff[best_idx].copy(),
                    "phi_b": phi_b_coeff[best_idx].copy(),
                    "amp_r": amp_r_coeff[best_idx].copy(),
                    "amp_b": amp_b_coeff[best_idx].copy(),
                    "duration_scale": float(duration_scale[best_idx]),
                }
                logger.info(
                    "Updated best eval action at epoch %d with robust_score %.6f (nominal_fidelity %.6f)",
                    best_eval_epoch,
                    best_eval_score,
                    best_eval_fidelity,
                )
            top_eval_actions = _update_top_eval_actions(
                top_eval_actions,
                epoch=epoch,
                metric=batch_best_score,
                phi_r=phi_r_coeff[best_idx],
                phi_b=phi_b_coeff[best_idx],
                amp_r=amp_r_coeff[best_idx],
                amp_b=amp_b_coeff[best_idx],
                duration_scale=float(duration_scale[best_idx]),
            )
            write_header_robust = not os.path.exists(eval_robust_log_path)
            with open(eval_robust_log_path, "a", encoding="utf-8") as f:
                if write_header_robust:
                    f.write(
                        "epoch,mean_reward_robust,mean_fidelity_nominal,mean_fidelity_robust,mean_penalty,mean_score\n"
                    )
                f.write(
                    f"{epoch},{np.mean(robust_reward_arr):.6f},{np.mean(robust_fid_nom_arr):.6f},"
                    f"{np.mean(robust_fid_rob_arr):.6f},{np.mean(robust_pen_arr):.6f},"
                    f"{np.mean(robust_score_arr):.6f}\n"
                )
        else:
            best_idx = int(np.argmax(fidelity_arr))
            batch_best_fidelity = float(fidelity_arr[best_idx])
            logger.info(
                "Eval fidelity mean %.6f | batch-best %.6f (idx=%d)",
                mean_fidelity,
                batch_best_fidelity,
                best_idx,
            )
            if batch_best_fidelity > best_eval_fidelity:
                best_eval_fidelity = batch_best_fidelity
                best_eval_score = batch_best_fidelity
                best_eval_epoch = epoch
                best_eval_action = {
                    "phi_r": phi_r_coeff[best_idx].copy(),
                    "phi_b": phi_b_coeff[best_idx].copy(),
                    "amp_r": amp_r_coeff[best_idx].copy(),
                    "amp_b": amp_b_coeff[best_idx].copy(),
                    "duration_scale": float(duration_scale[best_idx]),
                }
                logger.info(
                    "Updated best eval action at epoch %d with fidelity %.6f",
                    best_eval_epoch,
                    best_eval_fidelity,
                )
            top_eval_actions = _update_top_eval_actions(
                top_eval_actions,
                epoch=epoch,
                metric=batch_best_fidelity,
                phi_r=phi_r_coeff[best_idx],
                phi_b=phi_b_coeff[best_idx],
                amp_r=amp_r_coeff[best_idx],
                amp_b=amp_b_coeff[best_idx],
                duration_scale=float(duration_scale[best_idx]),
            )
        write_header = not os.path.exists(eval_log_path)
        with open(eval_log_path, "a", encoding="utf-8") as f:
            if write_header:
                f.write("epoch,mean_fidelity,std_fidelity\n")
            f.write(f"{epoch},{mean_fidelity:.6f},{std_fidelity:.6f}\n")

    logger.info("Sending message to RL agent server.")
    logger.info("Time stamp: %f", time.time())
    client_socket.send_data(reward_data)
