import logging
import os
import sys
import time

import numpy as np
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

if os.environ.get("DQ_FORCE_GPU", "0") == "1":
    os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")

from quantum_control_rl_server.remote_env_tools import Client
from trapped_ion_gkp_sim_function import (
    trapped_ion_gkp_sim,
    trapped_ion_gkp_sim_batch,
    characteristic_grid,
    prepare_characteristic_distribution,
    characteristic_norm,
)

logger = logging.getLogger("RL")
logger.propagate = False
logger.handlers = []
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)

client_socket = Client()
(host, port) = ("127.0.0.1", 5555)
client_socket.connect((host, port))

FAST_SMOKE = os.environ.get("FAST_SMOKE", "0") == "1"

N_STEPS = 120
N_SEGMENTS = 60
SEG_LEN = N_STEPS // N_SEGMENTS
T_STEP = 10.0
SAMPLE_EXTENT = 4.0
PLOT_EXTENT = float(os.environ.get("PLOT_EXTENT", "6.0"))
N_BOSON = 30
GKP_DELTA = float(os.environ.get("GKP_DELTA", "0.301"))
GKP_LOGICAL = os.environ.get("GKP_LOGICAL", "0")
_gkp_squeeze_env = os.environ.get("GKP_SQUEEZE_R", "").strip()
GKP_SQUEEZE_R = None if _gkp_squeeze_env == "" else float(_gkp_squeeze_env)
_gkp_kappa_env = os.environ.get("GKP_ENVELOPE_KAPPA", "").strip()
GKP_ENVELOPE_KAPPA = None if _gkp_kappa_env == "" else float(_gkp_kappa_env)
GKP_LATTICE_TRUNC = int(os.environ.get("GKP_LATTICE_TRUNC", "4"))

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

SMOOTH_LAMBDA = 0.0
SMOOTH_PHI_WEIGHT = 1.0
SMOOTH_AMP_WEIGHT = 0.2
REWARD_SCALE = 1.0
REWARD_CLIP = None
CHAR_REWARD_OBJECTIVE = os.environ.get("CHAR_REWARD_OBJECTIVE", "overlap_real").lower()

N_SHOTS_TRAIN = 0
N_SHOTS_EVAL = 0

ACTION_NOISE_PHI = 0.0
ACTION_NOISE_AMP = 0.0

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
_gkp_phase_env = os.environ.get("GKP_REL_PHASE", "").strip()
GKP_REL_PHASE = None if _gkp_phase_env == "" else float(_gkp_phase_env)
PHASE_CLIP = float(os.environ.get("PHASE_CLIP", str(np.pi)))
AMP_MIN = float(os.environ.get("AMP_MIN", "0.0"))
AMP_MAX = float(os.environ.get("AMP_MAX", "2.0"))

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
FINAL_REFINE_USE_LOC_CENTER = os.environ.get("FINAL_REFINE_USE_LOC_CENTER", "1") == "1"
FINAL_REFINE_USE_TRAIN_CENTER = os.environ.get("FINAL_REFINE_USE_TRAIN_CENTER", "1") == "1"
FINAL_REFINE_CENTER_SEED_STRIDE = int(
    os.environ.get("FINAL_REFINE_CENTER_SEED_STRIDE", "1000003")
)


def _build_characteristic_distribution(grid_size):
    all_points = []
    all_targets = []
    all_weights = []
    all_areas = []
    n_scales = len(CHAR_ALPHA_SCALES)
    for alpha_scale in CHAR_ALPHA_SCALES:
        points_i, target_i, weights_i, area_i = prepare_characteristic_distribution(
            alpha_cat=GKP_DELTA,
            gkp_delta=GKP_DELTA,
            n_boson=N_BOSON,
            extent=SAMPLE_EXTENT,
            grid_size=grid_size,
            cat_parity=GKP_LOGICAL,
            gkp_logical=GKP_LOGICAL,
            mix_uniform=CHAR_UNIFORM_MIX,
            alpha_scale=alpha_scale,
            cat_phase=GKP_REL_PHASE,
            gkp_squeeze_r=GKP_SQUEEZE_R,
            gkp_envelope_kappa=GKP_ENVELOPE_KAPPA,
            gkp_lattice_trunc=GKP_LATTICE_TRUNC,
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
    alpha_cat=GKP_DELTA,
    gkp_delta=GKP_DELTA,
    n_boson=N_BOSON,
    extent=SAMPLE_EXTENT,
    grid_size=FINAL_GRID_SIZE,
    cat_parity=GKP_LOGICAL,
    gkp_logical=GKP_LOGICAL,
    mix_uniform=CHAR_UNIFORM_MIX,
    alpha_scale=CHAR_ALPHA_SCALE,
    cat_phase=GKP_REL_PHASE,
    gkp_squeeze_r=GKP_SQUEEZE_R,
    gkp_envelope_kappa=GKP_ENVELOPE_KAPPA,
    gkp_lattice_trunc=GKP_LATTICE_TRUNC,
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
    "Characteristic sampling: start_mode=%s alpha_scales=%s radial_exp=%.2f gkp_phase=%s",
    CHAR_START_MODE,
    ",".join(f"{v:.3f}" for v in CHAR_ALPHA_SCALES),
    CHAR_RADIAL_EXP,
    "none" if GKP_REL_PHASE is None else f"{GKP_REL_PHASE:.3f}",
)
logger.info(
    "GKP params: delta=%.4f logical=%s squeeze_r=%s envelope_kappa=%s lattice_trunc=%d",
    GKP_DELTA,
    GKP_LOGICAL,
    "auto(-log(delta))" if GKP_SQUEEZE_R is None else f"{GKP_SQUEEZE_R:.4f}",
    "auto(delta)" if GKP_ENVELOPE_KAPPA is None else f"{GKP_ENVELOPE_KAPPA:.4f}",
    GKP_LATTICE_TRUNC,
)
logger.info(
    "Characteristic reward objective: %s | stage epochs: %d -> %d -> end",
    CHAR_REWARD_OBJECTIVE,
    TRAIN_STAGE1_EPOCHS,
    TRAIN_STAGE2_EPOCHS,
)
logger.info(
    "Characteristic point sampler: mode=%s radial_bins=%d uniform_mix=%.2f",
    CHAR_SAMPLER_MODE,
    CHAR_RADIAL_BINS,
    CHAR_UNIFORM_MIX,
)
logger.info(
    "Action clipping: phase_clip=%.3f amp_range=[%.3f, %.3f]",
    PHASE_CLIP,
    AMP_MIN,
    AMP_MAX,
)
logger.info(
    "Characteristic plotting extent: %.3f (sampling extent: %.3f)",
    PLOT_EXTENT,
    SAMPLE_EXTENT,
)
logger.info(
    "Final refinement setup: samples=%d rounds=%d topk=%d top_eval_centers=%d scale=%.3f decay=%.3f min_sigma=%.3f use_loc_center=%s use_train_center=%s",
    FINAL_REFINE_SAMPLES,
    FINAL_REFINE_ROUNDS,
    FINAL_REFINE_TOPK,
    FINAL_REFINE_TOP_EVAL_CENTERS,
    FINAL_REFINE_SCALE,
    FINAL_REFINE_DECAY,
    FINAL_REFINE_MIN_SIGMA,
    FINAL_REFINE_USE_LOC_CENTER,
    FINAL_REFINE_USE_TRAIN_CENTER,
)


def _smoothness_penalty(phi_r, phi_b, amp_r, amp_b):
    axis = 1 if np.ndim(phi_r) > 1 else 0
    dphi_r = np.diff(phi_r, axis=axis)
    dphi_b = np.diff(phi_b, axis=axis)
    damp_r = np.diff(amp_r, axis=axis)
    damp_b = np.diff(amp_b, axis=axis)
    phi_pen = 0.5 * (np.mean(dphi_r ** 2, axis=axis) + np.mean(dphi_b ** 2, axis=axis))
    amp_pen = 0.5 * (np.mean(damp_r ** 2, axis=axis) + np.mean(damp_b ** 2, axis=axis))
    return SMOOTH_PHI_WEIGHT * phi_pen + SMOOTH_AMP_WEIGHT * amp_pen


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


def _eval_fidelity_batch(phi_r_coeff, phi_b_coeff, amp_r_coeff, amp_b_coeff):
    phi_r_full = np.repeat(phi_r_coeff, SEG_LEN, axis=1)
    phi_b_full = np.repeat(phi_b_coeff, SEG_LEN, axis=1)
    amp_r_full = np.repeat(amp_r_coeff, SEG_LEN, axis=1)
    amp_b_full = np.repeat(amp_b_coeff, SEG_LEN, axis=1)
    _, fidelity_batch, _, _ = trapped_ion_gkp_sim_batch(
        phi_r_full,
        phi_b_full,
        amp_r=amp_r_full,
        amp_b=amp_b_full,
        n_boson=N_BOSON,
        omega=2 * np.pi * 0.002,
        t_step=T_STEP,
        alpha_cat=GKP_DELTA,
        cat_parity=GKP_LOGICAL,
        gkp_delta=GKP_DELTA,
        gkp_logical=GKP_LOGICAL,
        cat_phase=GKP_REL_PHASE,
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
        characteristic_objective=CHAR_REWARD_OBJECTIVE,
        gkp_squeeze_r=GKP_SQUEEZE_R,
        gkp_envelope_kappa=GKP_ENVELOPE_KAPPA,
        gkp_lattice_trunc=GKP_LATTICE_TRUNC,
    )
    return np.asarray(fidelity_batch, dtype=float)


def _refine_around_center(
    center_phi_r,
    center_phi_b,
    center_amp_r,
    center_amp_b,
    sigma_phi_r,
    sigma_phi_b,
    rng,
):
    center_phi_r = np.asarray(center_phi_r, dtype=float)
    center_phi_b = np.asarray(center_phi_b, dtype=float)
    center_amp_r = np.asarray(center_amp_r, dtype=float)
    center_amp_b = np.asarray(center_amp_b, dtype=float)
    sigma_phi_r = np.asarray(sigma_phi_r, dtype=float)
    sigma_phi_b = np.asarray(sigma_phi_b, dtype=float)

    best_phi_r = center_phi_r.copy()
    best_phi_b = center_phi_b.copy()
    best_amp_r = center_amp_r.copy()
    best_amp_b = center_amp_b.copy()
    best_fidelity = float(
        _eval_fidelity_batch(
            best_phi_r[None, :],
            best_phi_b[None, :],
            best_amp_r[None, :],
            best_amp_b[None, :],
        )[0]
    )

    n_rounds = max(1, FINAL_REFINE_ROUNDS)
    n_samples = max(0, FINAL_REFINE_SAMPLES)
    topk = max(1, FINAL_REFINE_TOPK)

    for ridx in range(n_rounds):
        if n_samples <= 0:
            break
        scale = FINAL_REFINE_SCALE * (FINAL_REFINE_DECAY ** ridx)
        n_cand = n_samples + 1

        cand_phi_r = np.repeat(best_phi_r[None, :], n_cand, axis=0)
        cand_phi_b = np.repeat(best_phi_b[None, :], n_cand, axis=0)
        cand_amp_r = np.repeat(best_amp_r[None, :], n_cand, axis=0)
        cand_amp_b = np.repeat(best_amp_b[None, :], n_cand, axis=0)

        noise_r = rng.normal(size=(n_samples, N_SEGMENTS))
        noise_b = rng.normal(size=(n_samples, N_SEGMENTS))
        cand_phi_r[1:, :] = best_phi_r[None, :] + scale * noise_r * sigma_phi_r[None, :]
        cand_phi_b[1:, :] = best_phi_b[None, :] + scale * noise_b * sigma_phi_b[None, :]
        cand_phi_r = np.clip(cand_phi_r, -PHASE_CLIP, PHASE_CLIP)
        cand_phi_b = np.clip(cand_phi_b, -PHASE_CLIP, PHASE_CLIP)
        cand_amp_r = np.clip(cand_amp_r, AMP_MIN, AMP_MAX)
        cand_amp_b = np.clip(cand_amp_b, AMP_MIN, AMP_MAX)

        cand_fidelity = _eval_fidelity_batch(cand_phi_r, cand_phi_b, cand_amp_r, cand_amp_b)
        order = np.argsort(cand_fidelity)[::-1]
        keep = order[: min(topk, len(order))]
        round_best_idx = int(order[0])
        round_best = float(cand_fidelity[round_best_idx])
        logger.info(
            "Final refine round %d/%d | scale=%.4f best=%.6f mean_topk=%.6f",
            ridx + 1,
            n_rounds,
            scale,
            round_best,
            float(np.mean(cand_fidelity[keep])),
        )

        if round_best > best_fidelity:
            best_fidelity = round_best
            best_phi_r = cand_phi_r[round_best_idx].copy()
            best_phi_b = cand_phi_b[round_best_idx].copy()
            best_amp_r = cand_amp_r[round_best_idx].copy()
            best_amp_b = cand_amp_b[round_best_idx].copy()

        sigma_phi_r = np.maximum(np.std(cand_phi_r[keep], axis=0), FINAL_REFINE_MIN_SIGMA)
        sigma_phi_b = np.maximum(np.std(cand_phi_b[keep], axis=0), FINAL_REFINE_MIN_SIGMA)

    return best_phi_r, best_phi_b, best_amp_r, best_amp_b, best_fidelity


def _update_top_eval_actions(
    records,
    epoch,
    fidelity,
    phi_r,
    phi_b,
    amp_r,
    amp_b,
):
    if FINAL_REFINE_TOP_EVAL_CENTERS <= 0:
        return records
    rec = {
        "epoch": int(epoch),
        "fidelity": float(fidelity),
        "phi_r": np.array(phi_r, dtype=float).copy(),
        "phi_b": np.array(phi_b, dtype=float).copy(),
        "amp_r": np.array(amp_r, dtype=float).copy(),
        "amp_b": np.array(amp_b, dtype=float).copy(),
    }
    filtered = [r for r in records if int(r["epoch"]) != int(epoch)]
    filtered.append(rec)
    filtered.sort(key=lambda x: float(x["fidelity"]), reverse=True)
    return filtered[: max(1, FINAL_REFINE_TOP_EVAL_CENTERS)]


done = False
eval_log_path = os.path.join(os.getcwd(), "eval_fidelity.csv")
if os.environ.get("CLEAR_EVAL_LOG", "1") == "1" and os.path.exists(eval_log_path):
    os.remove(eval_log_path)
best_eval_fidelity = -np.inf
best_eval_epoch = -1
best_eval_action = None
best_train_reward = -np.inf
best_train_epoch = -1
best_train_action = None
top_eval_actions = []
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
        locs = message["locs"]
        scales = message["scales"]
        for key in locs.keys():
            logger.info("locs[%s]:", key)
            logger.info(locs[key][0])
            logger.info("scales[%s]:", key)
            logger.info(scales[key][0])
        loc_phi_r = np.array(locs["phi_r"][0], dtype=float)
        loc_phi_b = np.array(locs["phi_b"][0], dtype=float)
        scale_phi_r = np.array(scales["phi_r"][0], dtype=float)
        scale_phi_b = np.array(scales["phi_b"][0], dtype=float)

        if best_eval_action is not None:
            logger.info(
                "Using best evaluation action from epoch %d with eval fidelity %.6f",
                best_eval_epoch,
                best_eval_fidelity,
            )
            base_phi_r = np.array(best_eval_action["phi_r"], dtype=float)
            base_phi_b = np.array(best_eval_action["phi_b"], dtype=float)
            base_amp_r = np.array(best_eval_action["amp_r"], dtype=float)
            base_amp_b = np.array(best_eval_action["amp_b"], dtype=float)
        else:
            base_phi_r = loc_phi_r
            base_phi_b = loc_phi_b
            amp_r_vals = np.array(locs.get("amp_r", [np.ones(N_SEGMENTS)])[0])
            amp_b_vals = np.array(locs.get("amp_b", [np.ones(N_SEGMENTS)])[0])
            base_amp_r = np.array(amp_r_vals, dtype=float)
            base_amp_b = np.array(amp_b_vals, dtype=float)

        # Multi-round local refinement in phase space. This directly optimizes
        # fidelity around one or more centers using batched simulator calls.
        sigma_phi_r = np.maximum(np.asarray(scale_phi_r, dtype=float), FINAL_REFINE_MIN_SIGMA)
        sigma_phi_b = np.maximum(np.asarray(scale_phi_b, dtype=float), FINAL_REFINE_MIN_SIGMA)
        centers = [(base_phi_r, base_phi_b, base_amp_r, base_amp_b, "best_eval")]
        if FINAL_REFINE_USE_LOC_CENTER:
            centers.append((loc_phi_r, loc_phi_b, base_amp_r, base_amp_b, "final_loc"))
        if FINAL_REFINE_USE_TRAIN_CENTER and best_train_action is not None:
            centers.append(
                (
                    np.array(best_train_action["phi_r"], dtype=float),
                    np.array(best_train_action["phi_b"], dtype=float),
                    np.array(best_train_action["amp_r"], dtype=float),
                    np.array(best_train_action["amp_b"], dtype=float),
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
                        f"top_eval_epoch_{int(rec['epoch'])}",
                    )
                )
                added_top_eval += 1
            logger.info(
                "Added %d historical top-eval centers for refinement",
                added_top_eval,
            )

        global_best = None
        global_best_fidelity = -np.inf
        global_best_label = "none"
        for cidx, (c_phi_r, c_phi_b, c_amp_r, c_amp_b, label) in enumerate(centers):
            # Use an independent RNG stream per center to avoid correlated
            # candidate sets when comparing multiple refinement centers.
            center_seed = FINAL_REFINE_SEED + cidx * FINAL_REFINE_CENTER_SEED_STRIDE
            rng_ref = np.random.default_rng(center_seed)
            logger.info("Final refinement center=%s | seed=%d", label, center_seed)
            b_phi_r, b_phi_b, b_amp_r, b_amp_b, b_fid = _refine_around_center(
                c_phi_r,
                c_phi_b,
                c_amp_r,
                c_amp_b,
                sigma_phi_r,
                sigma_phi_b,
                rng_ref,
            )
            logger.info("Final refinement center=%s | best fidelity %.6f", label, b_fid)
            if b_fid > global_best_fidelity:
                global_best_fidelity = b_fid
                global_best = (b_phi_r, b_phi_b, b_amp_r, b_amp_b)
                global_best_label = label

        if global_best is not None:
            base_phi_r, base_phi_b, base_amp_r, base_amp_b = global_best
            logger.info(
                "Final refinement selected center=%s with best sampled fidelity %.6f",
                global_best_label,
                global_best_fidelity,
            )

        phi_r_final = np.repeat(base_phi_r, SEG_LEN)
        phi_b_final = np.repeat(base_phi_b, SEG_LEN)
        amp_r_final = np.repeat(base_amp_r, SEG_LEN)
        amp_b_final = np.repeat(base_amp_b, SEG_LEN)

        _, final_fidelity, rho_final, rho_target = trapped_ion_gkp_sim(
            phi_r_final,
            phi_b_final,
            amp_r=amp_r_final,
            amp_b=amp_b_final,
            n_boson=N_BOSON,
            omega=2 * np.pi * 0.002,
            t_step=T_STEP,
            alpha_cat=GKP_DELTA,
            cat_parity=GKP_LOGICAL,
            gkp_delta=GKP_DELTA,
            gkp_logical=GKP_LOGICAL,
            cat_phase=GKP_REL_PHASE,
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
            characteristic_objective=CHAR_REWARD_OBJECTIVE,
            gkp_squeeze_r=GKP_SQUEEZE_R,
            gkp_envelope_kappa=GKP_ENVELOPE_KAPPA,
            gkp_lattice_trunc=GKP_LATTICE_TRUNC,
        )

        output_dir = os.path.join(os.getcwd(), "outputs")
        os.makedirs(output_dir, exist_ok=True)
        fidelity_path = os.path.join(output_dir, "final_fidelity.txt")
        with open(fidelity_path, "w", encoding="utf-8") as f:
            f.write(f"{final_fidelity:.6f}\n")
        logger.info("Final fidelity %.6f", final_fidelity)
        logger.info("Saved final fidelity to %s", fidelity_path)

        grid = np.linspace(-PLOT_EXTENT, PLOT_EXTENT, PLOT_GRID_SIZE)
        chi_target = characteristic_grid(rho_target, grid, grid)
        chi_final = characteristic_grid(rho_final, grid, grid)

        if plt is None:
            logger.warning("matplotlib unavailable; skip characteristic plot export.")
        else:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            im0 = axes[0].imshow(
                chi_target,
                extent=[-PLOT_EXTENT, PLOT_EXTENT, -PLOT_EXTENT, PLOT_EXTENT],
                origin="lower",
                cmap="RdBu_r",
            )
            axes[0].set_title("Target GKP characteristic")
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

    rng = np.random.default_rng(epoch)
    if epoch_type == "training":
        if ACTION_NOISE_PHI > 0.0:
            phi_r_coeff = phi_r_coeff + rng.normal(0.0, ACTION_NOISE_PHI, size=phi_r_coeff.shape)
            phi_b_coeff = phi_b_coeff + rng.normal(0.0, ACTION_NOISE_PHI, size=phi_b_coeff.shape)
        if ACTION_NOISE_AMP > 0.0:
            amp_r_coeff = amp_r_coeff + rng.normal(0.0, ACTION_NOISE_AMP, size=amp_r_coeff.shape)
            amp_b_coeff = amp_b_coeff + rng.normal(0.0, ACTION_NOISE_AMP, size=amp_b_coeff.shape)

    phi_r_before = phi_r_coeff.copy()
    phi_b_before = phi_b_coeff.copy()
    amp_r_before = amp_r_coeff.copy()
    amp_b_before = amp_b_coeff.copy()
    phi_r_coeff = np.clip(phi_r_coeff, -PHASE_CLIP, PHASE_CLIP)
    phi_b_coeff = np.clip(phi_b_coeff, -PHASE_CLIP, PHASE_CLIP)
    amp_r_coeff = np.clip(amp_r_coeff, AMP_MIN, AMP_MAX)
    amp_b_coeff = np.clip(amp_b_coeff, AMP_MIN, AMP_MAX)
    if epoch_type == "evaluation" or epoch % 20 == 0:
        clip_phi_r = float(np.mean(phi_r_before != phi_r_coeff))
        clip_phi_b = float(np.mean(phi_b_before != phi_b_coeff))
        clip_amp_r = float(np.mean(amp_r_before != amp_r_coeff))
        clip_amp_b = float(np.mean(amp_b_before != amp_b_coeff))
        logger.info(
            "Clip ratio phi_r=%.3f phi_b=%.3f amp_r=%.3f amp_b=%.3f",
            clip_phi_r,
            clip_phi_b,
            clip_amp_r,
            clip_amp_b,
        )

    phi_r = np.repeat(phi_r_coeff, SEG_LEN, axis=1)
    phi_b = np.repeat(phi_b_coeff, SEG_LEN, axis=1)
    amp_r = np.repeat(amp_r_coeff, SEG_LEN, axis=1)
    amp_b = np.repeat(amp_b_coeff, SEG_LEN, axis=1)
    if epoch_type == "evaluation":
        n_shots = N_SHOTS_EVAL
        sample_points, target_values, sample_weights = EVAL_POINTS, EVAL_TARGET, EVAL_WEIGHTS
        reward_norm = None
    else:
        n_shots = N_SHOTS_TRAIN
        sample_points, target_values, sample_weights, reward_norm = _select_train_points(
            epoch, rng
        )
        reward_norm = None

    if epoch_type == "evaluation":
        reward_data, fidelity_data, _, _ = trapped_ion_gkp_sim_batch(
            phi_r,
            phi_b,
            amp_r=amp_r,
            amp_b=amp_b,
            n_boson=N_BOSON,
            omega=2 * np.pi * 0.002,
            t_step=T_STEP,
            alpha_cat=GKP_DELTA,
            cat_parity=GKP_LOGICAL,
            gkp_delta=GKP_DELTA,
            gkp_logical=GKP_LOGICAL,
            cat_phase=GKP_REL_PHASE,
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
            characteristic_objective=CHAR_REWARD_OBJECTIVE,
            gkp_squeeze_r=GKP_SQUEEZE_R,
            gkp_envelope_kappa=GKP_ENVELOPE_KAPPA,
            gkp_lattice_trunc=GKP_LATTICE_TRUNC,
        )
    else:
        reward_data = trapped_ion_gkp_sim_batch(
            phi_r,
            phi_b,
            amp_r=amp_r,
            amp_b=amp_b,
            n_boson=N_BOSON,
            omega=2 * np.pi * 0.002,
            t_step=T_STEP,
            alpha_cat=GKP_DELTA,
            cat_parity=GKP_LOGICAL,
            gkp_delta=GKP_DELTA,
            gkp_logical=GKP_LOGICAL,
            cat_phase=GKP_REL_PHASE,
            sample_points=sample_points,
            target_values=target_values,
            sample_weights=sample_weights,
            sample_area=CHAR_AREA,
            reward_scale=REWARD_SCALE,
            reward_clip=REWARD_CLIP,
            reward_norm=reward_norm,
            n_shots=n_shots,
            reward_mode="characteristic",
            characteristic_objective=CHAR_REWARD_OBJECTIVE,
            gkp_squeeze_r=GKP_SQUEEZE_R,
            gkp_envelope_kappa=GKP_ENVELOPE_KAPPA,
            gkp_lattice_trunc=GKP_LATTICE_TRUNC,
        )
        smooth_pen = _smoothness_penalty(phi_r, phi_b, amp_r, amp_b)
        reward_data = reward_data - SMOOTH_LAMBDA * smooth_pen

    reward_arr = np.asarray(reward_data)
    logger.info("Reward shape %s", reward_arr.shape)
    logger.info("Reward min %.6f max %.6f", float(np.min(reward_arr)), float(np.max(reward_arr)))
    R = np.mean(reward_data)
    std_R = np.std(reward_data)
    logger.info("Average reward %.3f", R)
    logger.info("STDev reward %.3f", std_R)
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
            best_eval_epoch = epoch
            best_eval_action = {
                "phi_r": phi_r_coeff[best_idx].copy(),
                "phi_b": phi_b_coeff[best_idx].copy(),
                "amp_r": amp_r_coeff[best_idx].copy(),
                "amp_b": amp_b_coeff[best_idx].copy(),
            }
            logger.info(
                "Updated best eval action at epoch %d with fidelity %.6f",
                best_eval_epoch,
                best_eval_fidelity,
            )
        top_eval_actions = _update_top_eval_actions(
            top_eval_actions,
            epoch=epoch,
            fidelity=batch_best_fidelity,
            phi_r=phi_r_coeff[best_idx],
            phi_b=phi_b_coeff[best_idx],
            amp_r=amp_r_coeff[best_idx],
            amp_b=amp_b_coeff[best_idx],
        )
        write_header = not os.path.exists(eval_log_path)
        with open(eval_log_path, "a", encoding="utf-8") as f:
            if write_header:
                f.write("epoch,mean_fidelity,std_fidelity\n")
            f.write(f"{epoch},{mean_fidelity:.6f},{std_fidelity:.6f}\n")

    logger.info("Sending message to RL agent server.")
    logger.info("Time stamp: %f", time.time())
    client_socket.send_data(reward_data)
