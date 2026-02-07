import logging
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

if os.environ.get("DQ_FORCE_GPU", "0") == "1":
    os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")

from quantum_control_rl_server.remote_env_tools import Client
from trapped_ion_cat_sim_function import (
    trapped_ion_cat_sim,
    trapped_ion_cat_sim_batch,
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
N_BOSON = 30
ALPHA_CAT = 2.0

TRAIN_POINTS_STAGE1 = 120
TRAIN_POINTS_STAGE2 = 240
TRAIN_POINTS_STAGE3 = 960
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
_cat_phase_env = os.environ.get("CAT_REL_PHASE", "").strip()
CAT_REL_PHASE = None if _cat_phase_env == "" else float(_cat_phase_env)
PHASE_CLIP = float(os.environ.get("PHASE_CLIP", str(np.pi)))
AMP_MIN = float(os.environ.get("AMP_MIN", "0.0"))
AMP_MAX = float(os.environ.get("AMP_MAX", "2.0"))

CHAR_UNIFORM_MIX = float(os.environ.get("CHAR_UNIFORM_MIX", "0.5"))
FINAL_REFINE_SAMPLES = int(os.environ.get("FINAL_REFINE_SAMPLES", "64"))
FINAL_REFINE_SCALE = float(os.environ.get("FINAL_REFINE_SCALE", "1.0"))
FINAL_REFINE_SEED = int(os.environ.get("FINAL_REFINE_SEED", "1234"))


def _build_characteristic_distribution(grid_size):
    all_points = []
    all_targets = []
    all_weights = []
    all_areas = []
    n_scales = len(CHAR_ALPHA_SCALES)
    for alpha_scale in CHAR_ALPHA_SCALES:
        points_i, target_i, weights_i, area_i = prepare_characteristic_distribution(
            alpha_cat=ALPHA_CAT,
            n_boson=N_BOSON,
            extent=SAMPLE_EXTENT,
            grid_size=grid_size,
            cat_parity="even",
            mix_uniform=CHAR_UNIFORM_MIX,
            alpha_scale=alpha_scale,
            cat_phase=CAT_REL_PHASE,
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
    alpha_cat=ALPHA_CAT,
    n_boson=N_BOSON,
    extent=SAMPLE_EXTENT,
    grid_size=FINAL_GRID_SIZE,
    cat_parity="even",
    mix_uniform=CHAR_UNIFORM_MIX,
    alpha_scale=CHAR_ALPHA_SCALE,
    cat_phase=CAT_REL_PHASE,
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
    "Characteristic sampling: start_mode=%s alpha_scales=%s radial_exp=%.2f cat_phase=%s",
    CHAR_START_MODE,
    ",".join(f"{v:.3f}" for v in CHAR_ALPHA_SCALES),
    CHAR_RADIAL_EXP,
    "none" if CAT_REL_PHASE is None else f"{CAT_REL_PHASE:.3f}",
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


done = False
eval_log_path = os.path.join(os.getcwd(), "eval_fidelity.csv")
if os.environ.get("CLEAR_EVAL_LOG", "1") == "1" and os.path.exists(eval_log_path):
    os.remove(eval_log_path)
best_eval_fidelity = -np.inf
best_eval_epoch = -1
best_eval_action = None
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

        # Optional one-shot refinement: sample candidates from the final policy
        # distribution and pick the highest-fidelity action.
        if FINAL_REFINE_SAMPLES > 0:
            n_cand = FINAL_REFINE_SAMPLES + 1
            cand_phi_r = np.repeat(base_phi_r[None, :], n_cand, axis=0)
            cand_phi_b = np.repeat(base_phi_b[None, :], n_cand, axis=0)
            cand_amp_r = np.repeat(base_amp_r[None, :], n_cand, axis=0)
            cand_amp_b = np.repeat(base_amp_b[None, :], n_cand, axis=0)
            if FINAL_REFINE_SAMPLES > 0:
                rng_ref = np.random.default_rng(FINAL_REFINE_SEED)
                noise_r = rng_ref.normal(size=(FINAL_REFINE_SAMPLES, N_SEGMENTS))
                noise_b = rng_ref.normal(size=(FINAL_REFINE_SAMPLES, N_SEGMENTS))
                cand_phi_r[1:, :] = base_phi_r[None, :] + FINAL_REFINE_SCALE * noise_r * scale_phi_r[None, :]
                cand_phi_b[1:, :] = base_phi_b[None, :] + FINAL_REFINE_SCALE * noise_b * scale_phi_b[None, :]
                cand_phi_r = np.clip(cand_phi_r, -PHASE_CLIP, PHASE_CLIP)
                cand_phi_b = np.clip(cand_phi_b, -PHASE_CLIP, PHASE_CLIP)
                cand_amp_r = np.clip(cand_amp_r, AMP_MIN, AMP_MAX)
                cand_amp_b = np.clip(cand_amp_b, AMP_MIN, AMP_MAX)

            cand_phi_r_full = np.repeat(cand_phi_r, SEG_LEN, axis=1)
            cand_phi_b_full = np.repeat(cand_phi_b, SEG_LEN, axis=1)
            cand_amp_r_full = np.repeat(cand_amp_r, SEG_LEN, axis=1)
            cand_amp_b_full = np.repeat(cand_amp_b, SEG_LEN, axis=1)

            _, cand_fidelity, _, _ = trapped_ion_cat_sim_batch(
                cand_phi_r_full,
                cand_phi_b_full,
                amp_r=cand_amp_r_full,
                amp_b=cand_amp_b_full,
                n_boson=N_BOSON,
                omega=2 * np.pi * 0.002,
                t_step=T_STEP,
                alpha_cat=ALPHA_CAT,
                cat_parity="even",
                cat_phase=CAT_REL_PHASE,
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
            )
            cand_fidelity = np.asarray(cand_fidelity, dtype=float)
            best_cand_idx = int(np.argmax(cand_fidelity))
            base_phi_r = cand_phi_r[best_cand_idx].copy()
            base_phi_b = cand_phi_b[best_cand_idx].copy()
            base_amp_r = cand_amp_r[best_cand_idx].copy()
            base_amp_b = cand_amp_b[best_cand_idx].copy()
            logger.info(
                "Final refinement candidates=%d scale=%.3f | best sampled fidelity %.6f (idx=%d)",
                n_cand,
                FINAL_REFINE_SCALE,
                float(cand_fidelity[best_cand_idx]),
                best_cand_idx,
            )

        phi_r_final = np.repeat(base_phi_r, SEG_LEN)
        phi_b_final = np.repeat(base_phi_b, SEG_LEN)
        amp_r_final = np.repeat(base_amp_r, SEG_LEN)
        amp_b_final = np.repeat(base_amp_b, SEG_LEN)

        _, final_fidelity, rho_final, rho_target = trapped_ion_cat_sim(
            phi_r_final,
            phi_b_final,
            amp_r=amp_r_final,
            amp_b=amp_b_final,
            n_boson=N_BOSON,
            omega=2 * np.pi * 0.002,
            t_step=T_STEP,
            alpha_cat=ALPHA_CAT,
            cat_parity="even",
            cat_phase=CAT_REL_PHASE,
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
        )

        output_dir = os.path.join(os.getcwd(), "outputs")
        os.makedirs(output_dir, exist_ok=True)
        fidelity_path = os.path.join(output_dir, "final_fidelity.txt")
        with open(fidelity_path, "w", encoding="utf-8") as f:
            f.write(f"{final_fidelity:.6f}\n")
        logger.info("Final fidelity %.6f", final_fidelity)
        logger.info("Saved final fidelity to %s", fidelity_path)

        grid = np.linspace(-SAMPLE_EXTENT, SAMPLE_EXTENT, PLOT_GRID_SIZE)
        chi_target = characteristic_grid(rho_target, grid, grid)
        chi_final = characteristic_grid(rho_final, grid, grid)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        im0 = axes[0].imshow(
            chi_target,
            extent=[-SAMPLE_EXTENT, SAMPLE_EXTENT, -SAMPLE_EXTENT, SAMPLE_EXTENT],
            origin="lower",
            cmap="RdBu_r",
        )
        axes[0].set_title("Target cat characteristic")
        axes[0].set_xlabel("Re(alpha)")
        axes[0].set_ylabel("Im(alpha)")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(
            chi_final,
            extent=[-SAMPLE_EXTENT, SAMPLE_EXTENT, -SAMPLE_EXTENT, SAMPLE_EXTENT],
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
        reward_data, fidelity_data, _, _ = trapped_ion_cat_sim_batch(
            phi_r,
            phi_b,
            amp_r=amp_r,
            amp_b=amp_b,
            n_boson=N_BOSON,
            omega=2 * np.pi * 0.002,
            t_step=T_STEP,
            alpha_cat=ALPHA_CAT,
            cat_parity="even",
            cat_phase=CAT_REL_PHASE,
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
        )
    else:
        reward_data = trapped_ion_cat_sim_batch(
            phi_r,
            phi_b,
            amp_r=amp_r,
            amp_b=amp_b,
            n_boson=N_BOSON,
            omega=2 * np.pi * 0.002,
            t_step=T_STEP,
            alpha_cat=ALPHA_CAT,
            cat_parity="even",
            cat_phase=CAT_REL_PHASE,
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
        write_header = not os.path.exists(eval_log_path)
        with open(eval_log_path, "a", encoding="utf-8") as f:
            if write_header:
                f.write("epoch,mean_fidelity,std_fidelity\n")
            f.write(f"{epoch},{mean_fidelity:.6f},{std_fidelity:.6f}\n")

    logger.info("Sending message to RL agent server.")
    logger.info("Time stamp: %f", time.time())
    client_socket.send_data(reward_data)
