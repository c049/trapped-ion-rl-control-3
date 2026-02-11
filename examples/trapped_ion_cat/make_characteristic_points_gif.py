#!/usr/bin/env python3
import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from trapped_ion_cat_sim_function import prepare_characteristic_distribution


def _build_characteristic_distribution(
    alpha_cat,
    n_boson,
    sample_extent,
    grid_size,
    alpha_scales,
    mix_uniform,
    cat_phase,
):
    all_points = []
    all_targets = []
    all_weights = []
    n_scales = len(alpha_scales)
    for alpha_scale in alpha_scales:
        points_i, target_i, weights_i, _ = prepare_characteristic_distribution(
            alpha_cat=alpha_cat,
            n_boson=n_boson,
            extent=sample_extent,
            grid_size=grid_size,
            cat_parity="even",
            mix_uniform=mix_uniform,
            alpha_scale=alpha_scale,
            cat_phase=cat_phase,
        )
        all_points.extend(points_i)
        all_targets.append(target_i)
        all_weights.append(weights_i / float(n_scales))
    return all_points, np.concatenate(all_targets), np.concatenate(all_weights)


def _build_stage1_topk_indices(score, count):
    return np.argsort(score)[-count:]


def _sample_characteristic_points(
    rng,
    n_points,
    mode,
    char_points,
    char_target,
    char_weights,
    char_radii,
    radial_bins,
):
    if mode == "weighted":
        idx = rng.choice(len(char_points), size=n_points, replace=True, p=char_weights)
    elif mode == "uniform":
        idx = rng.choice(len(char_points), size=n_points, replace=True)
    elif mode == "radial_stratified":
        n_bins = max(1, radial_bins)
        r_max = float(np.max(char_radii))
        edges = np.linspace(0.0, r_max + 1e-12, n_bins + 1)
        idx_list = []
        bin_candidates = []
        bin_mass = np.zeros(n_bins, dtype=float)
        for bi in range(n_bins):
            lo = edges[bi]
            hi = edges[bi + 1]
            if bi == n_bins - 1:
                mask = (char_radii >= lo) & (char_radii <= hi)
            else:
                mask = (char_radii >= lo) & (char_radii < hi)
            candidates = np.flatnonzero(mask)
            bin_candidates.append(candidates)
            if candidates.size > 0:
                bin_mass[bi] = float(np.sum(char_weights[candidates]))

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
        for bi in range(n_bins):
            candidates = bin_candidates[bi]
            if candidates.size == 0:
                continue
            take = min(int(quotas[bi]), n_points - len(idx_list))
            if take <= 0:
                break
            local_w = char_weights[candidates]
            local_w_sum = float(np.sum(local_w))
            if local_w_sum > 0.0 and np.isfinite(local_w_sum):
                local_w = local_w / local_w_sum
                sampled = rng.choice(candidates, size=take, replace=True, p=local_w)
            else:
                sampled = rng.choice(candidates, size=take, replace=True)
            idx_list.extend(sampled.tolist())
        n_fill = n_points - len(idx_list)
        if n_fill > 0:
            fill = rng.choice(
                len(char_points),
                size=n_fill,
                replace=True,
                p=char_weights,
            )
            idx_list.extend(fill.tolist())
        idx = np.asarray(idx_list, dtype=int)
    else:
        raise ValueError(f"Unknown sampler mode: {mode}")
    return np.asarray([char_points[i] for i in idx], dtype=np.complex128)


def main():
    parser = argparse.ArgumentParser(
        description="Create a GIF of characteristic sample points across training epochs."
    )
    parser.add_argument("--num-epochs", type=int, default=2000)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--sample-extent", type=float, default=4.0)
    parser.add_argument("--plot-extent", type=float, default=6.0)
    parser.add_argument("--alpha-cat", type=float, default=2.0)
    parser.add_argument("--n-boson", type=int, default=30)
    parser.add_argument("--char-grid-size", type=int, default=61)
    parser.add_argument("--train-points-stage1", type=int, default=120)
    parser.add_argument("--train-points-stage2", type=int, default=240)
    parser.add_argument("--train-points-stage3", type=int, default=960)
    parser.add_argument("--train-stage1-epochs", type=int, default=120)
    parser.add_argument("--train-stage2-epochs", type=int, default=240)
    parser.add_argument("--char-start-mode", type=str, default="radial_topk")
    parser.add_argument("--char-radial-exp", type=float, default=1.0)
    parser.add_argument("--char-sampler-mode", type=str, default="radial_stratified")
    parser.add_argument("--char-radial-bins", type=int, default=12)
    parser.add_argument("--char-uniform-mix", type=float, default=0.2)
    parser.add_argument("--char-alpha-scales", type=str, default="1.0")
    parser.add_argument("--show-eval-points", action="store_true")
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/characteristic_points_evolution.gif",
    )
    args = parser.parse_args()

    alpha_scales = [float(v.strip()) for v in args.char_alpha_scales.split(",") if v.strip()]
    cat_phase = None

    char_points, char_target, char_weights = _build_characteristic_distribution(
        alpha_cat=args.alpha_cat,
        n_boson=args.n_boson,
        sample_extent=args.sample_extent,
        grid_size=args.char_grid_size,
        alpha_scales=alpha_scales,
        mix_uniform=args.char_uniform_mix,
        cat_phase=cat_phase,
    )
    char_radii = np.abs(np.asarray(char_points))

    topk_count = min(args.train_points_stage1, len(char_points))
    if args.char_start_mode == "topk":
        score = np.abs(char_target)
    else:
        radii = np.maximum(char_radii, 1e-6)
        score = np.abs(char_target) * (radii ** args.char_radial_exp)
    topk_idx = _build_stage1_topk_indices(score, topk_count)
    topk_points = np.asarray([char_points[i] for i in topk_idx], dtype=np.complex128)

    def select_points(epoch):
        if epoch < args.train_stage1_epochs:
            return topk_points, "stage1_topk"
        rng = np.random.default_rng(epoch)
        if epoch < args.train_stage2_epochs:
            return _sample_characteristic_points(
                rng,
                args.train_points_stage2,
                args.char_sampler_mode,
                char_points,
                char_target,
                char_weights,
                char_radii,
                args.char_radial_bins,
            ), "stage2_sampled"
        return _sample_characteristic_points(
            rng,
            args.train_points_stage3,
            args.char_sampler_mode,
            char_points,
            char_target,
            char_weights,
            char_radii,
            args.char_radial_bins,
        ), "stage3_sampled"

    eval_points = None
    if args.show_eval_points:
        eval_rng = np.random.default_rng(12345)
        eval_points = _sample_characteristic_points(
            eval_rng,
            args.train_points_stage3,
            args.char_sampler_mode,
            char_points,
            char_target,
            char_weights,
            char_radii,
            args.char_radial_bins,
        )

    frame_epochs = list(range(0, args.num_epochs + 1, args.stride))
    if frame_epochs[-1] != args.num_epochs:
        frame_epochs.append(args.num_epochs)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.set_xlim(-args.plot_extent, args.plot_extent)
    ax.set_ylim(-args.plot_extent, args.plot_extent)
    ax.set_aspect("equal")
    ax.set_xlabel("Re(alpha)")
    ax.set_ylabel("Im(alpha)")
    ax.set_title("Characteristic sample points over training")
    ax.grid(alpha=0.2, linewidth=0.5)

    if eval_points is not None:
        ax.scatter(
            np.real(eval_points),
            np.imag(eval_points),
            s=5,
            c="tab:cyan",
            alpha=0.15,
            label="eval points (fixed)",
            linewidths=0.0,
        )

    scat = ax.scatter([], [], s=8, c="gold", alpha=0.70, linewidths=0.0, label="train points")
    text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )
    ax.legend(loc="lower right")

    def init():
        scat.set_offsets(np.empty((0, 2)))
        text.set_text("")
        return scat, text

    def update(frame_idx):
        epoch = frame_epochs[frame_idx]
        points, stage = select_points(epoch)
        xy = np.column_stack((np.real(points), np.imag(points)))
        scat.set_offsets(xy)
        text.set_text(f"epoch={epoch}\\n{stage}\\npoints={len(points)}")
        return scat, text

    anim = FuncAnimation(
        fig,
        update,
        frames=len(frame_epochs),
        init_func=init,
        interval=100,
        blit=False,
    )

    out_path = args.output
    if not os.path.isabs(out_path):
        out_path = os.path.join(THIS_DIR, out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    anim.save(out_path, writer=PillowWriter(fps=args.fps))
    plt.close(fig)
    print(f"Saved GIF to {out_path}")
    print(f"Frames: {len(frame_epochs)} | Epoch stride: {args.stride}")


if __name__ == "__main__":
    main()

