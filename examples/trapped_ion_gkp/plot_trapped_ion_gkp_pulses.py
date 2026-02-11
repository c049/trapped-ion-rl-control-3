import argparse
import glob
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np


def _latest_h5(root_dir):
    h5_files = sorted(glob.glob(os.path.join(root_dir, "*.h5")))
    if not h5_files:
        raise SystemExit("No .h5 files found in this directory.")
    return h5_files[-1]


def _latest_run_group(h5_file):
    runs = [k for k in h5_file.keys() if k.isdigit()]
    if not runs:
        raise SystemExit("No numeric run groups found in the h5 file.")
    return sorted(runs, key=int)[-1]


def _load_locs(h5_file, run_group):
    locs_group = h5_file[run_group]["policy_distribution"]["locs"]
    locs = {}
    for key in locs_group.keys():
        arr = np.array(locs_group[key][-1]).squeeze()
        locs[key] = arr
    return locs


def _get_or_default(locs, key, n_segments, default_value):
    if key in locs:
        return np.asarray(locs[key], dtype=float)
    return np.full(n_segments, float(default_value), dtype=float)


def _expand_segments(seg_values, n_steps):
    n_segments = len(seg_values)
    if n_steps % n_segments != 0:
        raise ValueError(
            f"n_steps={n_steps} not divisible by n_segments={n_segments}."
        )
    seg_len = n_steps // n_segments
    return np.repeat(seg_values, seg_len)


def main():
    parser = argparse.ArgumentParser(
        description="Plot trapped-ion pulse sequences from the latest h5 file."
    )
    parser.add_argument("--h5", default=None, help="Path to the h5 log file.")
    parser.add_argument("--n-steps", type=int, default=120, help="Total steps.")
    parser.add_argument("--t-step", type=float, default=10.0, help="Time step.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for the plot (png). Defaults to outputs/pulse_sequences.png.",
    )
    args = parser.parse_args()

    h5_path = args.h5 or _latest_h5(os.getcwd())
    with h5py.File(h5_path, "r") as f:
        run = _latest_run_group(f)
        locs = _load_locs(f, run)

    required = ["phi_r", "phi_b"]
    for key in required:
        if key not in locs:
            raise SystemExit(f"Missing {key} in policy_distribution/locs.")

    n_segments = len(np.asarray(locs["phi_r"]).reshape(-1))
    phi_r_seg = _get_or_default(locs, "phi_r", n_segments, 0.0)
    phi_b_seg = _get_or_default(locs, "phi_b", n_segments, 0.0)
    amp_r_seg = _get_or_default(locs, "amp_r", n_segments, 1.0)
    amp_b_seg = _get_or_default(locs, "amp_b", n_segments, 1.0)

    phi_r = _expand_segments(phi_r_seg, args.n_steps)
    phi_b = _expand_segments(phi_b_seg, args.n_steps)
    amp_r = _expand_segments(amp_r_seg, args.n_steps)
    amp_b = _expand_segments(amp_b_seg, args.n_steps)

    t = np.arange(args.n_steps) * args.t_step

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    axes[0, 0].plot(t, phi_r)
    axes[0, 0].set_title("phi_r(t)")
    axes[0, 0].set_ylabel("rad")

    axes[0, 1].plot(t, phi_b)
    axes[0, 1].set_title("phi_b(t)")

    axes[1, 0].plot(t, amp_r)
    axes[1, 0].set_title("amp_r(t)")
    axes[1, 0].set_xlabel("time")
    axes[1, 0].set_ylabel("relative amplitude")

    axes[1, 1].plot(t, amp_b)
    axes[1, 1].set_title("amp_b(t)")
    axes[1, 1].set_xlabel("time")

    fig.tight_layout()

    output_path = args.output
    if output_path is None:
        output_dir = os.path.join(os.getcwd(), "outputs")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "pulse_sequences.png")

    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved pulse plot to {output_path}")


if __name__ == "__main__":
    main()
