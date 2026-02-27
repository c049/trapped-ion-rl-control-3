#!/usr/bin/env python3
import argparse
import csv
import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def _read_manifest(path: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _read_scalar_txt(path: str) -> Optional[float]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    try:
        return float(raw)
    except ValueError:
        return None


def _read_final_robust(path: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    for item in raw.split(","):
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        try:
            out[k.strip()] = float(v.strip())
        except ValueError:
            continue
    return out


def _read_sweep(path: str) -> Optional[np.ndarray]:
    if not os.path.exists(path):
        return None
    rows: List[List[float]] = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(
                [
                    float(row["detuning_frac"]),
                    float(row["detuning"]),
                    float(row["robust_fidelity"]),
                ]
            )
    if not rows:
        return None
    return np.asarray(rows, dtype=float)


def _sweep_stats(sweep: Optional[np.ndarray]) -> Dict[str, float]:
    if sweep is None or sweep.size == 0:
        return {}
    fids = sweep[:, 2]
    return {
        "sweep_mean": float(np.mean(fids)),
        "sweep_min": float(np.min(fids)),
        "sweep_max": float(np.max(fids)),
    }


def _compare_gain_stats(compare_csv: str) -> Dict[str, float]:
    if not os.path.exists(compare_csv):
        return {}
    robust = []
    baseline = []
    with open(compare_csv, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            robust.append(float(row["robust_fidelity"]))
            baseline.append(float(row["baseline_fidelity"]))
    if not robust:
        return {}
    robust_arr = np.asarray(robust, dtype=float)
    baseline_arr = np.asarray(baseline, dtype=float)
    diff = robust_arr - baseline_arr
    return {
        "compare_mean_robust": float(np.mean(robust_arr)),
        "compare_mean_baseline": float(np.mean(baseline_arr)),
        "compare_mean_gain": float(np.mean(diff)),
        "compare_min_gain": float(np.min(diff)),
        "compare_max_gain": float(np.max(diff)),
    }


def _stochastic_compare_stats(stochastic_compare_csv: str) -> Dict[str, float]:
    if not os.path.exists(stochastic_compare_csv):
        return {}
    robust_row: Optional[Dict[str, float]] = None
    baseline_row: Optional[Dict[str, float]] = None
    with open(stochastic_compare_csv, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            pulse = row.get("pulse", "").strip().lower()
            parsed = {}
            for k in [
                "mean_score",
                "std_score",
                "mean_f_nom",
                "std_f_nom",
                "mean_f_rob",
                "std_f_rob",
                "mean_penalty",
                "std_penalty",
            ]:
                try:
                    parsed[k] = float(row[k])
                except Exception:
                    parsed[k] = float("nan")
            if pulse == "robust":
                robust_row = parsed
            elif pulse == "baseline":
                baseline_row = parsed
    if robust_row is None or baseline_row is None:
        return {}
    return {
        "stoch_score_robust": float(robust_row["mean_score"]),
        "stoch_score_baseline": float(baseline_row["mean_score"]),
        "stoch_score_gain": float(robust_row["mean_score"] - baseline_row["mean_score"]),
        "stoch_frob_robust": float(robust_row["mean_f_rob"]),
        "stoch_frob_baseline": float(baseline_row["mean_f_rob"]),
        "stoch_frob_gain": float(robust_row["mean_f_rob"] - baseline_row["mean_f_rob"]),
    }


def _safe_float(v: str, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _plot_pareto(rows: List[Dict[str, float]], out_png: str) -> None:
    valid = [r for r in rows if np.isfinite(r["final_nominal"]) and np.isfinite(r["sweep_mean"])]
    if not valid:
        return
    x = np.array([r["final_nominal"] for r in valid], dtype=float)
    y = np.array([r["sweep_mean"] for r in valid], dtype=float)
    p = np.array([r["penalty"] for r in valid], dtype=float)

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.2))
    sc = ax.scatter(x, y, c=p, cmap="viridis", s=70)
    for r in valid:
        ax.annotate(
            f"p={r['penalty']:g}",
            (r["final_nominal"], r["sweep_mean"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=9,
        )
    ax.set_xlabel("Nominal fidelity F(delta=0)")
    ax.set_ylabel("Robust sweep mean fidelity")
    ax.set_title("Penalty Pareto: nominal vs robustness")
    ax.grid(alpha=0.25)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("ROBUST_FLOOR_PENALTY")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sweep-root",
        required=True,
        help="Root directory containing per-penalty subfolders",
    )
    args = parser.parse_args()

    sweep_root = os.path.abspath(args.sweep_root)
    if not os.path.isdir(sweep_root):
        raise FileNotFoundError(f"Sweep root not found: {sweep_root}")

    run_dirs = [
        os.path.join(sweep_root, d)
        for d in sorted(os.listdir(sweep_root))
        if os.path.isdir(os.path.join(sweep_root, d))
    ]
    if not run_dirs:
        raise RuntimeError(f"No run directories found under {sweep_root}")

    rows_out: List[Dict[str, float]] = []
    for run_dir in run_dirs:
        manifest = _read_manifest(os.path.join(run_dir, "manifest.env"))
        penalty = _safe_float(manifest.get("ROBUST_FLOOR_PENALTY", "nan"), float("nan"))
        nominal = _read_scalar_txt(os.path.join(run_dir, "final_fidelity.txt"))
        robust_info = _read_final_robust(os.path.join(run_dir, "final_robust_score.txt"))
        sweep = _read_sweep(os.path.join(run_dir, "dephasing_sweep_robust.csv"))
        sweep_stats = _sweep_stats(sweep)
        compare_stats = _compare_gain_stats(os.path.join(run_dir, "dephasing_compare.csv"))
        stochastic_stats = _stochastic_compare_stats(os.path.join(run_dir, "stochastic_compare.csv"))

        row: Dict[str, float] = {
            "penalty": penalty,
            "final_nominal": float("nan") if nominal is None else float(nominal),
            "final_score": robust_info.get("score", float("nan")),
            "final_f_nom": robust_info.get("f_nom", float("nan")),
            "final_f_rob": robust_info.get("f_rob", float("nan")),
            "final_penalty_term": robust_info.get("penalty", float("nan")),
            "sweep_mean": sweep_stats.get("sweep_mean", float("nan")),
            "sweep_min": sweep_stats.get("sweep_min", float("nan")),
            "sweep_max": sweep_stats.get("sweep_max", float("nan")),
            "compare_mean_robust": compare_stats.get("compare_mean_robust", float("nan")),
            "compare_mean_baseline": compare_stats.get("compare_mean_baseline", float("nan")),
            "compare_mean_gain": compare_stats.get("compare_mean_gain", float("nan")),
            "compare_min_gain": compare_stats.get("compare_min_gain", float("nan")),
            "compare_max_gain": compare_stats.get("compare_max_gain", float("nan")),
            "stoch_score_robust": stochastic_stats.get("stoch_score_robust", float("nan")),
            "stoch_score_baseline": stochastic_stats.get("stoch_score_baseline", float("nan")),
            "stoch_score_gain": stochastic_stats.get("stoch_score_gain", float("nan")),
            "stoch_frob_robust": stochastic_stats.get("stoch_frob_robust", float("nan")),
            "stoch_frob_baseline": stochastic_stats.get("stoch_frob_baseline", float("nan")),
            "stoch_frob_gain": stochastic_stats.get("stoch_frob_gain", float("nan")),
            "status": _safe_float(manifest.get("STATUS", "nan"), float("nan")),
        }
        row["run_dir"] = run_dir
        rows_out.append(row)

    rows_out.sort(
        key=lambda r: (np.inf if not np.isfinite(r["penalty"]) else r["penalty"])
    )

    summary_csv = os.path.join(sweep_root, "penalty_sweep_summary.csv")
    fieldnames = [
        "penalty",
        "final_nominal",
        "final_score",
        "final_f_nom",
        "final_f_rob",
        "final_penalty_term",
        "sweep_mean",
        "sweep_min",
        "sweep_max",
        "compare_mean_robust",
        "compare_mean_baseline",
        "compare_mean_gain",
        "compare_min_gain",
        "compare_max_gain",
        "stoch_score_robust",
        "stoch_score_baseline",
        "stoch_score_gain",
        "stoch_frob_robust",
        "stoch_frob_baseline",
        "stoch_frob_gain",
        "status",
        "run_dir",
    ]
    with open(summary_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows_out:
            w.writerow(row)

    pareto_png = os.path.join(sweep_root, "pareto_nominal_vs_robust_mean.png")
    _plot_pareto(rows_out, pareto_png)

    print(f"Wrote summary: {summary_csv}")
    print(f"Wrote Pareto:  {pareto_png}")


if __name__ == "__main__":
    main()
