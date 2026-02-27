#!/usr/bin/env python3
"""Build a detailed Sydney Uni weekly report PPT.

Design goals from meeting notes:
- Motivation -> Setup/Method -> Results -> Outlook story.
- Picture + words first, no code snippets.
- Explain RL loop clearly (episode/epoch/policy update/PPO).
- Emphasize characteristic-function reward and sampling-point evolution.
- Include all three state GIFs (Cat/GKP/Binomial).
- Summarize robust training (quasi-static + stochastic) with latest sweep results.
"""

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
PPT_DIR = ROOT / "ppt"
ASSET_DIR = PPT_DIR / "assets"

CAT_OUT = ROOT / "examples" / "trapped_ion_cat" / "outputs"
GKP_OUT = ROOT / "examples" / "trapped_ion_gkp" / "outputs"
BIN_OUT = ROOT / "examples" / "trapped_ion_binomial" / "outputs"
BIN_PRE_ROBUST_OUT = BIN_OUT / "archive_20260225_144627_pre_new_run"
BIN_SWEEP_ROOT = ROOT / "examples" / "trapped_ion_binomial" / "penalty_sweep"

STATIC_SWEEP = BIN_SWEEP_ROOT / "penalty_sweep_20260225_195742"
STOCH_SWEEP_COARSE = BIN_SWEEP_ROOT / "stochastic_penalty_sweep_20260226_023916"
STOCH_SWEEP_REFINE = BIN_SWEEP_ROOT / "stoch_validate_uniform_weight_local_20260226_170726"

PPT_PATH_DETAILED = PPT_DIR / "Sydney_Uni_Weekly_Report_2026-03-03_detailed.pptx"
PPT_PATH_ALIAS = PPT_DIR / "Sydney_Uni_Weekly_Report_2026-03-03.pptx"


@dataclass
class SweepEntry:
    mode: str
    tag: str
    penalty: float
    score: float
    f_nom: float
    f_rob: float
    penalty_term: float
    run_dir: Path


@dataclass
class DephasingStats:
    mean_robust: float
    mean_baseline: float
    mean_gain: float
    min_gain: float
    max_gain: float
    zero_gain: float


@dataclass
class HyperParamSummary:
    cat_num_epochs: str
    cat_policy_updates: str
    cat_n_steps: str
    cat_n_segments: str
    cat_stage_points: str
    cat_stage_epochs: str
    gkp_num_epochs: str
    gkp_policy_updates: str
    gkp_n_steps: str
    gkp_n_segments: str
    gkp_stage_points: str
    gkp_stage_epochs: str
    bin_num_epochs: str
    bin_policy_updates: str
    bin_n_steps: str
    bin_n_segments: str
    bin_stage_points: str
    bin_stage_epochs: str


@dataclass
class Metrics:
    cat_fid: float
    gkp_fid: float
    bin_pre_fid: float
    bin_pre_score: float
    bin_pre_f_rob: float
    bin_pre_run_dir: Path
    static_entries: List[SweepEntry]
    stoch_entries: List[SweepEntry]
    best_static: SweepEntry
    best_stoch: SweepEntry
    static_stats: DephasingStats
    stoch_stats: DephasingStats
    hyper: HyperParamSummary


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        fp = Path(p)
        if fp.exists():
            return ImageFont.truetype(str(fp), size=size)
    return ImageFont.load_default()


def _rr(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[int, int, int, int],
    radius: int,
    fill,
    outline,
    width: int = 3,
) -> None:
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def _arrow(
    draw: ImageDraw.ImageDraw,
    p1: Tuple[int, int],
    p2: Tuple[int, int],
    color,
    width: int = 5,
    head: int = 18,
) -> None:
    draw.line([p1, p2], fill=color, width=width)
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    norm = (dx * dx + dy * dy) ** 0.5 or 1.0
    ux = dx / norm
    uy = dy / norm
    px = -uy
    py = ux
    hx = x2 - int(head * ux)
    hy = y2 - int(head * uy)
    poly = [
        (x2, y2),
        (hx + int(0.6 * head * px), hy + int(0.6 * head * py)),
        (hx - int(0.6 * head * px), hy - int(0.6 * head * py)),
    ]
    draw.polygon(poly, fill=color)


def _wrap_text_lines(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
) -> List[str]:
    """Wrap text to fit pixel width for PIL drawing while preserving paragraphs."""
    out: List[str] = []
    for para in text.split("\n"):
        words = para.split()
        if not words:
            out.append("")
            continue
        current = words[0]
        for w in words[1:]:
            candidate = current + " " + w
            bb = draw.textbbox((0, 0), candidate, font=font)
            width = bb[2] - bb[0]
            if width <= max_width:
                current = candidate
            else:
                out.append(current)
                current = w
        out.append(current)
    return out


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> str:
    return "\n".join(_wrap_text_lines(draw, text, font, max_width))


def _fit_wrapped_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    max_width: int,
    max_height: int,
    start_size: int,
    min_size: int = 14,
    bold: bool = False,
    spacing: int = 6,
) -> Tuple[ImageFont.FreeTypeFont, str]:
    """Find a wrapped-text font size that fits within a box."""
    if max_width <= 2 or max_height <= 2:
        font = _font(min_size, bold=bold)
        return font, text

    for size in range(start_size, min_size - 1, -1):
        font = _font(size, bold=bold)
        wrapped = "\n".join(_wrap_text_lines(draw, text, font, max_width))
        bb = draw.multiline_textbbox((0, 0), wrapped, font=font, spacing=spacing)
        tw = bb[2] - bb[0]
        th = bb[3] - bb[1]
        if tw <= max_width and th <= max_height:
            return font, wrapped

    font = _font(min_size, bold=bold)
    wrapped = "\n".join(_wrap_text_lines(draw, text, font, max_width))
    return font, wrapped


def _draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
    fill,
    spacing: int = 6,
) -> None:
    wrapped = _wrap_text(draw, text, font, max_width)
    draw.multiline_text(xy, wrapped, font=font, fill=fill, spacing=spacing)


def _draw_text_in_box(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    text: str,
    *,
    start_size: int,
    min_size: int = 14,
    bold: bool = False,
    fill=(30, 46, 72),
    spacing: int = 6,
    padding: int = 0,
    align: str = "left",
    valign: str = "top",
) -> Tuple[int, int]:
    """Draw wrapped text inside a box with automatic font down-scaling."""
    x1, y1, x2, y2 = box
    inner_x1 = x1 + padding
    inner_y1 = y1 + padding
    inner_x2 = x2 - padding
    inner_y2 = y2 - padding
    max_width = max(2, inner_x2 - inner_x1)
    max_height = max(2, inner_y2 - inner_y1)
    font, wrapped = _fit_wrapped_text(
        draw,
        text,
        max_width=max_width,
        max_height=max_height,
        start_size=start_size,
        min_size=min_size,
        bold=bold,
        spacing=spacing,
    )
    bb = draw.multiline_textbbox((0, 0), wrapped, font=font, spacing=spacing, align=align)
    tw = bb[2] - bb[0]
    th = bb[3] - bb[1]

    if align == "center":
        tx = inner_x1 + max(0, (max_width - tw) // 2)
    elif align == "right":
        tx = inner_x1 + max(0, max_width - tw)
    else:
        tx = inner_x1

    if valign == "middle":
        ty = inner_y1 + max(0, (max_height - th) // 2)
    elif valign == "bottom":
        ty = inner_y1 + max(0, max_height - th)
    else:
        ty = inner_y1

    draw.multiline_text((tx, ty), wrapped, font=font, fill=fill, spacing=spacing, align=align)
    return tw, th


def _safe_float_text(path: Path, default: float = 0.0) -> float:
    try:
        return float(path.read_text(encoding="utf-8").strip())
    except Exception:
        return default


def _read_score_file(path: Path) -> Dict[str, float]:
    out = {}
    txt = path.read_text(encoding="utf-8", errors="ignore").strip()
    for item in txt.split(","):
        if "=" in item:
            k, v = item.split("=", 1)
            try:
                out[k.strip()] = float(v.strip())
            except Exception:
                continue
    return out


def _parse_penalty_token(token: str) -> float:
    t = token.strip()
    if t == "":
        return 0.0
    if t == "0":
        return 0.0
    if "p" in t:
        t = t.replace("p", ".")
    return float(t)


def _collect_sweep_entries(root: Path, prefix: str, mode: str) -> List[SweepEntry]:
    entries: List[SweepEntry] = []
    if not root.exists():
        return entries
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir() or not run_dir.name.startswith(prefix):
            continue
        score_file = run_dir / "final_robust_score.txt"
        if not score_file.exists():
            continue
        vals = _read_score_file(score_file)
        token = run_dir.name[len(prefix) :]
        try:
            penalty = _parse_penalty_token(token)
        except Exception:
            continue
        entries.append(
            SweepEntry(
                mode=mode,
                tag=run_dir.name,
                penalty=penalty,
                score=vals.get("score", 0.0),
                f_nom=vals.get("f_nom", 0.0),
                f_rob=vals.get("f_rob", 0.0),
                penalty_term=vals.get("penalty", 0.0),
                run_dir=run_dir,
            )
        )
    return entries


def _best_by_score(entries: List[SweepEntry]) -> SweepEntry:
    if not entries:
        raise RuntimeError("No sweep entries found.")
    return sorted(entries, key=lambda x: x.score, reverse=True)[0]


def _best_per_penalty(entries: List[SweepEntry]) -> List[SweepEntry]:
    keep: Dict[float, SweepEntry] = {}
    for e in entries:
        if e.penalty not in keep or e.score > keep[e.penalty].score:
            keep[e.penalty] = e
    return [keep[k] for k in sorted(keep)]


def _read_dephasing_stats(run_dir: Path) -> DephasingStats:
    csv_path = run_dir / "dephasing_compare.csv"
    if not csv_path.exists():
        return DephasingStats(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    df = pd.read_csv(csv_path)
    if not {"robust_fidelity", "baseline_fidelity", "detuning_frac"}.issubset(df.columns):
        return DephasingStats(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    gain = df["robust_fidelity"] - df["baseline_fidelity"]
    zero_idx = int((df["detuning_frac"].abs()).idxmin())
    return DephasingStats(
        mean_robust=float(df["robust_fidelity"].mean()),
        mean_baseline=float(df["baseline_fidelity"].mean()),
        mean_gain=float(gain.mean()),
        min_gain=float(gain.min()),
        max_gain=float(gain.max()),
        zero_gain=float(gain.iloc[zero_idx]),
    )


def _extract_env_default(path: Path, key: str, fallback: str = "-") -> str:
    if not path.exists():
        return fallback
    txt = path.read_text(encoding="utf-8", errors="ignore")
    patterns = [
        r'os\.environ\.get\("%s",\s*"([^"]+)"\)' % re.escape(key),
        r"os\.environ\.get\('%s',\s*'([^']+)'\)" % re.escape(key),
    ]
    for pat in patterns:
        m = re.search(pat, txt)
        if m:
            return m.group(1)
    return fallback


def _extract_literal_assignment(path: Path, key: str, fallback: str = "-") -> str:
    if not path.exists():
        return fallback
    txt = path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"^\s*%s\s*=\s*([0-9\.]+)\s*$" % re.escape(key), txt, re.MULTILINE)
    if m:
        return m.group(1)
    return fallback


def _read_hyperparams() -> HyperParamSummary:
    cat_server = ROOT / "examples" / "trapped_ion_cat" / "trapped_ion_cat_training_server.py"
    cat_client = ROOT / "examples" / "trapped_ion_cat" / "trapped_ion_cat_client.py"
    gkp_server = ROOT / "examples" / "trapped_ion_gkp" / "trapped_ion_gkp_training_server.py"
    gkp_client = ROOT / "examples" / "trapped_ion_gkp" / "trapped_ion_gkp_client.py"
    bin_server = ROOT / "examples" / "trapped_ion_binomial" / "trapped_ion_binomial_training_server.py"
    bin_client = ROOT / "examples" / "trapped_ion_binomial" / "trapped_ion_binomial_client.py"

    cat_s1_epoch = _extract_env_default(cat_client, "TRAIN_STAGE1_EPOCHS", "120")
    cat_s2_epoch = _extract_env_default(cat_client, "TRAIN_STAGE2_EPOCHS", "240")
    gkp_s1_epoch = _extract_env_default(gkp_client, "TRAIN_STAGE1_EPOCHS", "0")
    gkp_s2_epoch = _extract_env_default(gkp_client, "TRAIN_STAGE2_EPOCHS", "180")
    bin_s1_epoch = _extract_env_default(bin_client, "TRAIN_STAGE1_EPOCHS", "120")
    bin_s2_epoch = _extract_env_default(bin_client, "TRAIN_STAGE2_EPOCHS", "240")

    gkp_stage_epoch_text = (
        "S1=off, S2<%s" % gkp_s2_epoch if gkp_s1_epoch == "0" else "S1<%s, S2<%s" % (gkp_s1_epoch, gkp_s2_epoch)
    )

    return HyperParamSummary(
        cat_num_epochs=_extract_env_default(cat_server, "NUM_EPOCHS", "300"),
        cat_policy_updates=_extract_env_default(cat_server, "PPO_NUM_POLICY_UPDATES", "20"),
        cat_n_steps=_extract_literal_assignment(cat_client, "N_STEPS", "120"),
        cat_n_segments=_extract_literal_assignment(cat_client, "N_SEGMENTS", "60"),
        cat_stage_points=(
            "S1=%s, S2=%s, S3=%s"
            % (
                _extract_env_default(cat_client, "TRAIN_POINTS_STAGE1", "120"),
                _extract_env_default(cat_client, "TRAIN_POINTS_STAGE2", "240"),
                _extract_env_default(cat_client, "TRAIN_POINTS_STAGE3", "960"),
            )
        ),
        cat_stage_epochs=(
            "S1<%s, S2<%s"
            % (
                cat_s1_epoch,
                cat_s2_epoch,
            )
        ),
        gkp_num_epochs=_extract_env_default(gkp_server, "NUM_EPOCHS", "2000"),
        gkp_policy_updates=_extract_env_default(gkp_server, "PPO_NUM_POLICY_UPDATES", "10"),
        gkp_n_steps=_extract_env_default(gkp_server, "N_STEPS", "120"),
        gkp_n_segments=_extract_env_default(gkp_server, "N_SEGMENTS", "60"),
        gkp_stage_points=(
            "S1=%s, S2=%s, S3=%s"
            % (
                _extract_env_default(gkp_client, "TRAIN_POINTS_STAGE1", "120"),
                _extract_env_default(gkp_client, "TRAIN_POINTS_STAGE2", "240"),
                _extract_env_default(gkp_client, "TRAIN_POINTS_STAGE3", "960"),
            )
        ),
        gkp_stage_epochs=(
            gkp_stage_epoch_text
        ),
        bin_num_epochs=_extract_env_default(bin_server, "NUM_EPOCHS", "300"),
        bin_policy_updates=_extract_env_default(bin_server, "PPO_NUM_POLICY_UPDATES", "20"),
        bin_n_steps=_extract_env_default(bin_server, "N_STEPS", "120"),
        bin_n_segments=_extract_env_default(bin_server, "N_SEGMENTS", "60"),
        bin_stage_points=(
            "S1=%s, S2=%s, S3=%s"
            % (
                _extract_env_default(bin_client, "TRAIN_POINTS_STAGE1", "120"),
                _extract_env_default(bin_client, "TRAIN_POINTS_STAGE2", "240"),
                _extract_env_default(bin_client, "TRAIN_POINTS_STAGE3", "960"),
            )
        ),
        bin_stage_epochs=(
            "S1<%s, S2<%s"
            % (
                bin_s1_epoch,
                bin_s2_epoch,
            )
        ),
    )


def read_metrics() -> Metrics:
    static_entries_raw = _collect_sweep_entries(STATIC_SWEEP, "p_", "quasi-static")
    stoch_entries_raw = _collect_sweep_entries(STOCH_SWEEP_COARSE, "stoch_p_", "stochastic")
    stoch_entries_raw += _collect_sweep_entries(STOCH_SWEEP_REFINE, "stoch_p_", "stochastic")

    static_entries = _best_per_penalty(static_entries_raw)
    stoch_entries = _best_per_penalty(stoch_entries_raw)

    best_static = _best_by_score(static_entries_raw)
    best_stoch = _best_by_score(stoch_entries_raw)
    pre_vals = _read_score_file(BIN_PRE_ROBUST_OUT / "final_robust_score.txt") if (BIN_PRE_ROBUST_OUT / "final_robust_score.txt").exists() else {}

    return Metrics(
        cat_fid=_safe_float_text(CAT_OUT / "final_fidelity.txt"),
        gkp_fid=_safe_float_text(GKP_OUT / "final_fidelity.txt"),
        bin_pre_fid=_safe_float_text(BIN_PRE_ROBUST_OUT / "final_fidelity.txt", default=_safe_float_text(BIN_OUT / "final_fidelity.txt")),
        bin_pre_score=float(pre_vals.get("score", 0.0)),
        bin_pre_f_rob=float(pre_vals.get("f_rob", 0.0)),
        bin_pre_run_dir=BIN_PRE_ROBUST_OUT if BIN_PRE_ROBUST_OUT.exists() else BIN_OUT,
        static_entries=static_entries,
        stoch_entries=stoch_entries,
        best_static=best_static,
        best_stoch=best_stoch,
        static_stats=_read_dephasing_stats(best_static.run_dir),
        stoch_stats=_read_dephasing_stats(best_stoch.run_dir),
        hyper=_read_hyperparams(),
    )


def _make_missing_image(path: Path) -> None:
    img = Image.new("RGB", (1200, 700), (246, 248, 252))
    d = ImageDraw.Draw(img)
    d.text((80, 300), "Missing image", font=_font(64, bold=True), fill=(120, 130, 150))
    img.save(path)


def _safe_image(path: Path, fallback: Path) -> Path:
    if path.exists():
        return path
    return fallback


def build_hook_image(path: Path) -> None:
    w, h = 1600, 900
    img = Image.new("RGB", (w, h), (246, 250, 255))
    d = ImageDraw.Draw(img)
    f_title = _font(44, bold=True)
    f_sub = _font(30, bold=True)
    f_txt = _font(23)

    _draw_text_in_box(
        d,
        (60, 28, 1520, 120),
        "Main Question: Can model-free RL learn robust bosonic control pulses?",
        start_size=44,
        min_size=34,
        bold=True,
        fill=(20, 36, 58),
        spacing=8,
    )

    cards = [
        (70, 160, 520, 420, "What", "Prepare Cat / GKP / Binomial states with high fidelity."),
        (560, 160, 1010, 420, "Why", "Bosonic states support logical qubits, sensing, and tests of quantum physics."),
        (1050, 160, 1530, 420, "How", "Closed-loop RL: propose controls -> measure -> reward -> update policy."),
    ]
    cols = [
        ((232, 242, 255), (79, 128, 214)),
        ((236, 255, 243), (56, 160, 108)),
        ((255, 245, 234), (208, 136, 45)),
    ]
    for (x1, y1, x2, y2, title, body), (fill, outline) in zip(cards, cols):
        _rr(d, (x1, y1, x2, y2), 26, fill=fill, outline=outline, width=4)
        _draw_text_in_box(
            d,
            (x1 + 24, y1 + 20, x2 - 24, y1 + 75),
            title,
            start_size=30,
            min_size=24,
            bold=True,
            fill=(30, 46, 72),
            spacing=6,
        )
        _draw_text_in_box(
            d,
            (x1 + 24, y1 + 85, x2 - 24, y2 - 20),
            body,
            start_size=23,
            min_size=16,
            fill=(33, 52, 80),
            spacing=8,
        )

    _rr(d, (70, 470, 1530, 830), 30, fill=(255, 255, 255), outline=(124, 143, 173), width=3)
    _draw_text_in_box(
        d,
        (105, 510, 360, 560),
        "Presentation roadmap",
        start_size=30,
        min_size=24,
        bold=True,
        fill=(36, 55, 85),
    )
    _draw_text_in_box(
        d,
        (105, 565, 1490, 805),
        "1) Explain RL and optimization in plain physics language.\n"
        "2) Show characteristic-function sampling logic with GIFs.\n"
        "3) Present robust-training results under dephasing noise.\n"
        "4) Make clear what works now and what still needs improvement.",
        start_size=23,
        min_size=16,
        fill=(35, 55, 85),
        spacing=11,
    )
    img.save(path)


def build_applications_image(path: Path) -> None:
    w, h = 1600, 900
    img = Image.new("RGB", (w, h), (248, 251, 255))
    d = ImageDraw.Draw(img)
    f_title = _font(50, bold=True)
    f_sub = _font(32, bold=True)
    f_txt = _font(24)

    _draw_text_in_box(
        d,
        (60, 28, 1540, 112),
        "Why Bosonic State Preparation Matters",
        start_size=50,
        min_size=36,
        bold=True,
        fill=(22, 38, 60),
    )

    cols = [
        (80, 150, 520, 820, "Logical qubits", "Hardware-efficient encoding\nError-protected bosonic code space\nBetter fault-tolerant building blocks"),
        (580, 150, 1020, 820, "Quantum sensing", "Non-classical states improve sensitivity\nPhase-space structure can amplify signal\nUseful beyond digital computing"),
        (1080, 150, 1520, 820, "Fundamental physics", "Control of non-classical states\nBenchmark for quantum dynamics\nPlatform to test control principles"),
    ]
    fills = [(232, 243, 255), (236, 255, 245), (255, 247, 236)]
    outlines = [(78, 126, 211), (56, 160, 108), (206, 136, 46)]

    for (x1, y1, x2, y2, title, body), fill, outline in zip(cols, fills, outlines):
        _rr(d, (x1, y1, x2, y2), 24, fill=fill, outline=outline, width=4)
        _draw_text_in_box(
            d,
            (x1 + 24, y1 + 22, x2 - 24, y1 + 82),
            title,
            start_size=32,
            min_size=24,
            bold=True,
            fill=(29, 46, 72),
        )
        _draw_text_in_box(
            d,
            (x1 + 24, y1 + 95, x2 - 24, y2 - 330),
            body,
            start_size=24,
            min_size=16,
            fill=(35, 55, 84),
            spacing=10,
        )

        # simple icon motifs
        cx = (x1 + x2) // 2
        if "qubits" in title.lower():
            for k in range(3):
                r = 48 + 18 * k
                d.ellipse((cx - r, y2 - 210 - r, cx + r, y2 - 210 + r), outline=(75, 126, 210), width=4)
        elif "sensing" in title.lower():
            d.polygon([(cx - 90, y2 - 180), (cx, y2 - 300), (cx + 90, y2 - 180)], outline=(58, 156, 107), fill=(208, 241, 223))
            d.line((cx, y2 - 300, cx, y2 - 120), fill=(58, 156, 107), width=5)
        else:
            d.ellipse((cx - 115, y2 - 300, cx + 115, y2 - 70), outline=(202, 128, 42), width=5)
            d.line((cx - 115, y2 - 185, cx + 115, y2 - 185), fill=(202, 128, 42), width=4)
            d.line((cx, y2 - 300, cx, y2 - 70), fill=(202, 128, 42), width=4)

    img.save(path)


def build_motivation_image(path: Path) -> None:
    w, h = 1600, 900
    img = Image.new("RGB", (w, h), (247, 250, 255))
    d = ImageDraw.Draw(img)
    f_title = _font(50, bold=True)
    f_sub = _font(30, bold=True)
    f_txt = _font(26)

    _draw_text_in_box(
        d,
        (60, 28, 1540, 110),
        "Encoding Intuition: Multi-Qubit vs Bosonic Mode",
        start_size=50,
        min_size=36,
        bold=True,
        fill=(24, 35, 52),
    )

    left = (50, 120, 760, 840)
    right = (840, 120, 1550, 840)
    _rr(d, left, 30, fill=(236, 242, 255), outline=(86, 126, 214), width=4)
    _rr(d, right, 30, fill=(236, 255, 243), outline=(44, 164, 109), width=4)

    _draw_text_in_box(
        d,
        (90, 150, 700, 215),
        "Many physical qubits",
        start_size=30,
        min_size=22,
        bold=True,
        fill=(32, 66, 140),
    )
    _draw_text_in_box(
        d,
        (900, 150, 1500, 215),
        "One bosonic mode",
        start_size=30,
        min_size=22,
        bold=True,
        fill=(19, 112, 74),
    )

    gx0, gy0 = 110, 240
    step = 72
    for r in range(7):
        for c in range(7):
            x = gx0 + c * step
            y = gy0 + r * step
            fill = (31, 41, 55)
            if (r, c) in {(1, 1), (2, 4), (5, 2), (4, 5), (3, 3)}:
                fill = (91, 132, 255)
            d.ellipse((x - 18, y - 18, x + 18, y + 18), fill=fill)

    _draw_text_in_box(
        d,
        (95, 740, 720, 810),
        "Protection often needs many coupled qubits",
        start_size=26,
        min_size=18,
        fill=(20, 35, 58),
    )

    ox0, oy0 = 930, 255
    ow, oh = 520, 430
    _rr(d, (ox0, oy0, ox0 + ow, oy0 + oh), 24, fill=(248, 255, 251), outline=(65, 157, 107), width=3)
    for k in range(8):
        y = oy0 + 40 + k * 46
        d.line((ox0 + 45, y, ox0 + ow - 45, y), fill=(142, 196, 164), width=2)
    d.ellipse((ox0 + 210, oy0 + 120, ox0 + 310, oy0 + 220), outline=(26, 130, 89), width=5)
    d.ellipse((ox0 + 205, oy0 + 265, ox0 + 315, oy0 + 375), outline=(26, 130, 89), width=5)
    d.text((ox0 + 345, oy0 + 150), "|0_L>", font=f_sub, fill=(22, 100, 70))
    d.text((ox0 + 345, oy0 + 295), "|1_L>", font=f_sub, fill=(22, 100, 70))

    _draw_text_in_box(
        d,
        (895, 740, 1520, 810),
        "Richer controllable state space in one oscillator",
        start_size=26,
        min_size=18,
        fill=(14, 84, 56),
    )

    _arrow(d, (770, 450), (830, 450), color=(72, 98, 175), width=8, head=24)
    _draw_text_in_box(
        d,
        (560, 355, 1040, 440),
        "Hardware-efficient bosonic encoding",
        start_size=27,
        min_size=18,
        bold=True,
        fill=(53, 74, 128),
        align="center",
        valign="middle",
    )

    img.save(path)


def build_timeline_image(path: Path) -> None:
    w, h = 1600, 900
    img = Image.new("RGB", (w, h), (248, 251, 255))
    d = ImageDraw.Draw(img)
    f_title = _font(44, bold=True)
    f_year = _font(28, bold=True)
    f_txt = _font(23)

    d.text((70, 30), "Positioning This Work in Literature", font=f_title, fill=(24, 39, 60))

    y = 460
    d.line((140, y, 1460, y), fill=(90, 112, 145), width=6)
    points = [
        (220, "2001", "GKP code\nproposal\n(Gottesman-Kitaev-Preskill)"),
        (620, "2022", "Model-free RL\nquantum control\n(Sivak et al., PRX)"),
        (990, "2024", "Robust trapped-ion\nbosonic state prep\n(Matsos et al., PRL)"),
        (1360, "2026", "This project:\nRL workflow for\nCat/GKP/Binomial"),
    ]
    cols = [(79, 127, 210), (124, 92, 182), (53, 162, 107), (209, 133, 43)]
    for (x, year, txt), c in zip(points, cols):
        d.ellipse((x - 15, y - 15, x + 15, y + 15), fill=c)
        d.text((x - 35, y - 70), year, font=f_year, fill=(31, 47, 72))
        _rr(d, (x - 150, y + 40, x + 150, y + 225), 18, fill=(255, 255, 255), outline=(185, 199, 220), width=2)
        d.multiline_text((x - 132, y + 70), txt, font=f_txt, fill=(33, 50, 78), spacing=7)

    d.text((100, 120), "Use references you actually discuss in the talk; avoid citation overload.", font=f_txt, fill=(33, 53, 82))
    d.text((100, 160), "Story line: prior ideas -> our adaptation -> current robust-control results.", font=f_txt, fill=(33, 53, 82))

    img.save(path)


def build_literature_comparison_image(path: Path) -> None:
    w, h = 1600, 900
    img = Image.new("RGB", (w, h), (248, 251, 255))
    d = ImageDraw.Draw(img)
    f_title = _font(42, bold=True)
    f_sub = _font(27, bold=True)
    f_txt = _font(22)

    _draw_text_in_box(
        d,
        (70, 30, 1530, 100),
        "Key References and This Project",
        start_size=42,
        min_size=32,
        bold=True,
        fill=(24, 39, 60),
    )

    cards = [
        (
            80,
            150,
            510,
            790,
            "Sivak et al. (PRX 2022)",
            "Model-free RL for quantum control.\n"
            "Main message: feedback-driven policy\n"
            "optimization can work without analytic gradients.",
        ),
        (
            585,
            150,
            1015,
            790,
            "Matsos et al. (PRL 2024)",
            "Robust trapped-ion bosonic-state preparation.\n"
            "Main message: deterministic and noise-robust\n"
            "bosonic state preparation is experimentally feasible.",
        ),
        (
            1090,
            150,
            1520,
            790,
            "This Project",
            "Applies and explains an RL workflow for\n"
            "Cat/GKP/Binomial in trapped ions.\n"
            "Includes dephasing-robust static and\n"
            "stochastic training comparisons.",
        ),
    ]
    fills = [(236, 244, 255), (236, 255, 244), (255, 247, 236)]
    outlines = [(79, 128, 214), (57, 160, 108), (206, 136, 44)]

    for (x1, y1, x2, y2, title, body), fill, outline in zip(cards, fills, outlines):
        _rr(d, (x1, y1, x2, y2), 24, fill=fill, outline=outline, width=4)
        _draw_text_in_box(
            d,
            (x1 + 22, y1 + 22, x2 - 22, y1 + 86),
            title,
            start_size=27,
            min_size=20,
            bold=True,
            fill=(30, 47, 73),
        )
        _draw_text_in_box(
            d,
            (x1 + 22, y1 + 92, x2 - 22, y2 - 22),
            body,
            start_size=22,
            min_size=15,
            fill=(35, 55, 84),
            spacing=9,
        )

    _arrow(d, (510, 470), (585, 470), color=(96, 118, 149), width=7)
    _arrow(d, (1015, 470), (1090, 470), color=(96, 118, 149), width=7)

    img.save(path)


def build_char_reward_mechanism_image(path: Path) -> None:
    w, h = 1600, 900
    img = Image.new("RGB", (w, h), (248, 252, 255))
    d = ImageDraw.Draw(img)
    f_title = _font(44, bold=True)
    f_sub = _font(27, bold=True)
    f_txt = _font(22)

    _draw_text_in_box(
        d,
        (70, 30, 1530, 100),
        "Characteristic-Function Reward Workflow",
        start_size=44,
        min_size=32,
        bold=True,
        fill=(24, 39, 60),
    )

    boxes = [
        (90, 200, 480, 720, "1) Choose sample points", "Select points in phase space with staged sampling."),
        (560, 200, 950, 720, "2) Measure model response", "Evaluate characteristic values at sampled points."),
        (1030, 200, 1420, 720, "3) Compute reward", "Compare with target values and aggregate weighted mismatch."),
    ]
    fills = [(232, 243, 255), (236, 255, 244), (255, 247, 236)]
    outlines = [(79, 128, 214), (57, 160, 108), (206, 136, 44)]
    for (x1, y1, x2, y2, title, body), fill, outline in zip(boxes, fills, outlines):
        _rr(d, (x1, y1, x2, y2), 24, fill=fill, outline=outline, width=4)
        _draw_text_in_box(
            d,
            (x1 + 20, y1 + 20, x2 - 20, y1 + 84),
            title,
            start_size=27,
            min_size=20,
            bold=True,
            fill=(29, 46, 72),
        )
        _draw_text_in_box(
            d,
            (x1 + 20, y1 + 90, x2 - 20, y2 - 20),
            body,
            start_size=22,
            min_size=15,
            fill=(35, 55, 84),
            spacing=8,
        )

    _arrow(d, (480, 455), (560, 455), color=(94, 117, 148), width=7)
    _arrow(d, (950, 455), (1030, 455), color=(94, 117, 148), width=7)

    d.text((95, 770), "Reward improves when sampled characteristic values align with target structure.", font=f_txt, fill=(42, 62, 93))
    d.text((95, 805), "This gives a measurement-style objective compatible with model-free RL updates.", font=f_txt, fill=(42, 62, 93))

    img.save(path)


def build_sampling_strategy_image(path: Path) -> None:
    w, h = 1600, 900
    img = Image.new("RGB", (w, h), (249, 252, 255))
    d = ImageDraw.Draw(img)
    f_title = _font(42, bold=True)
    f_sub = _font(26, bold=True)
    f_txt = _font(21)

    _draw_text_in_box(
        d,
        (70, 30, 1530, 100),
        "Characteristic Point Selection Across Training",
        start_size=42,
        min_size=31,
        bold=True,
        fill=(24, 39, 60),
    )

    stages = [
        (100, 190, 480, 760, "Stage 1", "Top-k important points\nfor strong early signal"),
        (560, 190, 940, 760, "Stage 2", "Broader weighted sampling\nfor exploration + stability"),
        (1020, 190, 1400, 760, "Stage 3", "Dense sampling for\nhigh-fidelity fine matching"),
    ]
    fills = [(232, 243, 255), (236, 255, 244), (255, 247, 236)]
    outlines = [(79, 128, 214), (57, 160, 108), (206, 136, 44)]
    for (x1, y1, x2, y2, title, body), fill, outline in zip(stages, fills, outlines):
        _rr(d, (x1, y1, x2, y2), 24, fill=fill, outline=outline, width=4)
        _draw_text_in_box(
            d,
            (x1 + 22, y1 + 22, x2 - 22, y1 + 78),
            title,
            start_size=26,
            min_size=20,
            bold=True,
            fill=(29, 46, 72),
        )
        _draw_text_in_box(
            d,
            (x1 + 22, y1 + 84, x2 - 22, y1 + 170),
            body,
            start_size=21,
            min_size=15,
            fill=(35, 55, 84),
            spacing=8,
        )
        # visual density dots
        rng = np.random.default_rng(abs(hash(title)) % (2**32))
        n = {"Stage 1": 40, "Stage 2": 90, "Stage 3": 180}[title]
        for _ in range(n):
            px = int(rng.uniform(x1 + 35, x2 - 35))
            py = int(rng.uniform(y1 + 180, y2 - 35))
            d.ellipse((px - 2, py - 2, px + 2, py + 2), fill=(70, 105, 165))

    _arrow(d, (480, 470), (560, 470), color=(96, 118, 149), width=7)
    _arrow(d, (940, 470), (1020, 470), color=(96, 118, 149), width=7)

    d.text((100, 805), "The sampling policy becomes denser and more target-focused as training progresses.", font=f_txt, fill=(40, 60, 90))
    img.save(path)


def build_rl_loop_image(path: Path) -> None:
    w, h = 1600, 900
    img = Image.new("RGB", (w, h), (250, 251, 255))
    d = ImageDraw.Draw(img)
    f_title = _font(46, bold=True)
    f_box = _font(25, bold=True)
    f_txt = _font(23)

    d.text((70, 28), "Closed-Loop RL Optimization", font=f_title, fill=(25, 37, 57))

    boxes = {
        "policy": (95, 220, 500, 380),
        "pulse": (560, 120, 1040, 280),
        "measure": (1100, 220, 1510, 380),
        "reward": (930, 505, 1420, 675),
        "update": (220, 515, 760, 705),
    }

    colors = {
        "policy": ((232, 243, 255), (79, 130, 216)),
        "pulse": ((238, 253, 245), (54, 167, 110)),
        "measure": ((255, 245, 231), (214, 137, 45)),
        "reward": ((250, 240, 255), (160, 86, 198)),
        "update": ((236, 246, 251), (65, 151, 189)),
    }

    labels = {
        "policy": "Policy outputs\ncontrol parameters",
        "pulse": "Apply pulse sequence\nto quantum system",
        "measure": "Sample characteristic\nfunction points",
        "reward": "Compute scalar reward\n(target consistency)",
        "update": "PPO updates policy\nfor next epoch",
    }

    for key, rect in boxes.items():
        fill, outline = colors[key]
        _rr(d, rect, 26, fill=fill, outline=outline, width=4)
        x1, y1, x2, y2 = rect
        bb = d.multiline_textbbox((0, 0), labels[key], font=f_box, align="center", spacing=7)
        tw = bb[2] - bb[0]
        th = bb[3] - bb[1]
        d.multiline_text(
            (x1 + (x2 - x1 - tw) // 2, y1 + (y2 - y1 - th) // 2),
            labels[key],
            font=f_box,
            fill=(25, 35, 50),
            align="center",
            spacing=7,
        )

    _arrow(d, (500, 300), (560, 205), color=(67, 120, 201), width=7)
    _arrow(d, (1040, 205), (1100, 300), color=(66, 155, 104), width=7)
    _arrow(d, (1285, 380), (1180, 505), color=(195, 122, 39), width=7)
    _arrow(d, (930, 590), (760, 620), color=(137, 83, 178), width=7)
    _arrow(d, (220, 600), (95, 340), color=(58, 141, 177), width=7)

    d.text((120, 770), "One epoch = rollout episodes -> compute reward -> multiple policy updates -> evaluate", font=f_txt, fill=(34, 56, 94))
    d.text((120, 807), "The cycle repeats until the policy reaches the target fidelity and robustness regime.", font=f_txt, fill=(34, 56, 94))

    img.save(path)


def build_epoch_timeline_image(path: Path) -> None:
    w, h = 1600, 900
    img = Image.new("RGB", (w, h), (250, 252, 255))
    d = ImageDraw.Draw(img)
    f_title = _font(46, bold=True)
    f_box = _font(24, bold=True)
    f_txt = _font(22)

    d.text((70, 30), "Episode, Epoch, and Policy Update", font=f_title, fill=(24, 38, 58))

    y = 300
    x_starts = [80, 380, 690, 980, 1270]
    labels = [
        "1) Rollout\ntrajectories",
        "2) Collect\nmeasurement rewards",
        "3) Estimate\nadvantages",
        "4) K policy\nupdates (PPO)",
        "5) Evaluate\n& log",
    ]
    fills = [(232, 243, 255), (238, 253, 245), (255, 247, 236), (245, 239, 255), (236, 248, 250)]
    outlines = [(78, 125, 211), (53, 160, 106), (202, 137, 45), (152, 92, 186), (66, 148, 176)]

    for i, x in enumerate(x_starts):
        rect = (x, y, x + 240, y + 190)
        _rr(d, rect, 24, fill=fills[i], outline=outlines[i], width=4)
        bb = d.multiline_textbbox((0, 0), labels[i], font=f_box, align="center", spacing=8)
        tw = bb[2] - bb[0]
        th = bb[3] - bb[1]
        d.multiline_text((x + (240 - tw) // 2, y + (190 - th) // 2), labels[i], font=f_box, fill=(25, 35, 50), align="center", spacing=8)
        if i < len(x_starts) - 1:
            _arrow(d, (x + 240, y + 95), (x_starts[i + 1], y + 95), color=(95, 118, 155), width=6)

    d.text((110, 555), "Episode: one pulse sequence trial", font=f_txt, fill=(32, 52, 81))
    d.text((110, 592), "Epoch: many episodes + one optimization cycle", font=f_txt, fill=(32, 52, 81))
    d.text((110, 629), "Policy updates: repeated gradient steps inside one epoch", font=f_txt, fill=(32, 52, 81))
    d.text((110, 700), "Key hyperparameter: number of policy updates per epoch", font=_font(24, bold=True), fill=(52, 74, 115))

    img.save(path)


def build_ppo_image(path: Path) -> None:
    w, h = 1600, 900
    img = Image.new("RGB", (w, h), (248, 252, 255))
    d = ImageDraw.Draw(img)
    f_title = _font(46, bold=True)
    f_sub = _font(28, bold=True)
    f_txt = _font(23)

    _draw_text_in_box(
        d,
        (70, 28, 1530, 100),
        "PPO Optimization Cycle in This Project",
        start_size=46,
        min_size=32,
        bold=True,
        fill=(26, 40, 62),
    )

    steps = [
        (90, 200, 500, 760, "Collect", "Roll out current policy\nunder sampled noise\nand store trajectories"),
        (595, 200, 1005, 760, "Estimate", "Compute returns/advantages\nfrom reward signals\n(measurement-based)"),
        (1100, 200, 1510, 760, "Update", "Run K clipped policy updates\nfor stability, then\nstart next epoch"),
    ]
    fills = [(232, 243, 255), (238, 255, 246), (255, 247, 236)]
    outlines = [(75, 126, 210), (54, 160, 108), (206, 136, 44)]
    for (x1, y1, x2, y2, title, body), fill, outline in zip(steps, fills, outlines):
        _rr(d, (x1, y1, x2, y2), 26, fill=fill, outline=outline, width=4)
        _draw_text_in_box(
            d,
            (x1 + 26, y1 + 22, x2 - 26, y1 + 80),
            title,
            start_size=28,
            min_size=21,
            bold=True,
            fill=(31, 48, 74),
        )
        _draw_text_in_box(
            d,
            (x1 + 26, y1 + 95, x2 - 26, y2 - 20),
            body,
            start_size=23,
            min_size=16,
            fill=(36, 56, 86),
            spacing=10,
        )

    _arrow(d, (500, 470), (595, 470), color=(92, 116, 150), width=7)
    _arrow(d, (1005, 470), (1100, 470), color=(92, 116, 150), width=7)

    d.text((90, 130), "PPO performs stable iterative policy improvement using clipped objective updates.", font=f_txt, fill=(40, 60, 90))
    d.text((90, 815), "In this workflow, multiple policy updates are applied each epoch to convert rollout data into better controls.", font=f_txt, fill=(40, 60, 90))

    img.save(path)


def build_dynamics_vs_qutip_image(path: Path) -> None:
    w, h = 1600, 900
    img = Image.new("RGB", (w, h), (248, 251, 255))
    d = ImageDraw.Draw(img)
    f_title = _font(44, bold=True)
    f_sub = _font(28, bold=True)
    f_txt = _font(23)

    d.text((70, 30), "Why Use dynamiqs + GPU (instead of QuTiP here)", font=f_title, fill=(24, 39, 60))

    _rr(d, (110, 160, 760, 760), 26, fill=(236, 244, 255), outline=(79, 128, 214), width=4)
    _rr(d, (840, 160, 1490, 760), 26, fill=(236, 255, 244), outline=(57, 160, 108), width=4)

    d.text((140, 195), "QuTiP baseline", font=f_sub, fill=(38, 61, 98))
    d.multiline_text(
        (140, 260),
        "Great for prototyping\nand small-scale simulation,\nbut throughput can be limiting\nfor many RL rollouts.",
        font=f_txt,
        fill=(38, 61, 98),
        spacing=10,
    )

    d.text((870, 195), "dynamiqs + JAX path", font=f_sub, fill=(24, 108, 74))
    d.multiline_text(
        (870, 260),
        "Vectorized simulation\nand parallel rollout execution\nfit RL training loops better.\nGPU acceleration shortens\niteration time.",
        font=f_txt,
        fill=(24, 108, 74),
        spacing=10,
    )

    _arrow(d, (760, 460), (840, 460), color=(96, 118, 149), width=8)
    d.text((700, 410), "for large rollout batches", font=_font(22, bold=True), fill=(64, 82, 112))

    d.text((120, 810), "Practical implication: this toolchain improves parallel rollout throughput for RL training.", font=f_txt, fill=(40, 60, 90))

    img.save(path)


def build_hyperparam_image(metrics: Metrics, path: Path) -> None:
    h = metrics.hyper
    w, hh = 1600, 900
    img = Image.new("RGB", (w, hh), (249, 252, 255))
    d = ImageDraw.Draw(img)
    f_title = _font(44, bold=True)
    f_head = _font(24, bold=True)
    f_txt = _font(21)

    d.text((70, 30), "Training Parameters by Target State", font=f_title, fill=(24, 39, 60))

    _rr(d, (80, 120, 1520, 815), 24, fill=(255, 255, 255), outline=(120, 140, 172), width=3)

    # table grid
    x_cols = [110, 420, 780, 1140, 1490]
    y_rows = [170, 245, 320, 395, 470, 555, 640, 725]
    for x in x_cols:
        d.line((x, y_rows[0], x, y_rows[-1]), fill=(205, 216, 233), width=2)
    for y in y_rows:
        d.line((x_cols[0], y, x_cols[-1], y), fill=(205, 216, 233), width=2)

    d.text((128, 190), "Item", font=f_head, fill=(36, 55, 85))
    d.text((460, 190), "Cat", font=f_head, fill=(36, 55, 85))
    d.text((835, 190), "GKP", font=f_head, fill=(36, 55, 85))
    d.text((1170, 190), "Binomial", font=f_head, fill=(36, 55, 85))

    rows = [
        ("NUM_EPOCHS", h.cat_num_epochs, h.gkp_num_epochs, h.bin_num_epochs),
        ("Policy updates/epoch", h.cat_policy_updates, h.gkp_policy_updates, h.bin_policy_updates),
        ("N_STEPS", h.cat_n_steps, h.gkp_n_steps, h.bin_n_steps),
        ("N_SEGMENTS", h.cat_n_segments, h.gkp_n_segments, h.bin_n_segments),
        ("Characteristic points", h.cat_stage_points, h.gkp_stage_points, h.bin_stage_points),
        ("Stage boundaries", h.cat_stage_epochs, h.gkp_stage_epochs, h.bin_stage_epochs),
    ]

    y = 265
    for label, c1, c2, c3 in rows:
        d.text((128, y), label, font=f_txt, fill=(36, 55, 85))
        d.text((460, y), str(c1), font=f_txt, fill=(36, 55, 85))
        d.text((835, y), str(c2), font=f_txt, fill=(36, 55, 85))
        d.text((1170, y), str(c3), font=f_txt, fill=(36, 55, 85))
        y += 75

    d.text(
        (110, 760),
        "All values come from the current experiment configurations used in this report.",
        font=f_txt,
        fill=(40, 60, 92),
    )

    img.save(path)


def build_robust_modes_image(path: Path) -> None:
    w, h = 1600, 900
    img = Image.new("RGB", (w, h), (248, 252, 255))
    d = ImageDraw.Draw(img)
    f_title = _font(46, bold=True)
    f_sub = _font(27, bold=True)
    f_txt = _font(22)

    _draw_text_in_box(
        d,
        (70, 30, 1530, 100),
        "Dephasing-Robust Training Modes",
        start_size=46,
        min_size=33,
        bold=True,
        fill=(24, 39, 60),
    )

    # quasi-static box
    _rr(d, (100, 160, 760, 780), 26, fill=(236, 244, 255), outline=(79, 128, 214), width=4)
    _draw_text_in_box(
        d,
        (130, 195, 730, 248),
        "Quasi-static branch",
        start_size=27,
        min_size=21,
        bold=True,
        fill=(35, 61, 98),
    )
    _draw_text_in_box(
        d,
        (130, 255, 730, 700),
        "One detuning sample per trajectory (or a detuning grid across samples).\n"
        "Configuration used here:\n"
        "- grid-based detuning samples\n"
        "- Gaussian weighting for objective aggregation\n"
        "- nominal sample included in the objective",
        start_size=22,
        min_size=15,
        fill=(35, 61, 98),
        spacing=9,
    )

    # stochastic box
    _rr(d, (840, 160, 1500, 780), 26, fill=(236, 255, 244), outline=(57, 160, 108), width=4)
    _draw_text_in_box(
        d,
        (870, 195, 1470, 248),
        "Stochastic branch",
        start_size=27,
        min_size=21,
        bold=True,
        fill=(24, 108, 74),
    )
    _draw_text_in_box(
        d,
        (870, 255, 1470, 700),
        "Detuning is resampled across time segments.\n"
        "Configuration used here:\n"
        "- gamma_dt standard-deviation mode\n"
        "- Omega and t_step kept on a consistent physical scale\n"
        "- shared trajectories across each training batch",
        start_size=22,
        min_size=15,
        fill=(24, 108, 74),
        spacing=9,
    )

    _arrow(d, (760, 470), (840, 470), color=(95, 117, 148), width=8)
    d.text((770, 420), "same RL loop", font=_font(21, bold=True), fill=(62, 80, 109))

    d.text((110, 810), "Both modes optimize controls under noise; they differ in noise sampling model and objective aggregation.", font=f_txt, fill=(40, 60, 90))

    img.save(path)


def build_agent_io_image(path: Path) -> None:
    w, h = 1600, 900
    img = Image.new("RGB", (w, h), (249, 252, 255))
    d = ImageDraw.Draw(img)

    _draw_text_in_box(
        d,
        (70, 30, 1530, 100),
        "What the Agent Uses and What It Updates",
        start_size=44,
        min_size=32,
        bold=True,
        fill=(24, 39, 60),
    )

    cards = [
        (
            (90, 170, 730, 360),
            (232, 243, 255),
            (79, 128, 214),
            "Policy input (observation)",
            "Clock one-hot and constant context are fed to the policy network each step.",
        ),
        (
            (860, 170, 1510, 360),
            (236, 255, 244),
            (57, 160, 108),
            "Policy output (actions)",
            "Action dimensions include phi_r/phi_b and optional amp_r/amp_b plus duration_scale.",
        ),
        (
            (90, 430, 730, 750),
            (255, 247, 236),
            (206, 136, 44),
            "Reward signal from client side",
            "After a full trajectory, characteristic-function mismatch is converted to scalar reward.",
        ),
        (
            (860, 430, 1510, 750),
            (245, 239, 255),
            (153, 92, 186),
            "Update style",
            "PPO performs policy iteration (actor + value update from advantages), not value iteration on a discrete state table.",
        ),
    ]

    for rect, fill, outline, title, body in cards:
        x1, y1, x2, y2 = rect
        _rr(d, rect, 26, fill=fill, outline=outline, width=4)
        _draw_text_in_box(
            d,
            (x1 + 22, y1 + 20, x2 - 22, y1 + 76),
            title,
            start_size=27,
            min_size=20,
            bold=True,
            fill=(29, 46, 72),
        )
        _draw_text_in_box(
            d,
            (x1 + 22, y1 + 88, x2 - 22, y2 - 20),
            body,
            start_size=22,
            min_size=15,
            fill=(35, 55, 84),
            spacing=8,
        )

    _arrow(d, (730, 265), (860, 265), color=(95, 117, 148), width=7)
    _arrow(d, (1185, 360), (1185, 430), color=(95, 117, 148), width=7)
    _arrow(d, (860, 590), (730, 590), color=(95, 117, 148), width=7)
    _arrow(d, (410, 430), (410, 360), color=(95, 117, 148), width=7)

    _draw_text_in_box(
        d,
        (90, 780, 1510, 855),
        "Interpretation: controls are updated through repeated policy-improvement cycles driven by measurement-derived rewards.",
        start_size=22,
        min_size=15,
        fill=(40, 60, 90),
    )

    img.save(path)


def build_output_reading_guide_image(path: Path) -> None:
    w, h = 1600, 900
    img = Image.new("RGB", (w, h), (248, 252, 255))
    d = ImageDraw.Draw(img)

    _draw_text_in_box(
        d,
        (70, 30, 1530, 100),
        "How to Read the Result Figures",
        start_size=44,
        min_size=32,
        bold=True,
        fill=(24, 39, 60),
    )

    panels = [
        (
            (90, 180, 510, 780),
            (232, 243, 255),
            (79, 128, 214),
            "1) Fidelity vs Epoch",
            "Checks whether policy updates consistently improve control quality over training.",
        ),
        (
            (590, 180, 1010, 780),
            (236, 255, 244),
            (57, 160, 108),
            "2) Characteristic Map",
            "Compares target and final-state structure in phase space.",
        ),
        (
            (1090, 180, 1510, 780),
            (255, 247, 236),
            (206, 136, 44),
            "3) Pulse Sequence",
            "Shows the learned control profile that generates the final state.",
        ),
    ]

    for rect, fill, outline, title, body in panels:
        x1, y1, x2, y2 = rect
        _rr(d, rect, 24, fill=fill, outline=outline, width=4)
        _draw_text_in_box(
            d,
            (x1 + 18, y1 + 20, x2 - 18, y1 + 92),
            title,
            start_size=27,
            min_size=20,
            bold=True,
            fill=(30, 47, 73),
        )
        _draw_text_in_box(
            d,
            (x1 + 18, y1 + 105, x2 - 18, y1 + 210),
            body,
            start_size=21,
            min_size=15,
            fill=(35, 55, 84),
            spacing=8,
        )

        # simple symbolic visuals
        if "Fidelity" in title:
            d.line((x1 + 40, y2 - 120, x2 - 40, y2 - 120), fill=(87, 114, 149), width=3)
            d.line((x1 + 40, y2 - 220, x1 + 40, y2 - 80), fill=(87, 114, 149), width=3)
            pts = [
                (x1 + 55, y2 - 95),
                (x1 + 130, y2 - 130),
                (x1 + 210, y2 - 165),
                (x1 + 300, y2 - 185),
                (x1 + 370, y2 - 205),
            ]
            d.line(pts, fill=(60, 118, 204), width=6)
            for px, py in pts:
                d.ellipse((px - 5, py - 5, px + 5, py + 5), fill=(60, 118, 204))
        elif "Characteristic" in title:
            cx = (x1 + x2) // 2
            cy = y2 - 150
            for r in [30, 60, 90]:
                d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(55, 145, 100), width=4)
            d.line((cx - 110, cy, cx + 110, cy), fill=(55, 145, 100), width=3)
            d.line((cx, cy - 110, cx, cy + 110), fill=(55, 145, 100), width=3)
        else:
            x_start = x1 + 50
            y_mid = y2 - 165
            for i in range(14):
                x = x_start + i * 23
                up = int(35 * np.sin(i * 0.65))
                d.line((x, y_mid, x, y_mid - up), fill=(204, 132, 42), width=5)
            d.line((x1 + 40, y2 - 120, x2 - 40, y2 - 120), fill=(120, 132, 150), width=2)

    _arrow(d, (510, 480), (590, 480), color=(96, 118, 149), width=7)
    _arrow(d, (1010, 480), (1090, 480), color=(96, 118, 149), width=7)

    img.save(path)


def build_next_steps_image(path: Path) -> None:
    w, h = 1600, 900
    img = Image.new("RGB", (w, h), (249, 252, 255))
    d = ImageDraw.Draw(img)

    _draw_text_in_box(
        d,
        (70, 30, 1530, 100),
        "Execution Plan Before the Talk",
        start_size=42,
        min_size=31,
        bold=True,
        fill=(24, 39, 60),
    )

    stages = [
        ((120, 190, 500, 760), "1) Stability pass", "Finalize slide wording, figure consistency, and narrative transitions."),
        ((610, 190, 990, 760), "2) Dry run", "Practice timing and verify that RL workflow explanation is clear to physics audience."),
        ((1100, 190, 1480, 760), "3) Final polish", "Incorporate advisor feedback and lock the presentation for Tuesday report."),
    ]
    fills = [(232, 243, 255), (236, 255, 244), (255, 247, 236)]
    outlines = [(79, 128, 214), (57, 160, 108), (206, 136, 44)]
    for (rect, title, body), fill, outline in zip(stages, fills, outlines):
        x1, y1, x2, y2 = rect
        _rr(d, rect, 26, fill=fill, outline=outline, width=4)
        _draw_text_in_box(
            d,
            (x1 + 20, y1 + 20, x2 - 20, y1 + 92),
            title,
            start_size=27,
            min_size=20,
            bold=True,
            fill=(29, 46, 72),
        )
        _draw_text_in_box(
            d,
            (x1 + 20, y1 + 105, x2 - 20, y2 - 20),
            body,
            start_size=22,
            min_size=15,
            fill=(35, 55, 84),
            spacing=8,
        )

    _arrow(d, (500, 475), (610, 475), color=(96, 118, 149), width=7)
    _arrow(d, (990, 475), (1100, 475), color=(96, 118, 149), width=7)
    img.save(path)


def build_conclusion_image(metrics: Metrics, path: Path) -> None:
    w, h = 1600, 900
    img = Image.new("RGB", (w, h), (248, 252, 255))
    d = ImageDraw.Draw(img)
    _draw_text_in_box(
        d,
        (70, 30, 1530, 100),
        "Project Snapshot",
        start_size=44,
        min_size=32,
        bold=True,
        fill=(24, 39, 60),
    )

    cards = [
        (
            (100, 180, 730, 420),
            "Cross-target nominal performance",
            "Cat %.4f | GKP %.4f | Binomial baseline %.4f"
            % (metrics.cat_fid, metrics.gkp_fid, metrics.bin_pre_fid),
        ),
        (
            (860, 180, 1490, 420),
            "Best robust penalties",
            "Quasi-static p=%.2f | Stochastic p=%.2f"
            % (metrics.best_static.penalty, metrics.best_stoch.penalty),
        ),
        (
            (100, 500, 730, 760),
            "Robust quality now",
            "Static score %.4f (f_nom %.4f, f_rob %.4f)"
            % (metrics.best_static.score, metrics.best_static.f_nom, metrics.best_static.f_rob),
        ),
        (
            (860, 500, 1490, 760),
            "Main open direction",
            "Raise stochastic branch robustness while preserving near-zero-detuning fidelity.",
        ),
    ]
    fills = [(232, 243, 255), (236, 255, 244), (255, 247, 236), (245, 239, 255)]
    outlines = [(79, 128, 214), (57, 160, 108), (206, 136, 44), (153, 92, 186)]
    for (rect, title, body), fill, outline in zip(cards, fills, outlines):
        x1, y1, x2, y2 = rect
        _rr(d, rect, 24, fill=fill, outline=outline, width=4)
        _draw_text_in_box(
            d,
            (x1 + 20, y1 + 18, x2 - 20, y1 + 82),
            title,
            start_size=26,
            min_size=19,
            bold=True,
            fill=(29, 46, 72),
        )
        _draw_text_in_box(
            d,
            (x1 + 20, y1 + 90, x2 - 20, y2 - 20),
            body,
            start_size=23,
            min_size=15,
            fill=(35, 55, 84),
            spacing=8,
        )
    img.save(path)


def build_reference_scope_image(path: Path) -> None:
    w, h = 1600, 900
    img = Image.new("RGB", (w, h), (249, 252, 255))
    d = ImageDraw.Draw(img)
    _draw_text_in_box(
        d,
        (70, 30, 1530, 100),
        "Reference Scope Used in This Talk",
        start_size=42,
        min_size=31,
        bold=True,
        fill=(24, 39, 60),
    )

    rows = [
        ("Model-free RL context", "Sivak et al., PRX 2022"),
        ("Bosonic trapped-ion benchmark", "Matsos et al., PRL 2024"),
        ("Bosonic code motivation", "Gottesman-Kitaev-Preskill, PRA 2001"),
        ("Project-specific results", "Current Cat/GKP/Binomial outputs and robust sweeps"),
    ]
    y = 180
    for i, (topic, cite) in enumerate(rows):
        fill = (235, 244, 255) if i % 2 == 0 else (241, 248, 255)
        _rr(d, (120, y, 1480, y + 130), 18, fill=fill, outline=(164, 184, 214), width=2)
        _draw_text_in_box(
            d,
            (150, y + 18, 620, y + 112),
            topic,
            start_size=27,
            min_size=19,
            bold=True,
            fill=(32, 49, 75),
        )
        _draw_text_in_box(
            d,
            (640, y + 18, 1450, y + 112),
            cite,
            start_size=24,
            min_size=16,
            fill=(38, 58, 89),
        )
        y += 155
    img.save(path)


def build_agenda_visual_image(path: Path) -> None:
    w, h = 1600, 900
    img = Image.new("RGB", (w, h), (248, 252, 255))
    d = ImageDraw.Draw(img)
    _draw_text_in_box(
        d,
        (70, 30, 1530, 100),
        "Report Flow Overview",
        start_size=44,
        min_size=32,
        bold=True,
        fill=(24, 39, 60),
    )
    blocks = [
        ((120, 170, 720, 360), "Motivation", "Why bosonic state preparation and why RL now."),
        ((880, 170, 1480, 360), "Method", "Closed-loop optimization, PPO updates, and sampling strategy."),
        ((120, 430, 720, 720), "Results", "Cat/GKP/Binomial outcomes with robust-training sweeps."),
        ((880, 430, 1480, 720), "Outlook", "What works now, what to improve next, and action plan."),
    ]
    fills = [(232, 243, 255), (236, 255, 244), (255, 247, 236), (245, 239, 255)]
    outlines = [(79, 128, 214), (57, 160, 108), (206, 136, 44), (153, 92, 186)]
    for (rect, title, body), fill, outline in zip(blocks, fills, outlines):
        x1, y1, x2, y2 = rect
        _rr(d, rect, 24, fill=fill, outline=outline, width=4)
        _draw_text_in_box(
            d,
            (x1 + 22, y1 + 20, x2 - 22, y1 + 84),
            title,
            start_size=29,
            min_size=21,
            bold=True,
            fill=(30, 47, 73),
        )
        _draw_text_in_box(
            d,
            (x1 + 22, y1 + 95, x2 - 22, y2 - 22),
            body,
            start_size=23,
            min_size=16,
            fill=(35, 55, 84),
            spacing=8,
        )

    _arrow(d, (720, 265), (880, 265), color=(95, 117, 148), width=7)
    _arrow(d, (1180, 360), (1180, 430), color=(95, 117, 148), width=7)
    _arrow(d, (880, 575), (720, 575), color=(95, 117, 148), width=7)
    _arrow(d, (420, 430), (420, 360), color=(95, 117, 148), width=7)
    img.save(path)


def _plot_penalty_sweep(entries: List[SweepEntry], title: str, path: Path, best_penalty: float) -> None:
    if not entries:
        plt.figure(figsize=(11, 6))
        plt.title(title)
        plt.text(0.5, 0.5, "No data", ha="center", va="center")
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()
        return

    xs = np.array([e.penalty for e in entries], dtype=float)
    f_nom = np.array([e.f_nom for e in entries], dtype=float)
    f_rob = np.array([e.f_rob for e in entries], dtype=float)
    score = np.array([e.score for e in entries], dtype=float)

    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    ax.plot(xs, f_nom, "o-", lw=2.5, color="#2c6db7", label="Nominal fidelity")
    ax.plot(xs, f_rob, "o-", lw=2.5, color="#d2862c", label="Robust fidelity")
    ax.plot(xs, score, "o-", lw=2.5, color="#2f9b5c", label="Score")

    ax.set_xlabel("Penalty p")
    ax.set_ylabel("Metric")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.set_ylim(0.65, 1.01)
    ax.legend(loc="lower right")

    best_idx = int(np.argmin(np.abs(xs - best_penalty)))
    ax.axvline(xs[best_idx], color="#ba2d2d", linestyle="--", alpha=0.8)
    ax.text(xs[best_idx], 0.68, "best p=%.2f" % xs[best_idx], color="#ba2d2d")

    for x, y in zip(xs, score):
        ax.text(x, y + 0.01, "%.3f" % y, ha="center", fontsize=9)

    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_eval_curve_compare(static_run: Path, stoch_run: Path, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    used = False

    static_csv = static_run / "eval_fidelity.csv"
    if static_csv.exists():
        df = pd.read_csv(static_csv)
        if {"epoch", "mean_fidelity"}.issubset(df.columns):
            ax.plot(df["epoch"], df["mean_fidelity"], lw=2.0, color="#2c6db7", label="Static best (p=%.2f)" % _parse_penalty_token(static_run.name.split("_")[1]))
            used = True

    stoch_csv = stoch_run / "eval_fidelity.csv"
    if stoch_csv.exists():
        df = pd.read_csv(stoch_csv)
        if {"epoch", "mean_fidelity"}.issubset(df.columns):
            ax.plot(df["epoch"], df["mean_fidelity"], lw=2.0, color="#2f9b5c", label="Stochastic best (p=%.2f)" % _parse_penalty_token(stoch_run.name.split("_")[2]))
            used = True

    if not used:
        ax.text(0.5, 0.5, "Missing eval_fidelity.csv", transform=ax.transAxes, ha="center", va="center")

    ax.set_title("Binomial Evaluation Fidelity vs Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean evaluation fidelity")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_state_dashboard(metrics: Metrics, path: Path) -> None:
    names = ["Cat", "GKP", "Binomial\nstatic-best", "Binomial\nstoch-best"]
    vals = [
        metrics.cat_fid,
        metrics.gkp_fid,
        metrics.best_static.f_nom,
        metrics.best_stoch.f_nom,
    ]
    cols = ["#3a71bd", "#2f9b5c", "#cf8a32", "#6b55b7"]

    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    bars = ax.bar(names, vals, color=cols)
    ax.set_ylim(0.88, 1.01)
    ax.set_ylabel("Final nominal fidelity")
    ax.set_title("Current Best Fidelity Across Main Targets")
    ax.grid(axis="y", alpha=0.25)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2.0, v + 0.002, "%.4f" % v, ha="center", va="bottom", fontsize=10)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_static_vs_stoch(metrics: Metrics, path: Path) -> None:
    labels = [
        "Score",
        "Nominal fidelity",
        "Robust fidelity",
        "Mean gain\n(over baseline)",
        "Gain @ zero\ndetuning",
    ]
    static_vals = [
        metrics.best_static.score,
        metrics.best_static.f_nom,
        metrics.best_static.f_rob,
        metrics.static_stats.mean_gain,
        metrics.static_stats.zero_gain,
    ]
    stoch_vals = [
        metrics.best_stoch.score,
        metrics.best_stoch.f_nom,
        metrics.best_stoch.f_rob,
        metrics.stoch_stats.mean_gain,
        metrics.stoch_stats.zero_gain,
    ]

    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(12, 6.8))
    ax.bar(x - width / 2, static_vals, width, label="Static best (p=%.2f)" % metrics.best_static.penalty, color="#2c6db7")
    ax.bar(x + width / 2, stoch_vals, width, label="Stochastic best (p=%.2f)" % metrics.best_stoch.penalty, color="#2f9b5c")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(-0.08, 1.02)
    ax.grid(axis="y", alpha=0.25)
    ax.set_title("Best Static vs Best Stochastic Robust Training")
    ax.legend(loc="upper right")

    for i, (sv, tv) in enumerate(zip(static_vals, stoch_vals)):
        ax.text(i - width / 2, sv + 0.015, "%.3f" % sv, ha="center", fontsize=9)
        ax.text(i + width / 2, tv + 0.015, "%.3f" % tv, ha="center", fontsize=9)

    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_penalty_mode_comparison(metrics: Metrics, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8), sharex=False)

    s = sorted(metrics.static_entries, key=lambda e: e.penalty)
    t = sorted(metrics.stoch_entries, key=lambda e: e.penalty)
    xs = np.array([e.penalty for e in s], dtype=float) if s else np.array([])
    xt = np.array([e.penalty for e in t], dtype=float) if t else np.array([])

    configs = [
        ("Score", lambda e: e.score, (0.72, 1.01)),
        ("Nominal fidelity", lambda e: e.f_nom, (0.90, 1.01)),
        ("Robust fidelity", lambda e: e.f_rob, (0.75, 1.01)),
    ]

    for ax, (label, fn, ylim) in zip(axes, configs):
        if s:
            ys = np.array([fn(e) for e in s], dtype=float)
            ax.plot(xs, ys, "o-", color="#2c6db7", lw=2.2, label="Quasi-static")
            for x, y in zip(xs, ys):
                ax.text(x, y + 0.006, f"{y:.3f}", ha="center", fontsize=8, color="#1f4f87")
        if t:
            yt = np.array([fn(e) for e in t], dtype=float)
            ax.plot(xt, yt, "s-", color="#2f9b5c", lw=2.2, label="Stochastic")
            for x, y in zip(xt, yt):
                ax.text(x, y - 0.018, f"{y:.3f}", ha="center", fontsize=8, color="#1f6f43")

        ax.set_title(label)
        ax.set_xlabel("Penalty p")
        ax.set_ylim(*ylim)
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Metric value")
    axes[0].legend(loc="lower left")
    fig.suptitle("Penalty Sweep Comparison Across Modes", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_binomial_progression(metrics: Metrics, path: Path) -> None:
    labels = ["Pre-robust\ncheckpoint", "Best static\nrobust", "Best stochastic\nrobust"]
    score = [metrics.bin_pre_score, metrics.best_static.score, metrics.best_stoch.score]
    f_nom = [metrics.bin_pre_fid, metrics.best_static.f_nom, metrics.best_stoch.f_nom]
    f_rob = [metrics.bin_pre_f_rob, metrics.best_static.f_rob, metrics.best_stoch.f_rob]

    x = np.arange(len(labels))
    w = 0.25

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.bar(x - w, score, w, color="#486fb6", label="Score")
    ax.bar(x, f_nom, w, color="#2f9b5c", label="Nominal fidelity")
    ax.bar(x + w, f_rob, w, color="#d2862c", label="Robust fidelity")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.72, 1.01)
    ax.set_ylabel("Metric")
    ax.set_title("Binomial Training Progression")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right")

    for xi, vals in enumerate(zip(score, f_nom, f_rob)):
        for off, v in zip([-w, 0.0, w], vals):
            ax.text(xi + off, v + 0.006, f"{v:.3f}", ha="center", fontsize=9)

    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_eval_curve_single(run_dir: Path, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    csv_path = run_dir / "eval_fidelity.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if {"epoch", "mean_fidelity"}.issubset(df.columns):
            ax.plot(df["epoch"], df["mean_fidelity"], lw=2.2, color="#2c6db7")
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Mean evaluation fidelity")
            ax.grid(alpha=0.25)
            fig.savefig(path, dpi=220, bbox_inches="tight")
            plt.close(fig)
            return
    ax.text(0.5, 0.5, "Missing eval_fidelity.csv", transform=ax.transAxes, ha="center", va="center")
    ax.set_title(title)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def generate_assets(metrics: Metrics) -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    build_hook_image(ASSET_DIR / "hook_question.png")
    build_applications_image(ASSET_DIR / "applications_motivation.png")
    build_motivation_image(ASSET_DIR / "motivation_encoding.png")
    build_timeline_image(ASSET_DIR / "related_work_timeline.png")
    build_literature_comparison_image(ASSET_DIR / "literature_comparison.png")
    build_rl_loop_image(ASSET_DIR / "rl_loop.png")
    build_agent_io_image(ASSET_DIR / "agent_io.png")
    build_epoch_timeline_image(ASSET_DIR / "epoch_timeline.png")
    build_ppo_image(ASSET_DIR / "ppo_brief.png")
    build_char_reward_mechanism_image(ASSET_DIR / "char_reward_mechanism.png")
    build_sampling_strategy_image(ASSET_DIR / "sampling_strategy.png")
    build_output_reading_guide_image(ASSET_DIR / "output_reading_guide.png")
    build_agenda_visual_image(ASSET_DIR / "agenda_visual.png")
    build_next_steps_image(ASSET_DIR / "next_steps_plan.png")
    build_conclusion_image(metrics, ASSET_DIR / "conclusion_summary.png")
    build_reference_scope_image(ASSET_DIR / "reference_scope.png")
    build_hyperparam_image(metrics, ASSET_DIR / "hyperparam_compare.png")
    build_dynamics_vs_qutip_image(ASSET_DIR / "dynamics_vs_qutip.png")
    build_robust_modes_image(ASSET_DIR / "robust_modes.png")

    _plot_penalty_sweep(
        metrics.static_entries,
        "Quasi-static penalty sweep (fine scan)",
        ASSET_DIR / "static_penalty_sweep.png",
        metrics.best_static.penalty,
    )
    _plot_penalty_sweep(
        metrics.stoch_entries,
        "Stochastic penalty sweep (combined coarse+refine)",
        ASSET_DIR / "stoch_penalty_sweep.png",
        metrics.best_stoch.penalty,
    )
    _plot_eval_curve_compare(metrics.best_static.run_dir, metrics.best_stoch.run_dir, ASSET_DIR / "binomial_eval_compare.png")
    _plot_eval_curve_single(metrics.bin_pre_run_dir, ASSET_DIR / "binomial_pre_eval_curve.png", "Binomial baseline checkpoint: eval fidelity")
    _plot_state_dashboard(metrics, ASSET_DIR / "state_dashboard.png")
    _plot_static_vs_stoch(metrics, ASSET_DIR / "static_vs_stoch.png")
    _plot_penalty_mode_comparison(metrics, ASSET_DIR / "penalty_mode_compare.png")
    _plot_binomial_progression(metrics, ASSET_DIR / "binomial_progression.png")

    missing = ASSET_DIR / "_missing.png"
    if not missing.exists():
        _make_missing_image(missing)


def _set_title(slide, title: str, subtitle: str = "") -> None:
    tx = slide.shapes.add_textbox(Inches(0.45), Inches(0.10), Inches(12.3), Inches(0.95))
    tf = tx.text_frame
    tf.clear()
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(32)
    p.font.bold = True
    p.font.color.rgb = RGBColor(20, 33, 57)
    if subtitle:
        p2 = tf.add_paragraph()
        p2.text = subtitle
        p2.font.size = Pt(16)
        p2.font.color.rgb = RGBColor(75, 90, 114)


def _add_bullets(slide, x: float, y: float, w: float, h: float, lines: List[str], size: int = 19) -> None:
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    for i, t in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = t
        p.font.size = Pt(size)
        p.level = 0
        p.font.color.rgb = RGBColor(28, 42, 66)
        p.space_after = Pt(6)


def _add_footer(slide, page_num: int, ref_text: str = "") -> None:
    box_l = slide.shapes.add_textbox(Inches(0.35), Inches(7.10), Inches(10.8), Inches(0.26))
    tf_l = box_l.text_frame
    tf_l.clear()
    p = tf_l.paragraphs[0]
    p.text = ref_text if ref_text else "Sydney Uni weekly report | trapped-ion RL control"
    p.font.size = Pt(10)
    p.font.color.rgb = RGBColor(95, 110, 135)

    box_r = slide.shapes.add_textbox(Inches(12.15), Inches(7.08), Inches(0.9), Inches(0.28))
    tf_r = box_r.text_frame
    tf_r.clear()
    p2 = tf_r.paragraphs[0]
    p2.text = str(page_num)
    p2.font.size = Pt(11)
    p2.font.bold = True
    p2.font.color.rgb = RGBColor(95, 110, 135)


def _add_picture(slide, image_path: Path, x: float, y: float, w: Optional[float] = None, h: Optional[float] = None) -> None:
    fallback = ASSET_DIR / "_missing.png"
    image_path = _safe_image(image_path, fallback)
    if w is not None and h is not None:
        slide.shapes.add_picture(str(image_path), Inches(x), Inches(y), width=Inches(w), height=Inches(h))
    elif w is not None:
        slide.shapes.add_picture(str(image_path), Inches(x), Inches(y), width=Inches(w))
    elif h is not None:
        slide.shapes.add_picture(str(image_path), Inches(x), Inches(y), height=Inches(h))
    else:
        slide.shapes.add_picture(str(image_path), Inches(x), Inches(y))


def build_presentation(metrics: Metrics) -> int:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    def new_slide(title: str, subtitle: str = "", footer_ref: str = ""):
        s = prs.slides.add_slide(blank)
        _set_title(s, title, subtitle)
        _add_footer(s, len(prs.slides), footer_ref)
        return s

    cat_char = CAT_OUT / "char_target_vs_final.png"
    cat_train = CAT_OUT / "training_curve.png"
    cat_pulse = CAT_OUT / "pulse_sequences.png"
    cat_gif = CAT_OUT / "characteristic_points_evolution.gif"

    gkp_char = GKP_OUT / "char_target_vs_final.png"
    gkp_train = GKP_OUT / "training_curve.png"
    gkp_pulse = GKP_OUT / "pulse_sequences.png"
    gkp_gif = GKP_OUT / "characteristic_points_evolution.gif"

    # Use best static for robust binomial visualization, and global GIF from latest outputs.
    bin_pre_dir = metrics.bin_pre_run_dir
    bin_char_pre = bin_pre_dir / "char_target_vs_final.png"
    bin_pre_dephase = bin_pre_dir / "dephasing_compare.png"
    bin_pre_eval_curve = ASSET_DIR / "binomial_pre_eval_curve.png"

    bin_char_static = metrics.best_static.run_dir / "char_target_vs_final.png"
    bin_dephase_static = metrics.best_static.run_dir / "dephasing_compare.png"
    bin_char_stoch = metrics.best_stoch.run_dir / "char_target_vs_final.png"
    bin_dephase_stoch = metrics.best_stoch.run_dir / "dephasing_compare.png"
    bin_gif = BIN_OUT / "characteristic_points_evolution.gif"

    # 1
    s = new_slide(
        "Trapped-Ion Bosonic State Preparation via Model-Free Reinforcement Learning",
        "Motivation, method, and dephasing-robust training results",
        "Project overview",
    )
    _add_bullets(
        s,
        0.7,
        1.20,
        5.6,
        4.2,
        [
            "Core question: can RL learn high-fidelity and noise-robust pulses from measurement feedback?",
            "Targets: Cat, GKP, and Binomial bosonic states.",
            "Today: explain RL loop clearly, then show visual evidence and robust-training outcomes.",
            "The presentation focuses on physical interpretation of the optimization workflow.",
        ],
        size=19,
    )
    _add_picture(s, ASSET_DIR / "hook_question.png", 6.55, 1.05, w=6.25)
    _add_bullets(
        s,
        0.7,
        5.5,
        12.1,
        1.0,
        [
            "Main structure: Motivation -> Setup/Method -> RL optimization loop -> Results -> Outlook.",
        ],
        size=17,
    )

    # 2
    s = new_slide("Agenda", footer_ref="Report structure")
    _add_bullets(
        s,
        0.8,
        1.25,
        5.6,
        5.2,
        [
            "1) Why bosonic state preparation matters (logical qubits, sensing, fundamental physics).",
            "2) What model-free RL is doing in this project and why this is interesting.",
            "3) How optimization works: control parameters -> quantum feedback -> policy update loop.",
            "4) Characteristic-function reward and how sampling points are selected over training.",
            "5) Results for Cat, GKP, and Binomial states (images + GIFs).",
            "6) Dephasing-robust training: quasi-static and stochastic penalty sweeps.",
            "7) Conclusions, open issues, and immediate next steps.",
        ],
        size=18,
    )
    _add_picture(s, ASSET_DIR / "agenda_visual.png", 6.5, 1.05, w=6.2)

    # 3
    s = new_slide("Why This Problem Matters", footer_ref="Motivation")
    _add_bullets(
        s,
        0.7,
        1.12,
        5.3,
        4.7,
        [
            "Preparing non-classical bosonic states is a foundational control task.",
            "Better state preparation supports encoded qubits and practical logical operations.",
            "The same control ideas are relevant for sensing and broader quantum-state engineering.",
            "If we can automate pulse search with RL, we can scale exploration faster than manual tuning.",
        ],
        size=19,
    )
    _add_picture(s, ASSET_DIR / "applications_motivation.png", 6.35, 1.0, w=6.45)
    _add_bullets(
        s,
        0.7,
        6.0,
        12.0,
        0.8,
        ["Motivation links directly to encoded bosonic qubits, sensing relevance, and control of non-classical dynamics."],
        size=16,
    )

    # 4
    s = new_slide("Bosonic Encoding Intuition", footer_ref="Conceptual setup")
    _add_bullets(
        s,
        0.7,
        1.10,
        5.2,
        4.6,
        [
            "Conventional route: many physical qubits are combined for protection.",
            "Bosonic route: one oscillator mode provides a richer controllable state space.",
            "This project focuses on preparing bosonic target states with high accuracy and stability.",
            "Motivation organization follows Sivak (RL control) and bosonic-code context from group discussions.",
        ],
        size=19,
    )
    _add_picture(s, ASSET_DIR / "motivation_encoding.png", 6.2, 1.0, w=6.6)
    _add_bullets(
        s,
        0.7,
        5.95,
        12.0,
        0.8,
        ["This motivates focusing on pulse optimization for bosonic target-state preparation."],
        size=16,
    )

    # 5
    s = new_slide("Why Model-Free RL Here?", footer_ref="Method motivation")
    _add_bullets(
        s,
        0.8,
        1.20,
        7.0,
        4.8,
        [
            "We want an optimizer that can learn directly from measured feedback signals.",
            "Model-free RL does not require an analytic gradient of the full physical system.",
            "The policy can iteratively improve by comparing outcomes from many trajectories.",
            "This is a natural fit for closed-loop control where reward is computed from observables.",
            "In short: RL acts as an adaptive pulse-search strategy under realistic feedback constraints.",
        ],
        size=18,
    )
    _add_picture(s, ASSET_DIR / "rl_loop_measurement_feedback.png", 7.85, 1.35, w=4.95)

    # 6
    s = new_slide("Related Works and Positioning", footer_ref="Context and contribution")
    _add_bullets(
        s,
        0.7,
        1.1,
        5.1,
        4.4,
        [
            "Sivak et al. (PRX 2022): model-free RL can optimize quantum control directly from feedback.",
            "Matsos et al. (PRL 2024): robust and deterministic trapped-ion bosonic-state preparation was demonstrated experimentally.",
            "This project: apply and explain an RL workflow for trapped-ion Cat/GKP/Binomial preparation,",
            "including dephasing-robust static and stochastic training branches.",
        ],
        size=17,
    )
    _add_picture(s, ASSET_DIR / "literature_comparison.png", 6.1, 1.0, w=6.6)

    # 7
    s = new_slide("Project Scope in One Slide", footer_ref="Project scope")
    _add_bullets(
        s,
        0.8,
        1.15,
        6.8,
        4.8,
        [
            "State families covered: Cat, GKP, and Binomial.",
            "Optimization style: model-free RL with PPO policy updates.",
            "Reward style: characteristic-function agreement, not full-state tomography in loop.",
            "Robustness: dephasing-aware training with quasi-static and stochastic noise branches.",
            "Compute path: dynamiqs/JAX simulation with GPU acceleration and parallel rollout strategy.",
        ],
        size=18,
    )
    _add_picture(s, ASSET_DIR / "state_family_overview.png", 7.7, 1.45, w=5.1)

    # 8
    s = new_slide("How the RL Control Loop Works", footer_ref="Optimization loop")
    _add_bullets(
        s,
        0.7,
        1.02,
        5.4,
        3.9,
        [
            "The policy proposes pulse parameters for each trajectory.",
            "The simulator executes the pulse sequence and returns trajectory-level measurement reward.",
            "Reward is derived from characteristic-function mismatch at sampled points.",
            "PPO applies repeated policy updates, then the improved policy generates the next epoch.",
            "This is an iterative policy-improvement loop under measurement-style feedback.",
        ],
        size=17,
    )
    _add_picture(s, ASSET_DIR / "rl_loop.png", 6.45, 1.0, w=6.35)
    _add_bullets(
        s,
        0.7,
        5.35,
        12.0,
        0.8,
        ["Loop interpretation: the control policy is updated from rollout returns, not from analytical gradients of the physics model."],
        size=16,
    )

    # 9
    s = new_slide("Agent Inputs, Outputs, and Update Type", footer_ref="What is optimized each epoch")
    _add_bullets(
        s,
        0.7,
        1.05,
        5.1,
        4.7,
        [
            "Policy input: clock/step context and constant channel from the environment interface.",
            "Policy output: segmented pulse parameters (phi_r, phi_b, optional amplitudes, duration_scale).",
            "Client computes reward after full trajectory execution and sends scalar returns.",
            "Update type: policy iteration (actor + value update via PPO), not tabular value iteration.",
            "This distinction matters when explaining why PPO is used in this continuous-control setting.",
        ],
        size=17,
    )
    _add_picture(s, ASSET_DIR / "agent_io.png", 6.1, 1.0, w=6.7)

    # 10
    s = new_slide("Episode, Epoch, Policy Update", footer_ref="Training workflow")
    _add_picture(s, ASSET_DIR / "epoch_timeline.png", 0.95, 0.95, w=11.4)

    # 11
    s = new_slide("Proximal Policy Optimization (PPO)", footer_ref="Policy optimization method")
    _add_bullets(
        s,
        0.7,
        1.06,
        5.0,
        4.8,
        [
            "PPO is an on-policy actor-critic method for stable policy iteration.",
            "Each epoch: collect trajectories -> compute advantages with value network -> update actor and critic.",
            "Clipped policy objective constrains update size and prevents destructive policy jumps.",
            "In this project, policy-update count per epoch is a high-impact hyperparameter.",
            "We therefore report policy-update settings explicitly for Cat, GKP, and Binomial targets.",
        ],
        size=17,
    )
    _add_picture(s, ASSET_DIR / "ppo_brief.png", 6.2, 1.0, w=6.5)

    # 12
    s = new_slide("Why Different States Need Different Hyperparameters", footer_ref="State-dependent optimization behavior")
    _add_bullets(
        s,
        0.7,
        1.0,
        5.0,
        4.1,
        [
            "Different target geometries create different optimization difficulty.",
            "GKP needs longer training budget due to lattice-like structure constraints.",
            "Binomial uses stage-wise characteristic sampling to improve stability.",
            "Policy-update count and epoch budget matter more than minor entropy tuning here.",
        ],
        size=18,
    )
    _add_picture(s, ASSET_DIR / "hyperparam_compare.png", 6.1, 0.95, w=6.7)

    # 12
    s = new_slide("dynamiqs vs QuTiP and GPU Use", footer_ref="Simulation platform choice")
    _add_bullets(
        s,
        0.7,
        1.08,
        5.0,
        4.1,
        [
            "Main reason for dynamiqs path: better parallel rollout throughput.",
            "RL training repeatedly evaluates many trajectories, so batching matters.",
            "GPU acceleration reduces wall-clock cost for iterative optimization.",
            "Presentation-level message: this improves training practicality, not just implementation detail.",
        ],
        size=18,
    )
    _add_picture(s, ASSET_DIR / "dynamics_vs_qutip.png", 6.1, 1.0, w=6.7)

    # 13
    s = new_slide("Characteristic-Function Reward", footer_ref="Reward definition")
    _add_bullets(
        s,
        0.7,
        1.02,
        5.0,
        4.5,
        [
            "Reward is computed from sampled characteristic-function points.",
            "This acts like measurement-style feedback and avoids relying on full state information inside the loop.",
            "It also provides visual diagnostics: target vs final characteristic maps are easy to compare.",
            "This is where the RL loop connects directly to physics observables.",
        ],
        size=18,
    )
    _add_picture(s, ASSET_DIR / "char_reward_mechanism.png", 6.1, 1.0, w=6.7)

    # 14
    s = new_slide("Training Signal Scheduling", footer_ref="Stage-wise training setup")
    _add_bullets(
        s,
        0.7,
        1.05,
        5.0,
        4.6,
        [
            "Stage 1 uses a focused subset to provide strong initial learning signal.",
            "Stage 2 expands sampled coverage to improve stability and exploration.",
            "Stage 3 uses denser sampling for final high-fidelity refinement.",
            "This schedule is applied consistently across training epochs.",
            "Point-evolution GIFs are shown later in the results section.",
        ],
        size=18,
    )
    _add_picture(s, ASSET_DIR / "sampling_strategy.png", 6.1, 1.0, w=6.7)

    # 15
    s = new_slide("How to Read Training Outputs", footer_ref="Interpreting result figures")
    _add_bullets(
        s,
        0.7,
        1.0,
        5.2,
        4.8,
        [
            "Training curves show whether policy updates are producing consistent improvement.",
            "Characteristic maps show structural agreement between target and final state.",
            "Pulse panels show the learned control structure used to produce the final state.",
            "Dephasing comparison curves show robustness tradeoff under detuning noise.",
            "Together these plots provide a complete per-state result summary.",
        ],
        size=17,
    )
    _add_picture(s, ASSET_DIR / "output_reading_guide.png", 6.1, 1.0, w=6.7)
    _add_bullets(
        s,
        0.7,
        6.15,
        12.0,
        0.6,
        ["These figure types are used in the next slides to tell the Cat/GKP/Binomial story consistently."],
        size=15,
    )

    # 16
    s = new_slide("Cat State Results", footer_ref="Cat target")
    _add_bullets(
        s,
        0.7,
        1.03,
        12.0,
        1.0,
        ["Final cat fidelity: %.6f" % metrics.cat_fid],
        size=22,
    )
    _add_picture(s, cat_char, 0.7, 1.8, w=4.1)
    _add_picture(s, cat_train, 4.95, 1.8, w=4.1)
    _add_picture(s, cat_pulse, 9.2, 1.8, w=4.1)
    _add_bullets(
        s,
        0.7,
        6.35,
        12.0,
        0.7,
        ["Left: target vs final characteristic map | Middle: fidelity trend | Right: learned pulse structure"],
        size=16,
    )

    # 17
    s = new_slide("GKP State Results", footer_ref="GKP target")
    _add_bullets(
        s,
        0.7,
        1.03,
        12.0,
        1.0,
        ["Final GKP fidelity: %.6f" % metrics.gkp_fid],
        size=22,
    )
    _add_picture(s, gkp_char, 0.7, 1.8, w=4.1)
    _add_picture(s, gkp_train, 4.95, 1.8, w=4.1)
    _add_picture(s, gkp_pulse, 9.2, 1.8, w=4.1)
    _add_bullets(
        s,
        0.7,
        6.35,
        12.0,
        0.7,
        ["GKP requires stronger structural matching, which motivates larger epoch budget."],
        size=16,
    )

    # 18
    s = new_slide("Binomial Baseline Result (Before Robust Extension)", footer_ref="Binomial nominal checkpoint")
    _add_bullets(
        s,
        0.7,
        1.05,
        12.0,
        1.2,
        [
            "Baseline checkpoint fidelity: %.6f | score: %.6f | robust fidelity under sweep: %.6f."
            % (metrics.bin_pre_fid, metrics.bin_pre_score, metrics.bin_pre_f_rob),
            "This baseline motivated the dephasing-robust branch and the subsequent penalty sweeps.",
        ],
        size=18,
    )
    _add_picture(s, bin_char_pre, 0.7, 2.1, w=4.1)
    _add_picture(s, bin_pre_eval_curve, 4.95, 2.1, w=4.1)
    _add_picture(s, bin_pre_dephase, 9.2, 2.1, w=4.1)
    _add_bullets(
        s,
        0.7,
        6.35,
        12.0,
        0.7,
        ["Left: target vs final map | Middle: baseline evaluation trend | Right: baseline dephasing response"],
        size=16,
    )

    # 19
    s = new_slide("Characteristic-Point Evolution (GIFs)", footer_ref="Sampling-point evolution across targets")
    _add_bullets(
        s,
        0.7,
        1.0,
        12.0,
        0.8,
        [
            "All targets show staged movement from broad exploration toward informative phase-space regions.",
        ],
        size=20,
    )
    _add_picture(s, cat_gif, 0.55, 1.80, w=4.15)
    _add_picture(s, gkp_gif, 4.65, 1.80, w=4.15)
    _add_picture(s, bin_gif, 8.75, 1.80, w=4.15)
    _add_bullets(
        s,
        0.62,
        6.40,
        12.0,
        0.6,
        ["Cat                     GKP                     Binomial"],
        size=16,
    )

    # 20
    s = new_slide("Robust-Training Configuration", footer_ref="Dephasing-robust setup")
    _add_bullets(
        s,
        0.7,
        1.05,
        5.1,
        4.4,
        [
            "Quasi-static branch: grid detuning samples + gaussian weighting.",
            "Stochastic branch: segment-wise random detuning with gamma_dt scale.",
            "Physical scale alignment enforced: Omega_r=Omega_b=2π×2000 Hz and T_STEP=1e-5 s.",
            "Scale consistency rule: if Omega changes by factor k, T_STEP should change by 1/k.",
            "Duration scaling action is enabled to let optimizer adapt effective pulse duration.",
        ],
        size=17,
    )
    _add_picture(s, ASSET_DIR / "robust_modes.png", 6.2, 1.0, w=6.4)

    # 21
    s = new_slide("Quasi-Static Penalty Sweep (Fine Scan)", footer_ref="Points: p=0.1, 0.2, 0.3, 0.4")
    _add_bullets(
        s,
        0.7,
        1.02,
        5.0,
        3.8,
        [
            "Best quasi-static score occurs at p=%.2f." % metrics.best_static.penalty,
            "At p=%.2f: score=%.6f, f_nom=%.6f, f_rob=%.6f." % (
                metrics.best_static.penalty,
                metrics.best_static.score,
                metrics.best_static.f_nom,
                metrics.best_static.f_rob,
            ),
            "This branch currently achieves stronger robust-fidelity average than the stochastic branch.",
            "Mean gain over baseline in dephasing sweep: %.3f." % metrics.static_stats.mean_gain,
        ],
        size=17,
    )
    _add_picture(s, ASSET_DIR / "static_penalty_sweep.png", 6.1, 1.0, w=6.7)
    _add_picture(s, bin_char_static, 6.2, 4.95, w=3.2)
    _add_bullets(
        s,
        0.7,
        5.2,
        5.0,
        1.3,
        ["Static best improves average dephasing robustness while keeping high nominal fidelity."],
        size=15,
    )

    # 22
    s = new_slide("Stochastic Penalty Sweep (Combined)", footer_ref="Combined points include p=0.1/0.2/0.3/0.5/1.0")
    _add_bullets(
        s,
        0.7,
        1.02,
        5.0,
        3.8,
        [
            "Best stochastic score in current runs is at p=%.2f." % metrics.best_stoch.penalty,
            "At p=%.2f: score=%.6f, f_nom=%.6f, f_rob=%.6f." % (
                metrics.best_stoch.penalty,
                metrics.best_stoch.score,
                metrics.best_stoch.f_nom,
                metrics.best_stoch.f_rob,
            ),
            "Stochastic branch is improving but still behind best quasi-static robust average.",
            "Result suggests noise-model complexity currently needs more tuning/compute budget.",
        ],
        size=17,
    )
    _add_picture(s, ASSET_DIR / "stoch_penalty_sweep.png", 6.1, 1.0, w=6.7)
    _add_picture(s, bin_char_stoch, 6.2, 4.95, w=3.2)
    _add_bullets(
        s,
        0.7,
        5.2,
        5.0,
        1.3,
        ["Stochastic best currently trails static best but remains a valid path for realistic random-noise training."],
        size=15,
    )

    # 23
    s = new_slide("Penalty Sweep Comparison Across Modes", footer_ref="All tested p points in static and stochastic branches")
    _add_bullets(
        s,
        0.7,
        1.02,
        5.1,
        4.3,
        [
            "This figure aggregates all tested penalty points for both robust-training branches.",
            "Quasi-static and stochastic curves can favor different p regions.",
            "Current best points remain static p=%.2f and stochastic p=%.2f."
            % (metrics.best_static.penalty, metrics.best_stoch.penalty),
            "This is why we do coarse-to-fine sweeps instead of choosing one p value a priori.",
        ],
        size=17,
    )
    _add_picture(s, ASSET_DIR / "penalty_mode_compare.png", 5.9, 1.0, w=6.9)
    _add_picture(s, ASSET_DIR / "binomial_eval_compare.png", 0.9, 4.75, w=4.4)
    _add_bullets(
        s,
        6.05,
        5.85,
        6.7,
        1.0,
        ["Inset: epoch-wise evaluation trend for best static vs best stochastic run."],
        size=15,
    )

    # 24
    s = new_slide("Best Quasi-Static Run (p=%.2f)" % metrics.best_static.penalty, footer_ref="Dephasing robustness comparison against baseline")
    _add_bullets(
        s,
        0.7,
        1.05,
        5.0,
        4.4,
        [
            "Score: %.6f" % metrics.best_static.score,
            "Nominal fidelity: %.6f" % metrics.best_static.f_nom,
            "Robust fidelity: %.6f" % metrics.best_static.f_rob,
            "Mean gain vs baseline: %.6f" % metrics.static_stats.mean_gain,
            "Gain at zero detuning: %.6f" % metrics.static_stats.zero_gain,
            "Interpretation: robust improvement is strong across most detuning range, with slight nominal tradeoff.",
        ],
        size=17,
    )
    _add_picture(s, bin_dephase_static, 6.1, 1.1, w=6.7)
    _add_bullets(
        s,
        0.7,
        6.1,
        12.0,
        0.6,
        ["Current static best run remains the strongest robust candidate in this project snapshot."],
        size=15,
    )

    # 25
    s = new_slide("Best Stochastic Run (p=%.2f)" % metrics.best_stoch.penalty, footer_ref="Stochastic robust branch summary")
    _add_bullets(
        s,
        0.7,
        1.05,
        5.0,
        4.4,
        [
            "Score: %.6f" % metrics.best_stoch.score,
            "Nominal fidelity: %.6f" % metrics.best_stoch.f_nom,
            "Robust fidelity: %.6f" % metrics.best_stoch.f_rob,
            "Mean gain vs baseline: %.6f" % metrics.stoch_stats.mean_gain,
            "Gain at zero detuning: %.6f" % metrics.stoch_stats.zero_gain,
            "Interpretation: robust behavior is improved, but current setting still leaves room to reduce zero-noise gap.",
        ],
        size=17,
    )
    _add_picture(s, bin_dephase_stoch, 6.1, 1.1, w=6.7)
    _add_bullets(
        s,
        0.7,
        6.1,
        12.0,
        0.6,
        ["Stochastic robust training is promising and provides a clear direction for targeted follow-up runs."],
        size=15,
    )

    # 26
    s = new_slide("Static vs Stochastic: Current Best Comparison", footer_ref="Best-run metric comparison")
    _add_bullets(
        s,
        0.7,
        1.02,
        5.0,
        3.8,
        [
            "Static best currently has higher robust average and better overall score.",
            "Stochastic best remains valuable because it matches the intended random-noise training formulation.",
            "Difference does not mean stochastic is wrong; it indicates more tuning/compute may be needed.",
            "This comparison guides the next experimental schedule.",
        ],
        size=17,
    )
    _add_picture(s, ASSET_DIR / "static_vs_stoch.png", 6.1, 1.0, w=6.8)
    _add_bullets(
        s,
        0.7,
        5.75,
        12.0,
        0.8,
        ["Comparison highlights where stochastic training still needs improvement (especially near zero detuning)."],
        size=15,
    )

    # 27
    s = new_slide("Across Main Targets: Fidelity Snapshot", footer_ref="Cat/GKP and binomial robust candidates")
    _add_bullets(
        s,
        0.7,
        1.02,
        12.0,
        1.0,
        [
            "Cat and GKP already reach high nominal fidelity; binomial robustness is the main active optimization front.",
        ],
        size=18,
    )
    _add_picture(s, ASSET_DIR / "state_dashboard.png", 1.95, 1.45, w=9.4)
    _add_bullets(
        s,
        0.9,
        6.35,
        11.5,
        0.6,
        ["Cat and GKP are high-fidelity nominal targets; binomial is the primary robust-training benchmark."],
        size=15,
    )

    # 28
    s = new_slide("What We Learned So Far", footer_ref="Current conclusions")
    _add_bullets(
        s,
        0.8,
        1.15,
        6.9,
        4.8,
        [
            "Model-free RL can produce high-quality control pulses for multiple bosonic targets.",
            "Characteristic-function reward plus staged point sampling is practical and interpretable.",
            "Policy-update count and state-specific training budget strongly influence convergence quality.",
            "Quasi-static robust branch currently outperforms stochastic branch on average robust metrics.",
            "Stochastic branch is still meaningful and likely benefits from further targeted hyperparameter tuning.",
        ],
        size=18,
    )
    _add_picture(s, ASSET_DIR / "binomial_progression.png", 7.75, 1.55, w=5.0)

    # 29
    s = new_slide("Immediate Next Steps", footer_ref="Outlook")
    _add_bullets(
        s,
        0.8,
        1.15,
        7.0,
        4.8,
        [
            "Refine stochastic branch near p≈0.1 with controlled extra compute budget.",
            "Tune duration-related penalties to reduce zero-detuning nominal gap.",
            "Continue improving GKP with targeted long-horizon training settings.",
            "Prepare final speaking script with concept-first explanation and image-led storytelling.",
            "Keep citations tied to discussed content (Sivak PRX, Matsos PRL, related trapped-ion work).",
        ],
        size=18,
    )
    _add_picture(s, ASSET_DIR / "next_steps_plan.png", 7.8, 1.35, w=4.9)

    # 30
    s = new_slide("Conclusion", footer_ref="Summary")
    _add_bullets(
        s,
        0.8,
        1.2,
        7.0,
        4.6,
        [
            "This project demonstrates a clear and explainable RL control workflow for bosonic state preparation.",
            "The method works across Cat/GKP/Binomial targets and supports robust-training extensions.",
            "Current robust best points: quasi-static p=%.2f, stochastic p=%.2f." % (metrics.best_static.penalty, metrics.best_stoch.penalty),
            "The report now presents a complete story: motivation -> method -> results -> outlook.",
        ],
        size=18,
    )
    _add_picture(s, ASSET_DIR / "conclusion_summary.png", 7.8, 1.35, w=4.9)

    # 31
    s = new_slide("References", footer_ref="Only references cited in talk")
    _add_bullets(
        s,
        0.7,
        1.2,
        7.0,
        4.8,
        [
            "D. Gottesman, A. Kitaev, J. Preskill, Phys. Rev. A 64, 012310 (2001).",
            "V. V. Sivak et al., Model-Free Quantum Control with Reinforcement Learning, Phys. Rev. X 12, 011059 (2022).",
            "V. G. Matsos et al., Robust and Deterministic Preparation of Bosonic Logical States in a Trapped Ion, Phys. Rev. Lett. 133, 050602 (2024).",
            "Internal trapped-ion RL project outputs and dephasing-robust sweep logs (2026).",
            "Sydney Nano / Quantum Control Lab discussions and meeting guidance.",
        ],
        size=17,
    )
    _add_picture(s, ASSET_DIR / "reference_scope.png", 7.8, 1.35, w=4.9)

    # 32
    s = new_slide("Appendix: Presentation Flow Checklist", footer_ref="Appendix")
    _add_bullets(
        s,
        0.8,
        1.2,
        6.1,
        5.2,
        [
            "Define episode, epoch, and policy update before showing any result plots.",
            "When showing GIFs, explain that point concentration indicates increased sampling focus.",
            "For robust sweeps, report score, nominal fidelity, robust fidelity, and zero-detuning behavior.",
            "Separate method validity from current performance limits when discussing stochastic results.",
            "Finish with one clear impact statement and immediate follow-up plan.",
        ],
        size=17,
    )
    _add_bullets(
        s,
        7.0,
        1.55,
        5.3,
        4.8,
        [
            "Suggested transition cues:",
            "Motivation -> 'what problem is worth solving and why now?'",
            "Method -> 'how the RL loop changes pulse parameters over epochs'",
            "Results -> 'what the plots/GIFs prove and what remains open'",
            "Outlook -> 'specific next runs and expected impact'",
        ],
        size=16,
    )

    prs.save(str(PPT_PATH_DETAILED))
    shutil.copy2(PPT_PATH_DETAILED, PPT_PATH_ALIAS)
    return len(prs.slides)


def main() -> None:
    metrics = read_metrics()
    generate_assets(metrics)
    slide_count = build_presentation(metrics)

    print("Wrote detailed PPT:", PPT_PATH_DETAILED)
    print("Updated alias PPT:", PPT_PATH_ALIAS)
    print("Static best:", metrics.best_static.tag, "p=%.3f score=%.6f" % (metrics.best_static.penalty, metrics.best_static.score))
    print("Stochastic best:", metrics.best_stoch.tag, "p=%.3f score=%.6f" % (metrics.best_stoch.penalty, metrics.best_stoch.score))
    print("Slides:", slide_count)


if __name__ == "__main__":
    main()
