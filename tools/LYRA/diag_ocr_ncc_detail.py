"""
diag_ocr_ncc_detail.py — Regenerate NCC detail diagnostic for one frame
========================================================================
Shows the full pipeline for a single frame:
  Row 1: raw grayscale crop of CBD text region
  Row 2: upscaled image + binarization threshold
  Row 3: binary with blob boundaries + column projection
  Row 4: per-digit query crops (from binary blobs)
  Row 5: best-match template + NCC score for each digit

Usage (from repo root):
    python tools/LYRA/diag_ocr_ncc_detail.py [TIFF_ID] [FRAME_IDX]

Defaults: TIFF 8400, frame 7 (CBD 0464 — contains "4" at two positions)
"""

from pathlib import Path
import sys
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/LYRA"))
from lyra import detect_frames, _gauss_smooth
from build_digit_templates import (
    get_binary_text, find_digit_col_ranges, ncc_match,
    TMPL_W, TMPL_H,
    TEXT_Y0_FRAC, TEXT_Y1_FRAC, TEXT_X0_FRAC, TEXT_X1_FRAC,
    SCALE_X, SCALE_Y, THRESH_FRAC,
)

# ── Config ─────────────────────────────────────────────────────────────────
FLT = 125
RAW_DIR = ROOT / f"Data/ascope/raw/{FLT}"
OUT_DIR = ROOT / f"tools/LYRA/output/F{FLT}/step0"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TMPL_PATH = ROOT / "tools/LYRA/digit_templates.npy"
INDEX_CSV = ROOT / f"tools/LYRA/output/F{FLT}/step1/F{FLT}_frame_index.csv"

# Defaults
DEFAULT_TIFF_ID = "8400"
DEFAULT_FRAME = 7   # CBD 0464


def load_ground_truth():
    gt = {}
    with open(INDEX_CSV) as f:
        for row in csv.DictReader(f):
            tid = row["tiff_id"]
            fidx = int(row["frame_idx"])
            cbd = row.get("cbd", "")
            if tid not in gt:
                gt[tid] = {}
            if cbd:
                gt[tid][fidx] = cbd
    return gt


def main():
    tiff_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TIFF_ID
    frame_idx = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_FRAME

    # ── Load ───────────────────────────────────────────────────────────────
    templates = np.load(TMPL_PATH, allow_pickle=True).item()
    gt = load_ground_truth()
    gt_cbds = gt.get(tiff_id, {})
    true_cbd = gt_cbds.get(frame_idx, "????")
    true_str = f"{FLT:03d}{true_cbd}"

    matches = list(RAW_DIR.glob(f"*_{tiff_id:>07s}_*-reel_begin_end.tiff"))
    if not matches:
        matches = list(RAW_DIR.glob(f"*_{tiff_id}_*"))
    tiff_path = matches[0]

    print(f"Loading {tiff_path.name} ...")
    Image.MAX_IMAGE_PIXELS = None
    img = np.array(Image.open(tiff_path), dtype=np.float32)
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-9)
    H, W = img_norm.shape

    frames = detect_frames(img_norm)
    l, r = frames[frame_idx]
    crop_u8 = (img_norm[:, l:r+1] * 255).clip(0, 255).astype(np.uint8)
    fw = crop_u8.shape[1]
    print(f"Frame {frame_idx}: cols {l}–{r}, width {fw} px, true CBD: {true_str}")

    # ── Step 1: raw text crop ──────────────────────────────────────────────
    y0 = int(H * TEXT_Y0_FRAC)
    y1 = int(H * TEXT_Y1_FRAC)
    x0 = int(fw * TEXT_X0_FRAC)
    x1 = int(fw * TEXT_X1_FRAC)
    raw_band = crop_u8[y0:y1, x0:x1]

    # ── Step 2: upscale ───────────────────────────────────────────────────
    big = np.array(Image.fromarray(raw_band).resize(
        (raw_band.shape[1] * SCALE_X, raw_band.shape[0] * SCALE_Y),
        Image.LANCZOS))
    lo, hi = int(big.min()), int(big.max())
    thresh_val = lo + (hi - lo) * THRESH_FRAC

    # ── Step 3: binarize ──────────────────────────────────────────────────
    binary = get_binary_text(crop_u8, H, fw)

    # ── Step 4: blob detection (with "4" merge fix) ───────────────────────
    blobs_raw = find_digit_col_ranges(binary)
    n_blobs_raw = len(blobs_raw)

    # Adjust to 7 (same as recognize_frame)
    blobs = list(blobs_raw)
    while len(blobs) > 7:
        gaps = [(blobs[k+1][0] - blobs[k][1]) for k in range(len(blobs)-1)]
        m = int(np.argmin(gaps))
        merged = (blobs[m][0], blobs[m+1][1])
        blobs = blobs[:m] + [merged] + blobs[m+2:]
    if len(blobs) < 7:
        ws = [e - s for s, e in blobs]
        m = int(np.argmax(ws))
        s, e = blobs[m]
        mid = (s + e) // 2
        blobs = blobs[:m] + [(s, mid), (mid+1, e)] + blobs[m+1:]

    # ── Column projection for diagnostic ──────────────────────────────────
    col_proj = binary.sum(axis=0)
    smoothed = _gauss_smooth(col_proj, sigma=3)

    # ── NCC matching ──────────────────────────────────────────────────────
    n_digits = len(blobs)
    queries = []
    results = []
    for pos, (s, e) in enumerate(blobs):
        pad = 4
        crop_d = binary[:, max(0, s-pad): e+pad+1]
        if crop_d.shape[1] < 3:
            queries.append(None)
            results.append(("?", {}, "?"))
            continue
        query = np.array(Image.fromarray(
            (crop_d * 255).astype(np.uint8)).resize(
            (TMPL_W, TMPL_H), Image.LANCZOS), dtype=np.float32) / 255.0
        queries.append(query)
        scores = {d: ncc_match(query, t) for d, t in templates.items()}
        best = max(scores, key=scores.get)
        true_d = true_str[pos] if pos < len(true_str) else "?"
        results.append((best, scores, true_d))

    ocr_str = "".join(r[0] for r in results)

    # ── Build figure ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 22))
    fig.patch.set_facecolor("white")

    # Use gridspec: 5 logical rows with different height ratios
    gs = fig.add_gridspec(5, 1, height_ratios=[1, 1.2, 1.5, 2.5, 2.5],
                          hspace=0.35)

    # ── Row 1: raw crop ───────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(raw_band, cmap="gray", aspect="auto")
    ax1.set_title(f"Step 1: Raw CBD text region (Frame {frame_idx}, "
                  f"true: {true_str})", fontsize=11)
    ax1.axis("off")

    # ── Row 2: upscaled + threshold ───────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.imshow(big, cmap="gray", aspect="auto")
    ax2.set_title(f"Step 2: Upscaled ({SCALE_X}×h, {SCALE_Y}×v) → "
                  f"{big.shape[1]}×{big.shape[0]} px | "
                  f"Threshold: {int(thresh_val)} "
                  f"({int(THRESH_FRAC*100)}% of [{lo},{hi}])",
                  fontsize=11)
    ax2.axis("off")

    # ── Row 3: binary + blobs + column projection ─────────────────────────
    gs3 = gs[2].subgridspec(2, 1, height_ratios=[2, 1], hspace=0.15)

    ax3a = fig.add_subplot(gs3[0])
    ax3a.imshow(1 - binary, cmap="gray", aspect="auto")
    colors_blob = plt.cm.Set1(np.linspace(0, 1, max(n_digits, 1)))
    for idx, (s, e) in enumerate(blobs):
        c = colors_blob[idx % len(colors_blob)]
        ax3a.axvline(s, color=c, linewidth=1.0, alpha=0.8)
        ax3a.axvline(e, color=c, linewidth=1.0, alpha=0.8)
        ax3a.text((s + e) / 2, -10, str(idx), ha="center", fontsize=8,
                  color=c, fontweight="bold")
    ax3a.set_title(f"Step 3–4: Binary + {n_blobs_raw} raw blobs → "
                   f"{n_digits} after adjust", fontsize=11)
    ax3a.axis("off")

    ax3b = fig.add_subplot(gs3[1])
    ax3b.fill_between(range(len(col_proj)), col_proj, alpha=0.3,
                      color="steelblue", label="raw")
    ax3b.plot(smoothed, color="steelblue", linewidth=1, label="smoothed")
    threshold_line = smoothed.max() * 0.08
    ax3b.axhline(threshold_line, color="red", linewidth=0.8, linestyle="--",
                 alpha=0.7, label=f"threshold={threshold_line:.1f}")
    for idx, (s, e) in enumerate(blobs):
        c = colors_blob[idx % len(colors_blob)]
        ax3b.axvspan(s, e, alpha=0.15, color=c)
    ax3b.set_title("Column projection", fontsize=9)
    ax3b.legend(fontsize=7, loc="upper right")
    ax3b.set_xlim(0, len(col_proj))

    # ── Row 4: per-digit query images ─────────────────────────────────────
    gs4 = gs[3].subgridspec(1, n_digits, wspace=0.3)
    for pos in range(n_digits):
        ax = fig.add_subplot(gs4[pos])
        if queries[pos] is not None:
            ax.imshow(queries[pos], cmap="gray", vmin=0, vmax=1,
                      aspect="auto")
        best, scores, true_d = results[pos]
        marker = "OK" if best == true_d else "MISS"
        ax.set_title(f"Pos {pos}, true='{true_d}'\nbest='{best}' [{marker}]",
                     fontsize=8, color="green" if marker == "OK" else "red")
        ax.axis("off")

    # Add row label
    fig.text(0.01, 0.32, "Step 5a:\nQuery\ncrops",
             fontsize=10, fontweight="bold", va="center")

    # ── Row 5: best-match templates + scores ──────────────────────────────
    gs5 = gs[4].subgridspec(1, n_digits, wspace=0.3)
    for pos in range(n_digits):
        ax = fig.add_subplot(gs5[pos])
        best, scores, true_d = results[pos]
        if best in templates:
            ax.imshow(templates[best], cmap="gray", vmin=0, vmax=1,
                      aspect="auto")
        sorted_s = sorted(scores.items(), key=lambda x: -x[1])
        score_text = "\n".join(
            f"{'→' if d == true_d else ' '}{d}: {s:.3f}"
            for d, s in sorted_s[:5]
        )
        ax.set_title(f"Template '{best}'\nNCC={scores.get(best, 0):.3f}",
                     fontsize=8)
        ax.text(1.05, 0.95, score_text, transform=ax.transAxes,
                fontsize=6.5, fontfamily="monospace", va="top")
        ax.axis("off")

    fig.text(0.01, 0.12, "Step 5b:\nBest\ntemplate",
             fontsize=10, fontweight="bold", va="center")

    fig.suptitle(
        f"TIFF {tiff_id}, Frame {frame_idx} — OCR pipeline detail\n"
        f"True: {true_str} | OCR: {ocr_str}",
        fontsize=13, fontweight="bold", y=0.98)

    out_path = OUT_DIR / f"diag_ocr_ncc_detail_{tiff_id}_fr{frame_idx}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved → {out_path.relative_to(ROOT)}")
    print(f"  True: {true_str}")
    print(f"  OCR:  {ocr_str}")
    for pos, (best, scores, true_d) in enumerate(results):
        sorted_s = sorted(scores.items(), key=lambda x: -x[1])
        top3 = "  ".join(f"{d}={s:.3f}" for d, s in sorted_s[:3])
        marker = "OK" if best == true_d else "MISS"
        print(f"  pos {pos}: {top3}  [{marker}]")


if __name__ == "__main__":
    main()
