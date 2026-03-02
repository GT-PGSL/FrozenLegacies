"""
diag_ocr_groundtruth.py — Generate OCR diagnostic figures for F125 TIFFs
=========================================================================
Produces one PNG per TIFF showing each complete frame's CBD text region:
  - Left:   raw grayscale crop
  - Middle: binary image with blob boundaries (green lines)
  - Right:  OCR read, true CBD, per-digit NCC scores

Usage (from repo root):
    python tools/LYRA/diag_ocr_groundtruth.py [TIFF_ID ...]

If no TIFF_IDs given, processes: 7700, 7725, 8400, 8425
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
from lyra import detect_frames
from build_digit_templates import (
    get_binary_text, find_digit_col_ranges, ncc_match,
    TMPL_W, TMPL_H,
    TEXT_Y0_FRAC, TEXT_Y1_FRAC, TEXT_X0_FRAC, TEXT_X1_FRAC,
)

# ── Config ─────────────────────────────────────────────────────────────────
FLT = 125
RAW_DIR = ROOT / f"Data/ascope/raw/{FLT}"
OUT_DIR = ROOT / f"tools/LYRA/output/F{FLT}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INDEX_CSV = ROOT / f"tools/LYRA/output/F{FLT}/step1/F{FLT}_frame_index.csv"
TMPL_PATH = ROOT / "tools/LYRA/digit_templates.npy"

DEFAULT_TIFFS = ["7700", "7725", "8400", "8425"]


# ── Load ground truth from frame index ─────────────────────────────────────
def load_ground_truth():
    """Return dict: {tiff_id: {frame_idx: cbd_str}} from frame_index.csv."""
    gt = {}
    with open(INDEX_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = row["tiff_id"]
            fidx = int(row["frame_idx"])
            cbd = row.get("cbd", "")
            if tid not in gt:
                gt[tid] = {}
            if cbd:
                gt[tid][fidx] = cbd
    return gt


def make_diag_figure(tiff_path, tiff_id, gt_cbds, templates):
    """Generate diagnostic OCR figure for one TIFF."""
    print(f"\n  Loading {tiff_path.name} ...")
    Image.MAX_IMAGE_PIXELS = None
    img = np.array(Image.open(tiff_path), dtype=np.float32)
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-9)
    H, W = img_norm.shape

    frames = detect_frames(img_norm)
    widths = [r - l for l, r in frames]
    med_w = float(np.median(widths))

    complete = [(i, l, r) for i, (l, r) in enumerate(frames)
                if (r - l) >= 0.75 * med_w]
    n_comp = len(complete)
    print(f"  {n_comp} complete frames")

    # ── Figure layout: 3 columns per row ───────────────────────────────────
    fig, axes = plt.subplots(n_comp, 3, figsize=(16, 2.2 * n_comp))
    if n_comp == 1:
        axes = axes[np.newaxis, :]

    fig.patch.set_facecolor("white")

    for row, (fidx, l, r) in enumerate(complete):
        crop_u8 = (img_norm[:, l:r+1] * 255).clip(0, 255).astype(np.uint8)
        fw = crop_u8.shape[1]
        true_cbd = gt_cbds.get(fidx, "????")
        true_str = f"{FLT:03d}{true_cbd}"

        # ── Raw text region crop ───────────────────────────────────────────
        y0 = int(H * TEXT_Y0_FRAC)
        y1 = int(H * TEXT_Y1_FRAC)
        x0 = int(fw * TEXT_X0_FRAC)
        x1 = int(fw * TEXT_X1_FRAC)
        raw_band = crop_u8[y0:y1, x0:x1]

        ax_raw = axes[row, 0]
        ax_raw.imshow(raw_band, cmap="gray", aspect="auto")
        ax_raw.set_title(f"Fr {fidx} — true: {true_str}", fontsize=8)
        ax_raw.axis("off")

        # ── Binary + blob boundaries ───────────────────────────────────────
        binary = get_binary_text(crop_u8, H, fw)
        blobs = find_digit_col_ranges(binary)

        # Adjust to exactly 7 blobs (same logic as recognize_frame)
        while len(blobs) > 7:
            gaps = [(blobs[k+1][0] - blobs[k][1])
                    for k in range(len(blobs)-1)]
            m = int(np.argmin(gaps))
            merged = (blobs[m][0], blobs[m+1][1])
            blobs = blobs[:m] + [merged] + blobs[m+2:]
        if len(blobs) < 7:
            ws = [e - s for s, e in blobs]
            m = int(np.argmax(ws))
            s, e = blobs[m]
            mid = (s + e) // 2
            blobs = blobs[:m] + [(s, mid), (mid+1, e)] + blobs[m+1:]

        # Show binary as RGB so we can overlay colored lines
        bin_rgb = np.stack([binary, binary, binary], axis=-1)

        ax_bin = axes[row, 1]
        ax_bin.imshow(1 - bin_rgb, aspect="auto")  # invert: ink=white on black
        for s, e in blobs:
            ax_bin.axvline(s, color="lime", linewidth=0.5, alpha=0.7)
            ax_bin.axvline(e, color="lime", linewidth=0.5, alpha=0.7)
        ax_bin.set_title(f"{len(blobs)} blobs", fontsize=8)
        ax_bin.axis("off")

        # ── NCC per digit ──────────────────────────────────────────────────
        ax_ncc = axes[row, 2]
        ax_ncc.axis("off")

        if len(blobs) == 7:
            ocr_str = ""
            lines = []
            for pos, (s, e) in enumerate(blobs):
                pad = 4
                crop_d = binary[:, max(0, s-pad): e+pad+1]
                if crop_d.shape[1] < 3:
                    ocr_str += "?"
                    lines.append(f"  pos{pos}: ?")
                    continue
                query = np.array(Image.fromarray(
                    (crop_d * 255).astype(np.uint8)).resize(
                    (TMPL_W, TMPL_H), Image.LANCZOS),
                    dtype=np.float32) / 255.0
                scores = {d: ncc_match(query, t)
                          for d, t in templates.items()}
                best = max(scores, key=scores.get)
                ocr_str += best
                # Top 2 scores
                sorted_s = sorted(scores.items(), key=lambda x: -x[1])
                top2 = (f"{sorted_s[0][0]}={sorted_s[0][1]:.2f}  "
                        f"{sorted_s[1][0]}={sorted_s[1][1]:.2f}")
                true_d = true_str[pos] if pos < len(true_str) else "?"
                marker = "OK" if best == true_d else "MISS"
                lines.append(f"  p{pos}: {top2}  ({marker})")

            header = f"OCR: {ocr_str}\nTrue: {true_str}"
            match = "MATCH" if ocr_str == true_str else "MISMATCH"
            text = header + f"  [{match}]\n" + "\n".join(lines)
        else:
            text = f"blob count = {len(blobs)} (expected 7)"

        ax_ncc.text(0.02, 0.95, text, transform=ax_ncc.transAxes,
                    fontsize=6.5, fontfamily="monospace",
                    verticalalignment="top")

    fig.suptitle(
        f"TIFF {tiff_id}  —  OCR diagnostic  |  "
        f"True CBDs from frame_index",
        fontsize=11, fontweight="bold", y=1.0)
    fig.tight_layout()

    out_path = OUT_DIR / f"diag_ocr_{tiff_id}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {out_path.relative_to(ROOT)}")


# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tiff_ids = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_TIFFS

    print(f"Loading templates from {TMPL_PATH.name} ...")
    templates = np.load(TMPL_PATH, allow_pickle=True).item()
    print(f"  Digits: {sorted(templates.keys())}")

    gt = load_ground_truth()

    for tid in tiff_ids:
        # Find TIFF file
        matches = list(RAW_DIR.glob(f"*_{tid:>07s}_*-reel_begin_end.tiff"))
        if not matches:
            matches = list(RAW_DIR.glob(f"*_{tid}_*"))
        if not matches:
            print(f"\n  WARNING: no TIFF found for ID {tid} — skipping")
            continue
        tiff_path = matches[0]
        gt_cbds = gt.get(tid, {})
        if not gt_cbds:
            print(f"\n  WARNING: no ground truth CBDs for TIFF {tid}"
                  " — skipping")
            continue
        make_diag_figure(tiff_path, tid, gt_cbds, templates)

    print("\nDone.")
