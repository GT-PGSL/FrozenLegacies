"""
pick_calibration.py — LYRA interactive calibration picker (multi-frame)
========================================================================
For each complete frame in a TIFF, collect guidance clicks for:

  M — Main bang              (1 click  → red   vertical  line)
  R — Reference −60 dB      (1 click  → cyan  horizontal line)
  Y — Y-axis major grid line (2 clicks → lime  horizontal lines)
  X — X-axis major grid line (2 clicks → orange vertical   lines)

These clicks are GUIDANCE ONLY — approximate hints ±tens of pixels are fine.
The algorithm will snap to the true feature near each guide pick.

Keyboard shortcuts (case-insensitive):
  M  — switch to Main Bang mode
  R  — switch to Reference line −60 dB mode
  Y  — switch to Y-grid mode     (click two visible major lines)
  X  — switch to X-grid mode     (click two visible major lines)
  U  — undo last pick in the current mode
  N  — save picks for this frame and move to next
  S  — skip this frame (no picks saved)
  Q  — quit and save all collected picks so far

Usage
-----
Run from repo root:
    python tools/LYRA/pick_calibration.py [TIFF_PATH]

If TIFF_PATH is omitted, defaults to the F125 training TIFF.

Outputs
-------
  tools/LYRA/output/F{FLT}/F{FLT}_cal_picks.json
      Raw guidance picks per frame (mb_x, y_ref, y_grid x2, x_grid x2).

  tools/LYRA/output/F{FLT}/F{FLT}_cal_derived.json
      Derived constants: x_spacing_px, y_spacing_px, us_per_px, db_per_px,
      plus per-frame mb_guides and ref_guides for step2_calibrate.py.

Notes
-----
- Picks are saved incrementally: re-running resumes where you left off.
- The faint gray dashed lines are DEFAULT_CAL predictions (y_ref=1507,
  spacing=205 px) shown to help orient you.  They will be replaced by
  algorithm-derived positions in step2_calibrate.py.
"""

from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("QtAgg")   # QtAgg is more reliable than MacOSX for event handling on Apple Silicon
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/LYRA"))

from lyra import ensure_canonical_name, tiff_id

# ── Resolve TIFF path ──────────────────────────────────────────────────────────
if len(sys.argv) > 1:
    TIFF = Path(sys.argv[1])
    if not TIFF.is_absolute():
        TIFF = ROOT / TIFF
else:
    TIFF = ROOT / "Data/ascope/raw/125/40_0008400_0008424-reel_begin_end.tiff"

# Rename to canonical format if needed (e.g. F141 files renamed from reel convention)
TIFF = ensure_canonical_name(TIFF)

try:
    FLT = int(TIFF.parent.name)
except ValueError:
    FLT = 0


TIFF_ID    = tiff_id(TIFF)
OUT_DIR    = ROOT / f"tools/LYRA/output/F{FLT}"
STEP1_DIR  = OUT_DIR / "step1"
STEP1_DIR.mkdir(parents=True, exist_ok=True)
INDEX_CSV  = STEP1_DIR / f"F{FLT}_frame_index.csv"
PICKS_JSON = STEP1_DIR / f"F{FLT}_cal_picks.json"

# Display crop: show full display band plus context
Y_CROP_LO = 100    # top    (avoid sprocket holes)
Y_CROP_HI = 1950   # bottom (below −60 dB reference)

# Default calibration constants (for drawing faint guide lines)
_DB_REF_PX  = 1507    # default y_ref (−60 dB reference)
_PX_PER_DIV_Y = 205.0   # default Y spacing (10 dB / div)
_PX_PER_DIV_X = 205.54  # default X spacing (1.5 µs / div)

MODE_COLORS = {
    "mb":     "red",
    "ref":    "cyan",
    "y_grid": "limegreen",
    "x_grid": "orange",
}
MODE_LABELS = {
    "mb":     "Main Bang  [M] — 1 click (red)",
    "ref":    "Reference −60 dB  [R] — 1 click (cyan)",
    "y_grid": "Y-grid major line  [Y] — 2 clicks (lime)",
    "x_grid": "X-grid major line  [X] — 2 clicks (orange)",
}

# ── Load frame index ───────────────────────────────────────────────────────────
if not INDEX_CSV.exists():
    sys.exit(f"ERROR: frame index not found: {INDEX_CSV}\n"
             "Run step1_detect_frames.py first.")

index     = pd.read_csv(INDEX_CSV, dtype=str)
tiff_rows = index[(index["tiff"] == TIFF.name) &
                  (index["frame_type"] == "complete")].copy()

if len(tiff_rows) == 0:
    sys.exit(f"ERROR: no complete frames for {TIFF.name} in frame index.")

# ── Load image ─────────────────────────────────────────────────────────────────
print(f"\nLYRA Calibration Picker — {TIFF.name}")
print(f"  Flight F{FLT}  |  TIFF_ID {TIFF_ID}  |  Complete frames: {len(tiff_rows)}")
print(f"  Loading TIFF …")
Image.MAX_IMAGE_PIXELS = None
img      = np.array(Image.open(TIFF), dtype=np.float32)
img_norm = (img - img.min()) / (img.max() - img.min() + 1e-9)
H, W     = img_norm.shape
print(f"  Image: {W} × {H} px\n")
print(__doc__.split("Notes")[0])  # print the usage section

# ── Load existing picks (resume support) ──────────────────────────────────────
all_picks: dict = {}
if PICKS_JSON.exists():
    with open(PICKS_JSON) as f:
        all_picks = json.load(f)
    print(f"  Loaded {len(all_picks)} existing picks from {PICKS_JSON.name}")

# ── Interactive loop ───────────────────────────────────────────────────────────
def _row_fkey(row) -> str:
    cbd = str(getattr(row, "cbd", "") or "")
    return f"CBD{cbd}" if cbd else f"fr{row.frame_idx}"

_all_rows  = list(tiff_rows.itertuples())
frame_list = [r for r in _all_rows
              if not all_picks.get(_row_fkey(r), {}).get("exclude")]
n_excl     = len(_all_rows) - len(frame_list)
if n_excl:
    excl_keys = [_row_fkey(r) for r in _all_rows
                 if all_picks.get(_row_fkey(r), {}).get("exclude")]
    print(f"  Skipping {n_excl} excluded frame(s): {excl_keys}")
    print(f"  (To un-exclude, remove the 'exclude' key from cal_picks.json.)")
frame_i    = 0

while frame_i < len(frame_list):
    row      = frame_list[frame_i]
    cbd      = str(getattr(row, "cbd", "") or "")
    fkey     = f"CBD{cbd}" if cbd else f"fr{row.frame_idx}"
    left_px  = int(row.left_px)
    right_px = int(row.right_px)
    frame_w  = right_px - left_px + 1

    print(f"\n{'─'*60}")
    print(f"  [{frame_i+1}/{len(frame_list)}]  {fkey}  "
          f"cols {left_px}–{right_px}  (width {frame_w} px)")
    print(f"  M=main bang  R=ref −60dB  Y=y-grid  X=x-grid  "
          f"U=undo  N=next  S=skip  Q=quit")

    # Extract + display-stretch this frame
    frame    = img_norm[:, left_px : right_px + 1]
    crop     = frame[Y_CROP_LO : min(Y_CROP_HI, H), :]
    p2, p98  = np.percentile(crop, 2), np.percentile(crop, 98)
    disp     = np.clip((crop - p2) / (p98 - p2 + 1e-9), 0, 1)

    # Per-frame mutable state (dict avoids nonlocal issues in closures)
    init = all_picks.get(fkey, {})
    st = {
        "mode":   "mb",
        "picks":  {
            "mb":     init.get("mb"),
            "ref":    init.get("ref"),
            "y_grid": list(init.get("y_grid", [])),
            "x_grid": list(init.get("x_grid", [])),
        },
        "action": None,
        "drawn":  [],
    }

    # ── Build figure ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 8))
    fig.patch.set_facecolor("white")

    # Image displayed with y-axis in image coordinates (y increases downward)
    ax.imshow(disp, cmap="gray", vmin=0, vmax=1, aspect="auto",
              extent=[0, frame_w, Y_CROP_HI, Y_CROP_LO])
    ax.set_ylim(Y_CROP_HI, Y_CROP_LO)   # y increases downward
    ax.set_xlabel("X pixel (frame-relative)", fontsize=9)
    ax.set_ylabel("Y pixel (image)", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Faint default-cal guide lines (gray dashed) to help orient
    for k in range(-3, 7):
        y_guide = _DB_REF_PX - k * _PX_PER_DIV_Y
        if Y_CROP_LO < y_guide < Y_CROP_HI:
            ax.axhline(y_guide, color="0.65", lw=0.5, ls="--", alpha=0.5)
            db_val = -60 + k * 10
            ax.text(frame_w - 8, y_guide - 10, f"{db_val:+d} dB",
                    color="0.55", fontsize=6, ha="right", va="bottom")

    # Status bar at bottom of figure
    status_bar = fig.text(
        0.5, 0.005, "", ha="center", fontsize=10,
        bbox=dict(boxstyle="round", fc="wheat", alpha=0.85))

    def _update_status():
        """Update status bar text (no draw — caller must call _refresh)."""
        m   = st["mode"]
        p   = st["picks"]
        col = MODE_COLORS[m]
        lbl = MODE_LABELS[m]
        status_bar.set_text(
            f"[{frame_i+1}/{len(frame_list)}] {fkey}  |  {lbl}    "
            f"mb={1 if p['mb'] is not None else 0}  "
            f"ref={1 if p['ref'] is not None else 0}  "
            f"y_grid={len(p['y_grid'])}/2  x_grid={len(p['x_grid'])}/2"
        )
        status_bar.set_color(col)

    def _redraw():
        """Update drawn artists (no draw — caller must call _refresh)."""
        for art in st["drawn"]:
            try:
                art.remove()
            except Exception:
                pass
        st["drawn"].clear()
        p = st["picks"]

        if p["mb"] is not None:
            x = p["mb"]
            st["drawn"] += [
                ax.axvline(x, color="red", lw=2.0, ls="--", alpha=0.9),
                ax.text(x + 8, Y_CROP_LO + 60, f"MB\nx={x}",
                        color="red", fontsize=8, fontweight="bold", va="top"),
            ]

        if p["ref"] is not None:
            y = p["ref"]
            st["drawn"] += [
                ax.axhline(y, color="cyan", lw=2.0, ls="--", alpha=0.9),
                ax.text(10, y - 18, f"−60 dB  y={y}",
                        color="cyan", fontsize=8, fontweight="bold"),
            ]

        for y in p["y_grid"]:
            st["drawn"] += [
                ax.axhline(y, color="limegreen", lw=1.5, ls=":", alpha=0.9),
                ax.text(10, y - 14, f"Ygrid y={y}",
                        color="limegreen", fontsize=7),
            ]

        for x in p["x_grid"]:
            st["drawn"] += [
                ax.axvline(x, color="orange", lw=1.5, ls=":", alpha=0.9),
                ax.text(x + 5, Y_CROP_LO + 130, f"Xgrid\nx={x}",
                        color="orange", fontsize=7, va="top", rotation=90),
            ]

        ax.set_title(
            f"LYRA Calibration Picker — F{FLT}  {fkey}  "
            f"[{frame_i+1}/{len(frame_list)}]\n"
            f"Gray dashed = default-cal prediction  |  "
            f"M=main bang  R=ref−60dB  Y=y-grid  X=x-grid  "
            f"U=undo  N=next  S=skip  Q=quit",
            fontsize=8, loc="left",
        )

    def _refresh():
        """Single draw + flush — call once after all artist/text updates."""
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

    # ── Event handlers ────────────────────────────────────────────────────────
    def on_click(event):
        if event.inaxes != ax or event.button != 1:
            return
        x = int(round(event.xdata))
        y = int(round(event.ydata))
        p = st["picks"]
        m = st["mode"]

        if m == "mb":
            p["mb"] = x
            print(f"    MB:     x={x} px (frame-relative)")
        elif m == "ref":
            p["ref"] = y
            print(f"    Ref:    y={y} px  ({(-60 + (_DB_REF_PX - y) * 10.0/_PX_PER_DIV_Y):+.1f} dB by default cal)")
        elif m == "y_grid":
            if len(p["y_grid"]) < 2:
                p["y_grid"].append(y)
                p["y_grid"].sort()
                db_approx = -60 + (_DB_REF_PX - y) * 10.0 / _PX_PER_DIV_Y
                print(f"    Y-grid: y={y} px  (~{db_approx:+.1f} dB by default cal)  "
                      f"total {len(p['y_grid'])}/2")
            else:
                print(f"    Y-grid: already has 2 picks — press U to undo, or switch mode")
        elif m == "x_grid":
            if len(p["x_grid"]) < 2:
                p["x_grid"].append(x)
                p["x_grid"].sort()
                us_approx = (x - (p["mb"] or 800)) * 1.5 / _PX_PER_DIV_X
                print(f"    X-grid: x={x} px  (~{us_approx:+.2f} µs from MB guess)  "
                      f"total {len(p['x_grid'])}/2")
            else:
                print(f"    X-grid: already has 2 picks — press U to undo, or switch mode")

        _redraw()
        _update_status()
        _refresh()

    def on_key(event):
        key = (event.key or "").lower()
        p = st["picks"]

        if key == "m":
            st["mode"] = "mb"
            print(">> Mode: Main Bang  (click near the leftmost spike)")
        elif key == "r":
            st["mode"] = "ref"
            print(">> Mode: Reference line −60 dB  (click the bottom baseline)")
        elif key == "y":
            st["mode"] = "y_grid"
            p["y_grid"].clear()  # reset so user can re-pick both lines
            print(">> Mode: Y-grid  (click 2 horizontal major lines)")
        elif key == "x":
            st["mode"] = "x_grid"
            p["x_grid"].clear()  # reset so user can re-pick both lines
            print(">> Mode: X-grid  (click 2 vertical major lines)")
        elif key == "u":
            m = st["mode"]
            if m == "mb" and p["mb"] is not None:
                p["mb"] = None
                print("  Undo: MB cleared")
            elif m == "ref" and p["ref"] is not None:
                p["ref"] = None
                print("  Undo: ref cleared")
            elif m == "y_grid" and p["y_grid"]:
                v = p["y_grid"].pop()
                print(f"  Undo: y_grid y={v} removed")
            elif m == "x_grid" and p["x_grid"]:
                v = p["x_grid"].pop()
                print(f"  Undo: x_grid x={v} removed")
        elif key == "n":
            st["action"] = "next"
            plt.close(fig)
            return
        elif key == "s":
            st["action"] = "skip"
            plt.close(fig)
            return
        elif key == "q":
            st["action"] = "quit"
            plt.close(fig)
            return

        _redraw()
        _update_status()
        _refresh()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.subplots_adjust(bottom=0.06, top=0.92)
    _redraw()
    _update_status()
    _refresh()
    plt.show()   # blocks until window is closed or plt.close() is called

    # ── Post-window: process action ───────────────────────────────────────────
    p = st["picks"]

    def _save_picks(fkey, p):
        """Merge new picks with existing entry, preserving exclude/exclude_reason."""
        existing = all_picks.get(fkey, {})
        p_save   = dict(p)
        for k in ("exclude", "exclude_reason"):
            if k in existing:
                p_save[k] = existing[k]
        all_picks[fkey] = p_save

    if st["action"] == "quit":
        _save_picks(fkey, p)
        print(f"\n  Quitting — picks for {fkey} saved.")
        break
    elif st["action"] == "skip":
        print(f"  Skipped {fkey} (no picks saved).")
        frame_i += 1
        continue
    else:  # "next" or window closed without key
        _save_picks(fkey, p)
        print(f"  Saved {fkey}:  "
              f"mb={p['mb']}  ref={p['ref']}  "
              f"y_grid={p['y_grid']}  x_grid={p['x_grid']}")
        frame_i += 1

# ── Save raw picks ─────────────────────────────────────────────────────────────
with open(PICKS_JSON, "w") as f:
    json.dump(all_picks, f, indent=2)
print(f"\nRaw picks saved → {PICKS_JSON.relative_to(ROOT)}")
print(f"Total frames with picks: {len(all_picks)}")

# ── Derive spacing constants from picks ───────────────────────────────────────
# For each frame with 2 X or 2 Y picks, estimate the major-division spacing.
# If user picked non-adjacent lines, infer n_divs from the approximate spacing.
_APPROX_X = _PX_PER_DIV_X
_APPROX_Y = _PX_PER_DIV_Y

x_spacings: list[float] = []
y_spacings: list[float] = []

for fkey, p in all_picks.items():
    if p.get("exclude"):   # skip excluded frames from spacing derivation
        continue
    xg = sorted(p.get("x_grid", []))
    if len(xg) == 2:
        raw   = abs(xg[1] - xg[0])
        n_div = max(1, round(raw / _APPROX_X))
        x_spacings.append(raw / n_div)
        print(f"  {fkey}: X picks={xg}  raw_diff={raw:.0f} px  "
              f"n_div={n_div}  spacing={raw/n_div:.1f} px")

    yg = sorted(p.get("y_grid", []))
    if len(yg) == 2:
        raw   = abs(yg[1] - yg[0])
        n_div = max(1, round(raw / _APPROX_Y))
        y_spacings.append(raw / n_div)
        print(f"  {fkey}: Y picks={yg}  raw_diff={raw:.0f} px  "
              f"n_div={n_div}  spacing={raw/n_div:.1f} px")

x_spacing_px = float(np.median(x_spacings)) if x_spacings else _APPROX_X
y_spacing_px = float(np.median(y_spacings)) if y_spacings else _APPROX_Y

mb_guides  = {k: v["mb"]  for k, v in all_picks.items()
              if v.get("mb")  is not None and not v.get("exclude")}
ref_guides = {k: v["ref"] for k, v in all_picks.items()
              if v.get("ref") is not None}

derived = {
    "tiff":              TIFF.name,
    "flight":            FLT,
    "n_frames_picked":   len(all_picks),
    "x_spacing_px":      round(x_spacing_px, 3),
    "y_spacing_px":      round(y_spacing_px, 3),
    "us_per_px":         round(1.5  / x_spacing_px, 8),
    "db_per_px":         round(10.0 / y_spacing_px, 8),
    "x_spacings_per_frame": {
        k: round(abs(sorted(v.get("x_grid",[]))[1] - sorted(v.get("x_grid",[]))[0]) /
                 max(1, round(abs(sorted(v.get("x_grid",[]))[1] - sorted(v.get("x_grid",[]))[0]) / _APPROX_X)), 2)
        for k, v in all_picks.items()
        if len(v.get("x_grid", [])) == 2
    },
    "y_spacings_per_frame": {
        k: round(abs(sorted(v.get("y_grid",[]))[1] - sorted(v.get("y_grid",[]))[0]) /
                 max(1, round(abs(sorted(v.get("y_grid",[]))[1] - sorted(v.get("y_grid",[]))[0]) / _APPROX_Y)), 2)
        for k, v in all_picks.items()
        if len(v.get("y_grid", [])) == 2
    },
    "mb_guides":  mb_guides,
    "ref_guides": ref_guides,
}

DERIVED_JSON = STEP1_DIR / f"F{FLT}_cal_derived.json"
with open(DERIVED_JSON, "w") as f:
    json.dump(derived, f, indent=2)

print(f"\nDerived calibration → {DERIVED_JSON.relative_to(ROOT)}")
print(f"  x_spacing_px   : {x_spacing_px:.2f} px  → {derived['us_per_px']:.6f} µs/px")
print(f"  y_spacing_px   : {y_spacing_px:.2f} px  → {derived['db_per_px']:.6f} dB/px")
print(f"  n frames w/ MB : {len(mb_guides)}")
print(f"  n frames w/ ref: {len(ref_guides)}")
if x_spacings:
    print(f"  X spacings     : {[round(v,1) for v in x_spacings]}")
if y_spacings:
    print(f"  Y spacings     : {[round(v,1) for v in y_spacings]}")
print(f"\nNext: run step2_calibrate.py (it will load these picks automatically).")
