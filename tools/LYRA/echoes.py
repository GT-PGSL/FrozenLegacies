"""
echoes.py — LYRA Phase 4: Surface/bed echo extraction
============================================================
For each calibrated frame (from phase3 cal CSV) extract surface and bed echoes:

  • Noise floor estimation (pre-bang 75th-percentile baseline)
  • Echo peak detection  (prominence >= 5 dB, distance >= 80 px)
  • Envelope walking at +5 dB and +10 dB above noise floor
  • Waveform shape metrics: peakiness, asymmetry, trailing_tail, leading_rise
  • Derived geometry:
      h_air_m  = surface_twt / 2 × c_air   (air column height)
      h_ice_m  = (bed_twt - surface_twt) / 2 × c_ice   (ice thickness)
      h_eff_m  = h_air_m + h_ice_m / n_ice   (effective propagation distance)

Echo status:
  good       — surface + bed detected, bed SNR >= WEAK_BED_SNR_DB (5 dB)
  weak_bed   — bed detected but SNR < WEAK_BED_SNR_DB (marginal; inspect figure)
  no_bed     — surface detected, no bed echo found
  no_surface — surface not detected; geometry undefined

Per-frame calibration is taken directly from the phase3 cal CSV (mb_x, y_ref_px,
db_per_px, us_per_px).  Detection parameters (prominence, distance, Gaussian sigma)
are taken from DEFAULT_CAL.

Usage
-----
Normal mode (extract echoes):
    python tools/LYRA/echoes.py [TIFF_PATH]

Review mode (interactive override — click on image panel):
    python tools/LYRA/echoes.py TIFF --review          # bad frames only
    python tools/LYRA/echoes.py TIFF --review --all     # all frames

If TIFF_PATH is omitted, defaults to the F125 training TIFF (40_0008400…).

Outputs
-------
Per-flight echo CSV (updated incrementally):
    tools/LYRA/output/F{FLT}/phase4/F{FLT}_echoes.csv
      Columns: flight, tiff, cbd, file_id, echo_status,
               noise_floor_dB,
               surface_twt_us, surface_power_dB, surface_snr_dB,
               surface_width_10_us, surface_width_5_us, surface_peakiness,
               surface_asymmetry, surface_leading_rise_us, surface_trailing_tail_us,
               bed_twt_us, bed_power_dB, bed_snr_dB,
               bed_width_10_us, bed_width_5_us, bed_peakiness,
               bed_asymmetry, bed_leading_rise_us, bed_trailing_tail_us,
               h_air_m, h_ice_m, h_eff_m

Per-frame diagnostic figure (two-panel):
    tools/LYRA/output/F{FLT}/phase4/F{FLT}_{file_id}_echoes.png
      Left  : frame image with echo markers (pixel space, same scale as phase3)
      Right : waveform in physical units (TWT µs vs power dB)
"""

from pathlib import Path
import sys
import json
import argparse

import numpy as np
import pandas as pd

# -- Parse arguments before matplotlib (backend depends on --review) -----------
_parser = argparse.ArgumentParser(
    description="LYRA Phase 4: Surface/bed echo extraction",
    epilog="Review mode: --review shows bad frames for manual echo override",
)
_parser.add_argument("tiff", nargs="?", help="TIFF path or FLT/TIFF_ID format")
_parser.add_argument("--review", action="store_true",
                     help="Interactive review: click on image to override surface/bed picks")
_parser.add_argument("--all", action="store_true",
                     help="With --review: show all frames (default: only no_bed/no_surface/weak_bed)")
_args = _parser.parse_args()

import matplotlib
if not _args.review:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/LYRA"))

from lyra import (
    DEFAULT_CAL,
    extract_trace, estimate_noise_floor,
    detect_echoes, compute_echo_metrics,
    px_to_db, px_to_us,
    C_AIR_M_PER_US, C_ICE_M_PER_US,
    ensure_canonical_name, resolve_tiff_arg, tiff_id,
)

# -- Scientific constants -------------------------------------------------------
N_ICE           = 1.78   # refractive index of ice (Neal 1977)
WEAK_BED_SNR_DB = 5.0    # bed SNR threshold below which status = "weak_bed"

# -- Resolve TIFF path ----------------------------------------------------------
if _args.tiff:
    TIFF = resolve_tiff_arg(_args.tiff, ROOT)
else:
    TIFF = ROOT / "Data/ascope/raw/125/40_0008400_0008424-reel_begin_end.tiff"

TIFF = ensure_canonical_name(TIFF)

try:
    FLT = int(TIFF.parent.name.lstrip("Ff"))
except ValueError:
    FLT = 0


TIFF_ID   = tiff_id(TIFF)
OUT_DIR   = ROOT / f"tools/LYRA/output/F{FLT}"
PHASE1_DIR = OUT_DIR / "phase1"
PHASE3_DIR = OUT_DIR / "phase3"
PHASE4_DIR = OUT_DIR / "phase4"
PHASE4_DIR.mkdir(parents=True, exist_ok=True)

INDEX_CSV  = PHASE1_DIR / f"F{FLT}_frame_index.csv"
CAL_CSV    = PHASE3_DIR / f"F{FLT}_cal.csv"
ECHO_CSV   = PHASE4_DIR / f"F{FLT}_echoes.csv"

# -- Validate prerequisites -----------------------------------------------------
if not INDEX_CSV.exists():
    sys.exit(f"ERROR: frame index not found at {INDEX_CSV}\n"
             "Run phase 1 (detect_frames.py) first.")

if not CAL_CSV.exists():
    sys.exit(f"ERROR: phase 3 calibration CSV not found at {CAL_CSV}\n"
             "Run phase 3 (calibrate.py) first.")

# -- Load phase1 frame index ---------------------------------------------------
index     = pd.read_csv(INDEX_CSV, dtype=str)
tiff_rows = index[index["tiff"] == TIFF.name].copy()

if len(tiff_rows) == 0:
    sys.exit(f"ERROR: {TIFF.name} not found in frame index.\n"
             "Run phase 1 (detect_frames.py) for this TIFF first.")

# -- Load phase3 calibration CSV ------------------------------------------------
cal_df    = pd.read_csv(CAL_CSV, dtype=str)
tiff_cals = cal_df[cal_df["tiff"] == TIFF.name].copy()

if len(tiff_cals) == 0:
    sys.exit(f"ERROR: No phase 3 calibration rows for {TIFF.name} in {CAL_CSV}\n"
             "Run phase 3 (calibrate.py) for this TIFF first.")

# Build frame_idx -> calibration dict mapping (skip excluded frames)
frame_cal: dict[int, dict] = {}
for _, row in tiff_cals.iterrows():
    # Skip excluded frames (empty mb_x means no calibration was performed)
    if row.get("exclude_reason", "") not in ("", "nan") and str(row.get("exclude_reason", "")) not in ("", "nan"):
        print(f"  Skipping frame {row['frame_idx']} ({row.get('cbd','?')}): {row['exclude_reason']}")
        continue
    if not row["mb_x"] or str(row["mb_x"]).strip() == "":
        continue
    fidx = int(row["frame_idx"])
    frame_cal[fidx] = dict(
        mb_x        = int(float(row["mb_x"])),
        y_ref_px    = float(row["y_ref_px"]),
        db_per_px   = float(row["db_per_px"]),
        us_per_px   = float(row["us_per_px"]),
        cbd         = row["cbd"],
        file_id     = row["file_id"],
        tiff_id     = row.get("tiff_id", TIFF_ID),
    )

print(f"\nLYRA Step 3 — {TIFF.name}")
print(f"  Flight : F{FLT}  |  tiff_id : {TIFF_ID}")
print(f"  Calibrated frames in step2 CSV : {len(frame_cal)}\n")

# -- Load TIFF image ------------------------------------------------------------
print("Loading TIFF ...")
Image.MAX_IMAGE_PIXELS = None
img      = np.array(Image.open(TIFF), dtype=np.float32)
img_norm = (img - img.min()) / (img.max() - img.min() + 1e-9)
H, W_img = img_norm.shape
print(f"  Image  : {W_img} × {H} px\n")

# -- Build per-frame cal dict from step2 values + DEFAULT_CAL detection params -
def _make_cal(fc: dict) -> dict:
    """Merge per-frame step2 calibration values into DEFAULT_CAL."""
    cal = DEFAULT_CAL.copy()
    cal["mb_x"]     = fc["mb_x"]     # already in frame, but passed separately
    cal["y_ref_px"] = fc["y_ref_px"]
    cal["db_per_px"]= fc["db_per_px"]
    cal["us_per_px"]= fc["us_per_px"]
    # y_ref_db stays at -60 dB (reference is always picked at -60 dB line)
    return cal


# -- Echo overrides ------------------------------------------------------------
OVERRIDES_JSON = PHASE4_DIR / f"F{FLT}_echo_overrides.json"


def load_echo_overrides() -> dict:
    """Load manual echo overrides from JSON. Keys are file_id (e.g. 'CBD0465')."""
    if OVERRIDES_JSON.exists():
        with open(OVERRIDES_JSON) as f:
            return json.load(f)
    return {}


def save_echo_overrides(overrides: dict) -> None:
    """Write echo overrides to JSON."""
    with open(OVERRIDES_JSON, "w") as f:
        json.dump(overrides, f, indent=2)
    print(f"  Overrides -> {OVERRIDES_JSON.relative_to(ROOT)}  ({len(overrides)} frames)")


echo_overrides = load_echo_overrides()
if echo_overrides:
    print(f"  Echo overrides loaded: {len(echo_overrides)} frames  ({OVERRIDES_JSON.name})")


# ==============================================================================
# Review mode — interactive echo override picker
# ==============================================================================

def review_mode():
    """Interactive review: click on image panel to override surface/bed echo positions.

    Shows ALL calibrated frames in the TIFF. Users press [s] for surface mode,
    [b] for bed mode, then click on the left (image) panel to place the pick
    directly on the CRT trace overlay.
    """

    # Need existing echo CSV to know current statuses
    if not ECHO_CSV.exists():
        sys.exit("ERROR: No echo CSV found. Run normal mode first, then --review.")

    echo_df = pd.read_csv(ECHO_CSV)
    tiff_echos = echo_df[echo_df["tiff"] == TIFF.name].copy()

    if len(tiff_echos) == 0:
        sys.exit(f"ERROR: No echo rows for {TIFF.name}. Run normal mode first.")

    # Build frame queue — default: only bad frames + frames with overrides
    # Use --all to include good frames too
    BAD_STATUSES = {"no_bed", "no_surface", "weak_bed"}
    show_all = getattr(_args, "all", False)

    review_frames = []
    all_status_counts = {}

    for _, idx_row in tiff_rows.iterrows():
        frame_idx = int(idx_row["frame_idx"])
        if idx_row["frame_type"] == "partial":
            continue
        if frame_idx not in frame_cal:
            continue

        fc = frame_cal[frame_idx]
        fkey = fc["file_id"]

        erow = tiff_echos[tiff_echos["file_id"] == fkey]
        current_status = erow.iloc[0]["echo_status"] if len(erow) > 0 else "unknown"
        all_status_counts[current_status] = all_status_counts.get(current_status, 0) + 1

        if show_all or current_status in BAD_STATUSES:
            review_frames.append((frame_idx, fc, current_status, idx_row))

    # Print TIFF summary (all frames, not just filtered)
    total = sum(all_status_counts.values())
    print(f"\nTIFF {TIFF_ID}: {total} frames total")
    for s_name in ["good", "weak_bed", "no_bed", "no_surface"]:
        if s_name in all_status_counts:
            print(f"    {s_name:12s}: {all_status_counts[s_name]}")

    if not review_frames:
        print("No frames to review (all good). Use --all to review every frame.")
        return

    filter_label = "all frames" if show_all else "bad + overridden frames only"
    print(f"\nReview: {len(review_frames)} frames ({filter_label})")
    print(f"  Click on LEFT panel (image) to place pick on CRT trace")
    print(f"  Keys: [s]=surface  [b]=bed  click=set position")
    print(f"        [d]=clear overrides  [n/Right]=next  [p/Left]=prev  [q]=save & quit\n")

    frame_i = 0
    while 0 <= frame_i < len(review_frames):
        frame_idx, fc, current_status, idx_row = review_frames[frame_i]
        fkey = fc["file_id"]
        cbd = fc["cbd"]
        mb_x = fc["mb_x"]

        left_px = int(idx_row["left_px"])
        right_px = int(idx_row["right_px"])
        frame = img_norm[:, left_px:right_px + 1]
        frame_w = frame.shape[1]
        cal = _make_cal(fc)

        # Extract trace (fast, pure numpy)
        trace_y, trace_y_s, _ = extract_trace(frame, cal, robust=True, mb_x=mb_x)
        noise_floor_dB = estimate_noise_floor(trace_y_s, mb_x, cal)

        # Algorithmic detection (for reference)
        algo_surface_x, algo_bed_x = detect_echoes(trace_y_s, mb_x, noise_floor_dB, cal)

        # Current override (mutable copy)
        override = dict(echo_overrides.get(fkey, {}))

        # Mutable state for event handlers
        st = {"mode": "b", "override": override, "action": None, "drawn": []}

        us_per_px = cal["us_per_px"]

        # -- Build interactive figure ------------------------------------------
        fig = plt.figure(figsize=(18, 6), constrained_layout=True)
        fig.patch.set_facecolor("white")
        gs = fig.add_gridspec(1, 3, wspace=0.08)
        ax_img = fig.add_subplot(gs[0, :2])
        ax_wave = fig.add_subplot(gs[0, 2])

        # -- Left panel: frame image (click target) ---------------------------
        ax_img.imshow(frame, cmap="gray", vmin=0, vmax=1, aspect="auto",
                      extent=[0, frame_w, frame.shape[0], 0])
        ax_img.axhline(cal["y_ref_px"], color="cyan", lw=1.2, ls="--", alpha=0.8)
        ax_img.axvline(mb_x, color="red", lw=1.5, ls="--", alpha=0.9)
        x_all = np.arange(frame_w)
        ax_img.plot(x_all, trace_y_s, color="magenta", lw=0.8, alpha=0.7, label="trace")

        # Algo markers (faded reference)
        if algo_surface_x is not None:
            algo_srf_y = trace_y_s[algo_surface_x] if algo_surface_x < len(trace_y_s) else 0
            ax_img.plot(algo_surface_x, algo_srf_y, marker="^", color="lime",
                        markersize=7, alpha=0.35, zorder=4, label="algo srf")
            ax_img.axvline(algo_surface_x, color="lime", lw=0.5, ls=":", alpha=0.35)
        if algo_bed_x is not None:
            algo_bed_y = trace_y_s[algo_bed_x] if algo_bed_x < len(trace_y_s) else 0
            ax_img.plot(algo_bed_x, algo_bed_y, marker="^", color="orange",
                        markersize=7, alpha=0.35, zorder=4, label="algo bed")
            ax_img.axvline(algo_bed_x, color="orange", lw=0.5, ls=":", alpha=0.35)

        ax_img.set_ylim(DEFAULT_CAL["y_disp_hi"] + 50, DEFAULT_CAL["y_disp_lo"] - 50)
        ax_img.set_xlim(0, frame_w)
        ax_img.set_xlabel("Column (px) — click here to place pick", fontsize=8)
        ax_img.set_ylabel("Row (px)", fontsize=8)
        ax_img.spines["top"].set_visible(False)
        ax_img.spines["right"].set_visible(False)
        ax_img.legend(fontsize=7, loc="upper right", framealpha=0.6,
                      facecolor="black", labelcolor="white")

        # -- Right panel: waveform (reference only) ----------------------------
        twt_arr = px_to_us(np.arange(frame_w, dtype=float), mb_x, cal)
        power_arr = px_to_db(trace_y_s, cal)
        t_lo = float(cal["mb_skip_us"])
        t_hi = 25.0
        wf_mask = (twt_arr >= t_lo) & (twt_arr <= t_hi)

        ax_wave.plot(twt_arr[wf_mask], power_arr[wf_mask], color="black", lw=1.2)
        ax_wave.axhline(noise_floor_dB, color="gray", lw=1.0, ls="--", alpha=0.8,
                        label=f"NF {noise_floor_dB:.1f} dB")
        ax_wave.axhline(noise_floor_dB + 5.0, color="silver", lw=0.7, ls=":", alpha=0.8)
        ax_wave.axhline(noise_floor_dB + 10.0, color="darkgray", lw=0.7, ls=":", alpha=0.8)

        # Algo surface/bed on waveform (faded)
        if algo_surface_x is not None:
            algo_srf_twt = px_to_us(float(algo_surface_x), mb_x, cal)
            algo_srf_pwr = px_to_db(trace_y_s[algo_surface_x], cal)
            ax_wave.plot(algo_srf_twt, algo_srf_pwr, marker="^", color="limegreen",
                         markersize=7, alpha=0.35, zorder=4)
            ax_wave.axvline(algo_srf_twt, color="limegreen", lw=0.5, ls=":", alpha=0.35)
        if algo_bed_x is not None:
            algo_bed_twt = px_to_us(float(algo_bed_x), mb_x, cal)
            algo_bed_pwr = px_to_db(trace_y_s[algo_bed_x], cal)
            ax_wave.plot(algo_bed_twt, algo_bed_pwr, marker="^", color="darkorange",
                         markersize=7, alpha=0.35, zorder=4)
            ax_wave.axvline(algo_bed_twt, color="darkorange", lw=0.5, ls=":", alpha=0.35)

        p_floor = noise_floor_dB - 5.0
        p_ceil = noise_floor_dB + 50.0
        ax_wave.set_ylim(p_floor, p_ceil)
        ax_wave.set_xlim(t_lo, t_hi)
        ax_wave.set_xlabel("TWT from MB (µs)", fontsize=8)
        ax_wave.set_ylabel("Power (dB)", fontsize=8)
        ax_wave.spines["top"].set_visible(False)
        ax_wave.spines["right"].set_visible(False)
        ax_wave.tick_params(labelsize=7)
        ax_wave.legend(fontsize=6.5, loc="upper right", framealpha=0.85)

        # -- Title + status bar ------------------------------------------------
        mode_label = "BED" if st["mode"] == "b" else "SURFACE"
        status_color = {"good": "green", "weak_bed": "goldenrod",
                        "no_bed": "tomato", "no_surface": "red"}.get(current_status, "gray")
        title_obj = fig.suptitle(
            f"REVIEW  F{FLT} {fkey}  [{current_status}]  "
            f"({frame_i + 1}/{len(review_frames)})  "
            f"mode=[{mode_label}]",
            fontsize=10, color=status_color,
        )

        status_bar = fig.text(
            0.5, 0.005, "", ha="center", fontsize=9,
            bbox=dict(boxstyle="round", fc="wheat", alpha=0.85),
        )

        def _update_status():
            m = st["mode"]
            ov = st["override"]
            parts = [f"Mode: {'SURFACE' if m == 's' else 'BED'}"]
            if "surface_twt_us" in ov:
                parts.append(f"Srf: {ov['surface_twt_us']:.2f} µs")
            if "bed_twt_us" in ov:
                parts.append(f"Bed: {ov['bed_twt_us']:.2f} µs")
            if not ("surface_twt_us" in ov or "bed_twt_us" in ov):
                parts.append("No overrides")
            status_bar.set_text("  |  ".join(parts))
            col = "magenta" if m == "s" else "deepskyblue"
            status_bar.set_color(col)

        def _redraw_overrides():
            for art in st["drawn"]:
                try:
                    art.remove()
                except Exception:
                    pass
            st["drawn"].clear()

            ov = st["override"]
            if "surface_twt_us" in ov:
                twt = ov["surface_twt_us"]
                x_px = mb_x + int(round(twt / us_per_px))
                if 0 <= x_px < frame_w:
                    # Use clicked y if available, else fall back to smoothed trace
                    y_px = ov.get("surface_y_px", trace_y_s[x_px])
                    pwr = -(y_px - cal["y_ref_px"]) * cal["db_per_px"]
                    # Diamond on image panel (at clicked position)
                    st["drawn"].append(
                        ax_img.plot(x_px, y_px, marker="D", color="magenta",
                                    markersize=7, zorder=6, alpha=0.6,
                                    markeredgecolor="white", markeredgewidth=0.5)[0])
                    st["drawn"].append(
                        ax_img.axvline(x_px, color="magenta", lw=0.8, alpha=0.4))
                    # Mirror on waveform panel
                    st["drawn"].append(
                        ax_wave.plot(twt, pwr, marker="D", color="magenta",
                                     markersize=7, zorder=6, alpha=0.6,
                                     markeredgecolor="white", markeredgewidth=0.5)[0])
                    st["drawn"].append(
                        ax_wave.axvline(twt, color="magenta", lw=0.8, alpha=0.4))

            if "bed_twt_us" in ov:
                twt = ov["bed_twt_us"]
                x_px = mb_x + int(round(twt / us_per_px))
                if 0 <= x_px < frame_w:
                    # Use clicked y if available, else fall back to smoothed trace
                    y_px = ov.get("bed_y_px", trace_y_s[x_px])
                    pwr = -(y_px - cal["y_ref_px"]) * cal["db_per_px"]
                    # Diamond on image panel (at clicked position)
                    st["drawn"].append(
                        ax_img.plot(x_px, y_px, marker="D", color="cyan",
                                    markersize=7, zorder=6, alpha=0.6,
                                    markeredgecolor="white", markeredgewidth=0.5)[0])
                    st["drawn"].append(
                        ax_img.axvline(x_px, color="cyan", lw=0.8, alpha=0.4))
                    # Mirror on waveform panel
                    st["drawn"].append(
                        ax_wave.plot(twt, pwr, marker="D", color="cyan",
                                     markersize=7, zorder=6, alpha=0.6,
                                     markeredgecolor="white", markeredgewidth=0.5)[0])
                    st["drawn"].append(
                        ax_wave.axvline(twt, color="cyan", lw=0.8, alpha=0.4))

            _update_status()
            fig.canvas.draw_idle()

        def on_click(event):
            # Click on image panel -> convert pixel x to TWT, capture y for display
            if event.inaxes != ax_img or event.button != 1:
                return
            x_px = event.xdata
            y_px = event.ydata
            if x_px is None or x_px < mb_x:
                return
            x_px = int(round(x_px))
            if x_px >= frame_w:
                return
            twt = (x_px - mb_x) * us_per_px
            if st["mode"] == "s":
                st["override"]["surface_twt_us"] = round(twt, 3)
                st["override"]["surface_y_px"] = round(float(y_px), 1)
            elif st["mode"] == "b":
                st["override"]["bed_twt_us"] = round(twt, 3)
                st["override"]["bed_y_px"] = round(float(y_px), 1)
            _redraw_overrides()

        def on_key(event):
            k = event.key
            if k == "s":
                st["mode"] = "s"
                title_obj.set_text(
                    f"REVIEW  F{FLT} {fkey}  [{current_status}]  "
                    f"({frame_i + 1}/{len(review_frames)})  mode=[SURFACE]")
                _update_status()
                fig.canvas.draw_idle()
            elif k == "b":
                st["mode"] = "b"
                title_obj.set_text(
                    f"REVIEW  F{FLT} {fkey}  [{current_status}]  "
                    f"({frame_i + 1}/{len(review_frames)})  mode=[BED]")
                _update_status()
                fig.canvas.draw_idle()
            elif k == "d":
                st["override"].clear()
                _redraw_overrides()
            elif k in ("n", "right"):
                st["action"] = "next"
                plt.close(fig)
            elif k in ("p", "left"):
                st["action"] = "prev"
                plt.close(fig)
            elif k in ("q", "escape"):
                st["action"] = "quit"
                plt.close(fig)

        # Unbind matplotlib defaults that clash with review keys
        for action in ("keymap.save", "keymap.quit", "keymap.back",
                        "keymap.forward", "keymap.fullscreen"):
            try:
                plt.rcParams[action] = []
            except KeyError:
                pass
        fig.canvas.mpl_connect("button_press_event", on_click)
        fig.canvas.mpl_connect("key_press_event", on_key)

        _redraw_overrides()
        plt.show()

        # Save override for this frame
        if st["override"]:
            echo_overrides[fkey] = st["override"]
        elif fkey in echo_overrides:
            del echo_overrides[fkey]

        # Navigate
        if st["action"] == "next" or st["action"] is None:
            frame_i += 1
        elif st["action"] == "prev":
            frame_i = max(0, frame_i - 1)
        elif st["action"] == "quit":
            break

    # Persist overrides
    save_echo_overrides(echo_overrides)
    n_ov = len(echo_overrides)
    print(f"\nReview complete. {n_ov} override{'s' if n_ov != 1 else ''} saved.")


# ==============================================================================
# Branch: review mode vs normal extraction
# ==============================================================================

if _args.review:
    review_mode()
    print("\n" + "=" * 70)
    print("Re-running normal mode to apply overrides...")
    print("=" * 70 + "\n")
    # Reload overrides that review_mode just saved to disk
    echo_overrides = load_echo_overrides()
    # Fall through to normal extraction below


# -- Per-frame echo extraction --------------------------------------------------
print(f"  {'Fr':>3}  {'CBD':>6}  {'Status':^12}  {'NF':>6}  "
      f"{'Srf TWT':>8}  {'Srf dB':>7}  "
      f"{'Bed TWT':>8}  {'Bed dB':>7}  {'Bed SNR':>7}  "
      f"{'h_air':>7}  {'h_ice':>7}")
print("  " + "-" * 90)

echo_rows = []

for _, idx_row in tiff_rows.iterrows():
    frame_idx  = int(idx_row["frame_idx"])
    frame_type = idx_row["frame_type"]

    if frame_type == "partial":
        continue
    if frame_idx not in frame_cal:
        # Frame was excluded in step2 (or step2 hasn't been run for this frame)
        continue

    fc      = frame_cal[frame_idx]
    cbd     = fc["cbd"]
    file_id = fc["file_id"]
    mb_x    = fc["mb_x"]

    left_px  = int(idx_row["left_px"])
    right_px = int(idx_row["right_px"])
    frame    = img_norm[:, left_px : right_px + 1]
    frame_w  = frame.shape[1]

    cal = _make_cal(fc)

    # -- Extract trace and noise floor ------------------------------------------
    # Use robust mode: constrained-argmin rejects film-grain artefacts and
    # T/R ringing noise that would otherwise create fake peaks in the flat
    # region between the main bang and the first real echo.
    trace_y, trace_y_s, trace_info = extract_trace(frame, cal, robust=True, mb_x=mb_x)
    noise_floor_dB     = estimate_noise_floor(trace_y_s, mb_x, cal)

    # -- Detect surface and bed echoes ------------------------------------------
    surface_x, bed_x = detect_echoes(trace_y_s, mb_x, noise_floor_dB, cal)

    # -- Apply manual overrides (from echo_overrides.json) --------------------
    is_surface_override = is_bed_override = False
    override = echo_overrides.get(file_id, {})
    us_per_px = cal["us_per_px"]
    if "surface_twt_us" in override:
        surface_x = mb_x + int(round(override["surface_twt_us"] / us_per_px))
        surface_x = max(0, min(surface_x, frame_w - 1))
        is_surface_override = True
    if "bed_twt_us" in override:
        bed_x = mb_x + int(round(override["bed_twt_us"] / us_per_px))
        bed_x = max(0, min(bed_x, frame_w - 1))
        is_bed_override = True

    # -- Compute echo metrics ---------------------------------------------------
    surface = compute_echo_metrics(
        trace_y_s, surface_x, noise_floor_dB, mb_x, frame_w, cal, trace_y=trace_y
    ) if surface_x is not None else None

    bed = compute_echo_metrics(
        trace_y_s, bed_x, noise_floor_dB, mb_x, frame_w, cal, trace_y=trace_y
    ) if bed_x is not None else None

    # -- Classify echo_status ---------------------------------------------------
    if surface is None:
        echo_status = "no_surface"
    elif bed is None:
        echo_status = "no_bed"
    elif bed.peak_snr_dB < WEAK_BED_SNR_DB:
        echo_status = "weak_bed"
    else:
        echo_status = "good"

    # -- Bed envelope artifact flag -------------------------------------------
    # Check whether film artifacts exist in the post-bed region (between
    # the bed peak and the NF+5 trailing edge + 1 µs margin).  Artifacts
    # here bias the trailing edge, inflating width_5, asymmetry, and
    # trailing_tail.  Detection: compare constrained trace vs median-
    # filtered trace (art_diff_px from extract_trace).  Columns where the
    # median filter removed a >10 dB dip in this region are flagged.
    #
    # Artifacts elsewhere (pre-MB, surface-bed gap, pre-surface) are
    # benign for bed envelope estimation and are not flagged.
    bed_envelope_suspect = False
    art_max_dB = 0.0
    art_diff_px = trace_info.get("art_diff_px")
    if bed is not None and art_diff_px is not None:
        bed_trail_5 = getattr(bed, "trail_5_us", np.nan)
        if not np.isnan(bed_trail_5):
            # Check region: bed_peak to trail_5 + 1 µs
            us_per_px = cal["us_per_px"]
            db_per_px = cal["db_per_px"]
            post_bed_start = bed.peak_x
            post_bed_end   = min(len(art_diff_px),
                                 mb_x + int((bed_trail_5 + 1.0) / us_per_px))
            if post_bed_start < post_bed_end:
                region = art_diff_px[post_bed_start:post_bed_end]
                # >10 dB discrepancy = significant artifact
                art_thresh_px = 10.0 / db_per_px
                art_mask = region > art_thresh_px
                if np.any(art_mask):
                    art_max_dB = float(np.max(region[art_mask]) * db_per_px)
                    bed_envelope_suspect = True

    # -- Derived geometry -------------------------------------------------------
    h_air_m = h_ice_m = h_eff_m = np.nan
    if surface is not None:
        h_air_m = surface.peak_twt_us / 2.0 * C_AIR_M_PER_US
    if surface is not None and bed is not None:
        h_ice_m = (bed.peak_twt_us - surface.peak_twt_us) / 2.0 * C_ICE_M_PER_US
        h_eff_m = h_air_m + h_ice_m / N_ICE

    # -- Print summary line -----------------------------------------------------
    srf_twt = f"{surface.peak_twt_us:8.3f}" if surface else f"{'—':>8}"
    srf_db  = f"{surface.peak_power_dB:+7.1f}" if surface else f"{'—':>7}"
    bed_twt = f"{bed.peak_twt_us:8.3f}" if bed else f"{'—':>8}"
    bed_db  = f"{bed.peak_power_dB:+7.1f}" if bed else f"{'—':>7}"
    bed_snr = f"{bed.peak_snr_dB:+7.1f}" if bed else f"{'—':>7}"
    h_air_s = f"{h_air_m:7.0f}" if not np.isnan(h_air_m) else f"{'—':>7}"
    h_ice_s = f"{h_ice_m:7.0f}" if not np.isnan(h_ice_m) else f"{'—':>7}"

    ov_tag = ""
    if is_surface_override or is_bed_override:
        ov_tag = "  [M]"
    print(f"  {frame_idx:>3}  {str(cbd):>6}  {echo_status:^12}  "
          f"{noise_floor_dB:+6.1f}  "
          f"{srf_twt}  {srf_db}  "
          f"{bed_twt}  {bed_db}  {bed_snr}  "
          f"{h_air_s}  {h_ice_s}{ov_tag}")

    # -- Build output row -------------------------------------------------------
    def _e(echo, attr):
        return getattr(echo, attr, np.nan) if echo is not None else np.nan

    echo_rows.append(dict(
        flight      = FLT,
        tiff        = TIFF.name,
        tiff_id     = TIFF_ID,
        frame_idx   = frame_idx,
        cbd         = cbd,
        file_id     = file_id,
        mb_x        = mb_x,
        echo_status = echo_status,
        noise_floor_dB = round(noise_floor_dB, 2),
        # Surface echo
        surface_twt_us        = round(_e(surface, "peak_twt_us"),     4),
        surface_power_dB      = round(_e(surface, "peak_power_dB"),   2),
        surface_snr_dB        = round(_e(surface, "peak_snr_dB"),     2),
        surface_width_10_us   = round(_e(surface, "width_10_us"),     4),
        surface_width_5_us    = round(_e(surface, "width_5_us"),      4),
        surface_peakiness     = round(_e(surface, "peakiness"),       3),
        surface_asymmetry     = round(_e(surface, "asymmetry"),       3),
        surface_leading_rise_us  = round(_e(surface, "leading_rise_us"),  4),
        surface_trailing_tail_us = round(_e(surface, "trailing_tail_us"), 4),
        # Bed echo
        bed_twt_us            = round(_e(bed, "peak_twt_us"),    4),
        bed_power_dB          = round(_e(bed, "peak_power_dB"),  2),
        bed_snr_dB            = round(_e(bed, "peak_snr_dB"),    2),
        bed_width_10_us       = round(_e(bed, "width_10_us"),    4),
        bed_width_5_us        = round(_e(bed, "width_5_us"),     4),
        bed_peakiness         = round(_e(bed, "peakiness"),      3),
        bed_asymmetry         = round(_e(bed, "asymmetry"),      3),
        bed_leading_rise_us   = round(_e(bed, "leading_rise_us"),  4),
        bed_trailing_tail_us  = round(_e(bed, "trailing_tail_us"), 4),
        bed_envelope_suspect  = bed_envelope_suspect,
        artifact_max_dB       = round(art_max_dB, 1),
        # Geometry
        h_air_m = round(h_air_m, 1) if not np.isnan(h_air_m) else np.nan,
        h_ice_m = round(h_ice_m, 1) if not np.isnan(h_ice_m) else np.nan,
        h_eff_m = round(h_eff_m, 1) if not np.isnan(h_eff_m) else np.nan,
        # Override flags
        surface_override = is_surface_override,
        bed_override     = is_bed_override,
    ))

    # -- Generate two-panel diagnostic figure ----------------------------------
    fig = plt.figure(figsize=(18, 5), constrained_layout=True)
    fig.patch.set_facecolor("white")
    gs  = fig.add_gridspec(1, 3, wspace=0.08)
    ax_img  = fig.add_subplot(gs[0, :2])   # left 2/3: frame image
    ax_wave = fig.add_subplot(gs[0, 2])    # right 1/3: waveform

    # -- Left panel: frame image with echo markers ------------------------------
    fw = frame.shape[1]
    ax_img.imshow(frame, cmap="gray", vmin=0, vmax=1, aspect="auto",
                  extent=[0, fw, frame.shape[0], 0])

    # Noise floor reference row (cyan)
    y_ref_px = cal["y_ref_px"]
    ax_img.axhline(y_ref_px, color="cyan", lw=1.2, ls="--", alpha=0.8,
                   label=f"-60 dB  y={y_ref_px:.0f}")

    # Main bang (red dashed vertical)
    ax_img.axvline(mb_x, color="red", lw=1.5, ls="--", alpha=0.9,
                   label=f"MB  x={mb_x}")

    # Trace overlay (magenta, thin)
    x_all = np.arange(frame_w)
    ax_img.plot(x_all, trace_y_s, color="magenta", lw=0.8, alpha=0.7,
                label="trace")

    # Surface echo marker — diamond + "M" if manually overridden
    if surface is not None:
        srf_marker = "D" if is_surface_override else "^"
        srf_label  = f"Srf {surface.peak_twt_us:.2f}µs" + (" [M]" if is_surface_override else "")
        ax_img.plot(surface_x, surface.peak_y, marker=srf_marker, color="lime", markersize=9,
                    zorder=5, label=srf_label)
        ax_img.axvline(surface_x, color="lime", lw=0.8, ls=":", alpha=0.7)

    # Bed echo marker — diamond + "M" if manually overridden
    if bed is not None:
        bed_marker = "D" if is_bed_override else "^"
        bed_label  = f"Bed {bed.peak_twt_us:.2f}µs" + (" [M]" if is_bed_override else "")
        ax_img.plot(bed_x, bed.peak_y, marker=bed_marker, color="orange", markersize=9,
                    zorder=5, label=bed_label)
        ax_img.axvline(bed_x, color="orange", lw=0.8, ls=":", alpha=0.7)

    ax_img.set_ylim(DEFAULT_CAL["y_disp_hi"] + 50, DEFAULT_CAL["y_disp_lo"] - 50)
    ax_img.set_xlim(0, fw)
    ax_img.set_xlabel("Column (px, frame-relative)", fontsize=8)
    ax_img.set_ylabel("Row (px)", fontsize=8)
    ax_img.spines["top"].set_visible(False)
    ax_img.spines["right"].set_visible(False)
    ax_img.legend(fontsize=7, loc="upper right", framealpha=0.6,
                  facecolor="black", labelcolor="white")

    status_color = {"good": "lime", "weak_bed": "gold",
                    "no_bed": "tomato", "no_surface": "red"}.get(echo_status, "white")
    ax_img.set_title(
        f"LYRA Step 3 — F{FLT} {file_id}  |  "
        f"NF={noise_floor_dB:.1f} dB  "
        f"status=",
        fontsize=8, loc="left",
    )
    # Append colored status text
    ax_img.set_title(
        f"LYRA Step 3 — F{FLT} {file_id}  |  NF={noise_floor_dB:.1f} dB  "
        f"[{echo_status}]",
        fontsize=8, loc="left", color="black",
    )

    # -- Right panel: waveform in physical units --------------------------------
    # Convert trace from pixel-row to dB, columns to TWT
    twt_arr   = px_to_us(np.arange(frame_w, dtype=float), mb_x, cal)
    power_arr = px_to_db(trace_y_s, cal)

    # Show window: from mb_skip_us to 25 µs (covers typical RIS geometry)
    t_lo = float(cal["mb_skip_us"])
    t_hi = 25.0
    mask = (twt_arr >= t_lo) & (twt_arr <= t_hi)

    ax_wave.plot(twt_arr[mask], power_arr[mask],
                 color="black", lw=1.2, label="waveform")

    # Noise floor (gray dashed)
    ax_wave.axhline(noise_floor_dB, color="gray", lw=1.0, ls="--",
                    alpha=0.8, label=f"NF {noise_floor_dB:.1f} dB")

    # NF+5 and NF+10 thresholds (light gray dotted)
    ax_wave.axhline(noise_floor_dB + 5.0, color="silver", lw=0.7, ls=":",
                    alpha=0.8, label="NF+5 dB")
    ax_wave.axhline(noise_floor_dB + 10.0, color="darkgray", lw=0.7, ls=":",
                    alpha=0.8, label="NF+10 dB")

    # Surface echo
    if surface is not None:
        srf_mk = "D" if is_surface_override else "^"
        srf_lbl = (f"Srf {surface.peak_twt_us:.2f}µs  {surface.peak_power_dB:.1f}dB"
                   + (" [M]" if is_surface_override else ""))
        ax_wave.axvline(surface.peak_twt_us, color="limegreen", lw=1.0, ls="--",
                        alpha=0.8)
        ax_wave.plot(surface.peak_twt_us, surface.peak_power_dB,
                     marker=srf_mk, color="limegreen", markersize=9, zorder=5,
                     label=srf_lbl)
        # Envelope shading at +10 dB
        if not np.isnan(surface.lead_10_us) and not np.isnan(surface.trail_10_us):
            ax_wave.axvspan(surface.lead_10_us, surface.trail_10_us,
                            alpha=0.08, color="limegreen")

    # Bed echo
    if bed is not None:
        bed_mk = "D" if is_bed_override else "^"
        bed_lbl = (f"Bed {bed.peak_twt_us:.2f}µs  {bed.peak_power_dB:.1f}dB\n"
                   f"SNR={bed.peak_snr_dB:.1f}dB  peak={bed.peakiness:.2f}"
                   + (" [M]" if is_bed_override else ""))
        ax_wave.axvline(bed.peak_twt_us, color="darkorange", lw=1.0, ls="--",
                        alpha=0.8)
        ax_wave.plot(bed.peak_twt_us, bed.peak_power_dB,
                     marker=bed_mk, color="darkorange", markersize=9, zorder=5,
                     label=bed_lbl)
        # Envelope shading at +10 dB
        if not np.isnan(bed.lead_10_us) and not np.isnan(bed.trail_10_us):
            ax_wave.axvspan(bed.lead_10_us, bed.trail_10_us,
                            alpha=0.10, color="darkorange")

    # Axis formatting
    p_floor = noise_floor_dB - 5.0
    p_ceil  = noise_floor_dB + 50.0
    ax_wave.set_ylim(p_floor, p_ceil)
    ax_wave.set_xlim(t_lo, t_hi)
    ax_wave.set_xlabel("TWT from MB (µs)", fontsize=8)
    ax_wave.set_ylabel("Power (dB)", fontsize=8)
    ax_wave.spines["top"].set_visible(False)
    ax_wave.spines["right"].set_visible(False)
    ax_wave.legend(fontsize=6.5, loc="upper right", framealpha=0.85)
    ax_wave.tick_params(labelsize=7)
    ax_wave.set_title("Waveform", fontsize=8, loc="left")

    # Geometry annotation
    if not np.isnan(h_air_m) and not np.isnan(h_ice_m):
        ax_wave.text(
            0.02, 0.05,
            f"h_air={h_air_m:.0f} m\nh_ice={h_ice_m:.0f} m\nh_eff={h_eff_m:.0f} m",
            transform=ax_wave.transAxes, fontsize=7,
            va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="gray", alpha=0.8),
        )

    fig_path = PHASE4_DIR / f"F{FLT}_{file_id}_echoes.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# -- Update echo CSV (incremental, TIFF-specific rows) -------------------------
if echo_rows:
    new_df = pd.DataFrame(echo_rows)
    if ECHO_CSV.exists():
        existing = pd.read_csv(ECHO_CSV, dtype=str)
        existing = existing[existing["tiff"] != TIFF.name]
        merged   = pd.concat([existing, new_df.astype(str)], ignore_index=True)
    else:
        merged = new_df.astype(str)
    merged.to_csv(ECHO_CSV, index=False)

    # -- Summary statistics -----------------------------------------------------
    print(f"\n{'-' * 60}")
    print(f"  Echo CSV -> {ECHO_CSV.relative_to(ROOT)}")
    print(f"  Phase 4 figures -> {PHASE4_DIR.relative_to(ROOT)}/")
    print(f"  Frames processed : {len(echo_rows)}")

    statuses = [r["echo_status"] for r in echo_rows]
    for st in ["good", "weak_bed", "no_bed", "no_surface"]:
        n = statuses.count(st)
        if n:
            print(f"    {st:12s}: {n}")

    good_rows = [r for r in echo_rows if r["echo_status"] in ("good", "weak_bed")]
    if good_rows:
        h_airs = [r["h_air_m"] for r in good_rows
                  if not (isinstance(r["h_air_m"], float) and np.isnan(r["h_air_m"]))]
        h_ices = [r["h_ice_m"] for r in good_rows
                  if not (isinstance(r["h_ice_m"], float) and np.isnan(r["h_ice_m"]))]
        if h_airs:
            print(f"  h_air  : {np.mean(h_airs):.0f} ± {np.std(h_airs):.0f} m "
                  f"(n={len(h_airs)})")
        if h_ices:
            print(f"  h_ice  : {np.mean(h_ices):.0f} ± {np.std(h_ices):.0f} m "
                  f"(n={len(h_ices)})")

    # -- Generate review summary JSON (full flight, all TIFFs) -------------
    review_json_path = PHASE4_DIR / f"F{FLT}_echo_review.json"
    full_df = pd.read_csv(ECHO_CSV)
    review_data = {
        "flight": FLT,
        "total_frames": len(full_df),
        "summary": {},
    }
    for st_name in ["good", "weak_bed", "no_bed", "no_surface"]:
        st_df = full_df[full_df["echo_status"] == st_name]
        review_data["summary"][st_name] = len(st_df)

    # Per-status frame lists (only non-good statuses)
    for st_name in ["no_bed", "no_surface", "weak_bed"]:
        st_df = full_df[full_df["echo_status"] == st_name]
        if len(st_df) == 0:
            continue
        frames = []
        for _, row in st_df.iterrows():
            frames.append({
                "tiff_id": str(row.get("tiff_id", "")),
                "cbd": str(int(float(row["cbd"]))) if str(row["cbd"]).replace('.','',1).isdigit() else str(row["cbd"]),
                "file_id": str(row["file_id"]),
            })
        review_data[st_name] = frames

    # Per-TIFF breakdown: how many bad frames each TIFF has
    bad_df = full_df[full_df["echo_status"].isin(["no_bed", "no_surface", "weak_bed"])]
    if len(bad_df) > 0:
        tiff_breakdown = {}
        for tid, grp in bad_df.groupby("tiff_id"):
            counts = grp["echo_status"].value_counts().to_dict()
            tiff_breakdown[str(tid)] = {
                "total_bad": len(grp),
                **{k: int(v) for k, v in counts.items()},
            }
        # Sort by total_bad descending
        review_data["tiff_breakdown"] = dict(
            sorted(tiff_breakdown.items(), key=lambda x: x[1]["total_bad"], reverse=True)
        )

    with open(review_json_path, "w") as f:
        json.dump(review_data, f, indent=2)
    print(f"  Review JSON -> {review_json_path.relative_to(ROOT)}")

print("\nDone.")
