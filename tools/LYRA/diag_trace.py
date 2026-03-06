"""
diag_trace.py — Diagnostic visualisation of extract_trace internals
====================================================================
Produces a multi-panel figure for each target frame showing every
intermediate stage of the robust trace extraction algorithm:

  Panel 1: Raw frame image with ALL trace overlays
  Panel 2: Signal extent (per-column dark-pixel density)
  Panel 3: Pass-1 unconstrained trace vs coarse smooth (sigma=30)
  Panel 4: Constrained trace vs pass-1 (residual)
  Panel 5: Final trace_y_s vs CRT argmin (zoomed to echo region)

Usage:
    python tools/LYRA/diag_trace.py

Outputs:
    tools/LYRA/output/F127/diag/F127_CBD{N}_trace_diag.png
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/LYRA"))

from lyra import (
    DEFAULT_CAL, _gauss_smooth,
    detect_signal_extent, estimate_noise_floor,
    detect_echoes, px_to_db, px_to_us,
    ensure_canonical_name, tiff_id,
)

# -- Targets ------------------------------------------------------------------
TARGETS = [
    dict(tiff="Data/ascope/raw/127/47_0004850_0004874-reel_begin_end.tiff",
         cbd="0440", frame_idx=8, issue="surface trace misalignment"),
    dict(tiff="Data/ascope/raw/127/47_0004825_0004849-reel_begin_end.tiff",
         cbd="0458", frame_idx=5, issue="bed echo masked too early"),
]

FLT = 127
CAL_CSV = ROOT / f"tools/LYRA/output/F{FLT}/phase3/F{FLT}_cal.csv"
DIAG_DIR = ROOT / f"tools/LYRA/output/F{FLT}/diag"
DIAG_DIR.mkdir(parents=True, exist_ok=True)

cal_df = pd.read_csv(CAL_CSV, dtype=str)

for target in TARGETS:
    TIFF = ROOT / target["tiff"]
    TIFF = ensure_canonical_name(TIFF)
    cbd = target["cbd"]
    fidx = target["frame_idx"]

    # -- Load calibration for this frame -----------------------------------
    tiff_name = TIFF.name
    row = cal_df[(cal_df["tiff"] == tiff_name) &
                 (cal_df["frame_idx"] == str(fidx))].iloc[0]
    cal = DEFAULT_CAL.copy()
    cal["mb_x"]      = int(float(row["mb_x"]))
    cal["y_ref_px"]  = float(row["y_ref_px"])
    cal["db_per_px"] = float(row["db_per_px"])
    cal["us_per_px"] = float(row["us_per_px"])
    mb_x = cal["mb_x"]

    # -- Load frame --------------------------------------------------------
    idx_df = pd.read_csv(ROOT / f"tools/LYRA/output/F{FLT}/phase1/F{FLT}_frame_index.csv", dtype=str)
    idx_row = idx_df[(idx_df["tiff"] == tiff_name) &
                     (idx_df["frame_idx"] == str(fidx))].iloc[0]
    left_px = int(idx_row["left_px"])
    right_px = int(idx_row["right_px"])

    Image.MAX_IMAGE_PIXELS = None
    img = np.array(Image.open(TIFF), dtype=np.float32)
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-9)
    frame = img_norm[:, left_px:right_px + 1]
    H, W = frame.shape

    print(f"\n{'='*70}")
    print(f"CBD{cbd} — {target['issue']}")
    print(f"  TIFF: {tiff_name}  frame_idx={fidx}  mb_x={mb_x}")
    print(f"  Frame: {W}×{H} px  left={left_px} right={right_px}")

    # -- Reproduce extract_trace internals step by step --------------------
    y_lo = cal["y_disp_lo"]
    y_hi = min(cal["y_disp_hi"], H)
    band = frame[y_lo:y_hi, :]
    bH = band.shape[0]

    # Graticule masking
    mask_half = cal.get("graticule_mask_half_px", 5)
    y_sp_px = 10.0 / cal["db_per_px"]
    band_gm = band.astype(float).copy()
    n_lo_g = int(np.floor((y_lo - cal["y_ref_px"]) / y_sp_px)) - 1
    n_hi_g = int(np.ceil((y_hi - cal["y_ref_px"]) / y_sp_px)) + 1
    grat_rows_abs = []
    for _n in range(n_lo_g, n_hi_g + 1):
        gl_abs = cal["y_ref_px"] + _n * y_sp_px
        gl_rel = int(round(gl_abs - y_lo))
        if 0 <= gl_rel < bH:
            r0 = max(0, gl_rel - mask_half)
            r1 = min(bH, gl_rel + mask_half + 1)
            band_gm[r0:r1, :] = 1.0
            grat_rows_abs.append(gl_abs)
            grat_db = cal["y_ref_db"] + (cal["y_ref_px"] - gl_abs) * cal["db_per_px"]
            print(f"    Graticule line: y_abs={gl_abs:.1f}  ({grat_db:.0f} dB)  mask [{r0+y_lo},{r1+y_lo})")

    # Signal extent
    sig_start, sig_end = detect_signal_extent(frame, cal)
    print(f"  Signal extent: [{sig_start}, {sig_end}]  (W={W})")

    mb_skip_px = max(1, int(cal.get("mb_skip_us", 2.0) / cal["us_per_px"]))
    tr_end = min(W, mb_x + mb_skip_px)

    # Pass 1: unconstrained argmin
    trace_raw = np.argmin(band, axis=0).astype(float) + y_lo
    trace_gm = np.argmin(band_gm, axis=0).astype(float) + y_lo

    nf_row = float(cal["y_ref_px"])
    trace_pass1 = np.full(W, nf_row)
    trace_pass1[sig_start:sig_end + 1] = trace_gm[sig_start:sig_end + 1]
    pre_mb_start = max(sig_start, 0)
    pre_mb_end = min(mb_x, sig_end + 1)
    if pre_mb_start < pre_mb_end:
        trace_pass1[pre_mb_start:pre_mb_end] = trace_raw[pre_mb_start:pre_mb_end]
    trace_pass1[mb_x:tr_end] = nf_row

    # Coarse smooth
    coarse = _gauss_smooth(trace_pass1, sigma=30.0)
    max_jump = 250

    # Pass 2: constrained argmin
    trace_constrained = trace_pass1.copy()
    for col in range(sig_start, sig_end + 1):
        expected_row = coarse[col] - y_lo
        lo_row = max(0, int(expected_row - max_jump))
        hi_row = min(bH, int(expected_row + max_jump))
        if lo_row < hi_row:
            best_row = lo_row + np.argmin(band_gm[lo_row:hi_row, col])
            trace_constrained[col] = best_row + y_lo

    if pre_mb_start < pre_mb_end:
        trace_constrained[pre_mb_start:pre_mb_end] = trace_raw[pre_mb_start:pre_mb_end]
    trace_constrained[mb_x:tr_end] = nf_row

    # Fine smooth
    trace_y_s = _gauss_smooth(trace_constrained, sigma=5.0)

    # For comparison: what the unconstrained trace gives us
    trace_y = trace_pass1.copy()  # this is what gets returned as trace_y

    # -- Echo detection (to show what peaks are found) ---------------------
    noise_floor_dB = estimate_noise_floor(trace_y_s, mb_x, cal)
    surface_x, bed_x = detect_echoes(trace_y_s, mb_x, noise_floor_dB, cal)

    print(f"  Noise floor: {noise_floor_dB:.1f} dB")
    if surface_x is not None:
        srf_twt = (surface_x - mb_x) * cal["us_per_px"]
        srf_db = px_to_db(trace_y_s[surface_x], cal)
        srf_db_raw = px_to_db(trace_y[surface_x], cal)
        print(f"  Surface: x={surface_x}  twt={srf_twt:.3f} µs  "
              f"power(smoothed)={srf_db:.1f} dB  power(raw@same col)={srf_db_raw:.1f} dB")
    if bed_x is not None:
        bed_twt = (bed_x - mb_x) * cal["us_per_px"]
        bed_db = px_to_db(trace_y_s[bed_x], cal)
        print(f"  Bed: x={bed_x}  twt={bed_twt:.3f} µs  power={bed_db:.1f} dB")
    else:
        print(f"  Bed: NOT DETECTED")

    # -- Per-column dark-pixel density (for signal extent understanding) ---
    binary_thresh = cal.get("signal_binary_thresh", 0.30)
    binary = (band < binary_thresh).astype(np.uint8)
    # Exclude graticule rows
    grat_half_sig = cal.get("signal_grat_half", 8)
    for _n in range(n_lo_g, n_hi_g + 1):
        gl_abs = cal["y_ref_px"] + _n * y_sp_px
        gl_rel = int(round(gl_abs - y_lo))
        if 0 <= gl_rel < bH:
            r0 = max(0, gl_rel - grat_half_sig)
            r1 = min(bH, gl_rel + grat_half_sig + 1)
            binary[r0:r1, :] = 0
    col_count = binary.sum(axis=0).astype(float)
    col_smooth = _gauss_smooth(col_count, cal.get("signal_density_sigma", 30))
    density_thresh = cal.get("signal_density_thresh", 50)

    # -- Compute where constrained search window actually was --------------
    # At key columns (surface, bed), show the actual search window
    echo_cols = []
    if surface_x is not None:
        echo_cols.append(("Surface", surface_x))
    if bed_x is not None:
        echo_cols.append(("Bed", bed_x))

    for label, ecol in echo_cols:
        coarse_at_col = coarse[ecol]
        window_lo = max(y_lo, int(coarse_at_col - max_jump))
        window_hi = min(y_hi, int(coarse_at_col + max_jump))
        raw_argmin_at_col = trace_gm[ecol]
        constrained_at_col = trace_constrained[ecol]
        print(f"  {label} col={ecol}: coarse={coarse_at_col:.0f}  "
              f"window=[{window_lo},{window_hi}]  "
              f"raw_argmin={raw_argmin_at_col:.0f}  "
              f"constrained={constrained_at_col:.0f}  "
              f"diff={abs(raw_argmin_at_col - constrained_at_col):.0f} px")

    # Also check what happens OUTSIDE sig_end
    if bed_x is not None and bed_x > sig_end:
        print(f"  *** BED ECHO ({bed_x}) IS BEYOND sig_end ({sig_end})! ***")
    # Expected bed position from neighboring frames
    expected_bed_twt = 8.0  # µs, typical for F127
    expected_bed_x = mb_x + int(expected_bed_twt / cal["us_per_px"])
    if expected_bed_x > sig_end:
        print(f"  *** Expected bed position ({expected_bed_x}) "
              f"is beyond sig_end ({sig_end})! ***")
    else:
        print(f"  Expected bed position ({expected_bed_x}) "
              f"is within sig_end ({sig_end})")

    # -- 5-panel diagnostic figure -----------------------------------------
    fig, axes = plt.subplots(5, 1, figsize=(18, 22), constrained_layout=True)
    fig.patch.set_facecolor("white")

    x_all = np.arange(W)

    # -- Panel 1: Frame image with all trace overlays ----------------------
    ax = axes[0]
    ax.imshow(frame, cmap="gray", vmin=0, vmax=1, aspect="auto",
              extent=[0, W, H, 0])
    ax.plot(x_all, trace_gm, color="cyan", lw=0.5, alpha=0.5,
            label="raw argmin (grat-masked)")
    ax.plot(x_all, trace_pass1, color="blue", lw=0.7, alpha=0.6,
            label="pass-1 (gated)")
    ax.plot(x_all, coarse, color="yellow", lw=1.5, alpha=0.8,
            label="coarse smooth sigma=30")
    ax.plot(x_all, trace_constrained, color="red", lw=0.8, alpha=0.7,
            label="pass-2 constrained ±250")
    ax.plot(x_all, trace_y_s, color="magenta", lw=1.0, alpha=0.9,
            label="trace_y_s (final, sigma=5)")
    # Signal extent boundaries
    ax.axvline(sig_start, color="lime", lw=1.5, ls="--", label=f"sig_start={sig_start}")
    ax.axvline(sig_end, color="lime", lw=1.5, ls=":", label=f"sig_end={sig_end}")
    ax.axvline(mb_x, color="red", lw=2, ls="--", alpha=0.5, label=f"MB x={mb_x}")
    # Graticule lines
    for gl in grat_rows_abs:
        ax.axhline(gl, color="orange", lw=0.5, ls=":", alpha=0.5)
    # Echo markers
    if surface_x is not None:
        ax.axvline(surface_x, color="lime", lw=1, alpha=0.8)
        ax.plot(surface_x, trace_y_s[surface_x], "^", color="lime", ms=10, zorder=5)
    if bed_x is not None:
        ax.axvline(bed_x, color="orange", lw=1, alpha=0.8)
        ax.plot(bed_x, trace_y_s[bed_x], "^", color="orange", ms=10, zorder=5)
    ax.axvline(expected_bed_x, color="orange", lw=1, ls=":", alpha=0.5,
               label=f"expected bed x={expected_bed_x}")
    ax.set_ylim(y_hi + 50, y_lo - 50)
    ax.set_xlim(0, W)
    ax.legend(fontsize=6, loc="upper right", ncol=2, framealpha=0.7,
              facecolor="black", labelcolor="white")
    ax.set_title(f"Panel 1: CBD{cbd} — All trace overlays  |  {target['issue']}",
                 fontsize=9, fontweight="bold")
    ax.set_ylabel("Row (px)", fontsize=8)

    # -- Panel 2: Signal extent (dark-pixel density) -----------------------
    ax = axes[1]
    ax.plot(x_all, col_count, color="gray", lw=0.5, alpha=0.5, label="raw count")
    ax.plot(x_all, col_smooth, color="blue", lw=1.2, label="smoothed count")
    ax.axhline(density_thresh, color="red", lw=1, ls="--",
               label=f"threshold={density_thresh}")
    ax.axvline(sig_start, color="lime", lw=1.5, ls="--")
    ax.axvline(sig_end, color="lime", lw=1.5, ls=":")
    ax.axvline(mb_x, color="red", lw=1, ls="--", alpha=0.5)
    ax.axvline(expected_bed_x, color="orange", lw=1, ls=":", alpha=0.5,
               label=f"expected bed x={expected_bed_x}")
    ax.set_xlim(0, W)
    ax.set_ylabel("Dark pixel count", fontsize=8)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title("Panel 2: Signal extent — per-column dark-pixel density", fontsize=9)

    # -- Panel 3: Pass-1 trace vs coarse smooth ----------------------------
    ax = axes[2]
    # Plot as power (dB) for readability
    pass1_db = np.array([px_to_db(y, cal) for y in trace_pass1])
    coarse_db = np.array([px_to_db(y, cal) for y in coarse])
    raw_db = np.array([px_to_db(y, cal) for y in trace_gm])
    ax.plot(x_all, raw_db, color="cyan", lw=0.5, alpha=0.4, label="raw argmin")
    ax.plot(x_all, pass1_db, color="blue", lw=0.7, alpha=0.6, label="pass-1")
    ax.plot(x_all, coarse_db, color="yellow", lw=1.5, alpha=0.8, label="coarse sigma=30")
    ax.axhline(noise_floor_dB, color="gray", lw=0.7, ls="--", alpha=0.5)
    ax.axvline(sig_start, color="lime", lw=1, ls="--", alpha=0.5)
    ax.axvline(sig_end, color="lime", lw=1, ls=":", alpha=0.5)
    ax.axvline(mb_x, color="red", lw=1, ls="--", alpha=0.5)
    ax.set_xlim(0, W)
    ax.set_ylim(noise_floor_dB - 5, noise_floor_dB + 55)
    ax.set_ylabel("Power (dB)", fontsize=8)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title("Panel 3: Pass-1 (unconstrained) vs coarse smooth — power space",
                 fontsize=9)

    # -- Panel 4: Constrained vs pass-1 residual --------------------------
    ax = axes[3]
    constrained_db = np.array([px_to_db(y, cal) for y in trace_constrained])
    final_db = np.array([px_to_db(y, cal) for y in trace_y_s])
    residual = constrained_db - pass1_db
    ax.plot(x_all, pass1_db, color="blue", lw=0.5, alpha=0.4, label="pass-1")
    ax.plot(x_all, constrained_db, color="red", lw=0.7, alpha=0.6, label="constrained")
    ax.plot(x_all, final_db, color="magenta", lw=1.0, alpha=0.8, label="final (sigma=5)")
    ax.axhline(noise_floor_dB, color="gray", lw=0.7, ls="--", alpha=0.5)
    ax.axvline(sig_start, color="lime", lw=1, ls="--", alpha=0.5)
    ax.axvline(sig_end, color="lime", lw=1, ls=":", alpha=0.5)
    ax.axvline(mb_x, color="red", lw=1, ls="--", alpha=0.5)
    if surface_x is not None:
        ax.axvline(surface_x, color="lime", lw=1, alpha=0.5)
    if bed_x is not None:
        ax.axvline(bed_x, color="orange", lw=1, alpha=0.5)
    ax.axvline(expected_bed_x, color="orange", lw=1, ls=":", alpha=0.5)
    ax.set_xlim(0, W)
    ax.set_ylim(noise_floor_dB - 5, noise_floor_dB + 55)
    ax.set_ylabel("Power (dB)", fontsize=8)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title("Panel 4: Constrained vs pass-1 — where do they diverge?", fontsize=9)

    # -- Panel 5: Zoomed echo region — constrained search window analysis --
    ax = axes[4]
    # Zoom to the region around echoes: MB+2µs to MB+15µs
    zoom_lo = mb_x + int(2.0 / cal["us_per_px"])
    zoom_hi = min(W, mb_x + int(15.0 / cal["us_per_px"]))
    zoom_x = np.arange(zoom_lo, zoom_hi)

    # Plot in pixel space (y-row) to show search window
    ax.plot(zoom_x, trace_gm[zoom_lo:zoom_hi], color="cyan", lw=0.5, alpha=0.4,
            label="raw argmin (grat-masked)")
    ax.plot(zoom_x, trace_pass1[zoom_lo:zoom_hi], color="blue", lw=0.7, alpha=0.5,
            label="pass-1")
    ax.plot(zoom_x, coarse[zoom_lo:zoom_hi], color="yellow", lw=1.5, alpha=0.7,
            label="coarse sigma=30")
    # Show the ±250 search window as a shaded band around the coarse
    coarse_zoom = coarse[zoom_lo:zoom_hi]
    ax.fill_between(zoom_x, coarse_zoom - max_jump, coarse_zoom + max_jump,
                    alpha=0.1, color="yellow", label=f"±{max_jump} search window")
    ax.plot(zoom_x, trace_constrained[zoom_lo:zoom_hi], color="red", lw=0.8, alpha=0.7,
            label="constrained")
    ax.plot(zoom_x, trace_y_s[zoom_lo:zoom_hi], color="magenta", lw=1.2, alpha=0.9,
            label="trace_y_s (final)")
    # Graticule lines
    for gl in grat_rows_abs:
        if y_lo - 50 < gl < y_hi + 50:
            gdb = cal["y_ref_db"] + (cal["y_ref_px"] - gl) * cal["db_per_px"]
            ax.axhline(gl, color="orange", lw=0.7, ls=":", alpha=0.5)
            ax.text(zoom_lo + 5, gl - 5, f"{gdb:.0f} dB", fontsize=6, color="orange")
    if surface_x is not None and zoom_lo <= surface_x <= zoom_hi:
        ax.axvline(surface_x, color="lime", lw=1)
    if bed_x is not None and zoom_lo <= bed_x <= zoom_hi:
        ax.axvline(bed_x, color="orange", lw=1)
    ax.axvline(expected_bed_x, color="orange", lw=1, ls=":", alpha=0.5)
    ax.axvline(sig_end, color="lime", lw=1.5, ls=":", alpha=0.8,
               label=f"sig_end={sig_end}")
    ax.set_ylim(y_hi + 50, y_lo - 50)
    ax.set_xlim(zoom_lo, zoom_hi)
    ax.legend(fontsize=6, loc="upper right", ncol=2, framealpha=0.7,
              facecolor="black", labelcolor="white")
    ax.set_title("Panel 5: Echo region zoom — search window vs actual peaks (pixel space)",
                 fontsize=9)
    ax.set_xlabel("Column (px, frame-relative)", fontsize=8)
    ax.set_ylabel("Row (px) — lower = higher power", fontsize=8)

    fig_path = DIAG_DIR / f"F{FLT}_CBD{cbd}_trace_diag.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {fig_path.relative_to(ROOT)}")

print("\nDone.")
