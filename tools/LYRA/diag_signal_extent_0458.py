"""
Diagnostic: URSA-inspired signal-extent detection for F125 CBD 0458.

Shows per-column raw-pixel minimum and the longest-contiguous-run mask
that defines where the CRT black signal actually lives in the frame.
No changes to lyra.py — standalone only.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from scipy.ndimage import gaussian_filter1d

# -- Frame / calibration constants (from phase1 + phase3 CSVs) -----------------
TIFF_PATH  = "Data/ascope/raw/125/40_0008400_0008424-reel_begin_end.tiff"
FRAME_LEFT  = 1054          # phase1: CBD 0458 left_px
FRAME_RIGHT = 4061          # phase1: CBD 0458 right_px

MB_X        = 799           # phase3: mb_x  (frame-relative)
Y_REF_PX    = 1514.0        # phase3: y_ref_px
DB_PER_PX   = 0.047962      # phase3
US_PER_PX   = 0.00973047    # phase3 (2.0 us/div)
Y_LO        = 300           # DEFAULT_CAL y_disp_lo
Y_HI        = 1700          # DEFAULT_CAL y_disp_hi
GRAT_HALF   = 5             # graticule mask half-width (px)

# URSA-inspired threshold (raw uint8; CRT trace ~ 0-20; film grain ~ 30-80)
RAW_TRACE_THRESHOLD = 80    # columns with col_min_raw < this -> "has real signal"

# -- Load TIFF (raw uint8 AND normalised float) --------------------------------
Image.MAX_IMAGE_PIXELS = None
img_pil = Image.open(TIFF_PATH)
img_raw  = np.array(img_pil, dtype=np.uint8)          # raw pixel values
img_f    = img_raw.astype(np.float32)
img_norm = (img_f - img_f.min()) / (img_f.max() - img_f.min() + 1e-9)

# -- Extract frame slices ------------------------------------------------------
frame_raw  = img_raw [Y_LO:Y_HI, FRAME_LEFT:FRAME_RIGHT+1]   # uint8 band
frame_norm = img_norm[Y_LO:Y_HI, FRAME_LEFT:FRAME_RIGHT+1]   # float [0,1] band
bH, W = frame_raw.shape
print(f"Frame size: {W} × {bH} px  (raw dtype: {frame_raw.dtype})")

# -- Build graticule-row exclusion mask ----------------------------------------
y_sp_px = 10.0 / DB_PER_PX          # px per 10-dB major division
n_lo = int(np.floor((Y_LO - Y_REF_PX) / y_sp_px)) - 1
n_hi = int(np.ceil ((Y_HI - Y_REF_PX) / y_sp_px)) + 1
grat_rows = set()
for n in range(n_lo, n_hi + 1):
    gl_abs = Y_REF_PX + n * y_sp_px
    gl_rel = int(round(gl_abs - Y_LO))
    for r in range(max(0, gl_rel - GRAT_HALF), min(bH, gl_rel + GRAT_HALF + 1)):
        grat_rows.add(r)

non_grat_rows = [r for r in range(bH) if r not in grat_rows]
print(f"Graticule rows excluded from min check: {len(grat_rows)} / {bH}")

# -- Per-column raw minimum (excluding graticule rows) ------------------------
frame_raw_ng = frame_raw[non_grat_rows, :]   # shape (n_non_grat, W)
col_raw_min  = frame_raw_ng.min(axis=0)      # (W,)  uint8

# -- Signal column classification ----------------------------------------------
signal_mask = col_raw_min < RAW_TRACE_THRESHOLD   # True where CRT trace present
x = np.arange(W)

# -- Longest contiguous run of signal columns (URSA trim_signal_trace idea) ---
def longest_run(mask):
    """Return (start, end) inclusive of the longest True run in mask."""
    in_run, best_start, best_len = False, 0, 0
    cur_start = 0
    for i, v in enumerate(mask):
        if v and not in_run:
            cur_start = i
            in_run = True
        elif not v and in_run:
            if i - cur_start > best_len:
                best_len  = i - cur_start
                best_start = cur_start
            in_run = False
    if in_run and (W - cur_start) > best_len:
        best_start = cur_start
        best_len   = W - cur_start
    return best_start, best_start + best_len - 1

sig_start, sig_end = longest_run(signal_mask)
print(f"Signal extent: col {sig_start} – {sig_end}  "
      f"({(sig_end-sig_start+1)*US_PER_PX:.2f} µs)")
print(f"MB_X (frame-relative): {MB_X}")
print(f"Signal start TWT: {sig_start * US_PER_PX:.3f} µs")
print(f"Signal end   TWT: {sig_end   * US_PER_PX:.3f} µs")

# -- Build new trace using signal mask -----------------------------------------
# Graticule-masked normalised band (for argmin)
band_gm = frame_norm.copy()
for r in grat_rows:
    band_gm[r, :] = 1.0

# Argmin on graticule-masked band -> raw trace positions (band-relative rows)
trace_raw = np.argmin(band_gm, axis=0).astype(float)   # band-relative

# Apply signal mask: outside longest-run -> NF reference row
nf_row_rel = Y_REF_PX - Y_LO                           # band-relative NF row
trace_masked = trace_raw.copy()
outside_signal = np.ones(W, dtype=bool)
outside_signal[sig_start:sig_end+1] = False
trace_masked[outside_signal] = nf_row_rel

# Light smoothing
trace_masked_s = gaussian_filter1d(trace_masked, sigma=5.0)
trace_raw_s    = gaussian_filter1d(trace_raw,    sigma=5.0)

# Convert to absolute image rows for plotting (Y_LO offset)
trace_abs_new = trace_masked_s + Y_LO
trace_abs_old = trace_raw_s    + Y_LO

# -- Time / power axes ---------------------------------------------------------
t_axis  = x * US_PER_PX                              # µs
col_min_power = -(col_raw_min.astype(float) / 255.0 * 60)  # crude visual only

# -- Plot ----------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(14, 12),
                         gridspec_kw={"height_ratios": [3, 1.2, 3]})
fig.suptitle("F125 CBD 0458 — URSA-inspired signal-extent detection", fontsize=13)

# Panel 1: frame image + per-column raw min profile overlay
ax1 = axes[0]
ax1.imshow(frame_raw, cmap="gray", aspect="auto",
           extent=[0, W * US_PER_PX, Y_HI - Y_LO, 0])
ax1.axvline(sig_start * US_PER_PX, color="lime",  lw=1.5, label="signal start")
ax1.axvline(sig_end   * US_PER_PX, color="orange",lw=1.5, label="signal end")
ax1.axvline(MB_X       * US_PER_PX, color="cyan",  lw=1.0, ls="--", label="MB_X")
ax1.set_ylabel("Row (px, band-relative)")
ax1.set_title("Frame image with detected signal extent")
ax1.legend(fontsize=8, loc="upper right")
ax1.set_xlim(0, W * US_PER_PX)

# Panel 2: per-column raw minimum profile
ax2 = axes[1]
ax2.plot(t_axis, col_raw_min, lw=0.7, color="steelblue", label="col raw min (uint8)")
ax2.axhline(RAW_TRACE_THRESHOLD, color="red", lw=1.2, ls="--",
            label=f"threshold = {RAW_TRACE_THRESHOLD}")
ax2.fill_between(t_axis,
                 col_raw_min, RAW_TRACE_THRESHOLD,
                 where=(col_raw_min < RAW_TRACE_THRESHOLD),
                 alpha=0.3, color="green", label="signal cols")
ax2.axvline(sig_start * US_PER_PX, color="lime",  lw=1.5)
ax2.axvline(sig_end   * US_PER_PX, color="orange",lw=1.5)
ax2.set_ylabel("Raw min pixel (uint8)")
ax2.set_title(f"Per-column raw minimum (excl. graticule rows) — threshold = {RAW_TRACE_THRESHOLD}")
ax2.legend(fontsize=8, loc="upper right")
ax2.set_xlim(0, W * US_PER_PX)
ax2.set_ylim(0, 255)
ax2.invert_yaxis()   # 0 (black) at top, like image rows

# Panel 3: frame image + old trace (gray) + new masked trace (magenta)
ax3 = axes[2]
ax3.imshow(frame_raw, cmap="gray", aspect="auto",
           extent=[0, W * US_PER_PX, Y_HI - Y_LO, 0])
ax3.plot(t_axis, trace_abs_old - Y_LO, color="gray",    lw=0.8, alpha=0.6, label="old trace (unconstrained)")
ax3.plot(t_axis, trace_abs_new - Y_LO, color="magenta", lw=1.2, label="new trace (signal-masked)")
# Shade non-signal regions
ax3.axvspan(0,                    sig_start * US_PER_PX, alpha=0.15, color="red",  label="non-signal (forced to NF)")
ax3.axvspan(sig_end * US_PER_PX,  W * US_PER_PX,        alpha=0.15, color="red")
ax3.axvline(sig_start * US_PER_PX, color="lime",  lw=1.5)
ax3.axvline(sig_end   * US_PER_PX, color="orange",lw=1.5)
ax3.axvline(MB_X       * US_PER_PX, color="cyan",  lw=1.0, ls="--")
nf_row_plot = Y_REF_PX - Y_LO
ax3.axhline(nf_row_plot, color="yellow", lw=0.8, ls=":", alpha=0.7, label=f"NF ref ({Y_REF_PX:.0f} px)")
ax3.set_ylabel("Row (px, band-relative)")
ax3.set_xlabel("Two-way travel time (µs)")
ax3.set_title("Old vs. new trace — new trace constrained to detected signal region")
ax3.legend(fontsize=8, loc="upper right")
ax3.set_xlim(0, W * US_PER_PX)

plt.tight_layout()
out_path = "tools/LYRA/output/F125/phase4/diag_signal_extent_0458.png"
import os; os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {out_path}")
