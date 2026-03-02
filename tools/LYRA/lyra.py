"""
lyra.py — LYRA: Layered-echo Yield from Radiometric Archives
=============================================================

Algorithmic A-scope waveform extraction from raw TIFF film scans.
Named following the existing tool constellation convention:
  URSA (bear), ASTRA (stars), TERRA (earth), ARIES (ram), LYRA (lyre).

For each A-scope frame in a TIFF this module:
  1.  Detects frame boundaries (white inter-frame gaps)
  2.  Extracts the oscilloscope trace (darkest pixel per column)
  3.  Locates the main bang (origin of the time axis)
  4.  Estimates per-frame noise floor from pre-main-bang baseline
  5.  Finds surface and bed echo peaks via prominence-based detection
  6.  Walks leading / trailing edges at +5 dB and +10 dB above noise floor
  7.  Computes waveform shape metrics:
        peakiness        = peak_linear / mean_linear  (within +10 dB window)
        asymmetry        = trailing_width_10 / leading_width_10
        trailing_tail_us = trail_5 − trail_10  (subsurface scattering length)
        leading_rise_us  = peak_twt − lead_10  (onset steepness)
  8.  Returns a flat pandas DataFrame (one row per frame)

Grid calibration (validated vs Neal 1977 Fig 1.3a, CBD 0465, F125):
  X-axis : 1.5 µs / major division  (205.54 px/major; 4 minor ticks → 0.3 µs/minor)
  Y-axis : 10 dB  / major division  (205.0  px/major)
  Noise floor reference : y = 1507 px  → −60 dB

WARNING: Do NOT use ASTRA surface_us / bed_us for geometry — those times are
         2× too large (ASTRA assumed 3.0 µs/major; correct value is 1.5 µs/major).
         Use only LYRA-derived travel times.

Physical interpretation of waveform metrics:
  High peakiness + low asymmetry + short trailing_tail  → specular / liquid water
  Low peak power + long trailing_tail                   → saline / frozen marine ice
  Broad symmetric echo, moderate peakiness              → rough / incoherent bed
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


# ── TIFF canonical naming utility ─────────────────────────────────────────────

_CANONICAL_RE = re.compile(r'^\d+_\d+_\d+-reel_begin_end\.tiff$', re.IGNORECASE)


def _parse_rename_log(log_path: Path) -> dict[str, str]:
    """Parse a LYRA rename log. Returns {current_name: canonical_name} mapping.

    The log format has blocks like:
        Old: 43_0008375_0008399-reel_begin_end.tiff   ← canonical (target)
        New: F141-C0038_C0047.tiff                    ← current (on-disk) name
    """
    mapping: dict[str, str] = {}
    old_name = new_name = None
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line.startswith("Old: "):
            old_name = line[5:].strip()
        elif line.startswith("New: "):
            new_name = line[5:].strip()
            if old_name and new_name:
                mapping[new_name] = old_name
            old_name = new_name = None
    return mapping


def ensure_canonical_name(tiff_path: Path) -> Path:
    """Return the canonical-format path for a TIFF, renaming the file if needed.

    Canonical format: ``{reel}_{start}_{end}-reel_begin_end.tiff``
    (e.g. ``43_0008375_0008399-reel_begin_end.tiff``)

    If the file already matches that pattern, the original path is returned
    unchanged.  Otherwise the function searches for a ``*_rename_log*.txt``
    file in the same directory, parses it, renames the TIFF to its canonical
    name, and returns the new path.  An error is raised if the log is missing
    or the file is not listed in it.
    """
    if _CANONICAL_RE.match(tiff_path.name):
        return tiff_path  # already canonical — nothing to do

    log_files = sorted(tiff_path.parent.glob("*_rename_log*.txt"))
    if not log_files:
        raise FileNotFoundError(
            f"TIFF '{tiff_path.name}' is not in canonical format and no "
            f"'*_rename_log*.txt' was found in {tiff_path.parent}.\n"
            f"  Canonical format: {{reel}}_{{start}}_{{end}}-reel_begin_end.tiff"
        )

    mapping = _parse_rename_log(log_files[0])
    canonical = mapping.get(tiff_path.name)
    if canonical is None:
        raise KeyError(
            f"'{tiff_path.name}' not found in rename log '{log_files[0].name}'.\n"
            f"  Log covers: {sorted(mapping.keys())[:5]} …"
        )

    new_path = tiff_path.parent / canonical
    if new_path.exists():
        print(f"  [ensure_canonical_name] Already renamed: {new_path.name}")
        return new_path

    tiff_path.rename(new_path)
    print(f"  [ensure_canonical_name] Renamed:\n"
          f"      {tiff_path.name}\n"
          f"    → {new_path.name}")
    return new_path


def tiff_id(tiff_path: Path) -> str:
    """Extract 4-digit TIFF start number from canonical filename.

    E.g. ``40_0008400_0008424-reel_begin_end.tiff`` → ``"8400"``.
    """
    for part in tiff_path.stem.split("_"):
        if part.isdigit() and len(part) >= 5:
            return str(int(part[-8:]))[-4:]
    return tiff_path.stem[:8]  # fallback


# ── Pure-numpy replacements (no scipy dependency) ─────────────────────────────

def _connected_components_1d(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return (start, end) index pairs for each run of True in a boolean mask."""
    runs = []
    in_run = False
    start = 0
    for i, v in enumerate(mask.tolist()):
        if v and not in_run:
            start, in_run = i, True
        elif not v and in_run:
            runs.append((start, i - 1))
            in_run = False
    if in_run:
        runs.append((start, len(mask) - 1))
    return runs


def _find_peaks_numpy(arr: np.ndarray,
                      prominence: float = 0,
                      distance: int = 1) -> np.ndarray:
    """
    Find local minima in arr with minimum prominence and minimum distance.
    Equivalent (for our use case) to scipy.signal.find_peaks(-arr, ...).

    prominence : minimum depth of the minimum below surrounding neighbours
    distance   : minimum gap (pixels) between returned minima
    """
    n = len(arr)
    # Strict local minima
    candidates = [i for i in range(1, n - 1)
                  if arr[i] < arr[i - 1] and arr[i] < arr[i + 1]]
    if not candidates:
        return np.array([], dtype=int)
    candidates = np.array(candidates, dtype=int)

    # Distance filter: greedy — keep most prominent, suppress neighbours
    if distance > 1 and len(candidates) > 1:
        order = np.argsort(arr[candidates])   # most-prominent first
        kept: list[int] = []
        used: set[int] = set()
        for idx in order:
            p = int(candidates[idx])
            if p not in used:
                kept.append(p)
                for near in candidates:
                    if abs(int(near) - p) < distance:
                        used.add(int(near))
        candidates = np.array(sorted(kept), dtype=int)

    # Prominence filter
    if prominence > 0 and len(candidates) > 0:
        valid = []
        for p in candidates:
            d     = max(int(distance) * 3, 30)
            lo, hi = max(0, p - d), min(n - 1, p + d)
            lmax  = arr[lo:p].max()   if p   > lo else arr[p]
            rmax  = arr[p + 1:hi + 1].max() if hi > p else arr[p]
            if (min(lmax, rmax) - arr[p]) >= prominence:
                valid.append(p)
        candidates = np.array(valid, dtype=int)

    return candidates

# ── Physical constants ─────────────────────────────────────────────────────────
C_AIR_M_PER_US = 300.0   # m / µs  (speed of light in air)
C_ICE_M_PER_US = 168.0   # m / µs  (Millar 1981; Neal 1977 uses 169)

# ── Default grid calibration ───────────────────────────────────────────────────
# Measured algorithmically from raw TIFFs; validated vs Neal 1977 Fig 1.3a.
_X_SPACING_PX = 205.54   # pixels per major X division (measured from TIFF)
_Y_SPACING_PX = 205.0    # pixels per major Y division (user confirmed)

DEFAULT_CAL: dict = dict(
    us_per_px     = 1.5 / _X_SPACING_PX,   # 0.007299 µs/px   ← VALIDATED
    db_per_px     = 10.0 / _Y_SPACING_PX,  # 0.048780 dB/px
    y_ref_px      = 1507,    # row of −60 dB reference line
    y_ref_db      = -60.0,   # dB at the reference row
    y_disp_lo     = 300,     # top of usable display band (row index; lower = higher on image)
    y_disp_hi     = 1700,    # bottom of usable display band
    mb_x_guess    = 800,     # expected main bang column (frame-relative; heuristic)
    gauss_sigma   = 3,       # trace-smoothing Gaussian sigma (pixels)
    consec        = 5,       # consecutive threshold crossings to confirm trailing edge
    peak_min_prom_dB = 5.0,  # minimum peak prominence above noise floor for detection (dB)
    peak_min_dist_px = 80,   # minimum inter-peak distance (pixels)
    mb_skip_us    = 2.0,     # skip this many µs after main bang before searching for echoes
    min_bed_gap_us = 2.0,    # minimum surface→bed TWT gap (µs); ~168 m minimum ice thickness
    max_walk_us   = 17.0,    # rightward envelope-walk limit (µs from MB); covers all RIS depths
    graticule_mask_half_px = 5,  # ±px to suppress around each major graticule line
    signal_binary_thresh  = 0.30, # normalized pixel threshold for dark-pixel binary mask.
                                  # CRT trace ≈ 0.0–0.15; background ≈ 0.55–0.65.
    signal_density_sigma  = 30,   # Gaussian sigma for smoothing per-column dark-pixel count.
    signal_density_thresh = 50,   # smoothed dark-pixel count threshold for signal detection.
                                  # Used as the CAP in the adaptive threshold formula:
                                  #   thresh = max(floor, min(cap, fraction × peak_density))
                                  # For bright frames (peak ~200): thresh = 50 (same as original).
                                  # For faint frames (peak ~40): thresh = 12 (floor).
    signal_density_floor  = 12,   # minimum adaptive threshold; above film-grain noise (~5-6 px/col).
    signal_density_frac   = 0.25, # fraction of peak column density used for adaptive threshold.
    signal_grat_half      = 8,    # ±px to exclude around graticule rows in signal detection
                                  # (wider than graticule_mask_half_px to avoid dark-line leakage).
    surface_window_us   = 8.0,   # surface search window end (µs from MB); covers altitude ≤1200 m
    min_surface_lead_dB = 15.0,  # if strongest surface cand. > first by this, promote it
    quiet_gap_lead_us   = 1.0,   # µs after surface echo before checking for quiet gap
    quiet_gap_dB        = 12.0,  # trace must return within this many dB of NF in the gap
                                 # before a bed candidate is accepted.  Rejects surface-echo
                                 # trailing oscillations (which are continuous, never return
                                 # to NF); passes genuine thin-ice beds (surface echo decays
                                 # to NF within ≤1 µs of the surface peak, then quiet gap).
                                 # Not applied when gap < 1 µs wide (very thin ice, ≤84 m).
    min_bed_width_us    = 0.3,   # minimum bed echo width (µs) at NF + min_bed_width_dB.
                                 # Rejects very narrow film artifacts / spikes (< 0.2 µs).
                                 # Real bed echoes: ≥ 0.45 µs even for narrow returns
                                 # (F125 TIFF 7700 CBD0817: 0.45 µs at 10 dB threshold).
    min_bed_width_dB    = 10.0,  # dB above NF used for the width measurement.
)


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class EchoResult:
    """Waveform extraction result for one echo (surface or bed)."""
    # Peak location
    peak_x:        int   = 0      # column (frame-relative pixels)
    peak_y:        int   = 0      # row (pixels; lower = higher power)
    peak_twt_us:   float = np.nan # two-way travel time from main bang (µs)
    peak_power_dB: float = np.nan # power (dB, calibration reference frame)
    peak_snr_dB:   float = np.nan # SNR above measured noise floor (dB)

    # Envelope at +10 dB above noise floor
    lead_10_us:  float = np.nan   # leading edge TWT (µs from MB)
    trail_10_us: float = np.nan   # trailing edge TWT (µs from MB)
    width_10_us: float = np.nan   # width = trail_10 − lead_10

    # Envelope at +5 dB above noise floor (wider, captures subsurface tail)
    lead_5_us:   float = np.nan
    trail_5_us:  float = np.nan
    width_5_us:  float = np.nan

    # Derived shape metrics
    leading_rise_us:  float = np.nan  # peak_twt − lead_10  (onset steepness; lower → more specular)
    trailing_tail_us: float = np.nan  # trail_5 − trail_10  (subsurface scattering tail length)
    peakiness:        float = np.nan  # peak_linear / mean_linear within +10 dB window (>1 = peaked)
    asymmetry:        float = np.nan  # (peak_twt − lead_10) / (trail_10 − peak_twt) (>1 = trailing-heavy)


@dataclass
class LyraFrame:
    """Complete LYRA result for one A-scope frame."""
    frame_idx:   int = 0
    frame_left:  int = 0   # left pixel column in original TIFF
    frame_right: int = 0   # right pixel column in original TIFF
    frame_w:     int = 0
    mb_x:        int = 0   # main bang column (frame-relative)
    noise_floor_dB: float = -60.0
    surface: EchoResult = field(default_factory=EchoResult)
    bed:     EchoResult = field(default_factory=EchoResult)
    surface_detected: bool = False
    bed_detected:     bool = False
    quality:          str  = "ok"  # "ok" | "no_surface" | "no_bed" | "failed"


# ── Unit conversions ───────────────────────────────────────────────────────────

def px_to_db(y, cal: dict):
    """Row pixel → power (dB). Lower y (higher on image) = higher power."""
    return cal["y_ref_db"] + (cal["y_ref_px"] - y) * cal["db_per_px"]


def db_to_px(db: float, cal: dict) -> float:
    """Power (dB) → row pixel."""
    return cal["y_ref_px"] - (db - cal["y_ref_db"]) / cal["db_per_px"]


def px_to_us(x, mb_x: int, cal: dict):
    """Column pixel (frame-relative) → TWT (µs from main bang)."""
    return (x - mb_x) * cal["us_per_px"]


def us_to_px(us: float, mb_x: int, cal: dict) -> int:
    """TWT (µs from main bang) → column pixel (frame-relative)."""
    return int(round(mb_x + us / cal["us_per_px"]))


# ── Gaussian smoothing ─────────────────────────────────────────────────────────

def _gauss_smooth(arr: np.ndarray, sigma: float) -> np.ndarray:
    r = int(4 * sigma + 0.5)
    x = np.arange(-r, r + 1, dtype=float)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    return np.convolve(arr.astype(float), k, mode="same")


# ── Frame boundary detection ───────────────────────────────────────────────────

def detect_frames(img_norm: np.ndarray,
                  band_frac: tuple = (0.20, 0.80),
                  smooth_sigma: float = 20,
                  gap_threshold: float = 0.90,
                  min_gap_px: int = 80,
                  min_frame_px: int = 200,
                  buffer_px: int = 5,
                  expected_frames: int = 14) -> list[tuple[int, int]]:
    """
    Detect A-scope frame boundaries from inter-frame white gaps.

    Ports the working approach from extract_noise_floor_ursa_frames.py:
      1. Compute column means in central band (20%–80% of height)
      2. Gaussian smooth
      3. Normalize to [0, 1]           ← CRITICAL: do NOT threshold raw values
      4. Threshold normalized at 0.90 → gap mask
      5. Find gap CENTERS (mean col of each run ≥ min_gap_px wide)
      6. Frames = intervals between gap centers, with buffer_px inward

    Parameters
    ----------
    img_norm        : (H, W) float array, values in [0, 1]
    band_frac       : (lo, hi) fractional row range for gap analysis
    smooth_sigma    : Gaussian sigma for column-mean smoothing
    gap_threshold   : threshold on normalized [0,1] profile (default 0.90)
    min_gap_px      : minimum gap width (pixels) to count as a separator
    min_frame_px    : minimum frame width (pixels) to keep
    buffer_px       : pixels to trim inward from each gap center
    expected_frames : expected number of frames per TIFF (default 14 = 12 complete
                      + 2 partial); used both for the widest-gap filter (keeps at
                      most expected_frames-1 gaps) and fallback if too few gaps found

    Returns
    -------
    List of (left, right) frame boundary tuples, sorted left-to-right.
    """
    H, W = img_norm.shape
    y_lo = int(H * band_frac[0])
    y_hi = int(H * band_frac[1])

    col_mean   = img_norm[y_lo:y_hi, :].mean(axis=0)
    col_smooth = _gauss_smooth(col_mean, smooth_sigma)

    # Normalize to [0, 1] — this is the key step that makes thresholding work
    c_min, c_max = col_smooth.min(), col_smooth.max()
    if c_max > c_min:
        normalized = (col_smooth - c_min) / (c_max - c_min)
    else:
        normalized = np.zeros_like(col_smooth)

    gap_mask = normalized > gap_threshold

    # Find centers of sufficiently wide gap regions; track widths for filtering
    runs = _connected_components_1d(gap_mask)
    gap_info: list[tuple[int, int]] = []   # (width_px, center_px)
    for s, e in runs:
        w = e - s + 1
        if w >= min_gap_px:
            gap_info.append((w, int(round((s + e) / 2))))

    # Filter to true inter-frame gaps using bimodal width distribution.
    #
    # True inter-frame gaps (white film base between A-scopes) are always much
    # wider than any intra-frame feature (graticule lines, CRT edge artifacts,
    # partial-frame edges).  Sorting by width descending and looking for the
    # largest *relative* drop between adjacent entries reliably separates the two
    # populations — e.g. F141 TIFF 8525 shows a 91 % drop from the 12th true gap
    # (1024 px) to a reel-edge artifact (92 px) that would otherwise create a
    # spurious "partial" frame at the beginning.
    #
    # The bimodal cut is applied only when the drop is dramatic (> 50 %) and
    # enough gaps remain (≥ 3) to form a sensible frame sequence.  After the
    # bimodal cut the hard cap max_gaps = expected_frames − 1 still applies as
    # a safety upper bound, so the old behaviour is preserved for TIFFs where
    # gap widths are all similar (no dramatic drop).
    gap_info.sort(key=lambda t: -t[0])   # widest first
    if len(gap_info) > 1:
        best_cut   = len(gap_info)
        best_ratio = 0.0
        for i in range(1, len(gap_info)):
            ratio = 1.0 - gap_info[i][0] / gap_info[i - 1][0]
            if ratio > best_ratio:
                best_ratio = ratio
                best_cut   = i
        if best_ratio > 0.50 and best_cut >= 3:
            print(f"  [detect_frames] bimodal gap cut: keeping {best_cut} wide gaps "
                  f"(drop {best_ratio:.0%} at rank {best_cut}: "
                  f"{gap_info[best_cut-1][0]}→{gap_info[best_cut][0]} px)")
            gap_info = gap_info[:best_cut]

    max_gaps = expected_frames - 1
    if len(gap_info) > max_gaps:
        gap_info = gap_info[:max_gaps]

    gap_centers = sorted([c for _, c in gap_info])

    # Fallback: if too few gaps, estimate from expected frame count
    if len(gap_centers) < min(3, expected_frames - 1):
        est_w = W / expected_frames if expected_frames > 0 else W
        gap_centers = [int((i + 0.5) * est_w) for i in range(expected_frames - 1)]

    # Build frames as intervals between gap centers
    frame_edges = [0] + gap_centers + [W]
    frames = []
    for i in range(len(frame_edges) - 1):
        left  = int(frame_edges[i])  + (buffer_px if i > 0 else 0)
        right = int(frame_edges[i + 1]) - (buffer_px if i < len(frame_edges) - 2 else 0)
        if right - left > min_frame_px:
            frames.append((max(0, left), min(W - 1, right)))

    # ── Post-process: split frames wider than 1.5× median ─────────────────────
    # If a separator was missed (e.g. dim gap at reel start), one entry may span
    # two physical A-scopes (~2× normal width).  Find the sub-separator by
    # searching for the brightest column in the middle 30–70% of the oversized
    # frame, then split there.  Only triggered when a frame is genuinely 1.5×
    # the median width — does not affect normally detected frames.
    if len(frames) >= 2:
        widths = [r - l for l, r in frames]
        med_w  = float(np.median(widths))
        new_frames = []
        for l, r in frames:
            if (r - l) > 1.5 * med_w:
                # Search middle 30%–70% for the sub-separator peak
                sub_l    = l + int(0.30 * (r - l))
                sub_r    = l + int(0.70 * (r - l))
                sub_peak = sub_l + int(np.argmax(col_smooth[sub_l:sub_r]))
                left_end  = max(l,     sub_peak - buffer_px - 1)
                right_beg = min(W - 1, sub_peak + buffer_px + 1)
                new_frames.append((l, left_end))
                new_frames.append((right_beg, r))
                print(f"  [detect_frames] oversized frame {l}–{r} (w={r-l}) split "
                      f"at col {sub_peak} → {l}–{left_end} + {right_beg}–{r}")
            else:
                new_frames.append((l, r))
        frames = sorted(new_frames)

    return frames


# ── Signal extent detection ────────────────────────────────────────────────────

def detect_signal_extent(frame: np.ndarray,
                         cal: dict) -> tuple[int, int]:
    """
    Detect the horizontal extent of the CRT A-scope signal within a frame.

    The CRT beam sweeps left-to-right, creating a continuous dark trace on the
    phosphor screen.  Outside the sweep, only film grain exists — no distinct
    dark line.  This function finds the first and last columns where the CRT
    trace is present.

    Algorithm
    ---------
    1. Binary threshold on the globally-normalised frame (< signal_binary_thresh)
       to isolate dark pixels (CRT trace + some film grain).
    2. Exclude graticule rows (±signal_grat_half px around each 10-dB major
       grid line) so that horizontal grid lines do not inflate the count.
    3. Sum the dark-pixel count per column and smooth with a Gaussian (σ = signal_density_sigma).
    4. Adaptive threshold: max(floor, min(cap, fraction × peak_density)).
       Bright frames (peak ~200) → cap = 50 (same as original fixed threshold).
       Faint frames (peak ~40) → floor = 12 (above film-grain noise ~5-6 px/col).
    5. Signal columns are those where the smoothed count ≥ adaptive threshold.
    6. Return the first and last such column.

    Parameters
    ----------
    frame : (H, W) float array, globally normalised [0, 1]
        The full-height frame (not the y_disp_lo:y_disp_hi band).
    cal   : calibration dict (must include y_ref_px, db_per_px, y_disp_lo, y_disp_hi).

    Returns
    -------
    (sig_start, sig_end) : column indices (frame-relative, inclusive).
        If detection fails, returns (0, W-1) as a safe fallback.
    """
    y_lo = cal["y_disp_lo"]
    y_hi = min(cal["y_disp_hi"], frame.shape[0])
    band = frame[y_lo:y_hi, :]
    bH, W = band.shape

    # ── Binary: dark pixels below threshold ──────────────────────────────────
    binary_thresh = cal.get("signal_binary_thresh", 0.30)
    binary = (band < binary_thresh).astype(np.uint8)

    # ── Exclude graticule rows ───────────────────────────────────────────────
    grat_half = cal.get("signal_grat_half", 8)
    y_sp_px   = 10.0 / cal["db_per_px"]
    n_lo = int(np.floor((y_lo - cal["y_ref_px"]) / y_sp_px)) - 1
    n_hi = int(np.ceil ((y_hi - cal["y_ref_px"]) / y_sp_px)) + 1
    for _n in range(n_lo, n_hi + 1):
        gl_abs = cal["y_ref_px"] + _n * y_sp_px
        gl_rel = int(round(gl_abs - y_lo))
        if 0 <= gl_rel < bH:
            r0 = max(0, gl_rel - grat_half)
            r1 = min(bH, gl_rel + grat_half + 1)
            binary[r0:r1, :] = 0

    # ── Per-column dark-pixel count, smoothed ────────────────────────────────
    col_count  = binary.sum(axis=0).astype(float)
    sigma      = cal.get("signal_density_sigma", 30)
    col_smooth = _gauss_smooth(col_count, sigma)

    # ── Adaptive threshold ─────────────────────────────────────────────────
    # Fixed threshold (50) works for bright frames but excludes real echo
    # columns in faint CRT traces (e.g. F127 CBD0440: mean 18.5 dark px/col).
    # Adaptive formula: max(floor, min(cap, fraction × peak_density)).
    # For bright frames (peak ~200): cap=50 reached → identical to original.
    # For faint frames (peak ~40): floor=12 reached → catches real signal.
    # Film grain noise ≈ 5-6 dark px/col; floor=12 stays safely above this.
    cap      = cal.get("signal_density_thresh", 50)
    floor    = cal.get("signal_density_floor",  12)
    fraction = cal.get("signal_density_frac",   0.25)
    peak_density   = float(np.max(col_smooth)) if len(col_smooth) > 0 else 0.0
    density_thresh = max(floor, min(cap, fraction * peak_density))

    # ── Signal extent: first/last column above threshold ─────────────────────
    above = col_smooth >= density_thresh
    signal_cols = np.where(above)[0]

    if len(signal_cols) == 0:
        # Fallback: no signal detected → treat entire frame as signal
        return 0, W - 1

    return int(signal_cols[0]), int(signal_cols[-1])


# ── Trace extraction ───────────────────────────────────────────────────────────

def extract_trace(frame: np.ndarray,
                  cal: dict,
                  robust: bool = False,
                  mb_x: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract oscilloscope trace from one A-scope frame.

    Finds the darkest (minimum intensity) pixel per column within the
    oscilloscope display band, then applies Gaussian smoothing.

    The A-scope trace appears as a dark line on a lighter background in the
    scanned TIFF (CRT phosphor → film → scan inversion).

    Parameters
    ----------
    robust : bool
        If True, use a constrained-argmin approach that rejects film-grain
        artefacts and T/R ringing noise.  Requires mb_x.

        Algorithm
        ---------
        1. Compute unconstrained argmin per column.
        2. Pre-filter: replace the T/R ringing region
           [0, mb_x + mb_skip_px] and frame edges (±20 px) with the noise
           floor row so the coarse trajectory is not biased by ringing.
        3. Apply a coarse Gaussian smooth (σ=30) to the pre-filtered trace
           to obtain a stable "expected" row position for every column.
        4. Constrained argmin: at each column, restrict the search band to
           [coarse − max_jump, coarse + max_jump] (max_jump = 250 px).
           This rejects isolated film-grain dark pixels far from the true
           trace while still following legitimate echo peaks.
        5. Apply fine Gaussian smooth (σ=5) to the constrained result.

        Why it works: film grain creates isolated 1–3 px dark hits; real
        echoes are 50–500 px wide.  The coarse smooth (σ=30) is largely
        unaffected by sparse grain hits (their total weight ≪ 1) but
        accurately tracks the broad echo envelope.  The ±250 px search
        window is wide enough to capture strong echo peaks (surface rise
        ≈ 900–1000 px) yet narrow enough to exclude stray grain.

    mb_x : int | None
        Main-bang column (frame-relative).  Required in robust mode.

    Returns
    -------
    trace_y   : (W,) row positions (pixels; lower = higher power)
    trace_y_s : (W,) Gaussian-smoothed row positions
    """
    H, W   = frame.shape
    y_lo   = cal["y_disp_lo"]
    y_hi   = min(cal["y_disp_hi"], H)
    band   = frame[y_lo:y_hi, :]
    bH     = band.shape[0]

    # ── Graticule masking ──────────────────────────────────────────────────────
    # Major graticule lines (10 dB horizontal rules) are printed with dense ink
    # and appear DARKER than the CRT echo trace in the globally-normalised TIFF.
    # Without masking, argmin picks a graticule line instead of the signal peak,
    # producing a systematic ~200 px (≈10 dB) power underestimation.
    #
    # Fix: set ±graticule_mask_half_px around every known major grid row to 1.0
    # (white → never selected by argmin).  Grid positions are computed exactly
    # from step-2 calibration (y_ref_px, db_per_px); the formula is
    # flight-independent and adapts to F125 / F127 / F141 automatically.
    mask_half = cal.get("graticule_mask_half_px", 3)
    y_sp_px   = 10.0 / cal["db_per_px"]          # px per 10 dB major division
    band_gm   = band.astype(float).copy()         # graticule-masked band
    n_lo_g = int(np.floor((y_lo - cal["y_ref_px"]) / y_sp_px)) - 1
    n_hi_g = int(np.ceil( (y_hi - cal["y_ref_px"]) / y_sp_px)) + 1
    for _n in range(n_lo_g, n_hi_g + 1):
        gl_abs = cal["y_ref_px"] + _n * y_sp_px   # absolute image row
        gl_rel = int(round(gl_abs - y_lo))          # band-relative row
        if 0 <= gl_rel < bH:
            r0 = max(0, gl_rel - mask_half)
            r1 = min(bH, gl_rel + mask_half + 1)
            band_gm[r0:r1, :] = 1.0               # white → never picked by argmin

    if robust and mb_x is not None:
        # ── Robust trace extraction — signal-extent–based masking ────────────
        #
        # Root cause of spurious peaks: outside the CRT sweep region (before
        # signal start, after signal end), direct argmin picks random dark
        # film-grain pixels, producing a wiggly trace with local minima
        # misidentified as echoes.
        #
        # Fix: detect_signal_extent() finds the horizontal range where the CRT
        # trace is actually present (per-column dark-pixel density above
        # threshold).  Outside that range, force trace_y to the NF reference
        # row.  Inside, use graticule-masked argmin.
        #
        # Pre-MB columns (0..mb_x) within the signal region are kept with raw
        # argmin (no graticule masking) — this feeds estimate_noise_floor().
        sig_start, sig_end = detect_signal_extent(frame, cal)

        mb_skip_px = max(1, int(cal.get("mb_skip_us", 2.0) / cal["us_per_px"]))
        tr_end     = min(W, mb_x + mb_skip_px)

        # ── Pass 1: unconstrained argmin ─────────────────────────────────────
        trace_raw = np.argmin(band,    axis=0).astype(float) + y_lo  # no grat mask
        trace_gm  = np.argmin(band_gm, axis=0).astype(float) + y_lo  # grat masked

        # Assemble pass-1 trace: NF outside signal, argmin inside
        nf_row = float(cal["y_ref_px"])
        trace_pass1 = np.full(W, nf_row)
        trace_pass1[sig_start:sig_end+1] = trace_gm[sig_start:sig_end+1]
        # Pre-MB within signal: raw argmin (for NF estimation)
        pre_mb_start = max(sig_start, 0)
        pre_mb_end   = min(mb_x, sig_end + 1)
        if pre_mb_start < pre_mb_end:
            trace_pass1[pre_mb_start:pre_mb_end] = trace_raw[pre_mb_start:pre_mb_end]
        # T/R ringing zone → NF
        trace_pass1[mb_x:tr_end] = nf_row

        # ── Pass 2: coarse smooth → constrained argmin within signal ─────────
        # The coarse smooth is now stable because non-signal columns are at NF,
        # not at random grain positions.  Within the signal region, constrain
        # the argmin to ±max_jump of the expected (coarse) position — this
        # rejects isolated film-grain dark pixels while following the real trace.
        coarse = _gauss_smooth(trace_pass1, sigma=30.0)
        max_jump = 250  # px; wide enough for echo peaks (~900 px rise)

        trace_y = trace_pass1.copy()
        for col in range(sig_start, sig_end + 1):
            expected_row = coarse[col] - y_lo  # band-relative
            lo_row = max(0,  int(expected_row - max_jump))
            hi_row = min(bH, int(expected_row + max_jump))
            if lo_row < hi_row:
                best_row = lo_row + np.argmin(band_gm[lo_row:hi_row, col])
                trace_y[col] = best_row + y_lo

        # Re-apply pre-MB raw argmin (not constrained — needed for NF estimation)
        if pre_mb_start < pre_mb_end:
            trace_y[pre_mb_start:pre_mb_end] = trace_raw[pre_mb_start:pre_mb_end]
        # Re-apply T/R ringing zone → NF
        trace_y[mb_x:tr_end] = nf_row

        # ── Fine smooth (σ=5) for stable envelope walking ────────────────────
        # Smooth the constrained trace (grain-resistant) for peak detection
        # and envelope walking.
        trace_constrained = trace_y
        trace_y_s = _gauss_smooth(trace_constrained, sigma=5.0)

        # Return the UNCONSTRAINED pass-1 trace as trace_y for peak power
        # reading.  The constrained argmin cannot reach sharp echo peaks
        # (coarse σ=30 dampens a surface peak by ~600 px, exceeding the
        # ±250 px search window).  The unconstrained argmin correctly tracks
        # the CRT trace at echo peaks; grain contamination in flat regions
        # does not matter because peak power is only read at detected peaks.
        trace_y = trace_pass1

    else:
        trace_y   = np.argmin(band_gm, axis=0).astype(float) + y_lo
        trace_y_s = _gauss_smooth(trace_y, cal["gauss_sigma"])

    return trace_y.astype(int), trace_y_s


# ── Main bang detection ────────────────────────────────────────────────────────

def detect_mb(trace_y_s: np.ndarray, cal: dict,
              search_frac: float = 0.25,
              guide_x: int | None = None) -> int:
    """
    Locate the main bang (transmitter pulse) — leftmost maximum power column.

    Three-tier detection strategy:

    1. GUIDED (guide_x from per-frame user pick):
       Narrow search window [guide_x−100, guide_x+50].  Baseline sampled from
       200 px BEFORE the window.  Finds the first 5-column sustained crossing
       20 dB above baseline, then the first local minimum ≥15 dB below baseline.

    2. BROAD FALLBACK (when guided path fails, or no guide):
       Full-frame search using heavy Gaussian smoothing (σ=20) to suppress
       film grain, then global argmin = maximum-power column.  Verified ≥10 dB
       above noise.  Handles variable MB positions (e.g. F127: MB ranges
       254–1140 px across frames in the same TIFF).

    3. LAST RESORT: guide_x or mb_x_guess (800 px).

    Parameters
    ----------
    trace_y_s   : smoothed trace Y positions (from extract_trace)
    cal         : calibration dict (uses db_per_px, y_ref_db, mb_x_guess)
    search_frac : unused; kept for API compatibility
    guide_x     : approximate main bang column from user picks (optional)

    Returns
    -------
    mb_x : column index (frame-relative)
    """
    W         = len(trace_y_s)
    db_per_px = max(cal.get("db_per_px", 0.049), 1e-6)
    mb_x: int | None = None

    # ── TIER 1: GUIDED PATH (narrow window around user click) ─────────────
    if guide_x is not None:
        lo = max(30, guide_x - 100)
        hi = min(W,  guide_x + 50)
        if hi > lo:
            # Baseline from 200 px BEFORE window — avoids leading-edge corruption
            bl_start = max(30, lo - 200)
            bl_end   = max(bl_start + 10, lo)
            baseline = float(np.percentile(trace_y_s[bl_start:bl_end], 75))

            region   = trace_y_s[lo:hi]
            thresh_y = baseline - 20.0 / db_per_px

            _run = 0
            crossing = None
            for _i, _y in enumerate(region.tolist()):
                if _y < thresh_y:
                    _run += 1
                    if _run >= 5:
                        crossing = _i - 4   # start of sustained run
                        break
                else:
                    _run = 0

            if crossing is not None:
                min_depth = 15.0 / db_per_px
                sub = region[crossing : min(len(region), crossing + 300)]
                for i in range(1, len(sub) - 1):
                    if sub[i] < sub[i - 1] and sub[i] < sub[i + 1]:
                        if (baseline - sub[i]) >= min_depth:
                            mb_x = lo + crossing + i
                            break

    # ── Fallback tiers (when guided narrow-window found nothing) ─────────
    # Noise floor reference for all fallback tiers
    global_noise_y = float(np.percentile(trace_y_s, 75))
    min_depth_px   = 10.0 / db_per_px

    # ── TIER 2: GUIDE POSITION CHECK ─────────────────────────────────────
    # When the guided narrow-window algorithm fails but guide_x exists,
    # check whether guide_x itself sits near a strong signal.  This handles
    # flights with STABLE MB positions (F125 ~800, F141 ~540) where
    # tiff_mb_estimate is correct but the narrow-window algorithm couldn't
    # find a clean crossing (e.g. noisy trace, edge effects).
    # We search ±20 px around guide_x for the local minimum and verify
    # it's ≥10 dB above noise.  This avoids falling through to the broad
    # search, which can latch onto pre-trigger artifacts (observed in F141:
    # artifact at ~150 px, −17.8 dB, stronger than the real MB at ~540 px).
    if mb_x is None and guide_x is not None:
        check_lo  = max(0, guide_x - 20)
        check_hi  = min(W, guide_x + 20)
        local_min = float(np.min(trace_y_s[check_lo:check_hi]))
        if (global_noise_y - local_min) >= min_depth_px:
            mb_x = check_lo + int(np.argmin(trace_y_s[check_lo:check_hi]))

    # ── TIER 3: BROAD FALLBACK (full-frame search) ────────────────────────
    # Used when guide_x was WRONG (tiff_mb_estimate far from real MB,
    # as observed in F127 where MB ranges 254–1140 px across frames due
    # to inconsistent oscilloscope trigger timing).  Guide position check
    # (Tier 2) found noise floor at guide_x → guide_x is wrong →
    # search the full frame.
    #
    # Strategy: heavy smooth (σ=20) suppresses film grain, then global
    # argmin finds the strongest signal.  Not used when guide_x is
    # correct (Tier 2 catches that case), so pre-trigger artifacts at
    # x < 200 are only a concern when guide_x was already wrong.
    if mb_x is None:
        broad_smooth = _gauss_smooth(trace_y_s.astype(float), sigma=20)
        search_lo    = 50
        search_hi    = max(search_lo + 1, W - 50)
        candidate    = search_lo + int(np.argmin(broad_smooth[search_lo:search_hi]))
        if (global_noise_y - broad_smooth[candidate]) >= min_depth_px:
            mb_x = candidate

    # ── TIER 4: LAST RESORT ───────────────────────────────────────────────
    if mb_x is None:
        mb_x = guide_x if guide_x is not None else cal.get("mb_x_guess", 800)

    # Sanity: main bang must be well above the noise-floor reference.
    # Threshold relaxed from 15 dB to 10 dB to accommodate weaker frames
    # (e.g. F127 CBD0421 at −45.9 dB = 14.1 dB above noise floor).
    mb_power = float(px_to_db(float(trace_y_s[mb_x]), cal))
    if mb_power < cal["y_ref_db"] + 10:
        mb_x = guide_x if guide_x is not None else cal.get("mb_x_guess", 800)

    return mb_x


# ── Noise floor estimation ─────────────────────────────────────────────────────

def estimate_noise_floor(trace_y_s: np.ndarray,
                         mb_x: int,
                         cal: dict) -> float:
    """
    Estimate per-frame noise floor from the pre-main-bang baseline.

    The region before the main bang contains no real signal, so the
    trace sits at the system noise floor level.

    Returns
    -------
    noise_floor_dB : estimated noise floor in dB
    """
    pre_mb = trace_y_s[:max(1, mb_x - 30)]
    if len(pre_mb) >= 10:
        # Median of the highest-Y (lowest-power) portion → noise baseline
        p75 = float(np.percentile(pre_mb, 75))
        baseline_px = float(np.median(pre_mb[pre_mb >= p75]))
        return float(px_to_db(baseline_px, cal))
    return float(cal["y_ref_db"])


# ── Echo peak detection ────────────────────────────────────────────────────────

def detect_echoes(trace_y_s: np.ndarray,
                  mb_x: int,
                  noise_floor_dB: float,
                  cal: dict) -> tuple[int | None, int | None]:
    """
    Detect surface and bed echo peak columns.

    Surface: first prominent peak after main bang decay.
    Bed:     second prominent peak, at least 0.3 µs after surface.

    Returns
    -------
    surface_x : frame-relative column of surface peak, or None
    bed_x     : frame-relative column of bed peak, or None
    """
    W = len(trace_y_s)

    # Skip main bang + its decay
    x_start = mb_x + int(cal["mb_skip_us"] / cal["us_per_px"])
    x_end   = min(W - 1, mb_x + int(20.0 / cal["us_per_px"]))

    if x_start >= x_end:
        return None, None

    region = trace_y_s[x_start:x_end]

    # find_peaks on −trace (minima in y = maxima in power)
    prom_px  = cal["peak_min_prom_dB"] / cal["db_per_px"]
    dist_px  = cal["peak_min_dist_px"]
    peak_rel = _find_peaks_numpy(region, prominence=prom_px, distance=dist_px)

    if len(peak_rel) == 0:
        return None, None

    peak_cols   = peak_rel + x_start
    peak_powers = np.array([float(px_to_db(trace_y_s[c], cal)) for c in peak_cols])

    # Must be above noise floor by at least 3 dB
    valid       = peak_powers - noise_floor_dB > 3.0
    peak_cols   = peak_cols[valid]
    peak_powers = peak_powers[valid]

    if len(peak_cols) == 0:
        return None, None

    # ── Surface selection ──────────────────────────────────────────────────────
    # Default: first peak after mb_skip_us.  Robustness guard: if a later peak
    # within the surface search window is substantially stronger (>min_surface_
    # lead_dB), prefer it.  This rejects weak T/R ringing artefacts that slip
    # past mb_skip_us (e.g. CBD0461 false peak at 2.77 µs, 30 dB below real
    # surface at 5.68 µs).  The surface_window_us bound prevents accidentally
    # promoting the bed echo (which is outside the window).
    srf_win_px   = mb_x + int(cal.get("surface_window_us", 8.0) / cal["us_per_px"])
    srf_mask     = peak_cols <= srf_win_px
    srf_cands    = peak_cols[srf_mask]
    srf_powers_c = peak_powers[srf_mask]

    if len(srf_cands) == 0:           # no peak in window → fall back to first
        srf_cands    = peak_cols[:1]
        srf_powers_c = peak_powers[:1]

    if len(srf_cands) >= 2:
        first_power = srf_powers_c[0]
        best_idx    = int(np.argmax(srf_powers_c))
        best_power  = srf_powers_c[best_idx]
        # Promote strongest candidate if first peak is a weak artefact
        if best_power - first_power > cal.get("min_surface_lead_dB", 15.0):
            surface_x = int(srf_cands[best_idx])
        else:
            surface_x = int(srf_cands[0])
    else:
        surface_x = int(srf_cands[0])

    if len(peak_cols) < 2:
        return surface_x, None

    # Bed: first peak at least min_bed_gap_us after surface (~168 m minimum ice)
    # that also passes the quiet-gap check.
    #
    # Quiet-gap check: somewhere in the window [surface_x + lead_px, cand_x],
    # the smoothed trace must return within quiet_gap_dB of the noise floor.
    # This rejects candidates that are continuous with the surface echo (e.g.
    # phosphor afterglow, trailing ringing, film blemishes in the inter-echo
    # region) while accepting genuine bed echoes, which are preceded by a quiet
    # region where the surface echo has fully decayed.
    # The check is skipped when the gap is < 1 µs (very thin ice, h < 84 m).
    min_gap_px    = int(cal.get("min_bed_gap_us", 2.0) / cal["us_per_px"])
    candidates    = peak_cols[peak_cols >= surface_x + min_gap_px]
    if len(candidates) == 0:
        return surface_x, None

    quiet_lead_px   = int(cal.get("quiet_gap_lead_us", 1.0)  / cal["us_per_px"])
    quiet_gap_px    = cal.get("quiet_gap_dB",   12.0) / cal["db_per_px"]
    nf_row          = float(cal["y_ref_px"])
    min_bed_w_px    = int(cal.get("min_bed_width_us",  0.5) / cal["us_per_px"])
    width_thresh_y  = nf_row - cal.get("min_bed_width_dB", 10.0) / cal["db_per_px"]
    W               = len(trace_y_s)

    # Collect ALL candidates that pass quiet-gap and width checks, then
    # select the most prominent one.  Previous logic took the first passing
    # candidate, which misidentified weak intermediate features (surface
    # ringing, shallow internal reflectors) as the bed when the real bed
    # echo was further away and much stronger.
    valid_beds = []   # list of (column, power_dB)
    for cand in candidates:
        c      = int(cand)
        gap_lo = surface_x + quiet_lead_px
        gap_hi = c

        # ── Quiet-gap check ───────────────────────────────────────────────────
        # Skip when gap is < 1 µs (very thin ice, h ≤ 84 m)
        if gap_hi - gap_lo >= int(1.0 / cal["us_per_px"]):
            gap_trace = trace_y_s[gap_lo:gap_hi]
            if float(np.max(gap_trace)) < nf_row - quiet_gap_px:
                continue   # trace never returns to NF → continuous artifact

        # ── Width check ───────────────────────────────────────────────────────
        # The smoothed trace must stay above width_thresh_y for at least
        # min_bed_w_px consecutive columns around the candidate.
        # Real bed echoes: ≥ 0.83 µs wide (F125 TIFF 8400 validated data).
        # Film artifacts / surface-echo ringing: < 0.45 µs wide.
        w_lo    = max(0, c - min_bed_w_px * 3)
        w_hi    = min(W, c + min_bed_w_px * 3)
        above   = (trace_y_s[w_lo:w_hi] < width_thresh_y).astype(np.int8)
        # Maximum consecutive run of True (columns clearly above threshold)
        padded  = np.zeros(len(above) + 2, dtype=np.int8)
        padded[1:-1] = above
        diff    = np.diff(padded)
        starts  = np.where(diff > 0)[0]
        ends    = np.where(diff < 0)[0]
        max_run = 0 if len(starts) == 0 else int((ends - starts).max())
        if max_run < min_bed_w_px:
            continue   # too narrow → film artifact or surface-echo ringing

        cand_power = float(px_to_db(trace_y_s[c], cal))
        valid_beds.append((c, cand_power))

    # Select the most prominent (strongest power) bed candidate
    if valid_beds:
        bed_x = int(max(valid_beds, key=lambda b: b[1])[0])
    else:
        bed_x = None

    return surface_x, bed_x


# ── Envelope walker ────────────────────────────────────────────────────────────

def walk_envelope(trace_y_s: np.ndarray,
                  x_peak: int,
                  thresh_db: float,
                  mb_x: int,
                  frame_w: int,
                  cal: dict) -> tuple[float, float, float]:
    """
    Walk outward from a confirmed peak to find echo leading/trailing edges
    at a given power threshold.

    Leading edge  : walk LEFT until trace crosses threshold (first crossing).
    Trailing edge : walk RIGHT until `consec` consecutive columns are below
                    threshold (confirming sustained return to baseline).

    Parameters
    ----------
    thresh_db : absolute threshold in dB (e.g. noise_floor_dB + 10)

    Returns
    -------
    lead_us, trail_us, width_us : TWT values (µs from main bang)
    """
    thresh_y = db_to_px(thresh_db, cal)   # pixel row at threshold power
    consec   = cal["consec"]

    # Physical TWT bound for the rightward walk: cap at max_walk_us from MB.
    # Prevents the envelope extending into frame separators (all-white columns
    # look like "very high power" because argmin returns y_disp_lo there) or
    # beyond the maximum realistic two-way travel time for RIS geometry.
    max_walk_us = cal.get("max_walk_us", 17.0)
    x_right_max = min(frame_w - 1,
                      mb_x + int(max_walk_us / cal["us_per_px"]))

    # Walk LEFT from peak
    lead = x_peak
    for x in range(x_peak, max(0, x_peak - 1500), -1):
        if trace_y_s[x] >= thresh_y:   # trace at or below threshold
            lead = x
            break
    else:
        lead = max(0, x_peak - 1500)

    # Walk RIGHT — require `consec` consecutive below-threshold columns
    trail = x_peak
    count = 0
    for x in range(x_peak, x_right_max):
        if trace_y_s[x] >= thresh_y:
            count += 1
            if count >= consec:
                trail = x - consec
                break
        else:
            count = 0
    else:
        trail = x_right_max

    lead_us  = float(px_to_us(lead,  mb_x, cal))
    trail_us = float(px_to_us(trail, mb_x, cal))
    return lead_us, trail_us, float(trail_us - lead_us)


# ── Peakiness ─────────────────────────────────────────────────────────────────

def compute_peakiness(trace_y_s: np.ndarray,
                      x_peak: int,
                      lead_px: int,
                      trail_px: int,
                      cal: dict) -> float:
    """
    Peakiness = peak power (linear) / mean power (linear) within ±10 dB window.

    > 1 : sharply peaked (specular reflector / liquid water)
    ≈ 1 : flat-topped (rough / noise-dominated)
    """
    x0 = max(0, lead_px)
    x1 = min(len(trace_y_s) - 1, trail_px) + 1
    if x1 <= x0 + 2:
        return np.nan

    powers_dB  = np.array([float(px_to_db(trace_y_s[x], cal)) for x in range(x0, x1)])
    powers_lin = 10.0 ** (powers_dB / 10.0)
    peak_lin   = 10.0 ** (float(px_to_db(int(round(trace_y_s[x_peak])), cal)) / 10.0)

    mean_lin = float(np.mean(powers_lin))
    if mean_lin <= 0:
        return np.nan
    return float(peak_lin / mean_lin)


# ── Per-echo full metrics ──────────────────────────────────────────────────────

def compute_echo_metrics(trace_y_s: np.ndarray,
                         x_peak: int,
                         noise_floor_dB: float,
                         mb_x: int,
                         frame_w: int,
                         cal: dict,
                         trace_y: np.ndarray | None = None) -> EchoResult:
    """
    Compute complete waveform metrics for one detected echo peak.

    Parameters
    ----------
    trace_y : optional raw (pre-smooth) trace from extract_trace.
        When supplied, peak power is read from the raw argmin within ±2 px
        of x_peak rather than from the smoothed trace.  The smoothed trace
        (trace_y_s) is still used for all envelope-walking operations.
        Without trace_y, peak power falls back to trace_y_s[x_peak].
    """
    # Peak power: use the raw trace (or a local window around x_peak) so that
    # Gaussian smoothing does not bias the peak value downward.
    # Window ±15 px: the smoothed trace can shift the peak column by several
    # pixels (asymmetric echoes), so a narrow ±2 px window may miss the true
    # minimum.  ±15 px covers the peak of any real echo while remaining small
    # enough to avoid picking up adjacent features.
    if trace_y is not None:
        lo  = max(0, x_peak - 15)
        hi  = min(len(trace_y), x_peak + 16)
        peak_y = int(np.min(trace_y[lo:hi]))
    else:
        peak_y = int(round(trace_y_s[x_peak]))
    peak_twt   = float(px_to_us(x_peak, mb_x, cal))
    peak_power = float(px_to_db(peak_y, cal))
    peak_snr   = peak_power - noise_floor_dB

    result = EchoResult(
        peak_x=x_peak, peak_y=peak_y,
        peak_twt_us=peak_twt,
        peak_power_dB=peak_power,
        peak_snr_dB=peak_snr,
    )

    # Envelope at +10 dB above measured noise floor
    thresh_10 = noise_floor_dB + 10.0
    if peak_power > thresh_10:
        l10, t10, w10 = walk_envelope(
            trace_y_s, x_peak, thresh_10, mb_x, frame_w, cal)
        result.lead_10_us  = l10
        result.trail_10_us = t10
        result.width_10_us = w10
        result.leading_rise_us = peak_twt - l10

        lead_10_px  = us_to_px(l10, mb_x, cal)
        trail_10_px = us_to_px(t10, mb_x, cal)
        result.peakiness = compute_peakiness(
            trace_y_s, x_peak, lead_10_px, trail_10_px, cal)

        trailing_w = t10 - peak_twt
        leading_w  = peak_twt - l10
        if leading_w > 1e-6:
            result.asymmetry = trailing_w / leading_w

    # Envelope at +5 dB above measured noise floor
    thresh_5 = noise_floor_dB + 5.0
    if peak_power > thresh_5:
        l5, t5, w5 = walk_envelope(
            trace_y_s, x_peak, thresh_5, mb_x, frame_w, cal)
        result.lead_5_us  = l5
        result.trail_5_us = t5
        result.width_5_us = w5
        if not np.isnan(result.trail_10_us):
            result.trailing_tail_us = t5 - result.trail_10_us

    return result


# ── Y grid lines: extrapolation (graticule too faint to detect directly) ──────

def compute_y_grid_lines(y_ref_px: float,
                          y_spacing_px: float,
                          y_disp_lo: int,
                          y_disp_hi: int,
                          n_above: int = 6,
                          n_below: int = 6,
                          ) -> list[float]:
    """
    Compute Y (horizontal) grid line positions by extrapolation.

    CRITICAL: The −60 dB reference line sits BETWEEN two major graticule
    divisions at −55 dB and −65 dB, not on a division.  The offset from
    y_ref_px to the nearest graticule line above (−55 dB) is half a major
    division ≈ y_spacing_px / 2 ≈ 102.5 px (validated from F125 Frame 8:
    y_ref=1507, nearest grid=1404 → offset=103 px, from detect_gridlines3.py).

    Graticule lines: …, −45, −55, −65, −75, … dB
    NOT at:         …, −50, −60, −70, −80, … dB

    Formula:
        y_first = y_ref_px − y_spacing_px / 2    (→ −55 dB)
        y_grid[k] = y_first + k × y_spacing_px

    Returns
    -------
    Sorted list of Y row positions within [y_disp_lo, y_disp_hi].
    """
    # First graticule above the reference line (−55 dB)
    y_first = y_ref_px - y_spacing_px / 2.0
    lines = []
    for k in range(-n_above, n_below + 1):
        y = y_first + k * y_spacing_px
        if y_disp_lo <= y <= y_disp_hi:
            lines.append(float(y))
    return sorted(lines)


# ── X grid lines: argmax in below-baseline band (validated approach) ──────────

def detect_xgrid_lines(frame: np.ndarray,
                        mb_x: int,
                        x_spacing_px: float,
                        cal: dict,
                        n_major: int = 14,
                        x_band: tuple[int, int] = (1600, 1695),
                        gauss_sigma: float = 10.0,
                        half_search: int = 35,
                        inlier_thresh_px: float = 8.0,
                        guide_x_grid: list[int] | None = None,
                        d_from_mb: float | None = None,
                        ) -> tuple[np.ndarray, float]:
    """
    Detect X (vertical) major grid lines using column-mean in a y-band
    between the noise floor and the bottom display line.

    Critical band selection: y=[1600, 1695] sits BETWEEN the noise floor
    (y_ref ≈ 1512) and the bottom major horizontal display line (y ≈ 1717).
    In this region only MAJOR vertical grid lines are visible — minor tick
    notches exist only along horizontal major lines (y≈1717, 1512, …) and
    do not appear inside this intermediate band.  The previous band (1550, 1800)
    included y≈1717 where minor tick notches at every x_spacing/5 ≈ 41 px
    scored higher than the diffuse major vertical lines, causing the phase
    search to lock onto minor tick positions (offset 1–4 minor ticks from
    the true major tick phase).

    Key design note: the graticule is NOT anchored at mb_x.  The first tick to
    the right of mb_x is typically 69–88 px away (F125 data), well beyond the
    ±half_search snap window.  The algorithm therefore:
      1. Finds the FIRST tick by argmax over the full first division
         [mb_x, mb_x + x_spacing_px] in the below-baseline strip.
      2. Builds all subsequent guides as first_x + k * x_spacing_px.
      3. Applies ±half_search snap to each guide (handles inter-frame drift).

    Parameters
    ----------
    frame         : (H, W) float32 frame array [0, 1]
    mb_x          : main bang column (anchor for guide positions)
    x_spacing_px  : expected major-division spacing in pixels
    cal           : calibration dict (uses y_disp_lo, y_disp_hi)
    n_major       : how many major lines to search for (rightward from mb_x)
    x_band        : (y_lo, y_hi) rows of below-baseline band
    gauss_sigma   : smoothing sigma for column mean profile
    half_search   : ±px search window around each guide position
    inlier_thresh_px : residual threshold for inlier-only linear fit

    Returns
    -------
    x_fitted     : (n_major,) array of fitted X positions (float)
    spacing_px   : spacing from inlier linear fit (px / major division)
    """
    H, W = frame.shape
    y_lo = min(x_band[0], H - 2)
    y_hi = min(x_band[1], H)

    if y_hi <= y_lo:
        guides = np.array([mb_x + k * x_spacing_px for k in range(n_major)
                           if 0 < mb_x + k * x_spacing_px < W])
        return guides.astype(float), x_spacing_px

    below      = frame[y_lo:y_hi, :]
    col_mean   = np.mean(below, axis=0)
    col_smooth = _gauss_smooth(col_mean, gauss_sigma)

    # ── Find the first grid tick ───────────────────────────────────────────
    # The tick marks appear as bright columns in the below-baseline band.
    #
    # Priority A: use clicked x_grid guide positions (most accurate).
    # The user may click on any major division; back-calculate first_x from it.
    # If 2+ clicks are provided, also re-measure x_spacing_px from their spacing —
    # this corrects for per-TIFF oscilloscope time-base differences that would
    # otherwise cause progressive drift (each tick adds the per-division error).
    #
    # Priority B: grid phase search — NO reliance on mb_x.
    # Film frame widths vary significantly between frames (e.g. 2595–3250 px in
    # F127), so the CRT display starts at different frame-relative positions.
    # mb_x also jitters ±10–30 px due to detection noise.  Instead, we scan all
    # possible grid phases x0 in [0, x_spacing_px) and pick the one that maximises
    # the mean of col_smooth at {x0, x0+sp, x0+2sp, …}.  This is robust to any
    # per-frame CRT registration variation or mb_x inconsistency.
    if guide_x_grid:
        # Priority A
        x_sorted = sorted(float(v) for v in guide_x_grid)
        if len(x_sorted) >= 2:
            dx     = x_sorted[-1] - x_sorted[0]
            n_div  = max(1, round(dx / x_spacing_px))   # divisions between clicks
            x_spacing_px = dx / n_div                   # measured spacing for this TIFF
        x_seed      = x_sorted[0]
        k_seed      = max(0, round((x_seed - mb_x) / x_spacing_px))
        first_guess = x_seed - k_seed * x_spacing_px
        snap_lo = max(0, int(first_guess) - 15)
        snap_hi = min(W, int(first_guess) + 16)
        if snap_hi > snap_lo:
            first_x = float(snap_lo + np.argmax(col_smooth[snap_lo:snap_hi]))
        else:
            first_x = first_guess
    elif d_from_mb is not None:
        # Priority A2: D_anchor — constant distance from mb_x to a known major tick.
        #
        # Physical basis: the oscilloscope sweep is triggered from the radar transmit
        # pulse, so all waveform content is time-locked to the graticule.  D = tick_x
        # − mb_x is constant across frames within a TIFF (same oscilloscope time base).
        #
        # Strategy: predict the anchor tick's frame-relative position as
        # anchor_pred = mb_x + d_from_mb, snap ±half_search around it to find the
        # actual tick, then back-calculate first_x = anchor_x − k_anchor * spacing.
        # The anchor lands in the right portion of the frame (past the bright left-side
        # artifact), giving a reliable snap even when major ticks are faint.
        k_anchor    = max(0, round(d_from_mb / x_spacing_px))
        anchor_pred = float(mb_x) + d_from_mb
        a_lo        = max(0, int(anchor_pred) - half_search)
        a_hi        = min(W, int(anchor_pred) + half_search + 1)
        if a_hi > a_lo:
            anchor_x = float(a_lo + np.argmax(col_smooth[a_lo:a_hi]))
        else:
            anchor_x = anchor_pred
        first_x = anchor_x - k_anchor * x_spacing_px
    else:
        # Priority B: grid phase search — right-portion only.
        #
        # The left side of every frame (~first 35% of width) contains a large
        # bright artifact in the below-baseline band (from the inter-frame
        # boundary, CRT unblank, and main-bang region bleeding downward).
        # Using the full frame causes the phase search to lock onto that bright
        # region rather than the faint but periodic graticule tick marks.
        #
        # Fix: score only positions in the right 65% of the frame (x > W*0.35).
        # This region is after the main bang and early echoes, where the trace
        # sits at the noise floor and the only periodic features are the CRT
        # graticule ticks.  Typically ≥8 major ticks remain in this region —
        # sufficient for reliable phase estimation.
        #
        # Coarse sweep over all phases (~200 steps), then 0.1 px fine refinement.
        step     = max(0.5, x_spacing_px / 200.0)
        k_arr    = np.arange(n_major, dtype=float)
        W_sm     = len(col_smooth)
        x_start  = 200                         # exclude only inter-frame border (~0-200 px)
        min_ticks = 4                          # require at least this many valid ticks
        best_score = -np.inf
        best_x0    = 0.0
        for x0 in np.arange(0.0, x_spacing_px, step):
            idxs  = np.round(x0 + k_arr * x_spacing_px).astype(int)
            valid = (idxs >= x_start) & (idxs < W_sm)
            if valid.sum() >= min_ticks:
                s = col_smooth[idxs[valid]].mean()
                if s > best_score:
                    best_score = s
                    best_x0    = x0
        # Fine refinement ±2 coarse steps around best_x0
        for x0 in np.arange(max(0.0,          best_x0 - 2 * step),
                             min(x_spacing_px, best_x0 + 2 * step + 0.01), 0.1):
            idxs  = np.round(x0 + k_arr * x_spacing_px).astype(int)
            valid = (idxs >= x_start) & (idxs < W_sm)
            if valid.sum() >= min_ticks:
                s = col_smooth[idxs[valid]].mean()
                if s > best_score:
                    best_score = s
                    best_x0    = x0
        first_x = float(best_x0)

    # Guide positions anchored at first_x
    guides = []
    for k in range(n_major):
        gx = first_x + k * x_spacing_px
        if 0 < gx < W:
            guides.append(gx)
    guides = np.array(guides)

    if len(guides) < 2:
        return guides.astype(float), x_spacing_px

    # Raw detection: ±half_search snap around each guide (accounts for spacing drift)
    raw_x = np.empty(len(guides), dtype=float)
    # k=0 already resolved above (first_x); still snap to refine sub-pixel position
    for i, g in enumerate(guides):
        lo = max(0, int(g) - half_search)
        hi = min(W, int(g) + half_search)
        if hi > lo:
            raw_x[i] = lo + float(np.argmax(col_smooth[lo:hi]))
        else:
            raw_x[i] = g

    # Initial linear fit across all guides
    k_all   = np.arange(len(raw_x), dtype=float)
    c_all   = np.polyfit(k_all, raw_x, 1)
    resid   = raw_x - np.polyval(c_all, k_all)

    # Inlier-only refit (validated approach)
    inliers = np.abs(resid) <= inlier_thresh_px
    if inliers.sum() >= 2:
        c_in    = np.polyfit(k_all[inliers], raw_x[inliers], 1)
        spacing = abs(float(c_in[0]))
    else:
        c_in    = c_all
        spacing = abs(float(c_all[0]))

    x_fitted = np.polyval(c_in, k_all)
    return x_fitted, spacing


# ── Per-frame calibration ──────────────────────────────────────────────────────

def detect_frame_calibration(
        frame: np.ndarray,
        default_cal: dict,
        guides: dict | None = None,
) -> dict:
    """
    Per-frame calibration using validated algorithms.

    Steps
    -----
    A. Main bang  (mb_x): validated argmin approach — find the leftmost
       column where the trace is deflected highest.  If a guide pick is
       provided, narrows search to ±200 px around it (find_mainbang2.py).

    B. Y reference (y_ref_px, −60 dB): pre-main-bang trace median sits at
       the system noise floor = the −60 dB reference line.

    C. X grid lines: column-mean argmax in below-baseline band y=[1550,1800].
       Grid tick marks are bright; we fit an inlier-only linear model
       (detect_gridlines3.py).

    D. Y grid lines: PURE EXTRAPOLATION from y_ref_px + k × y_spacing_px.
       The horizontal graticule lines are too faint to detect algorithmically
       (confirmed empirically — detect_gridlines3.py comment: "No algorithmic
       row-mean detection — graticule lines too faint").

    Parameters
    ----------
    frame       : (H, W) float32 array [0, 1], already cropped to this frame
    default_cal : baseline calibration dict (DEFAULT_CAL or previous per-frame)
    guides      : optional dict from pick_calibration.py output, with keys:
                    'mb'          – int, approximate main bang x (frame-rel)
                    'ref'         – int, approximate y_ref_px row
                    'x_spacing_px'– float, X major-division spacing
                    'y_spacing_px'– float, Y major-division spacing

    Returns
    -------
    cal : updated dict with added keys:
        mb_x           – detected main bang column (frame-relative pixels)
        mb_power_dB    – power at main bang (dB)
        y_ref_px       – noise-floor row = −60 dB anchor (pixels)
        db_per_px      – Y scale (dB/px; from y_spacing_px)
        hgrid_lines    – list of Y grid line rows (extrapolated)
        hgrid_spacing  – Y major-division spacing used (px)
        xgrid_lines    – list of X grid line cols (detected)
        x_spacing_px   – X major-division spacing (px; from inlier fit)
        cal_source_y   – always 'extrapolated'
    """
    guides = guides or {}
    cal    = default_cal.copy()

    # Derive spacing constants: prefer guide, then default_cal, then DEFAULT_CAL
    x_spacing_px = float(guides.get("x_spacing_px",
                         default_cal.get("x_spacing_px",
                         1.5 / default_cal["us_per_px"])))
    y_spacing_px = float(guides.get("y_spacing_px",
                         default_cal.get("y_spacing_px",
                         10.0 / default_cal["db_per_px"])))

    # ── A: Main bang ──────────────────────────────────────────────────────
    trace_y, trace_y_s = extract_trace(frame, cal)
    guide_mb           = guides.get("mb")
    mb_x               = detect_mb(trace_y_s, cal, guide_x=guide_mb)
    mb_power_dB        = float(px_to_db(float(trace_y_s[mb_x]), cal))

    cal["mb_x"]        = mb_x
    cal["mb_power_dB"] = mb_power_dB

    # ── B: Y reference (noise floor = −60 dB) ────────────────────────────
    guide_ref = guides.get("ref")
    if guide_ref is not None:
        # Guide provided: use it directly (user confirmed the reference line)
        y_ref_px = float(guide_ref)
    else:
        # No guide: sample from the MIDDLE of the pre-bang region.
        # Avoid frame-boundary artifacts (cols 0–30) and the main bang's
        # leading edge (last ~50 px before mb_x).
        # Safe window: mb_x//4 → mb_x//2, clamped away from the bang.
        pre_lo = max(30, mb_x // 4)
        pre_hi = min(max(pre_lo + 10, mb_x // 2), max(pre_lo + 5, mb_x - 50))
        if pre_hi > pre_lo + 5:
            pre_trace = trace_y_s[pre_lo:pre_hi]
            y_ref_px  = float(np.percentile(pre_trace, 75))
            # Sanity: must lie within the oscilloscope display band
            if not (cal["y_disp_lo"] < y_ref_px < cal["y_disp_hi"]):
                warnings.warn(
                    f"y_ref estimate {y_ref_px:.0f} px outside display band "
                    f"[{cal['y_disp_lo']}, {cal['y_disp_hi']}] — using default")
                y_ref_px = float(default_cal["y_ref_px"])
        else:
            y_ref_px = float(default_cal["y_ref_px"])

    cal["y_ref_px"] = y_ref_px
    cal["y_ref_db"] = -60.0

    # ── C: X grid lines (argmax in below-baseline band) ──────────────────
    x_fitted, x_spacing_detected = detect_xgrid_lines(
        frame, mb_x, x_spacing_px, cal,
        guide_x_grid=guides.get("x_grid"),
        d_from_mb=guides.get("d_from_mb"))
    cal["xgrid_lines"]  = x_fitted.tolist()
    cal["x_spacing_px"] = x_spacing_detected

    # ── D: Y grid lines (pure extrapolation) ─────────────────────────────
    hgrid_lines = compute_y_grid_lines(
        y_ref_px, y_spacing_px,
        cal["y_disp_lo"], cal["y_disp_hi"])
    cal["hgrid_lines"]  = hgrid_lines
    cal["hgrid_spacing"] = y_spacing_px
    cal["db_per_px"]     = 10.0 / y_spacing_px
    cal["cal_source_y"]  = "extrapolated"

    return cal


# ── Main entry: process one TIFF ──────────────────────────────────────────────

def process_tiff(tiff_path: str | Path,
                 cal: dict | None = None,
                 frame_bounds: list[tuple[int, int]] | None = None,
                 verbose: bool = True) -> list[LyraFrame]:
    """
    Process all A-scope frames in a TIFF and extract waveform metrics.

    Parameters
    ----------
    tiff_path    : path to raw A-scope TIFF
    cal          : calibration dict (uses DEFAULT_CAL if None)
    frame_bounds : pre-computed (left, right) boundaries; auto-detected if None
    verbose      : print per-frame progress

    Returns
    -------
    list of LyraFrame, one per detected frame
    """
    if cal is None:
        cal = DEFAULT_CAL.copy()
    tiff_path = Path(tiff_path)

    if verbose:
        print(f"\nLYRA v1.0 — processing {tiff_path.name}")
        print("─" * 70)

    Image.MAX_IMAGE_PIXELS = None
    img      = np.array(Image.open(tiff_path), dtype=np.float32)
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-9)
    H, W     = img_norm.shape

    if verbose:
        print(f"  Image : {W} × {H} px")

    if frame_bounds is None:
        frame_bounds = detect_frames(img_norm)
    if verbose:
        print(f"  Frames detected : {len(frame_bounds)}")
        print()

    results: list[LyraFrame] = []

    for idx, (left, right) in enumerate(frame_bounds):
        frame   = img_norm[:, left : right + 1]
        frame_w = frame.shape[1]
        lf      = LyraFrame(
            frame_idx=idx, frame_left=left, frame_right=right, frame_w=frame_w)

        try:
            trace_y, trace_y_s = extract_trace(frame, cal)
            mb_x               = detect_mb(trace_y_s, cal)
            lf.mb_x            = mb_x
            lf.noise_floor_dB  = estimate_noise_floor(trace_y_s, mb_x, cal)

            surface_x, bed_x = detect_echoes(
                trace_y_s, mb_x, lf.noise_floor_dB, cal)

            if surface_x is not None:
                lf.surface          = compute_echo_metrics(
                    trace_y_s, surface_x, lf.noise_floor_dB, mb_x, frame_w, cal)
                lf.surface_detected = True
            else:
                lf.quality = "no_surface"

            if bed_x is not None:
                lf.bed          = compute_echo_metrics(
                    trace_y_s, bed_x, lf.noise_floor_dB, mb_x, frame_w, cal)
                lf.bed_detected = True
            else:
                if lf.quality == "ok":
                    lf.quality = "no_bed"

        except Exception as exc:
            warnings.warn(f"Frame {idx}: {exc}")
            lf.quality = "failed"

        results.append(lf)

        if verbose:
            s = f"  [{idx:2d}] MB={lf.mb_x:4d}px  NF={lf.noise_floor_dB:+.1f} dB"
            if lf.surface_detected:
                s += (f"  | surf {lf.surface.peak_twt_us:.2f}µs"
                      f" {lf.surface.peak_power_dB:+.1f}dB"
                      f" SNR={lf.surface.peak_snr_dB:.1f}dB")
            if lf.bed_detected:
                s += (f"  | bed {lf.bed.peak_twt_us:.2f}µs"
                      f" {lf.bed.peak_power_dB:+.1f}dB"
                      f" SNR={lf.bed.peak_snr_dB:.1f}dB"
                      f"  peak={lf.bed.peakiness:.2f}"
                      f"  asym={lf.bed.asymmetry:.2f}")
            if lf.quality != "ok":
                s += f"  [{lf.quality}]"
            print(s)

    return results


# ── DataFrame serialisation ────────────────────────────────────────────────────

def to_dataframe(results: list[LyraFrame],
                 tiff_path: str | Path | None = None) -> pd.DataFrame:
    """
    Flatten list of LyraFrame into a tidy DataFrame (one row per frame).

    Column naming convention: {echo}_{metric}  (e.g. bed_peakiness)
    """
    tiff_name = Path(tiff_path).name if tiff_path else ""
    rows = []

    for lf in results:
        row: dict = dict(
            tiff             = tiff_name,
            frame_idx        = lf.frame_idx,
            frame_left       = lf.frame_left,
            frame_right      = lf.frame_right,
            frame_w          = lf.frame_w,
            mb_x             = lf.mb_x,
            noise_floor_dB   = lf.noise_floor_dB,
            surface_detected = lf.surface_detected,
            bed_detected     = lf.bed_detected,
            quality          = lf.quality,
        )

        # Echo columns with echo prefix
        for prefix, echo in [("surface", lf.surface), ("bed", lf.bed)]:
            for k, v in asdict(echo).items():
                row[f"{prefix}_{k}"] = v

        # Derived geometry (physical distances)
        if lf.surface_detected:
            row["h_air_m"] = lf.surface.peak_twt_us / 2.0 * C_AIR_M_PER_US
        else:
            row["h_air_m"] = np.nan

        if lf.surface_detected and lf.bed_detected:
            twt_ice        = lf.bed.peak_twt_us - lf.surface.peak_twt_us
            row["h_ice_m"] = twt_ice / 2.0 * C_ICE_M_PER_US
        else:
            row["h_ice_m"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)
