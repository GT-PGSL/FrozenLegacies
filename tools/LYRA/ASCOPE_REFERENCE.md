# ASCOPE_REFERENCE.md — A-scope Physics, Calibration & LYRA Pipeline

Reference document for the LYRA (Layered-echo Yield from Radiometric Archives) pipeline. Covers the physical basis of 60 MHz A-scope radar data from the 1974--75 SPRI/TUD/NSF Antarctic surveys, the calibration constants used in LYRA, and the algorithms behind each processing step.

All values validated against Neal 1977 Fig 1.3a (CBD 0465, F125).

---

## 1. The A-scope Record

### 1.1 What the Film Shows

Each A-scope frame is a single radar pulse return, photographed from the CRT screen of a Tektronix 465 oscilloscope onto 35 mm film. The horizontal axis is **time** (proportional to distance from the aircraft) and the vertical axis is **received power** (in dB). A bright phosphor trace sweeps left-to-right across the screen once per radar pulse; the 35 mm camera captures each sweep as one frame.

After scanning to TIFF, each frame is approximately **3000 px wide** and **2400 px tall**. A typical TIFF reel contains 12--13 consecutive frames separated by bright inter-frame borders (~200 px wide).

### 1.2 Signal Sequence (left to right within one frame)

```
~200 px     : inter-frame border (bright vertical stripe)
~490-520 px : PRF timing pulse (triggers the oscilloscope sweep; NOT t=0)
~630-650 px : T/R switch transient (system artifact; NOT the main bang)
~720-830 px : MAIN BANG — transmitted pulse leaking into receiver
              First local minimum = correct main bang pick → defines t = 0
              Second local minimum = T/R switch ringing (may be deeper — ignore it)
>830 px     : Surface echo → then bed echo (further right = deeper)
```

**t = 0 for all travel times = the MAIN BANG position, NOT the first graticule line.**

### 1.3 Pixel Conventions

- **X increases to the right** (increasing time / depth)
- **Y increases downward** in the image (decreasing power)
  - Higher on the image = stronger signal
  - Lower on the image = weaker signal (closer to noise floor)
- The CRT echo trace appears as a **dark line on a lighter background** (phosphor → film → scan inversion)

---

## 2. Radar System Parameters (Neal 1977 Table 1.1)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Center frequency | 60 MHz (lambda = 5 m) | |
| Transmit power P_T | 1 kW = 60 dBm | |
| Pulse width (F125) | 125 ns | 14 MHz receiver bandwidth |
| Antenna gain G | 12 dB (raw) | |
| Cable loss | 1 dB | |
| T/R switch loss | 0.3 dB | |
| **Effective gain G_eff** | **10.7 dB** (one-way) | 12 - 1 - 0.3 |
| In-flight attenuator | Variable: 20, 30, or 40 dB | 40 dB confirmed for F125 |
| MDS (minimum detectable signal) | -101 dBm at 14 MHz BW | |
| Dynamic range | 70 dB | Noise floor to main bang |
| c_air | 300 m/us | |
| c_ice | 168 m/us | Millar 1981 (Neal uses 169) |
| n_ice | 1.78 | Refractive index of ice |

---

## 3. Grid Calibration

The oscilloscope CRT has a calibrated graticule overlaid on the phosphor screen. Both axes use regularly spaced major divisions, visible as faint lines on the scanned film.

### 3.1 Axis Scales

| Axis | Major division | Pixel spacing | Physical value |
|------|----------------|---------------|----------------|
| X (time) | 1 major = 1.5 us | ~205 px | 1.5 us two-way travel time |
| X (time) | 1 minor = 0.3 us | ~41 px | 4 minor ticks between majors |
| Y (power) | 1 major = 10 dB | ~205 px | 10 dB received power |

**ASTRA calibration error**: ASTRA assumed 3.0 us/major division. The correct value is **1.5 us/major**. All ASTRA travel times (surface_us, bed_us) are therefore **2x too large**. Never use ASTRA geometry. Use LYRA only.

### 3.2 Calibration Constants (LYRA DEFAULT_CAL)

```python
US_PER_PX  = 1.5 / 205.54   # = 0.007299 us/px
DB_PER_PX  = 10.0 / 205.0   # = 0.04878 dB/px
Y_REF_PX   = 1507            # frame-relative y-pixel of the -60 dB reference line
Y_REF_DB   = -60.0           # dB value at the reference row
MB_X_GUESS = 800             # expected main bang column (frame-relative)
Y_DISP_LO  = 300             # top of usable CRT display region (px)
Y_DISP_HI  = 1700            # bottom of usable CRT display region (px)
```

### 3.3 Per-Flight Calibration Variations

The DEFAULT_CAL values above are tuned to F125. Other flights have slightly different oscilloscope settings, producing different pixel spacings, MB positions, and noise-floor reference rows. LYRA Step 2 measures these per-frame from user guide picks and graticule detection.

| Flight | db_per_px | y_spacing (px) | x_spacing (px) | D_anchor (px) | y_ref (px) | MB position (px) |
|--------|-----------|----------------|-----------------|----------------|------------|-------------------|
| F125 | 0.04762 | 210 | 202--209 | 485 | 1517--1521 | ~800 |
| F127 | 0.04988 | 200.5 | ~205 | 375 | ~1512 | ~800 (variable: 254--1140) |
| F141 | 0.05025 | 199 | 201--209 | 121 | 1603--1606 | ~540 |

Key points:
- **D_anchor** = distance from main bang to first x-grid line (constant within a TIFF); physically justified because the oscilloscope sweep is time-locked to the radar transmit pulse.
- **F127 MB position is highly variable** (254--1140 px within a single TIFF) due to inconsistent oscilloscope trigger timing. LYRA handles this with a multi-tier MB detection algorithm.
- **F141 MB at ~540 px** (not 800) because the oscilloscope was set up differently. LYRA's `tiff_mb_estimate` (median of guided MB picks) handles cross-flight portability.

---

## 4. Power Interpretation (Y-axis)

### 4.1 From Pixels to Relative dB

The Y-axis represents **relative power in dB** (not absolute dBm). Power increases upward (lower y-pixel = stronger signal).

```
signal_dB = Y_REF_DB - (y_px - Y_REF_PX) * DB_PER_PX
```

Example: a surface echo at y_px = 568 has:
```
signal_dB = -60.0 - (568 - 1507) * 0.04878 = -60.0 - (-45.84) = -14.2 dB
```

### 4.2 From Relative dB to Absolute dBm

```
P_dBm = signal_dB + (MDS_dBm - noisefloor_dB) + attenuator_dB
      = signal_dB + (-101 - (-60)) + 40
      = signal_dB - 1     [for F125 standard conditions]
```

The noise floor (~-60 dB) maps to the MDS (-101 dBm) plus the attenuator setting (+40 dB). The calibration offset C = MDS - noisefloor_dB = -101 - (-60) = -41 dBm.

### 4.3 Key Power Levels

| Feature | Typical y (px) | Power (dB) | Description |
|---------|---------------|------------|-------------|
| Main bang | ~310 | ~+10 dB | Transmitted pulse; top of dynamic range |
| Surface echo | ~570 | -14 dB | Aircraft-to-surface return |
| Bed echo | ~770 | -24 dB | Through-ice return from bed |
| Noise floor | ~1507 | -60 dB | System noise baseline |

---

## 5. Travel Time Interpretation (X-axis)

### 5.1 Two-Way Travel Time (TWT)

All times are measured from the **main bang** position (t = 0), not from the left edge of the frame:

```
twt_us = (echo_x_px - mb_x_px) * US_PER_PX
```

### 5.2 Derived Geometry

```
h_air  = twt_surface / 2 * c_air       # aircraft altitude (m)
h_ice  = (twt_bed - twt_surface) / 2 * c_ice   # ice thickness (m)
h_eff  = h_air + h_ice / n_ice         # effective propagation distance (m)
```

Where: c_air = 300 m/us, c_ice = 168 m/us, n_ice = 1.78.

---

## 6. CBD 0465 Reference Measurements

Frame 8 of F125 TIFF `40_0008400_0008424`, validated against Neal 1977 Fig 1.3a. Use as ground truth for testing any algorithm changes.

| Echo | Frame x (px) | Frame y (px) | Power (dB) | TWT (us) | Physical |
|------|-------------|-------------|------------|----------|----------|
| Surface | 1579 | 568 | -14.2 | 5.69 | h_air = 853 m |
| Bed | 2329 | 771 | -24.1 | 11.16 | h_ice = 460 m |

- Noise floor: -60.0 dB (y_ref_px = 1507)
- Surface SNR: +45.8 dB
- Bed SNR: +35.9 dB (Neal's annotation: ~40 dB)
- MB-to-bed TWT: 11.16 us (Neal's annotation: ~10 us, +12% within hand-drawn annotation uncertainty)

---

## 7. LYRA Pipeline Overview

LYRA processes raw TIFF scans in three steps. Each step reads the outputs of the previous step.

```
Step 1: Frame detection + CBD assignment
    Input:  raw TIFF
    Output: frame_index.csv (frame boundaries, CBD numbers)

    pick_calibration.py (interactive): user clicks MB, ref, x-grid, y-grid
    Output: cal_picks.json (guide picks for Step 2)

Step 2: Per-frame grid calibration
    Input:  TIFF + frame_index.csv + cal_picks.json
    Output: cal.csv (mb_x, y_ref_px, db_per_px, us_per_px per frame)

Step 3: Echo extraction + waveform metrics
    Input:  TIFF + cal.csv
    Output: echoes.csv + per-frame diagnostic figures
```

### 7.1 Step 1: Frame Detection + CBD Assignment

**Frame detection** (`detect_frames()` in lyra.py):
1. Compute column-mean brightness across the full TIFF; normalize to [0, 1]
2. Threshold at 0.90 to find bright inter-frame borders
3. Identify gaps (runs of bright columns) wider than `min_gap_px`
4. **Bimodal gap filter**: sort gaps by width descending, find the largest relative drop (>50%) between adjacent gap widths, and discard all gaps below the cutoff. This separates true inter-frame separators (e.g. ~1025 px wide) from false intra-frame gaps (e.g. ~92 px from film blemishes). Required for F141.
5. Keep the widest `expected_frames - 1` gaps (note: number of gaps = number of frames minus 1)

**CBD assignment** (`--method` flag):
- `segment` (default): Structural 7-segment OCR reads the 7-digit label printed on each frame
- `ncc`: NCC template matching (older, 79% raw accuracy)
- `manual`: Reads CBDs from ASTRA picks CSV (human-verified ground truth)
- `--override FR:CBD`: Set anchor CBDs at specific frames; sequential propagation fills between anchors
- `--cbd-start N`: Bypass all OCR; assign N, N+1, N+2...

After raw OCR, `apply_sequential_constraint()` enforces monotonic CBD ordering and corrects isolated misreads.

**CBD numbering note**: TIFF filenames contain **reel positions** (e.g. `47_0004850_0004874`), not CBD numbers. The CBD-to-reel mapping is flight-specific and sometimes non-monotonic.

### 7.2 Interactive Guide Picks (`pick_calibration.py`)

Before Step 2, the user provides guide clicks on a few frames to anchor the calibration:

| Key | Pick type | Purpose |
|-----|-----------|---------|
| M | Main bang | x-position of t = 0 (first minimum of transmitted pulse) |
| R | Reference | y-position of a known dB line (establishes y_ref for the TIFF) |
| X | X-grid | x-position of one or more major graticule lines |
| Y | Y-grid | y-position of major graticule lines (confirms dB spacing) |

Picks from one frame propagate to other frames in the same TIFF:
- **MB position** → `tiff_mb_estimate` (median of guided frames with mb > 200 px)
- **Reference** → `tiff_anchor` (y_ref propagates as constant within TIFF)
- **D_anchor** → D = first_x_grid - mb_x (constant per TIFF, propagated to unguided frames)

Frames can be excluded by adding `"exclude": true` to the JSON (tilted graticule, CRT window shift, faint/missing trace, reel-begin artifact).

### 7.3 Step 2: Per-Frame Grid Calibration

For each frame, Step 2 determines:

1. **Main bang position** (`detect_mb()` — 4-tier algorithm):
   - Tier 1 (Guided): narrow window around guide_x; threshold crossing 20 dB above baseline; collect **all** local minima >= 15 dB below baseline, then pick the one **closest to guide_x** (not the first found left-to-right). This prevents left-of-guide artifacts — e.g. the PRF timing pulse at ~490 px — from being mis-identified as the main bang when the real MB is at ~800 px.
   - Tier 2 (Guide check): check +/-20 px around guide for signal >= 10 dB above noise. Handles stable-MB flights.
   - Tier 3 (Broad fallback): heavy smooth sigma=20, full-frame argmin. Handles variable-MB flights (F127).
   - Tier 4 (Last resort): use guide_x or default 800 px.
   - **Sanity check**: mb_power must be >= y_ref_db + 10 dB; if not, revert to guide.

   Physical basis: the main bang is the **first** peak — the transmitted pulse leaking into the receiver. The second peak (which may be deeper) is T/R switch ringing and occurs *after* t = 0. Always pick the first acceptable minimum.

2. **X-grid lines** (`detect_xgrid_lines()` — priority system):
   - Priority A: user guide x_grid clicks (exact positions)
   - Priority A2: D_anchor propagation (D = first_x_grid - mb_x = constant within TIFF)
   - Priority B: automated phase search over x0 in [0, x_spacing_px), starting at x = 200 px
   - **Quality gate**: if mb_power < TIFF_median - 8 dB, strip D_anchor and fall back to Priority B (prevents corrupted MB from propagating bad x-grid to all frames)

3. **Y-grid lines** and **noise floor reference** (from user guide picks + graticule detection)

Output: `F{FLT}_cal.csv` with per-frame columns `mb_x`, `y_ref_px`, `db_per_px`, `us_per_px`, `mb_power_dB`, and detection metadata.

### 7.4 Step 3: Echo Extraction + Waveform Metrics

Step 3 reads the per-frame calibration from Step 2, extracts the CRT trace from the image, and detects surface and bed echoes.

**Trace extraction** (`extract_trace(robust=True)` in lyra.py):

The CRT echo trace is the dark line on the film. For each column, LYRA finds the darkest pixel (argmin) within the display band.

Robust mode (default for Step 3) prevents film-grain noise from corrupting the trace:
1. Raw argmin per column
2. Pre-filter: replace columns in [0, mb_x + skip] and near frame edges with noise floor value
3. Coarse Gaussian smooth (sigma=30) on pre-filtered trace → stable "expected position" per column
4. Constrained argmin: reject any pixel > 250 px from the expected position (catches sporadic film-grain dark pixels that are nowhere near the real CRT trace)
5. **Running median filter (window=11 columns)**: applied to the constrained trace before smoothing. Rejects sporadic film artifacts (1--5 columns of anomalous dark pixels from dust, scratches, or emulsion defects) while preserving real echo features (which are hundreds of columns wide). This replaced an earlier two-pass approach (pass 1 at +/-250, pass 2 at +/-150) that was too restrictive for deep-ice bed echoes where the sigma=30 coarse smooth dampens narrow peaks and a +/-150 band could not reach them.
6. Fine Gaussian smooth (sigma=5) on median-filtered trace → `trace_y_s` (the magenta line in diagnostic figures)

**Important distinction — `trace_y` vs `trace_y_s`**: The raw unconstrained argmin trace (`trace_y`, black line in diagnostic figures) is used **only** for reading peak power at detected echo positions. The constrained + median-filtered + smoothed trace (`trace_y_s`, magenta line) is used for **all** envelope metrics: width, peakiness, asymmetry, trailing_tail, leading_rise. This means that any film artifact surviving into `trace_y_s` will affect the envelope shape characterization, even if peak power is correctly read from the raw trace. The artifact flag (see below) detects this situation.

**Graticule masking**: before argmin, all known graticule row positions (computed from y_ref_px and db_per_px) are masked to 1.0 (max brightness). This prevents the 10-dB major grid lines from being picked as the "darkest pixel" — a critical fix because global TIFF normalization makes graticule lines darker than the actual CRT trace in some frames.

**Signal extent detection** (`detect_signal_extent()` — adaptive threshold):

Determines where the CRT sweep starts and ends (signal vs film-grain-only columns):
1. Binary threshold on normalized frame (< 0.30) to find dark pixels
2. Exclude graticule rows (+/-8 px) from the count
3. Count dark pixels per column; smooth with Gaussian (sigma=30)
4. **Adaptive threshold**: `max(floor=12, min(cap=50, 0.25 * peak_density))`
   - Bright frames (peak ~200 dark px/col): threshold = 50 (original behavior)
   - Faint frames (peak ~40 dark px/col): threshold = 12
   - Film grain noise is ~5--6 dark px/col; floor of 12 stays safely above this
5. Signal columns = those with smoothed count >= threshold

This adaptive threshold fixed 22 F127 frames where faint CRT traces had silently wrong bed picks.

**Echo detection** (`detect_echoes()` in lyra.py):

1. Find all peaks in the smoothed trace after mb_skip_us (2 us) with prominence >= 5 dB and separation >= 80 px
2. Require peaks to be > 3 dB above noise floor
3. **Surface selection**: first peak in the surface window (0 to 8 us from MB). If a later peak in the window is > 15 dB stronger than the first, promote it (rejects weak T/R ringing artifacts)
4. **Bed selection**: scan all peaks at least min_bed_gap_us (2 us, ~168 m ice thickness) after the surface. Each candidate must pass:
   - **Quiet-gap check**: between surface_x + 1 us and the candidate, the trace must return within 12 dB of the noise floor at some point. Rejects surface-echo trailing oscillations (which never return to baseline).
   - **Width check**: the echo must be at least 0.3 us wide at NF + 10 dB. Rejects narrow film artifacts.
5. Among all passing candidates, select the one with the **strongest power** (most prominent).

**Envelope walking** (`walk_envelope()`):

From each detected peak, walk left (leading edge) and right (trailing edge) at two thresholds: NF + 5 dB and NF + 10 dB. The rightward walk is bounded by max_walk_us (17 us from MB) to prevent extension into frame separators.

**Bed envelope artifact flag** (`bed_envelope_suspect`):

Film artifacts (dust, scratches, emulsion defects) occasionally create spurious dark pixels within the bed echo envelope region. Even after the running median filter, some artifacts can survive and distort envelope metrics like asymmetry and width.

LYRA detects this automatically by comparing the constrained trace (before median filtering) to the median-filtered trace (after filtering). If the median filter changed any column by more than 10 dB within the bed echo envelope (from bed peak to trailing_5 + 1 us), that frame is flagged:

- `bed_envelope_suspect = True`: the bed echo's envelope shape may be contaminated
- `artifact_max_dB`: the largest discrepancy found (in dB)

How it works: the running median filter replaces each column's trace value with the median of its 11-column neighborhood. If one column has a genuine echo, its neighbors will too, so the median barely changes the value. But if one column has a film artifact (a random dark pixel far from the real trace), its 10 neighbors will outvote it, and the median will snap back to the real echo position. The difference between the "before median" and "after median" value at that column tells us how much the artifact displaced the trace. If that displacement exceeds 10 dB and falls within the bed echo envelope, the frame is flagged.

Typical prevalence: ~9--11% of frames flagged across F125 and F127. Flagged frames should be inspected in the diagnostic figures before including their envelope metrics in scientific analysis. Peak power and SNR are unaffected (read from the raw trace).

**Waveform shape metrics** (`compute_echo_metrics()`):

| Metric | Definition | Physical interpretation |
|--------|------------|----------------------|
| peak_power_dB | Power at echo peak (from raw trace) | Reflection strength |
| SNR_dB | peak_power_dB - noise_floor_dB | Signal quality |
| width_10_us | Echo width at NF + 10 dB | Specular: narrow; rough: wide |
| width_5_us | Echo width at NF + 5 dB | Includes trailing scattering |
| peakiness | peak_linear / mean_linear (within +10 dB window) | > 1: sharp; ~ 1: flat |
| asymmetry | trailing_width / leading_width (at +10 dB) | > 1: trailing tail; 1: symmetric |
| trailing_tail_us | trail_5 - trail_10 | Subsurface scattering length |
| leading_rise_us | peak_twt - lead_10 | Onset steepness |

**Physical interpretation of waveform shapes**:
- High peakiness + low asymmetry + short trailing_tail → specular reflector / liquid water at bed
- Low peak power + long trailing_tail → saline / frozen marine ice at bed
- Broad symmetric echo, moderate peakiness → rough / incoherent bed

**Echo status categories**:

| Status | Meaning |
|--------|---------|
| good | Surface + bed detected; bed SNR >= 5 dB |
| weak_bed | Bed detected but SNR < 5 dB (marginal; inspect diagnostic figure) |
| no_bed | Surface detected but no bed echo found |
| no_surface | No surface detected; all geometry undefined |

**Output CSV columns** (`F{FLT}_echoes.csv`):

```
flight, tiff, cbd, file_id, echo_status, noise_floor_dB,
surface_twt_us, surface_power_dB, surface_snr_dB,
surface_width_10_us, surface_width_5_us, surface_peakiness,
surface_asymmetry, surface_leading_rise_us, surface_trailing_tail_us,
bed_twt_us, bed_power_dB, bed_snr_dB,
bed_width_10_us, bed_width_5_us, bed_peakiness,
bed_asymmetry, bed_leading_rise_us, bed_trailing_tail_us,
h_air_m, h_ice_m, h_eff_m,
bed_envelope_suspect, artifact_max_dB
```

The last two columns are the artifact flag:
- `bed_envelope_suspect`: boolean — True if film artifacts may have contaminated the bed echo envelope shape metrics
- `artifact_max_dB`: float — maximum discrepancy (dB) between pre- and post-median-filter trace within the bed envelope. 0.0 if no artifact detected.

---

## 8. Geometric Spreading and the Radar Equation

### 8.1 Why 20*log10, not 40*log10

The radar equation relates the received power P_R to the basal reflection coefficient R_0. The key question is how the signal power decreases with distance between the aircraft and the bed.

For **incoherent** (rough/diffuse) scatterers, the reflected power is uncorrelated across the reflecting surface, so power falls off as 1/r^4 round-trip and the spreading term uses **40*log10(r)**.

For **specular** (coherent) reflectors, the bed acts like a mirror — the reflected wavefront maintains its phase coherence, and power falls off as 1/r^2 round-trip, so the spreading term uses **20*log10(r)**.

At 60 MHz (lambda = 5 m), the first Fresnel zone radius at typical Antarctic ice shelf depths is:

```
r_Fresnel = sqrt(lambda * h_eff / 2)
```

For CBD 0465 (h_eff = 1111 m): `r_Fresnel = sqrt(5 * 1111 / 2) = 53 m`. Over the flat Ross Ice Shelf, the ice-bed interface is smooth at this scale, so the reflection is coherent/specular. This is confirmed independently by Neal (1979) and Millar (1981), who both use 20*log10 for Ross Ice Shelf data.

Using 40*log10 would systematically overestimate the spreading loss and produce R_0 values that are too high (too reflective).

### 8.2 The Full Radar Equation

```
R_0(dB) = P_R(dBm) - P_T(dBm) - G_spread(dB) + L_diel(dB)
```

Each term:

**P_R** — absolute received power at the bed echo (dBm):
```
P_R = bed_power_dB + C + A_att

    bed_power_dB  = power read from the A-scope (relative dB scale on CRT)
    C             = MDS_dBm - noisefloor_dB = -101 - (-60) = -41 dBm
    A_att         = attenuator setting = 40 dB (for F125)

→ P_R = bed_power_dB - 1    [for F125 standard conditions]
```

**P_T** — transmit power = 60 dBm (1 kW)

**G_spread** — geometric spreading term (specular, 20*log10):
```
G_spread = 20*log10(lambda * G_linear / (8 * pi * h_eff))

    lambda   = 5 m (60 MHz)
    G_linear = 10^(G_eff/10) = 10^(10.7/10) = 11.75  (one-way effective gain as linear ratio)
    h_eff    = h_air + h_ice / n_ice   (effective one-way propagation path in meters)
```

Note: G_eff = 10.7 dB is the effective one-way antenna gain (12 dB raw - 1 dB cable - 0.3 dB T/R switch). The same antenna is used for transmit and receive, so both G factors (Tx and Rx) are captured by a single G^2 in the power equation, which becomes 2*G in the dB-domain spreading term.

Numerically:
```
G_spread = 20*log10(5 * 11.75 / (8 * pi * h_eff))
         = 20*log10(58.75 / (25.13 * h_eff))
         = 20*log10(2.338 / h_eff)
```

**L_diel** — two-way dielectric absorption loss through the ice column (dB):
```
L_diel = 2 * absorption_rate * h_ice
```

Typical absorption rate for cold Antarctic ice at 60 MHz: ~1--3 dB per 100 m (depends on temperature profile). For saline/marine ice, absorption increases dramatically (Paren model: epsilon'' = 0.021 + 0.0115 * [Cl-]_ppm → ~0.27 dB/m at 70 ppm chloride).

### 8.3 Worked Example: CBD 0465 (F125)

From LYRA Step 3:
```
bed_power_dB  = -24.1 dB   (read from A-scope)
h_air         = 853 m       (from surface TWT = 5.69 us)
h_ice         = 460 m       (from bed-surface TWT difference)
h_eff         = 853 + 460/1.78 = 853 + 258 = 1111 m
```

Absolute received power:
```
P_R = -24.1 - 1 = -25.1 dBm
```

Geometric spreading:
```
G_spread = 20*log10(2.338 / 1111) = 20*log10(0.002104) = 20 * (-2.677) = -53.5 dB
```

R_0 (neglecting dielectric loss):
```
R_0 = P_R - P_T - G_spread
    = (-25.1) - (60) - (-53.5)
    = -31.6 dB
```

With dielectric correction (assuming ~1.5 dB/100 m for the ~460 m column):
```
L_diel = 2 * 1.5 * 4.6 = 13.8 dB
R_0_corrected = -31.6 + 13.8 = -17.8 dB
```

This is consistent with the ASTRA survey-wide F125 mean of -18.1 dB. Values well below the seawater reference (-0.77 dB) indicate dielectric attenuation from saline or marine ice at the base of the Ross Ice Shelf.

### 8.4 Reference Value

Smooth ice over seawater: R_0 = -0.77 dB (Neal 1979). Values significantly below this indicate dielectric attenuation from saline ice, frozen marine ice, or other basal anomalies.

### 8.5 Measurement Uncertainty

Total R_0 uncertainty: **+/- 5 dB** (Neal 1977 Section 6.3):
- +/- 3.5 dB from power measurement (CRT reading precision, calibration drift)
- < 3 dB from dielectric loss model (temperature profile uncertainty)

---

## 9. Determining the Attenuator Setting

### 9.1 Background

The SPRI 60 MHz radar had a manually adjustable in-flight attenuator with settings of 20, 30, or 40 dB. The attenuator was inserted between the receiver and the oscilloscope display to keep the dynamic range on-screen. The setting was **not recorded digitally** — it was a rotary switch the operator adjusted during flight.

Knowing the attenuator setting is critical for converting the relative dB scale on the A-scope to absolute received power (dBm). An error of 10 dB in the assumed attenuator propagates directly into R_0.

### 9.2 Method: Fresnel Surface Echo Comparison

The attenuator is determined by comparing the **measured** surface echo power on the A-scope to the **predicted** surface echo power from the radar equation, assuming a known surface reflectivity.

Over the Ross Ice Shelf, the ice-air surface is flat and smooth at 60 MHz wavelengths (5 m). The Fresnel reflection coefficient for the air-ice interface is:

```
R_surface = ((n_ice - 1) / (n_ice + 1))^2 = ((1.78 - 1) / (1.78 + 1))^2 = (0.78/2.78)^2 = 0.0787
→ R_surface(dB) = 10*log10(0.0787) = -11.0 dB
```

From the radar equation with known P_T, G, lambda, and h_air (from TWT), we can predict the absolute P_R at the surface. The difference between the predicted P_R and the observed A-scope reading gives the attenuator + calibration offset.

### 9.3 F125 Confirmation: 40 dB

For F125, the attenuator was confirmed at **40 dB** through two independent methods:

1. **Fresnel surface echo method** (LYRA Frames 1--4 of TIFF 8400): measured surface echo yields attenuator = 39.6 +/- 0.7 dB. Only Frames 1--4 are used because a Y-axis calibration drift of ~200 px occurs at Frame 5, introducing a ~10 dB systematic error.

2. **ESM Figure cross-check**: Neal's PhD thesis Figures 5.4a and 5.4b (8 mm ESM record) show the attenuator scale markings at 40 dB for the flight segments corresponding to F125 — direct visual confirmation.

### 9.4 Other Flights

For the remaining 11 flights, 40 dB is assumed based on the F125 validation. This is reasonable because:
- The operator typically used 40 dB for open-ice-shelf surveys to keep the strong surface echo on-screen
- 20 dB would saturate the display over the ice shelf; 30 dB would clip the surface echo for low-altitude passes
- Cross-flight consistency of mb_power (~-46 dB on the A-scope) across all flights supports a uniform setting

The +/-5 dB total uncertainty in R_0 (Neal 1977 Section 6.3) already accounts for possible attenuator mis-identification.

---

## 10. Determining Flight Altitude

### 10.1 Method: Surface Echo Travel Time

Flight altitude is determined directly from the A-scope for every frame — there is no external altimeter record in the LYRA pipeline.

The radar pulse travels from the aircraft to the ice surface and back. The two-way travel time (TWT) to the surface echo, measured from the main bang position, gives:

```
h_air = twt_surface / 2 * c_air = twt_surface / 2 * 300   [meters]
```

Where c_air = 300 m/us (speed of electromagnetic waves in air).

### 10.2 Typical Values

For the 1974--75 SPRI Ross Ice Shelf surveys:
- Typical survey altitude: 800--1000 m above the ice surface
- Surface TWT: 5.3--6.7 us (corresponding to 795--1005 m)
- Example: CBD 0465 (F125), surface at x = 1579 px, MB at x = 800 px → TWT = 5.69 us → h_air = 853 m

### 10.3 Ice Thickness

Ice thickness follows the same principle using the bed echo:

```
twt_ice = twt_bed - twt_surface    [two-way time through ice only]
h_ice   = twt_ice / 2 * c_ice = twt_ice / 2 * 168    [meters]
```

Where c_ice = 168 m/us (speed in ice; Millar 1981). The refractive index n_ice = c_air / c_ice = 300 / 168 = 1.78 converts ice thickness to an equivalent air-path for the geometric spreading correction:

```
h_eff = h_air + h_ice / n_ice
```

### 10.4 Why Not Use a Separate Altimeter?

The SPRI aircraft had both a radar altimeter and a barometric altimeter, but:
1. Those records are not digitized or paired to individual A-scope frames
2. The A-scope surface echo provides altitude at the exact moment of each radar pulse — no time interpolation needed
3. Over ice shelves, surface elevation varies slowly, so the per-pulse TWT measurement is the most direct and reliable altitude source

---

## 11. Frame Exclusion Criteria

Some frames cannot be reliably calibrated. Record in `F{FLT}_cal_picks.json` with `"exclude": true` and `"exclude_reason"`:

| Criterion | How to identify |
|-----------|----------------|
| Reel-begin artifact | First frame of a TIFF; mb_power >> -46 dB (e.g. -5 to -8 dB). Film leader before CRT settles. Exclude if mb_power > -25 dB. |
| Tilted graticule | Graticule lines not parallel in the frame; no reliable grid reference |
| CRT window shift | ~39 px systematic offset in the sub-baseline band; Priority B finds wrong x-grid phase |
| Faint/missing trace | mb_power far below the cluster median (-46 dB); main bang undetectable |
| Mid-reel splice | 2 partial frames detected mid-TIFF; post-splice frames may have mb at noise floor |

---

## 12. Running the Pipeline

### 12.1 Standard Workflow

```bash
# From repository root:

# Phase 1: detect frames and assign CBDs
python tools/LYRA/detect_frames.py Data/ascope/raw/125/40_0008400_0008424-reel_begin_end.tiff

# Phase 2: pick calibration guides (MB, ref, x-grid, y-grid)
python tools/LYRA/pick_calibration.py Data/ascope/raw/125/40_0008400_0008424-reel_begin_end.tiff

# Phase 3: per-frame grid calibration
python tools/LYRA/calibrate.py Data/ascope/raw/125/40_0008400_0008424-reel_begin_end.tiff

# Phase 4: echo extraction
python tools/LYRA/echoes.py Data/ascope/raw/125/40_0008400_0008424-reel_begin_end.tiff
```

### 12.2 Step 1 Options

```bash
# Use structural OCR (default)
--method segment

# Use NCC template matching
--method ncc

# Use ASTRA CSV ground truth
--method manual

# Set specific CBD anchors (propagates sequentially between anchors)
--override 3:434 10:444

# Bypass OCR entirely: assign N, N+1, N+2...
--cbd-start 432
```

### 12.3 Output Location

All outputs are in `tools/LYRA/output/F{FLT}/`:

```
F125/
├── phase1/
│   ├── F125_frame_index.csv       # Frame boundaries + CBDs
│   ├── F125_cal_picks.json        # User guide picks
│   ├── F125_*_contact.png         # Contact sheet overview
│   └── F125_*_ocr_diag.png        # OCR diagnostic figure
├── phase3/
│   ├── F125_cal.csv               # Per-frame calibration
│   └── F125_CBD*_cal.png          # Calibration diagnostic figures
└── phase4/
    ├── F125_echoes.csv            # Echo extraction results
    └── F125_*_echoes.png          # Two-panel diagnostic figures
```

### 12.4 Troubleshooting

- **Fixes have no effect**: stale `.pyc` files. Run `find . -path "*/__pycache__/lyra*" -delete`
- **Step 2 blocks with "no ref pick"**: run `pick_calibration.py`, press R on any frame, Q to save, then re-run Step 2
- **OCR gives wrong CBDs on F125 77xx TIFFs**: 3-digit CBDs cause the hundreds digit to be dropped. Use `--cbd-start N` or `--method manual`.
- **Non-canonical TIFF filenames** (e.g. F141 files named `F141-C{start}_C{end}.tiff`): LYRA's `ensure_canonical_name()` auto-renames using `*_rename_log*.txt` in the same directory.

---

## 13. Glossary

| Term | Definition |
|------|-----------|
| A-scope | Radar display: single pulse return, power vs time |
| CBD | Counter Block Display number — unique identifier for each A-scope frame along a flight track |
| CRT | Cathode Ray Tube (the oscilloscope display screen) |
| D_anchor | Distance (px) from main bang to first x-grid line; constant per TIFF |
| h_air | One-way air column height: aircraft altitude above ice surface |
| h_eff | Effective propagation distance: h_air + h_ice / n_ice |
| h_ice | One-way ice thickness |
| L_diel | Two-way dielectric loss through the ice column |
| Main bang | Transmitted pulse leaking into receiver; defines t = 0 |
| MDS | Minimum Detectable Signal (-101 dBm at 14 MHz BW) |
| NF | Noise floor (-60 dB relative) |
| R_0 | Basal reflection coefficient (dB); specular ice/seawater = -0.77 dB |
| SNR | Signal-to-noise ratio (peak power minus noise floor, in dB) |
| TIFF | Scanned film reel; each TIFF typically contains 12-13 A-scope frames |
| TWT | Two-way travel time (us from main bang to echo and back) |
| y_ref | Y-pixel corresponding to a known dB level (calibration anchor) |

---

## 14. Key References

- **Neal, C. S. (1977)**. *Radio echo studies of the Ross Ice Shelf*. PhD thesis, University of Cambridge. Primary source for all system parameters, calibration, and saline ice physics.
- **Neal, C. S. (1979)**. The dynamics of the Ross Ice Shelf revealed by radio echo-sounding. *J. Glaciol.*, 24(90), 295--307. Confirms 20*log radar equation and R_0 reference value.
- **Millar, D. H. M. (1981)**. *Radio echo layering in polar ice sheets*. PhD thesis, University of Cambridge. G_two-way = 21.4 dB = 2 x 10.7 dB; c_ice = 168 m/us.
- **Schroeder, D. M. et al. (2021)**. Archival radar data and the future of ice sheet research. Dynamic range 70 dB for A-scope.
- **Jezek, K. C. & Bentley, C. R. (1983)**. Field studies of bottom crevasses in the Ross Ice Shelf, Antarctica. *J. Glaciol.*, 29(101), 157--167. Derives R_0 from same SPRI 60 MHz A-scope data.
