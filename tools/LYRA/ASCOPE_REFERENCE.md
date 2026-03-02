# ASCOPE_REFERENCE.md — A-scope Physics & Calibration Constants

Quick reference for LYRA development. All values validated against Neal 1977 Fig 1.3a (CBD 0465).

---

## Signal Sequence on Each A-scope Frame (left → right)

```
~200 px     : inter-frame border (bright vertical stripe)
~490-520 px : PRF timing pulse (triggers oscilloscope sweep; NOT t=0)
~630-650 px : T/R switch transient (system artifact; NOT the main bang)
~720-830 px : MAIN BANG (transmitted pulse leaking into receiver) ← t=0 for TWT
              first local minimum = correct main bang pick
              second local minimum = T/R switch ringing (may be deeper — ignore)
>830 px     : surface echo, then bed echo
```

**t=0 for all travel times = MAIN BANG position, NOT the first graticule line.**

---

## Grid Calibration (F125 validated; applies broadly across TIFF reels)

| Axis | Division | Pixel spacing | Physical value |
|------|----------|---------------|----------------|
| X (time) | major | **205.54 px** | **1.5 µs** |
| X (time) | minor | 41.1 px | 0.3 µs (4 minors between majors) |
| Y (power) | major | **205.0 px** | **10 dB** |

**Key constants:**
```python
US_PER_PX  = 1.5 / 205.54   # = 0.007299 µs/px
DB_PER_PX  = 10.0 / 205.0   # = 0.04878 dB/px
Y_REF_PX   = 1507            # frame-relative y-pixel → −60 dB (F125 training TIFF)
Y_REF_DB   = -60.0
MB_X_GUESS = 800             # frame-relative x-pixel for main bang (starting guess)
Y_DISP_LO  = 300             # top of CRT display region
Y_DISP_HI  = 1700            # bottom of CRT display region
```

**ASTRA was wrong**: ASTRA assumed 3.0 µs/major division → all ASTRA travel times are 2× too large. Never use ASTRA `surface_us` / `bed_us` for geometry. Use LYRA algorithmic pipeline only.

---

## Y-axis: Power Interpretation

- Y-axis is **relative dB**, not dBm. Conversion to absolute:
  ```
  P_dBm = signal_dB + (MDS_dBm − noisefloor_dB) + attenuator_dB
        = signal_dB + (−101 − (−60)) + 40
        = signal_dB − 1     [for F125 standard conditions]
  ```
- `signal_dB` increases upward (higher on frame = brighter = more power)
- `y_px` increases downward → `signal_dB = Y_REF_DB − (y_px − Y_REF_PX) × DB_PER_PX`
- Noise floor sits at ~y_px=1507 (−60 dB reference line)
- Main bang sits at ~y_px=310 (−70 dB + 70 dB dynamic range... near top of display)

---

## X-axis: Travel Time Interpretation

- TWT = **two-way travel time** from transmitted pulse to echo
- All times measured from MAIN BANG (t=0), not from frame left edge
- `twt_us = (echo_x_px − mb_x_px) × US_PER_PX`
- One-way ranges: `h_air = twt_surface/2 × c_air`, `h_ice = twt_bed/2 × c_ice`
  - c_air = 300 m/µs, c_ice = 168 m/µs, n_ice = 1.78

---

## CBD 0465 Reference Measurements (Frame 8, F125 TIFF `40_0008400_0008424`)

Validated against Neal 1977 Fig 1.3a. Use as ground truth for algorithm checks.

| Echo | Frame x (px) | Frame y (px) | Power (dB) | TWT from MB (µs) | Physical |
|------|-------------|-------------|------------|-------------------|---------|
| Surface | 1579 | 568 | −14.20 dB | 5.69 µs | h_air = 853 m |
| Bed | 2329 | 771 | −24.10 dB | 11.16 µs | h_ice = 460 m |

- Noise floor: −60.0 dB (y_ref_px = 1507)
- Surface SNR: +45.8 dB above noise floor
- Bed SNR: +35.9 dB above noise floor (Neal's label: ~40 dB ✓)
- MB→bed TWT: 11.16 µs (Neal's label: ~10 µs ✓, +12% within annotation uncertainty)
- Frame 8 absolute boundaries in TIFF: left=22097 px, right=25111 px

**Echo envelope widths** (ABS threshold above noise floor, CONSEC=5):

| Threshold | Surface width | Bed width | Notes |
|-----------|--------------|-----------|-------|
| +10 dB | 1.693 µs | 1.168 µs | Primary metric |
| +5 dB | 2.817 µs | 5.19 µs | Bed trailing likely ocean cavity echo |

---

## detect_mb() Algorithm (lyra.py)

```
1. With guide_x: search window lo=max(30, guide_x−100), hi=min(W, guide_x+50)
2. Baseline sampled BEFORE window: trace_y_s[max(30,lo−200):lo]  ← avoid leading edge
3. Threshold crossing: 20 dB below baseline, sustained ≥5 consecutive columns
4. Accept first local min ≥15 dB below baseline (inside MB envelope)
   → do NOT compare to global_min (ringing peak can be deeper)
5. Fallback if no crossing: guide_x (not argmin of full frame)
```

Physical reason: MAIN BANG = first peak = transmitted pulse. Second peak = T/R switch ringing (occurs after t=0). Always pick the FIRST acceptable minimum.

---

## LYRA Step 2: x-grid Priority System

```
Priority A  : user guide x_grid clicks → exact positions
Priority A2 : D_anchor (D = first_x_grid − mb_x = const within TIFF) → propagate to frames without x_grid clicks
              D_anchor is physically justified: oscilloscope sweep is time-locked to radar transmit pulse
Priority B  : phase search over x0 ∈ [0, x_spacing_px), x_start=200
              x_start=200 excludes only inter-frame border; allows first graticule tick at ~520 px to be found
```

**Quality gate**: if `mb_power_dB < TIFF_median_mb − 8 dB`, strip D_anchor → fall back to Priority B.
Reason: corrupted mb_x propagates wrong x-grid via D_anchor to all frames in the TIFF.

---

## Frame Exclusion Criteria

Record in `F{FLT}_cal_picks.json` with `"exclude": true` and `"exclude_reason"`:
- **Tilted graticule on film**: graticule lines not parallel → no reliable grid reference
- **CRT window shift**: ~39 px systematic offset in below-baseline band → Priority B finds wrong phase
- **Faint/missing trace**: mb_power_dB ≪ −42 dB (cluster median) and mb undetectable

---

## CBD Numbering

- TIFF filenames contain REEL positions (e.g., `47_0004850_0004874`) — NOT CBD numbers
- NCC OCR templates (trained on F125 font) may fail on other flights → use `--cbd-start N` override:
  ```bash
  python tools/LYRA/step1_detect_frames.py Data/ascope/raw/127/47_0004850_0004874-reel_begin_end.tiff --cbd-start 432
  ```
- CBD start numbers: look up in `Data/ascope/picks/{FLT}/{FLT}_CombinedASTRAPicks.csv` for the CBD range
  covering those reel positions (cross-reference by h_ice_m or lat values if needed)
