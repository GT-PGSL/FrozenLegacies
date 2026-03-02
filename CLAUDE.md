# CLAUDE.md — FrozenLegacies Project Context

## Project Summary

Derive **bed-echo character** (reflection coefficient R₀, power variance V_p, fading length τ_p) from archival SPRI/TUD/NSF 60 MHz airborne radar A-scope data (1974–75 Antarctic surveys). Compare with modern inverted basal drag over the Ross Ice Shelf and Siple Coast ice streams. The 50-year temporal baseline is the unique scientific contribution.

**Current phase**: LYRA — fully algorithmic A-scope waveform extraction pipeline replacing manual ASTRA picks.

## Key Technical Decisions

- **Primary data**: A-scope 35mm film (Tektronix 465, broadband) → ASTRA/LYRA. Z-scope is retired (Z→A calibration null: r = −0.05).
- **ASTRA is retired for geometry**: ASTRA travel times are 2× too large (assumed 3 µs/div; correct = 1.5 µs/div). ASTRA R₀ peak picks are still valid as cross-check only.
- **LYRA replaces ASTRA**: fully algorithmic frame detection → grid calibration → echo extraction → waveform metrics. See `tools/LYRA/ASCOPE_REFERENCE.md` for physics constants.
- **Attenuator**: 40 dB confirmed for F125 (Fresnel surface echo validation + ESM Figs 5.4a/5.4b). Assumed 40 dB for all 12 flights; "50 dB" Method A estimates for F107/F137/F143 are likely Y-axis calibration drift not real differences.
- **R₀ uncertainty**: ±5 dB total (±3.5 dB power + <3 dB dielectric; Neal 1977 §6.3).
- **Radar equation**: `R₀(dB) = 10log(P_R/P_T) − 20log(λG/8πh_eff) + L_diel` — 20log (specular), NOT 40log.

## LYRA Pipeline — Current Status (Feb 2026)

**Canonical code**: `tools/LYRA/lyra.py` (ONE true source — never edit copies elsewhere)

**Step runners**:
- Step 1 (frame detection + CBD assignment):
  `python tools/LYRA/step1_detect_frames.py <tiff> [--method {manual,segment,ncc}] [--override FR:CBD ...] [--cbd-start N]`
  → `tools/LYRA/output/F{FLT}/step1/F{FLT}_frame_index.csv` + `_step1_contact.png` + `_step1_ocr_diag.png`
  - `--method manual`: reads CBDs from `Data/ascope/picks/{FLT}/{FLT}_CombinedASTRAPicks.csv` (human-verified ground truth)
  - `--method segment` (default): structural 7-segment OCR (`segment_ocr.py`, 89% raw → 100% corrected on F125)
  - `--method ncc`: NCC template matching (`build_digit_templates.py`, 79% raw accuracy)
  - `--override FR:CBD`: anchor-based override with sequential propagation (e.g. `--override 3:434 10:444` fills all frames between anchors)
  - `--cbd-start N`: bypass all OCR, assign N, N+1, N+2... (last resort)
  - CSV columns: `cbd`, `ocr_method` (segment/ncc/manual/override), `ocr_raw` (raw string before correction)
- Interactive guide picks: `python tools/LYRA/pick_calibration.py <tiff>`
  → `tools/LYRA/output/F{FLT}/step1/F{FLT}_cal_picks.json`
- Step 2 (per-frame calibration + figures): `python tools/LYRA/step2_calibrate.py <tiff>`
  → `tools/LYRA/output/F{FLT}/step2/F{FLT}_CBD{N}_step2.png` + `F{FLT}_step2_cal.csv`
- Step 3 (echo extraction + waveform metrics): `python tools/LYRA/step3_echoes.py <tiff>`
  → `tools/LYRA/output/F{FLT}/step3/F{FLT}_step3_echoes.csv` + per-frame `_step3.png`
  - Outputs: surface/bed TWT, power, SNR, width, peakiness, asymmetry, h_air_m, h_ice_m, h_eff_m
  - echo_status: "good" / "no_surface" / "no_bed" / "weak_bed" (bed SNR < 5 dB)

**Step 2 internals** (`detect_frame_calibration()` in lyra.py):
- mb_x: tiff_mb_estimate guide (median of guided-frame MB picks where mb > 200 px) → threshold crossing (20 dB below baseline, ≥5 cols) → guide fallback
- x-grid: Priority A (user guide) → A2 (D_anchor: D = first_x_grid − mb_x, const per TIFF) → B (phase search, x_start=200)
- Quality gate: if mb_power_dB < TIFF_median − 8 dB → strip D_anchor → re-run with Priority B
- tiff_mb_estimate: median MB position from guided frames (mb > 200 px, excluding reel-begin artifacts). Propagated as soft guide to unguided frames — fixes cross-flight MB position drift (e.g. F141 MB at ~540 px vs F125 DEFAULT_CAL 800 px).
- Exclude flag: frames with `"exclude": true` in cal_picks.json are skipped entirely; stale PNGs auto-deleted

**Validated on F125** (5 TIFFs: 7700, 7725, 8300, 8400, 8425 → 60 frames): all good.
**Validated on F127** (3 TIFFs: 4825, 4850, 4875 → 17 frames; 1 frame excluded per TIFF).
**Validated on F141** (1 TIFF: 8525 → 12 frames, CBDs 0110–0121): stress-test PASSED (Feb 26 2026).

**Fixes added (Mar 2 2026)**:

- **Adaptive signal-extent threshold** in `detect_signal_extent()`: fixed threshold=50 failed for faint CRT traces (F127: mean 18.5 dark px/col vs 117.8 for bright frames). Adaptive formula: `max(floor=12, min(cap=50, 0.25 × peak_density))`. Bright frames unchanged; faint frames auto-adapt. Fixed 22/38 F127 bed picks; 0/98 regressions. DEFAULT_CAL: `signal_density_floor=12`, `signal_density_frac=0.25`.

**Fixes added (Feb 26 2026)**:
- **Bimodal gap filter** in `detect_frames()`: sorts gaps by width, finds largest relative drop (>50%) between adjacent widths, cuts there. Removes false intra-frame gaps without discarding real inter-frame separators. Required for F141 (false gap at 92 px vs true gaps at ~1025 px).
- **`ensure_canonical_name()`** in lyra.py: checks TIFF filename against `^\d+_\d+_\d+-reel_begin_end\.tiff`; if non-canonical, parses `*_rename_log*.txt` in same directory and renames the file. Called at start of all three step scripts. Required for F141 (files named `F141-C{start}_C{end}.tiff`).
- **`pick_calibration.py` exclude flag preservation**: `_save_picks()` helper merges back `exclude`/`exclude_reason` when saving picks; excluded frames are pre-filtered from the interactive frame list. Previously overwrote exclude flags on N/Q key press.
- **`tiff_mb_estimate`** in `step2_calibrate.py`: computed as median of guided MB picks > 200 px; propagated as soft MB guide to unguided frames. Fixes cross-flight MB position portability.

**CBD assignment tips**:
- F125 TIFFs 7600–7799: OCR (both segment and NCC) drops hundreds digit for 3-digit CBDs → use `--cbd-start N` or `--method manual`
- F127: full ASTRA CSV coverage (26 TIFFs) → `--method manual` works perfectly
- If segment OCR is off-by-1 or has skipped CBDs: use `--override FR:CBD` to set anchor(s), sequential propagation fills the rest

**F125 TIFF CBD mapping (from ASTRA pattern, confirmed):**
- 7600: CBD 860–871; 7625: CBD 847–858; 7650: CBD 835–845; 7675: CBD 822–833
- 7700: CBD 810–821 (ASTRA missing, interpolated); 7725: CBD 797–808 (ASTRA missing, interpolated)
- 7750: CBD ~785–796 (ASTRA missing); 7775: CBD 772–783
- Pattern: CBD decreases as TIFF number increases for the 76xx–77xx range; ~12–13 CBDs per TIFF step.
- For 82xx–84xx range: CBD INCREASES as TIFF number increases (opposite direction — different reel).

**Calibration constants confirmed across TIFFs:**
- F125: db/px=0.04762 (y_sp=210 px/major), x_sp≈202–209 px/major, mb≈−46 dB, D_anchor=485 px, y_ref≈1517–1521 px, MB_x≈800 px
- F127: db/px=0.04988 (y_sp=200.5 px/major), x_sp≈205 px/major, mb≈−46 dB, D_anchor=375 px, y_ref≈1512 px, MB_x≈800 px
- F141: db/px=0.05025 (y_sp=199 px/major), x_sp≈201–209 px/major, mb≈−46 dB, D_anchor=121 px, y_ref≈1603–1606 px, MB_x≈540 px (frame-relative; NOT 800)

**Reel-begin artifact**: First complete frame of many TIFFs shows mb_power >> −46 dB (e.g. −5 to −8 dB; far above normal). Cause: film leader before CRT settles. **Always exclude if mb_power > −25 dB in first frame.**

**Mid-reel splice**: 2 partial frames detected mid-reel; subsequent frames may show mb at noise floor → exclude. Post-splice CBD assignment is sequential (splice partials count as real positions).

**LYRA incremental testing plan** (DO NOT SKIP STEPS):

1. ✅ Frame 8 of F125 TIFF 8400 — grid calibration validated vs Neal 1977 Fig 1.3a
2. ✅ All 12 frames of F125 TIFF 8400 — MB picks correct
3. ✅ F127 TIFF 4850 — x_start=200 fix, quality gate, --cbd-start override, 2 frames excluded
4. ✅ F125 TIFFs 8300/8425/7700/7725 + F127 4825/4875 — all 5 TIFFs validated; 60 F125 + 17 F127 frames
5. ✅ F141 TIFF 8525 — stress-test PASSED: bimodal gap filter, ensure_canonical_name, tiff_mb_estimate
6. ✅ Step 3 (echo extraction) — robust trace, graticule masking, envelope TWT bound, surface selection; validated on F125 TIFF 8400 (12 frames)
7. ✅ Step 1 OCR integration — structural 7-segment OCR (segment_ocr.py), --method {manual,segment,ncc}, --override with anchor propagation; validated on F125 TIFF 8400 + F127 TIFF 4850
8. **NEXT**: run step3 on remaining validated TIFFs (F125 7700/8300/8425, F127, F141); then batch-process remaining flights

**If lyra fixes seem to have no effect**: check for stale .pyc files: `find . -path "*/core/__pycache__/lyra*" -delete`

## Implementation Phases — Scientific Pipeline (ASTRA-based R₀, completed)

All 12 flights analyzed with ASTRA picks → `Radiometric_Calibrator/run_flight.py`. 5,072 valid CBDs.

| FLT | n_ok | R₀ mean | R₀ std | V_p | τ_p | h_sal@70ppm | % below ref |
|-----|------|---------|--------|-----|-----|-------------|-------------|
| 103 | 307 | **−7.22** | 5.24 | **0.484** | 3.9 km | 12.0 m | 96% |
| 107 | 423 | −15.34 | 9.31 | 2.468 | 3.6 km | 27.3 m | 93% |
| 114 | 206 | **−25.45** | 8.98 | 3.537 | 5.0 km | **45.7 m** | **100%** |
| 115 | 405 | −18.24 | 6.85 | 2.064 | 2.6 km | 32.4 m | 99% |
| 125 | 565 | −18.14 | 11.21 | 1.818 | 2.9 km | 32.5 m | 94% |
| 126 | 223 | **−26.13** | **13.49** | **4.016** | 3.7 km | **47.1 m** | 97% |
| 127 | 570 | −13.81 | 9.05 | 2.129 | 2.7 km | 24.4 m | 93% |
| 128 | 760 | −11.67 | 9.41 | 2.163 | 3.4 km | 20.7 m | 88% |
| 137 | 146 | **−6.45** | 6.86 | 1.408 | 3.5 km | 11.6 m | 80% |
| 138 | 323 | −10.50 | 9.24 | 2.535 | 3.4 km | 18.9 m | 85% |
| 141 | 498 | −16.09 | 10.48 | 2.460 | 2.9 km | 28.9 m | 91% |
| 143 | 646 | −14.49 | 9.02 | 2.990 | 3.8 km | 25.7 m | 92% |

Survey-wide: mean −14.96 dB, std 10.44, 92% below ref (−0.77 dB), h_saline mean 26.6 m.

**Key anomalies**: F103 near-specular (V_p=0.484, possible widespread melt); F114 entirely Zone a/b (100% below ref, −85.4°S); F126 most heterogeneous + darkest; F137 48 suspicious R₀ > +5 dB (possible attenuator issue). Geographic trend: brighter R₀ NW margin, darker SE interior.

**Phase scripts**: `run_flight.py` (pipeline), `make_pub_figures.py` (F125 figs), `make_survey_figures.py` (5 cross-flight figs), `phase4_geographic.py` (EPSG:3031 maps + RIGGS overlay).

## System Parameters (Neal 1977 Table 1.1)

| Parameter | Value |
|-----------|-------|
| P_T | 60 dBm (1 kW) |
| Attenuator A_att | 40 dB (options: 20/30/40 dB; 40 dB confirmed F125) |
| MDS @ 14 MHz BW | −101 dBm |
| MDS @ 4 MHz BW | −106 dBm |
| MDS @ 1 MHz BW | −112 dBm |
| Pulse width (F125) | 125 ns → 14 MHz BW |
| G_eff (one-way) | 10.7 dB (12 dB antenna − 1 dB cable − 0.3 dB T/R switch) |
| λ | 5 m (60 MHz) |
| c_air | 300 m/µs |
| c_ice | 168 m/µs (Millar 1981); Neal uses 169 m/µs |
| n_ice | 1.78 |
| Dynamic range | 70 dB (noise floor → main bang) |

## Full R₀ Workflow

```latex
h_air   = twt_surface/2 × c_air
h_ice   = twt_bed/2 × c_ice       [twt = two-way time from MAIN BANG]
h_eff   = h_air + h_ice/n_ice     [refraction correction]

C       = MDS_dBm − noisefloor_dB  [power calibration offset; typically −41 dBm]
P_R     = bed_dB + C + A_att       [absolute received power, dBm]

L_diel  = 2∫α(T(z))dz             [two-way dielectric loss; Johari & Charette 1975]

R₀(dB)  = (P_R − P_T) − 20·log(λG/8π·h_eff) + L_diel
```
Reference: smooth ice/seawater R₀ = −0.77 dB (Neal 1979). Confirmed 20·log (specular), NOT 40·log.

## Saline Ice — Key Physics (Neal 1977 Ch.4; Paren model at 60 MHz)

- Attenuation: ε''_s = 2.1×10⁻² + 1.15×10⁻² × [Cl⁻]_ppm → **~0.27 dB/m at 70 ppm**
- Saline ice Fresnel R₀ = −0.80 dB (vs −0.77 pure ice) — all anomalous low R₀ is attenuation, not interface change
- Amery chlorinity: ~70 ppm @20°C, ~140 ppm @0°C; RIS brine enters via bottom crevasses
- Neal 1977 zones: a,b < −20 dB (frozen seawater, 10+ m); c,d,e = −10 to −20 dB; NW > 0 dB (basal melt ≤1.5 m/yr)
- Melt sensitivity: ±4.2 dB per ±0.15 m/yr; min 10 m saline layer for 30 dB two-way loss

## Repository Structure

```
FrozenLegacies/
├── tools/LYRA/               # LYRA pipeline
│   ├── lyra.py               # Core library (ONE true source)
│   ├── step1_detect_frames.py # Frame detection + CBD assignment
│   ├── pick_calibration.py   # Interactive guide picks
│   ├── step2_calibrate.py    # Per-frame grid calibration
│   ├── step3_echoes.py       # Echo extraction + waveform metrics
│   ├── segment_ocr.py        # Structural 7-segment OCR engine
│   ├── build_digit_templates.py # NCC templates + shared adjust_blobs_to_7()
│   └── output/F{FLT}/        # Per-flight outputs (step1/, step2/, step3/)
├── tools/ASTRA/              # ASTRA_Mac.py (retired for geometry; picks still used for R₀ cross-check)
├── Data/ascope/
│   ├── picks/{FLT}/{FLT}_CombinedASTRAPicks.csv
│   └── raw/{FLT}/            # Raw A-scope TIFFs
├── Navigation_Files/         # 61 per-flight CSVs: CBD, LAT, LON, THK, SRF; 66,141 CBDs total
├── Radiometric_Calibrator/   # Phase 1–4 R₀ pipeline (run_flight.py, core/, make_*figures.py)
├── docs/figures/             # All publication figures
├── docs/BedEcho_Implementation_Plan.tex  # Research notebook (v16, 36+ pages)
└── docs/Papers/              # Reference papers and dissertations
```

A-scope picks: flights 103, 107, 114, 115, 125, 126, 127, 128, 137, 138, 141, 143
Navigation: use individual CSVs in `Navigation_Files/`, not AllFlights.csv (subset only)

## Literature — Critical Cross-Checks (all PASS)

- **Neal 1977**: primary source; P_T, G, MDS, attenuator, saline ice eqs, R₀ zones all confirmed ✓
- **Neal 1979**: confirms 20·log radar equation, c_ice=169, R₀_ref=−0.77 dB ✓
- **Schroeder 2021**: A-scope dynamic range 70 dB; Z-scope calibration varies → Z-scope retired ✓
- **Millar 1981**: G_two-way=21.4 dB=2×10.7 dB ✓; c_ice=168 ✓; Vostok 86.8 dB validates L_diel model ✓
- **Rose 1978**: INS uncertainty F125 ~3–7 km/CBD; radar eq. 20·log confirmed ✓
- **Neal 1982**: roughness metrics (V_p, τ_p, φ₀) — Phase 3 method
- Dissertations referenced: Neal 1977, Rose 1978, Millar 1981, Oswald 1975, Hargreaves 1977, Jankowski 1981

## Navigation & Positioning

- INS = Litton LTN-21; Schuler period 84.4 min → lat error ±6.5 km, lon ±1.0 km (F103 dual-INS)
- F125: 1 nav fix only (Roosevelt Is., 3.3 km shift at t=7:35) → **~3–7 km position uncertainty per CBD**
- Altitude uncertainty ~±20–25 m → ~0.1 dB effect on R₀ (second order)

## Working Conventions

- **No flattery** (wasted tokens); **no empty criticism** (must offer mitigation); **always add vector**
- **Figures**: Nature/Science standards — white bg, no top/right spines, ColorBrewer palette, ≥300 dpi, bold panel labels, no chartjunk. Use `make_pub_figures.py` as template.
- **American spelling** (color, analyze, behavior)
- **LaTeX** for all documentation (user prefers Overleaf)
- Python in VS Code on MacBook Air M3
