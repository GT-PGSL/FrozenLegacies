#!/usr/bin/env python3
"""
Diagnostic: measure sig_start and D_mb = mb_x - sig_start across flights.

Tests the hypothesis that D_mb is more stable than mb_x across frames,
because frame-to-frame mb_x variation is caused by film frame boundary
shifts, not by actual oscilloscope timing changes.

Usage:
    python tools/LYRA/diag_sig_start.py 127      # single flight
    python tools/LYRA/diag_sig_start.py 125 127   # multiple flights
"""
import sys, os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "LYRA"))
import lyra

DEFAULT_CAL = {
    "y_ref_px":   1507,
    "y_ref_db":   -60.0,
    "db_per_px":  0.04878,
    "us_per_px":  2.0 / 205.54,
    "y_disp_lo":  300,
    "y_disp_hi":  1700,
    "mb_x_guess": 800,
    "mb_skip_us": 2.0,
}


def analyze_flight(flt_str):
    flt = flt_str.lstrip("Ff")
    raw_dir = ROOT / "Data" / "ascope" / "raw" / flt
    out_dir = ROOT / "tools" / "LYRA" / "output" / f"F{flt}"

    # Load frame index
    fi_path = out_dir / "phase1" / f"F{flt}_frame_index.csv"
    if not fi_path.exists():
        print(f"  [!] No frame_index.csv for F{flt}")
        return None
    fi = pd.read_csv(fi_path)

    # Load cal CSV if available
    cal_path = out_dir / "phase3" / f"F{flt}_cal.csv"
    cal_df = None
    if cal_path.exists():
        cal_df = pd.read_csv(cal_path)

    results = []  # list of dicts

    # Group by TIFF
    tiff_groups = fi.groupby("tiff")
    n_tiffs = len(tiff_groups)

    for tiff_idx, (tiff_name, tiff_rows) in enumerate(tiff_groups):
        tiff_path = raw_dir / tiff_name
        if not tiff_path.exists():
            continue

        # Load TIFF
        img = np.array(Image.open(tiff_path)).astype(float)
        if img.ndim == 3:
            img = img.mean(axis=2)
        # Global normalize
        vmin, vmax = float(img.min()), float(img.max())
        if vmax > vmin:
            img = (img - vmin) / (vmax - vmin)

        tiff_id = tiff_name.split("_")[1].lstrip("0") or "0"

        complete_rows = tiff_rows[tiff_rows["frame_type"] == "complete"]
        for _, row in complete_rows.iterrows():
            left  = int(row["left_px"])
            right = int(row["right_px"])
            frame = img[:, left:right+1]
            fidx  = int(row["frame_idx"])
            cbd   = row.get("cbd", "")

            # Build cal for this frame
            cal = dict(DEFAULT_CAL)
            mb_x_from_cal = None

            if cal_df is not None:
                # Match by tiff + frame_idx or cbd
                mask = cal_df["tiff"] == tiff_name
                if "frame_idx" in cal_df.columns:
                    match = cal_df[mask & (cal_df["frame_idx"] == fidx)]
                elif "cbd" in cal_df.columns and cbd:
                    match = cal_df[mask & (cal_df["cbd"] == int(cbd))]
                else:
                    match = pd.DataFrame()

                if len(match) == 1:
                    cr = match.iloc[0]
                    cal["y_ref_px"]  = float(cr.get("y_ref_px", cal["y_ref_px"]))
                    cal["db_per_px"] = float(cr.get("db_per_px", cal["db_per_px"]))
                    cal["us_per_px"] = float(cr.get("us_per_px", cal["us_per_px"]))
                    if "mb_x" in cr and not pd.isna(cr["mb_x"]):
                        mb_x_from_cal = int(cr["mb_x"])

            # Detect signal extent
            try:
                sig_start, sig_end = lyra.detect_signal_extent(frame, cal)
            except Exception as e:
                print(f"    [!] sig_extent failed F{flt} {tiff_id} fr{fidx}: {e}")
                continue

            d_mb = None
            if mb_x_from_cal is not None:
                d_mb = mb_x_from_cal - sig_start

            results.append({
                "flight":    flt,
                "tiff_id":   tiff_id,
                "tiff_name": tiff_name,
                "frame_idx": fidx,
                "cbd":       cbd,
                "sig_start": sig_start,
                "sig_end":   sig_end,
                "mb_x":      mb_x_from_cal,
                "D_mb":      d_mb,
            })

        if (tiff_idx + 1) % 5 == 0 or tiff_idx == n_tiffs - 1:
            print(f"  F{flt}: processed {tiff_idx+1}/{n_tiffs} TIFFs "
                  f"({len(results)} frames so far)")

    return pd.DataFrame(results) if results else None


def print_stats(df, label):
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    n = len(df)
    print(f"  Total frames: {n}")

    # sig_start stats
    ss = df["sig_start"]
    print(f"\n  sig_start (CRT trace start column):")
    print(f"    mean={ss.mean():.1f}  std={ss.std():.1f}  "
          f"min={ss.min()}  max={ss.max()}  range={ss.max()-ss.min()}")

    # Per-TIFF sig_start stats
    print(f"\n  Per-TIFF sig_start (within-TIFF std):")
    for tid, grp in df.groupby("tiff_id"):
        s = grp["sig_start"]
        print(f"    TIFF {tid:>5s}: n={len(s):3d}  "
              f"mean={s.mean():7.1f}  std={s.std():6.1f}  "
              f"range=[{s.min()}, {s.max()}]  span={s.max()-s.min()}")

    # mb_x stats (if available)
    has_mb = df["mb_x"].notna()
    if has_mb.any():
        mb = df.loc[has_mb, "mb_x"]
        print(f"\n  mb_x (frame-relative main bang column):")
        print(f"    mean={mb.mean():.1f}  std={mb.std():.1f}  "
              f"min={mb.min():.0f}  max={mb.max():.0f}  range={mb.max()-mb.min():.0f}")

        # Per-TIFF mb_x stats
        print(f"\n  Per-TIFF mb_x (within-TIFF std):")
        for tid, grp in df[has_mb].groupby("tiff_id"):
            m = grp["mb_x"]
            print(f"    TIFF {tid:>5s}: n={len(m):3d}  "
                  f"mean={m.mean():7.1f}  std={m.std():6.1f}  "
                  f"range=[{m.min():.0f}, {m.max():.0f}]  span={m.max()-m.min():.0f}")

        # D_mb stats
        dm = df.loc[has_mb, "D_mb"]
        print(f"\n  D_mb = mb_x - sig_start:")
        print(f"    mean={dm.mean():.1f}  std={dm.std():.1f}  "
              f"min={dm.min():.0f}  max={dm.max():.0f}  range={dm.max()-dm.min():.0f}")

        # Per-TIFF D_mb stats
        print(f"\n  Per-TIFF D_mb (within-TIFF std):")
        for tid, grp in df[has_mb].groupby("tiff_id"):
            d = grp["D_mb"]
            print(f"    TIFF {tid:>5s}: n={len(d):3d}  "
                  f"mean={d.mean():7.1f}  std={d.std():6.1f}  "
                  f"range=[{d.min():.0f}, {d.max():.0f}]  span={d.max()-d.min():.0f}")

        # Key comparison: std(mb_x) vs std(D_mb) per TIFF
        print(f"\n  STABILITY COMPARISON: std(mb_x) vs std(D_mb) per TIFF:")
        print(f"    {'TIFF':>5s}  {'n':>3s}  {'std(mb_x)':>10s}  {'std(D_mb)':>10s}  {'reduction':>10s}")
        for tid, grp in df[has_mb].groupby("tiff_id"):
            m_std = grp["mb_x"].std()
            d_std = grp["D_mb"].std()
            pct = (1 - d_std / m_std) * 100 if m_std > 0 else 0
            marker = " <-- BETTER" if d_std < m_std else ""
            print(f"    {tid:>5s}  {len(grp):3d}  {m_std:10.2f}  {d_std:10.2f}  "
                  f"{pct:+8.1f}%{marker}")


if __name__ == "__main__":
    flights = sys.argv[1:] if len(sys.argv) > 1 else ["127"]
    print(f"Analyzing flights: {', '.join(flights)}")

    all_dfs = []
    for flt in flights:
        print(f"\n--- Processing F{flt} ---")
        df = analyze_flight(flt)
        if df is not None and len(df) > 0:
            all_dfs.append(df)
            print_stats(df, f"F{flt}")

    if len(all_dfs) > 1:
        combined = pd.concat(all_dfs, ignore_index=True)
        print_stats(combined, "ALL FLIGHTS COMBINED")
