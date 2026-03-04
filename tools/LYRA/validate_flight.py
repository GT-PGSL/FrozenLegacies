#!/usr/bin/env python3
"""LYRA Phase 5 validation: compare ice thickness with BEDMAP1.

Usage:
    python tools/LYRA/validate_flight.py <flight_number>

Reads the Phase 4 echoes CSV for the given flight, joins with Navigation CSVs
for lat/lon, matches against BEDMAP1 ice thickness data, and generates a
3-panel along-track validation figure.

Output:
    tools/LYRA/output/F{FLT}/validation/F{FLT}_validation.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

# ── Paths (relative to repo root) ────────────────────────────────────────────

REPO = Path(__file__).resolve().parent.parent.parent  # FrozenLegacies/
NAV_DIR = REPO / "Navigation_Files"
BEDMAP_SHP = REPO / "Data" / "BEDMAP" / "BedMap1" / "bedmap1_clip.shp"
OUTPUT_BASE = Path(__file__).resolve().parent / "output"

# ── Constants ─────────────────────────────────────────────────────────────────

MATCH_RADIUS_M = 20_000  # max distance (m) for BEDMAP1 nearest-neighbour match
NAV_MISSING = 9999       # sentinel for missing THK/SRF in Navigation CSVs

STATUS_COLORS = {
    "good":       "#2ca02c",   # green
    "weak_bed":   "#ff7f0e",   # orange
    "no_bed":     "#aaaaaa",   # gray
    "no_surface": "#d62728",   # red
}
STATUS_ORDER = ["good", "weak_bed", "no_bed", "no_surface"]

BEDMAP_COLOR = "#2166ac"   # dark blue
NAV_THK_COLOR = "#999999"  # light gray

_TRANSFORM = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)


# ── Publication figure styling ────────────────────────────────────────────────

def _setup_rcparams():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica Neue", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9.5,
        "axes.titleweight": "bold",
        "axes.titlepad": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.framealpha": 0.90,
        "legend.edgecolor": "#cccccc",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.alpha": 0.3,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def _panel_label(ax, label, x=0.015, y=0.97):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="top", ha="left",
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5, alpha=0.7))


# ── Data loading ──────────────────────────────────────────────────────────────

def load_echoes(flt: int) -> pd.DataFrame:
    """Load all step3 echoes for a flight."""
    csv_path = OUTPUT_BASE / f"F{flt}" / "phase4" / f"F{flt}_echoes.csv"
    if not csv_path.exists():
        sys.exit(f"Phase 4 output not found: {csv_path}")
    df = pd.read_csv(csv_path)
    # Strip whitespace from column names (CSV writer sometimes adds spaces)
    df.columns = df.columns.str.strip()
    return df.sort_values("cbd").reset_index(drop=True)


def load_navigation(flt: int) -> pd.DataFrame:
    """Load navigation CSV for a flight."""
    nav_path = NAV_DIR / f"{flt}.csv"
    if not nav_path.exists():
        sys.exit(f"Navigation file not found: {nav_path}")
    nav = pd.read_csv(nav_path)
    nav.columns = nav.columns.str.strip()
    return nav


def load_bedmap() -> tuple[np.ndarray, np.ndarray]:
    """Load BEDMAP1 shapefile and return (coords_xy, ice_thickness) arrays.

    coords_xy: (N, 2) array in EPSG:3031
    ice_thickness: (N,) array in metres
    """
    if not BEDMAP_SHP.exists():
        sys.exit(f"BEDMAP1 shapefile not found: {BEDMAP_SHP}")
    gdf = gpd.read_file(BEDMAP_SHP)
    coords = np.column_stack([gdf["PS_x"].values, gdf["PS_y"].values])
    thk = gdf["Ice_thickn"].values.astype(float)
    return coords, thk


# ── Core logic ────────────────────────────────────────────────────────────────

def merge_with_nav(echoes: pd.DataFrame, nav: pd.DataFrame) -> pd.DataFrame:
    """Join echoes with navigation on CBD."""
    merged = echoes.merge(nav, left_on="cbd", right_on="CBD", how="left")
    # Mark missing nav rows
    merged["has_nav"] = merged["LAT"].notna()
    return merged


def match_bedmap(lats: np.ndarray, lons: np.ndarray,
                 bedmap_xy: np.ndarray, bedmap_thk: np.ndarray,
                 radius_m: float = MATCH_RADIUS_M
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Find nearest BEDMAP1 ice thickness for each (lat, lon) point.

    Returns:
        bedmap_ice: (N,) array of matched ice thickness (NaN if no match within radius)
        bedmap_dist: (N,) array of distances in metres
    """
    # Project flight points to EPSG:3031
    x_ps, y_ps = _TRANSFORM.transform(lons, lats)
    flight_xy = np.column_stack([x_ps, y_ps])

    # Build KDTree on BEDMAP1 points
    tree = cKDTree(bedmap_xy)
    dists, idxs = tree.query(flight_xy)

    # Apply radius filter
    bedmap_ice = np.where(dists <= radius_m, bedmap_thk[idxs], np.nan)
    bedmap_dist = dists

    return bedmap_ice, bedmap_dist


# ── Figure generation ─────────────────────────────────────────────────────────

def make_validation_figure(merged: pd.DataFrame, flt: int, out_path: Path):
    """Generate 3-panel along-track validation figure."""

    _setup_rcparams()

    cbd = merged["cbd"].values
    status = merged["echo_status"].values
    h_ice = merged["h_ice_m"].values
    h_air = merged["h_air_m"].values
    bed_snr = merged["bed_snr_dB"].values if "bed_snr_dB" in merged.columns else None

    # Count statistics
    n_total = len(merged)
    n_good = int((status == "good").sum())
    n_weak = int((status == "weak_bed").sum())
    n_nobed = int((status == "no_bed").sum())
    n_nosurf = int((status == "no_surface").sum())

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True,
                             gridspec_kw={"hspace": 0.12})
    ax_thk, ax_alt, ax_snr = axes

    # ── Panel (a): Ice Thickness ──────────────────────────────────────────

    # BEDMAP1 reference (plot first, behind LYRA dots)
    if "bedmap_ice" in merged.columns:
        bm_mask = merged["bedmap_ice"].notna()
        if bm_mask.any():
            ax_thk.scatter(cbd[bm_mask], merged["bedmap_ice"].values[bm_mask],
                           s=12, c=BEDMAP_COLOR, marker="D", alpha=0.5,
                           zorder=2, label="BEDMAP1")

    # Navigation THK reference
    if "THK" in merged.columns:
        nav_mask = merged["THK"].notna() & (merged["THK"] != NAV_MISSING)
        if nav_mask.any():
            ax_thk.plot(cbd[nav_mask], merged["THK"].values[nav_mask],
                        color=NAV_THK_COLOR, linewidth=1.0, alpha=0.7,
                        zorder=1, label="Nav CSV")

    # LYRA h_ice (plot on top, by status)
    for st in STATUS_ORDER:
        mask = status == st
        if not mask.any():
            continue
        vals = h_ice[mask]
        valid = np.isfinite(vals)
        if valid.any():
            ax_thk.scatter(cbd[mask][valid], vals[valid],
                           s=20, c=STATUS_COLORS[st], edgecolors="k",
                           linewidths=0.3, zorder=3, label=f"LYRA ({st})")

    ax_thk.set_ylabel("Ice thickness (m)")
    ax_thk.legend(loc="upper right", ncol=2, markerscale=1.2)
    _panel_label(ax_thk, "(a)")

    # ── Panel (b): Aircraft Altitude ──────────────────────────────────────

    for st in STATUS_ORDER:
        mask = status == st
        if not mask.any():
            continue
        vals = h_air[mask]
        valid = np.isfinite(vals)
        if valid.any():
            ax_alt.scatter(cbd[mask][valid], vals[valid],
                           s=15, c=STATUS_COLORS[st], edgecolors="k",
                           linewidths=0.3, zorder=3)

    ax_alt.set_ylabel("h_air (m)")
    _panel_label(ax_alt, "(b)")

    # ── Panel (c): Bed SNR ────────────────────────────────────────────────

    if bed_snr is not None:
        # Weak-bed threshold line
        ax_snr.axhline(5.0, color="#d62728", linewidth=0.8, linestyle="--",
                       alpha=0.6, zorder=1, label="weak_bed threshold (5 dB)")

        for st in STATUS_ORDER:
            mask = status == st
            if not mask.any():
                continue
            vals = bed_snr[mask]
            valid = np.isfinite(vals)
            if not valid.any():
                continue

            # Check for artifact flag
            edge_colors = np.full(valid.sum(), "k", dtype=object)
            if "bed_envelope_suspect" in merged.columns:
                suspect = merged["bed_envelope_suspect"].values[mask][valid]
                edge_colors[suspect == True] = "#d62728"  # noqa: E712

            ax_snr.scatter(cbd[mask][valid], vals[valid],
                           s=15, c=STATUS_COLORS[st], edgecolors=edge_colors,
                           linewidths=np.where(edge_colors == "#d62728", 1.2, 0.3),
                           zorder=3)

        ax_snr.legend(loc="upper right", fontsize=6.5)

    ax_snr.set_ylabel("Bed SNR (dB)")
    ax_snr.set_xlabel("CBD number")
    _panel_label(ax_snr, "(c)")

    # ── Title and summary ─────────────────────────────────────────────────

    summary = f"F{flt} \u2014 LYRA Phase 5 Validation"
    summary += f"  [{n_good} good"
    if n_weak:
        summary += f", {n_weak} weak"
    summary += f", {n_nobed} no_bed"
    if n_nosurf:
        summary += f", {n_nosurf} no_surface"
    summary += f" / {n_total} total]"
    fig.suptitle(summary, fontsize=10, fontweight="bold", y=0.995)

    # BEDMAP match summary in bottom-right
    if "bedmap_ice" in merged.columns:
        n_matched = int(merged["bedmap_ice"].notna().sum())
        if n_matched > 0:
            median_dist = merged.loc[merged["bedmap_ice"].notna(),
                                     "bedmap_dist"].median()
            ax_snr.text(0.99, 0.02,
                        f"BEDMAP1: {n_matched} matches (median {median_dist/1000:.1f} km)",
                        transform=ax_snr.transAxes, fontsize=6.5,
                        ha="right", va="bottom", color=BEDMAP_COLOR)

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Validation figure \u2192 {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LYRA Phase 5 validation with BEDMAP1 comparison")
    parser.add_argument("flight", type=int, help="Flight number (e.g. 126)")
    parser.add_argument("--radius", type=float, default=MATCH_RADIUS_M,
                        help=f"BEDMAP1 match radius in metres (default {MATCH_RADIUS_M})")
    args = parser.parse_args()

    flt = args.flight
    print(f"LYRA Validation \u2014 F{flt}")

    # Load data
    print("  Loading phase 4 echoes ...")
    echoes = load_echoes(flt)
    print(f"    {len(echoes)} frames")

    print("  Loading navigation ...")
    nav = load_navigation(flt)
    print(f"    {len(nav)} CBDs with coordinates")

    print("  Merging echoes with navigation ...")
    merged = merge_with_nav(echoes, nav)
    n_with_nav = int(merged["has_nav"].sum())
    print(f"    {n_with_nav}/{len(merged)} frames matched to coordinates")

    # BEDMAP1 matching
    print("  Loading BEDMAP1 ...")
    bedmap_xy, bedmap_thk = load_bedmap()
    print(f"    {len(bedmap_thk)} points loaded")

    # Match only frames with valid coordinates
    has_coords = merged["has_nav"] & merged["LAT"].notna() & merged["LON"].notna()
    merged["bedmap_ice"] = np.nan
    merged["bedmap_dist"] = np.nan

    if has_coords.any():
        lats = merged.loc[has_coords, "LAT"].values
        lons = merged.loc[has_coords, "LON"].values
        bm_ice, bm_dist = match_bedmap(lats, lons, bedmap_xy, bedmap_thk,
                                       radius_m=args.radius)
        merged.loc[has_coords, "bedmap_ice"] = bm_ice
        merged.loc[has_coords, "bedmap_dist"] = bm_dist

        n_matched = int(np.isfinite(bm_ice).sum())
        print(f"    {n_matched} BEDMAP1 matches within {args.radius/1000:.0f} km")
    else:
        print("    No coordinates available for BEDMAP matching")

    # Generate figure
    out_path = OUTPUT_BASE / f"F{flt}" / "validation" / f"F{flt}_validation.png"
    print("  Generating validation figure ...")
    make_validation_figure(merged, flt, out_path)

    # Summary statistics
    good = merged[merged["echo_status"] == "good"]
    if len(good) > 0 and "bedmap_ice" in merged.columns:
        both = good[good["bedmap_ice"].notna() & good["h_ice_m"].notna()]
        if len(both) > 0:
            diff = both["h_ice_m"].values - both["bedmap_ice"].values
            print(f"\n  Ice thickness comparison (LYRA vs BEDMAP1, n={len(both)} good frames):")
            print(f"    Mean difference: {np.mean(diff):+.0f} m")
            print(f"    Std difference:  {np.std(diff):.0f} m")
            print(f"    Median abs diff: {np.median(np.abs(diff)):.0f} m")

    print("\nDone.")


if __name__ == "__main__":
    main()
