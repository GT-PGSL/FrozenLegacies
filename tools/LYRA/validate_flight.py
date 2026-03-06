#!/usr/bin/env python3
"""LYRA Phase 5 validation: compare ice thickness with ASTRA and BEDMAP1.

Usage:
    python tools/LYRA/validate_flight.py <flight_number>

Reads the Phase 4 echoes CSV for the given flight, compares against:
  - ASTRA manual picks on the same A-scope frames (d = 0; primary validation)
  - RIGGS ice thickness (independent seismic/radar; Bentley 1984)
  - BEDMAP1 SPRI (same-source, spatial context only)

Output:
    tools/LYRA/output/F{FLT}/validation/F{FLT}_validation.png
    tools/LYRA/output/F{FLT}/validation/F{FLT}_validation.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

# Import basemap helpers from plot_flight_tracks
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plot_flight_tracks import load_basemap_ps, _graticule_line, _meridian_line, _geo_label

# -- Paths (relative to repo root) --------------------------------------------

REPO = Path(__file__).resolve().parent.parent.parent  # FrozenLegacies/
NAV_DIR = REPO / "Navigation_Files"
BEDMAP_SHP = REPO / "Data" / "BEDMAP" / "BedMap1" / "bedmap1_clip.shp"
RIGGS_XLSX = REPO / "Data" / "RIGGS" / "RIGGS_H_Ice.xlsx"
OUTPUT_BASE = Path(__file__).resolve().parent / "output"

# -- Constants -----------------------------------------------------------------

MATCH_RADIUS_M = 20_000  # max distance (m) for nearest-neighbour match
NAV_MISSING = 9999       # sentinel for missing THK/SRF in Navigation CSVs
ASTRA_DIR = REPO / "Data" / "ascope" / "picks"

# ASTRA timing correction: ASTRA assumed 3 µs/div but correct is 1.5 µs/div
C_ICE = 168.0   # m/µs
C_AIR = 300.0   # m/µs

# BEDMAP1 mission IDs ---------------------------------------------------------
# SPRI_7475 is the same SPRI/NSF/TUD 1974-75 data that LYRA reprocesses
# (comparison is circular). All other missions are independent.
SAME_SOURCE_MISSIONS = {"SPRI_7475"}

# ColorBrewer-based palette ----------------------------------------------------
STATUS_COLORS = {
    "good":       "#1b9e77",   # teal-green (ColorBrewer Dark2)
    "weak_bed":   "#d95f02",   # orange
    "no_bed":     "#bdbdbd",   # gray
    "no_surface": "#e7298a",   # magenta-pink
}
STATUS_LABELS = {
    "good":       "Good echo",
    "weak_bed":   "Weak bed (SNR < 5 dB)",
    "no_bed":     "No bed echo",
    "no_surface": "No surface echo",
}
STATUS_ORDER = ["good", "weak_bed", "no_bed", "no_surface"]

# Reference dataset colors
RIGGS_SEIS_COLOR  = "#d62728"   # red (seismic — independent, high accuracy)
RIGGS_RADAR_COLOR = "#ff7f0e"   # orange (RIGGS 35 MHz radar)
SPRI_COLOR        = "#2166ac"   # blue (same-source, circular)

_TRANSFORM = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)


# -- Publication figure styling ------------------------------------------------

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


# -- Data loading --------------------------------------------------------------

def load_echoes(flt: int) -> pd.DataFrame:
    """Load all step3 echoes for a flight."""
    csv_path = OUTPUT_BASE / f"F{flt}" / "phase4" / f"F{flt}_echoes.csv"
    if not csv_path.exists():
        sys.exit(f"Phase 4 output not found: {csv_path}")
    df = pd.read_csv(csv_path)
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


def load_bedmap() -> gpd.GeoDataFrame:
    """Load BEDMAP1 shapefile with full attribution."""
    if not BEDMAP_SHP.exists():
        sys.exit(f"BEDMAP1 shapefile not found: {BEDMAP_SHP}")
    return gpd.read_file(BEDMAP_SHP)


def load_astra(flt: int) -> pd.DataFrame | None:
    """Load ASTRA manual picks for a flight and compute corrected h_ice/h_air.

    ASTRA assumed 3 us/div but the correct calibration is 1.5 us/div,
    so all travel times must be divided by 2.  Then convert one-way:
        h_ice = (bed_us - surface_us) / 4 * c_ice
        h_air = surface_us / 4 * c_air
    (divide by 4 = /2 for ASTRA error * /2 for two-way -> one-way)

    Returns DataFrame with columns: CBD, h_ice_astra, h_air_astra
    or None if no ASTRA data exists.
    """
    csv_path = ASTRA_DIR / str(flt) / f"{flt}_CombinedASTRAPicks.csv"
    if not csv_path.exists():
        return None
    astra = pd.read_csv(csv_path)
    astra.columns = astra.columns.str.strip()

    # Need both surface and bed travel times, and a valid CBD
    has_both = (astra["surface_us"].notna() & astra["bed_us"].notna()
                & (astra["bed_us"] > 0) & astra["CBD"].notna())
    astra = astra[has_both].copy()
    if astra.empty:
        return None

    # Apply /2 timing correction + one-way conversion (/4 total)
    astra["h_ice_astra"] = (astra["bed_us"] - astra["surface_us"]) / 4 * C_ICE
    astra["h_air_astra"] = astra["surface_us"] / 4 * C_AIR

    # Keep only valid ice thicknesses
    astra = astra[astra["h_ice_astra"] > 0].copy()

    # Ensure CBD is integer to match LYRA's echoes CSV
    astra["CBD"] = astra["CBD"].astype(int)

    # Average multiple ASTRA picks per CBD (some have >1 A-scope per frame)
    astra = astra.groupby("CBD")[["h_ice_astra", "h_air_astra"]].mean().reset_index()

    return astra


def load_riggs_stations() -> pd.DataFrame | None:
    """Load RIGGS ice thickness stations (Bentley 1984, Table 5).

    Uses seismic thickness where available (+/-10 m accuracy, Bamber &
    Bentley 1994), with RIGGS radar (35 MHz) as fallback.  Both are
    independent of SPRI 60 MHz.

    Returns DataFrame with: Station, h_ice, source, LAT_dd, LON_dd, x_ps, y_ps
    or None if the XLSX is missing.
    """
    if not RIGGS_XLSX.exists():
        return None
    df = pd.read_excel(RIGGS_XLSX)

    # Convert NM/NR strings to NaN
    for col in ["Radar_H_ice_m", "Seismic_H_ice_m"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop stations without coordinates
    df = df[df["LAT_dd"].notna() & df["LON_dd"].notna()].copy()

    # Prefer seismic, fall back to radar
    df["h_ice"] = df["Seismic_H_ice_m"].fillna(df["Radar_H_ice_m"])
    df["source"] = np.where(df["Seismic_H_ice_m"].notna(), "seismic", "radar")
    df = df[df["h_ice"].notna()].copy()

    # Project to polar stereographic
    x_ps, y_ps = _TRANSFORM.transform(df["LON_dd"].values, df["LAT_dd"].values)
    df["x_ps"] = x_ps
    df["y_ps"] = y_ps

    return df[["Station", "h_ice", "source", "LAT_dd", "LON_dd",
               "x_ps", "y_ps"]].reset_index(drop=True)


# -- Core logic ----------------------------------------------------------------

def merge_with_nav(echoes: pd.DataFrame, nav: pd.DataFrame) -> pd.DataFrame:
    """Join echoes with navigation on CBD."""
    merged = echoes.merge(nav, left_on="cbd", right_on="CBD", how="left")
    merged["has_nav"] = merged["LAT"].notna()
    return merged


def match_reference(lats: np.ndarray, lons: np.ndarray,
                    ref_xy: np.ndarray, ref_thk: np.ndarray,
                    radius_m: float = MATCH_RADIUS_M
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find nearest reference ice thickness for each (lat, lon) point.

    Returns:
        ref_ice: (N,) matched ice thickness (NaN if no match within radius)
        ref_dist: (N,) distances in metres
        ref_idx: (N,) index into ref arrays (-1 if no match)
    """
    x_ps, y_ps = _TRANSFORM.transform(lons, lats)
    flight_xy = np.column_stack([x_ps, y_ps])

    tree = cKDTree(ref_xy)
    dists, idxs = tree.query(flight_xy)

    within = dists <= radius_m
    ref_ice = np.where(within, ref_thk[idxs], np.nan)
    ref_idx = np.where(within, idxs, -1)
    return ref_ice, dists, ref_idx


def _extract_ref_arrays(gdf: gpd.GeoDataFrame
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Extract (xy_coords, thickness) arrays from a BEDMAP1 subset."""
    xy = np.column_stack([gdf["PS_x"].values, gdf["PS_y"].values])
    thk = gdf["Ice_thickn"].values.astype(float)
    return xy, thk


# -- Inset map helper ---------------------------------------------------------

def _draw_inset_map(ax_map, nav, merged, status, riggs_df, spri_gdf, flt):
    """Draw spatial context map on the given inset axes.

    Uses Natural Earth basemap from plot_flight_tracks, with graticules,
    geographic labels, flight track, RIGGS stations, and BEDMAP1 SPRI.
    """
    # Load basemap
    land = load_basemap_ps()
    land.plot(ax=ax_map, color="#e4eef5", edgecolor="#7aadbb",
              linewidth=0.4, zorder=1)

    # Determine map extent from flight coordinates (with margin).
    # Expand the shorter dimension so the map fills the panel height
    # while keeping equal-aspect geographic proportions.
    has_coords = merged["LAT"].notna() & merged["LON"].notna()
    if has_coords.any():
        fx, fy = _TRANSFORM.transform(
            merged.loc[has_coords, "LON"].values,
            merged.loc[has_coords, "LAT"].values)
        margin = 150_000  # 150 km buffer
        x_lim = (fx.min() - margin, fx.max() + margin)
        y_lim = (fy.min() - margin, fy.max() + margin)
    else:
        x_lim = (-700_000, 500_000)
        y_lim = (-1_400_000, -400_000)

    # Get the panel's physical aspect ratio from the figure geometry
    fig = ax_map.get_figure()
    bbox = ax_map.get_position()
    panel_w = bbox.width * fig.get_figwidth()
    panel_h = bbox.height * fig.get_figheight()
    panel_aspect = panel_h / panel_w  # physical H/W

    x_range = x_lim[1] - x_lim[0]
    y_range = y_lim[1] - y_lim[0]
    data_aspect = y_range / x_range

    if data_aspect < panel_aspect:
        # Map is wider than tall — expand y to fill height
        needed_y = x_range * panel_aspect
        y_mid = (y_lim[0] + y_lim[1]) / 2
        y_lim = (y_mid - needed_y / 2, y_mid + needed_y / 2)
    else:
        # Map is taller than wide — expand x to fill width
        needed_x = y_range / panel_aspect
        x_mid = (x_lim[0] + x_lim[1]) / 2
        x_lim = (x_mid - needed_x / 2, x_mid + needed_x / 2)

    ax_map.set_xlim(*x_lim)
    ax_map.set_ylim(*y_lim)

    # Graticules (sparse for inset)
    lon_arr = np.linspace(-180, 180, 721)
    lat_arr = np.linspace(-90, -60, 200)
    for lat in [-85, -80, -75]:
        lx, ly = _graticule_line(lon_arr, lat)
        ax_map.plot(lx, ly, color="#cccccc", linewidth=0.3,
                    zorder=0, linestyle="--")
    for lon in range(-180, 181, 30):
        lx, ly = _meridian_line(lat_arr, lon)
        ax_map.plot(lx, ly, color="#cccccc", linewidth=0.3,
                    zorder=0, linestyle=":")

    # Full flight track from navigation
    if nav is not None:
        nav_valid = nav[nav["LAT"].notna() & nav["LON"].notna()]
        if len(nav_valid) > 0:
            tx, ty = _TRANSFORM.transform(nav_valid["LON"].values,
                                          nav_valid["LAT"].values)
            ax_map.plot(tx, ty, color="#555555", linewidth=0.8,
                        alpha=0.5, zorder=2, label=f"F{flt} track")

    # LYRA frames (colored by status)
    if has_coords.any():
        fst = status[has_coords]
        for st in STATUS_ORDER:
            sm = fst == st
            if sm.any():
                ax_map.scatter(fx[sm], fy[sm], s=6, c=STATUS_COLORS[st],
                               edgecolors="white", linewidths=0.15, zorder=4)

    # BEDMAP1 SPRI points (same-source, spatial context)
    if spri_gdf is not None and len(spri_gdf) > 0:
        ax_map.scatter(spri_gdf["PS_x"].values, spri_gdf["PS_y"].values,
                       s=2, facecolors="none", edgecolors=SPRI_COLOR,
                       linewidths=0.2, marker="D", alpha=0.2, zorder=1.5,
                       label="BEDMAP1 SPRI")

    # RIGGS stations — seismic vs radar, filled if matched to flight
    if riggs_df is not None and len(riggs_df) > 0:
        # Determine which stations were matched (used in left panel comparison)
        matched_stns = set()
        if "riggs_stn_idx" in merged.columns:
            idx_vals = merged["riggs_stn_idx"].dropna().values
            matched_stns = set(int(i) for i in idx_vals if i >= 0)
        # Map riggs_df row positions to their original index for matching
        riggs_idx = riggs_df.index.values
        is_matched = np.isin(riggs_idx, list(matched_stns))

        seis = riggs_df[riggs_df["source"] == "seismic"]
        radar = riggs_df[riggs_df["source"] == "radar"]
        seis_matched = np.isin(seis.index.values, list(matched_stns))
        radar_matched = np.isin(radar.index.values, list(matched_stns))

        # Plot unmatched (hollow) then matched (filled) for each source
        if len(seis) > 0:
            seis_label_used = False
            if (~seis_matched).any():
                ax_map.scatter(seis.loc[~seis_matched, "x_ps"].values,
                               seis.loc[~seis_matched, "y_ps"].values,
                               s=12, facecolors="none",
                               edgecolors=RIGGS_SEIS_COLOR,
                               linewidths=0.6, marker="s", zorder=6,
                               label="RIGGS seismic")
                seis_label_used = True
            if seis_matched.any():
                ax_map.scatter(seis.loc[seis_matched, "x_ps"].values,
                               seis.loc[seis_matched, "y_ps"].values,
                               s=16, facecolors=RIGGS_SEIS_COLOR,
                               edgecolors=RIGGS_SEIS_COLOR,
                               linewidths=0.6, marker="s", zorder=6.5,
                               label=("_nolegend_" if seis_label_used
                                      else "RIGGS seismic"))
        if len(radar) > 0:
            radar_label_used = False
            if (~radar_matched).any():
                ax_map.scatter(radar.loc[~radar_matched, "x_ps"].values,
                               radar.loc[~radar_matched, "y_ps"].values,
                               s=10, facecolors="none",
                               edgecolors=RIGGS_RADAR_COLOR,
                               linewidths=0.5, marker="s", zorder=5.5,
                               label="RIGGS radar")
                radar_label_used = True
            if radar_matched.any():
                ax_map.scatter(radar.loc[radar_matched, "x_ps"].values,
                               radar.loc[radar_matched, "y_ps"].values,
                               s=14, facecolors=RIGGS_RADAR_COLOR,
                               edgecolors=RIGGS_RADAR_COLOR,
                               linewidths=0.5, marker="s", zorder=6,
                               label=("_nolegend_" if radar_label_used
                                      else "RIGGS radar"))

    # Geographic labels (only those visible in extent)
    for lon, lat, text, fs in [
        (-155, -79.5, "Ross Ice\nShelf", 6),
        (-130, -80.5, "Siple Coast", 5.5),
        (170, -78.5, "Ross Sea", 5.5),
    ]:
        gx, gy = _TRANSFORM.transform(np.array([lon]), np.array([lat]))
        if x_lim[0] < gx[0] < x_lim[1] and y_lim[0] < gy[0] < y_lim[1]:
            ax_map.text(gx[0], gy[0], text, fontsize=fs, color="#557799",
                        ha="center", va="center", fontstyle="italic",
                        zorder=7,
                        path_effects=[pe.withStroke(linewidth=1.5,
                                                    foreground="white")])

    # Scale bar
    sb_len = 100_000  # 100 km
    sb_x0 = x_lim[0] + 0.06 * (x_lim[1] - x_lim[0])
    sb_y0 = y_lim[0] + 0.06 * (y_lim[1] - y_lim[0])
    ax_map.plot([sb_x0, sb_x0 + sb_len], [sb_y0, sb_y0],
                color="k", linewidth=1.2, solid_capstyle="butt", zorder=8)
    for xv in [sb_x0, sb_x0 + sb_len]:
        ax_map.plot([xv, xv], [sb_y0 - 8_000, sb_y0 + 8_000],
                    color="k", linewidth=0.8, zorder=8)
    ax_map.text(sb_x0 + sb_len / 2, sb_y0 + 15_000, "100 km",
                fontsize=4.5, ha="center", va="bottom")

    # Axes formatting
    ax_map.set_aspect("equal")
    ax_map.legend(loc="upper left", fontsize=6, markerscale=0.8,
                  handletextpad=0.3, borderpad=0.4,
                  framealpha=0.85, edgecolor="#cccccc")

    # Tick labels in km, matching left panel font sizes
    def _m2km(x, _pos=None):
        return f"{x / 1e3:.0f}"
    ax_map.xaxis.set_major_formatter(FuncFormatter(_m2km))
    ax_map.yaxis.set_major_formatter(FuncFormatter(_m2km))
    ax_map.xaxis.set_major_locator(plt.MultipleLocator(300_000))
    ax_map.yaxis.set_major_locator(plt.MultipleLocator(200_000))
    ax_map.tick_params(labelsize=7, direction="in", pad=2)
    ax_map.set_xlabel("PS easting (km)", fontsize=8)
    ax_map.set_ylabel("PS northing (km)", fontsize=8)

    # Enable all spines for the map inset
    for spine in ax_map.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)


# -- Figure generation ---------------------------------------------------------

def make_validation_figure(merged: pd.DataFrame, flt: int, out_path: Path,
                           nav: pd.DataFrame | None = None,
                           riggs_df: pd.DataFrame | None = None,
                           spri_gdf: gpd.GeoDataFrame | None = None):
    """Generate along-track validation figure with inset map.

    Main panel: Along-track ice thickness (LYRA vs RIGGS and BEDMAP1 SPRI).
    Inset: Spatial map with flight track, RIGGS stations, BEDMAP1 SPRI.
    """

    _setup_rcparams()

    cbd = merged["cbd"].values
    status = merged["echo_status"].values
    h_ice = merged["h_ice_m"].values

    # Count statistics
    counts = {st: int((status == st).sum()) for st in STATUS_ORDER}

    # -- Layout: along-track panel + side map -----------------------------

    fig = plt.figure(figsize=(12, 4.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.8, 1], wspace=0.30,
                          left=0.06, right=0.97, top=0.88, bottom=0.12)
    ax_thk = fig.add_subplot(gs[0, 0])

    # -- Main panel: Along-track ice thickness -----------------------------

    # RIGGS (independent) — seismic vs radar distinction
    has_riggs = "riggs_ice" in merged.columns
    if has_riggs:
        rm = merged["riggs_ice"].notna()
        if rm.any():
            # Split by source type
            has_source = "riggs_source" in merged.columns
            if has_source:
                rm_seis = rm & (merged["riggs_source"] == "seismic")
                rm_radar = rm & (merged["riggs_source"] == "radar")
            else:
                rm_seis = rm
                rm_radar = pd.Series(False, index=merged.index)

            # Seismic RIGGS
            if rm_seis.any():
                n_seis_pts = int(rm_seis.sum())
                if "riggs_stn_idx" in merged.columns:
                    n_seis_stns = int(merged.loc[rm_seis, "riggs_stn_idx"].nunique())
                else:
                    n_seis_stns = n_seis_pts
                md_seis = merged.loc[rm_seis, "riggs_dist"].median()
                ax_thk.scatter(cbd[rm_seis], merged["riggs_ice"].values[rm_seis],
                               s=30, facecolors="none",
                               edgecolors=RIGGS_SEIS_COLOR,
                               linewidths=0.9, marker="s", alpha=0.8, zorder=2,
                               label=f"RIGGS seismic ($n_{{\\mathrm{{stn}}}}$="
                                     f"{n_seis_stns}, "
                                     f"$\\tilde{{d}}$={md_seis/1000:.0f} km)")

            # Radar RIGGS
            if rm_radar.any():
                n_radar_pts = int(rm_radar.sum())
                if "riggs_stn_idx" in merged.columns:
                    n_radar_stns = int(merged.loc[rm_radar, "riggs_stn_idx"].nunique())
                else:
                    n_radar_stns = n_radar_pts
                md_radar = merged.loc[rm_radar, "riggs_dist"].median()
                ax_thk.scatter(cbd[rm_radar],
                               merged["riggs_ice"].values[rm_radar],
                               s=25, facecolors="none",
                               edgecolors=RIGGS_RADAR_COLOR,
                               linewidths=0.8, marker="s", alpha=0.8, zorder=2,
                               label=f"RIGGS radar ($n_{{\\mathrm{{stn}}}}$="
                                     f"{n_radar_stns}, "
                                     f"$\\tilde{{d}}$={md_radar/1000:.0f} km)")

            # Station name annotations (one per unique station)
            if ("riggs_stn_name" in merged.columns
                    and "riggs_stn_idx" in merged.columns):
                labeled = set()
                for _, row in merged[rm].iterrows():
                    stn_idx = row.get("riggs_stn_idx", -1)
                    stn_name = row.get("riggs_stn_name", "")
                    src = row.get("riggs_source", "seismic")
                    clr = (RIGGS_SEIS_COLOR if src == "seismic"
                           else RIGGS_RADAR_COLOR)
                    if stn_idx >= 0 and stn_idx not in labeled and stn_name:
                        ax_thk.annotate(stn_name,
                                        (row["cbd"], row["riggs_ice"]),
                                        textcoords="offset points",
                                        xytext=(4, 4), fontsize=5.5,
                                        color=clr, alpha=0.8)
                        labeled.add(stn_idx)

    # BEDMAP1 SPRI (same-source, spatial context only)
    has_spri = "spri_ice" in merged.columns
    if has_spri:
        sm = merged["spri_ice"].notna()
        if sm.any():
            n_spri = int(sm.sum())
            md_spri = merged.loc[sm, "spri_dist"].median()
            ax_thk.scatter(cbd[sm], merged["spri_ice"].values[sm],
                           s=18, facecolors="none", edgecolors=SPRI_COLOR,
                           linewidths=0.7, marker="D", alpha=0.5, zorder=1.5,
                           label=f"BEDMAP1 SPRI ($n$={n_spri}, "
                                 f"$\\tilde{{d}}$={md_spri/1000:.0f} km)")

    # LYRA h_ice by status
    for st in STATUS_ORDER:
        mask = status == st
        if not mask.any():
            continue
        vals = h_ice[mask]
        valid = np.isfinite(vals)
        if valid.any():
            ax_thk.scatter(cbd[mask][valid], vals[valid],
                           s=18, c=STATUS_COLORS[st],
                           edgecolors="white", linewidths=0.3, zorder=3)

    # No-detection rug marks
    for st, ypos in [("no_bed", 0.02), ("no_surface", 0.06)]:
        mask = status == st
        if not mask.any():
            continue
        for c in cbd[mask]:
            ax_thk.axvline(c, ymin=0, ymax=ypos, color=STATUS_COLORS[st],
                           linewidth=0.6, alpha=0.6)
        ax_thk.plot([], [], color=STATUS_COLORS[st], linewidth=1.5,
                    label=f"{STATUS_LABELS[st]} ({counts[st]})")

    ax_thk.set_ylabel("Ice thickness (m)")
    ax_thk.set_xlabel("CBD number")
    ax_thk.legend(loc="upper right", markerscale=1.2, handletextpad=0.4,
                  fontsize=6.5)

    # -- Title -------------------------------------------------------------

    subtitle_parts = []
    for st in STATUS_ORDER:
        if counts[st] > 0:
            subtitle_parts.append(f"{counts[st]} {STATUS_LABELS[st].lower()}")
    # Center title and subtitle over the along-track panel
    panel_center = (ax_thk.get_position().x0 + ax_thk.get_position().x1) / 2
    fig.suptitle(f"F{flt} \u2014 Along-track validation",
                 fontsize=10, fontweight="bold", y=0.99, x=panel_center)
    fig.text(panel_center, 0.935, " | ".join(subtitle_parts),
             ha="center", fontsize=7, color="#666666")

    # -- Map panel ---------------------------------------------------------

    ax_map = fig.add_subplot(gs[0, 1])
    _draw_inset_map(ax_map, nav, merged, status, riggs_df, spri_gdf, flt)

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Validation figure \u2192 {out_path}")


# -- Main ----------------------------------------------------------------------

def _comparison_stats(lyra_vals: np.ndarray, ref_vals: np.ndarray) -> dict:
    """Compute comparison statistics between LYRA and a reference dataset."""
    diff = lyra_vals - ref_vals
    stats = {
        "n": int(len(diff)),
        "mean_diff_m": round(float(np.mean(diff)), 1),
        "std_diff_m": round(float(np.std(diff)), 1),
        "rmse_m": round(float(np.sqrt(np.mean(diff**2))), 1),
        "median_abs_diff_m": round(float(np.median(np.abs(diff))), 1),
    }
    if len(diff) >= 3:
        r = np.corrcoef(ref_vals, lyra_vals)[0, 1]
        stats["correlation_r"] = round(float(r), 3)
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="LYRA Phase 5 validation with BEDMAP1 comparison")
    parser.add_argument("flight", type=int, help="Flight number (e.g. 126)")
    parser.add_argument("--radius", type=float, default=MATCH_RADIUS_M,
                        help=f"BEDMAP1 match radius in metres "
                             f"(default {MATCH_RADIUS_M})")
    args = parser.parse_args()

    flt = args.flight
    report: dict = {"flight": flt, "match_radius_km": args.radius / 1000}
    print(f"LYRA Validation \u2014 F{flt}")

    # -- Echoes ------------------------------------------------------------
    print("  Loading phase 4 echoes ...")
    echoes = load_echoes(flt)
    print(f"    {len(echoes)} frames")

    # -- Navigation --------------------------------------------------------
    print("  Loading navigation ...")
    nav = load_navigation(flt)
    print(f"    {len(nav)} CBDs with coordinates")

    print("  Merging echoes with navigation ...")
    merged = merge_with_nav(echoes, nav)
    n_with_nav = int(merged["has_nav"].sum())
    print(f"    {n_with_nav}/{len(merged)} frames matched to coordinates")

    # Echo status breakdown
    status_counts = merged["echo_status"].value_counts().to_dict()
    cbd_vals = merged["cbd"].dropna()
    tiffs = merged["tiff_id"].unique() if "tiff_id" in merged.columns else []
    good_ice = merged.loc[merged["echo_status"] == "good", "h_ice_m"]

    report["echoes"] = {
        "total_frames": len(merged),
        "n_tiffs": len(tiffs),
        "cbd_range": ([int(cbd_vals.min()), int(cbd_vals.max())]
                      if len(cbd_vals) > 0 else None),
        "echo_status": {k: int(v) for k, v in status_counts.items()},
    }
    if len(good_ice) > 0:
        report["echoes"]["h_ice_good_frames"] = {
            "min": round(float(good_ice.min()), 1),
            "max": round(float(good_ice.max()), 1),
            "mean": round(float(good_ice.mean()), 1),
            "median": round(float(good_ice.median()), 1),
        }

    # Per-status frame lists (for quick review of problem frames)
    phase4_dir = OUTPUT_BASE / f"F{flt}" / "phase4"
    for st in ("no_bed", "no_surface", "weak_bed"):
        subset = merged[merged["echo_status"] == st]
        if len(subset) == 0:
            continue
        frames = []
        for _, row in subset.iterrows():
            cbd = int(row["cbd"]) if pd.notna(row["cbd"]) else None
            tid = row.get("tiff_id", "")
            file_id = row.get("file_id", "")
            entry = {"cbd": cbd, "tiff_id": str(tid)}
            # Include diagnostic PNG path if it exists
            if file_id:
                png = phase4_dir / f"F{flt}_{file_id}_echoes.png"
                if png.exists():
                    entry["diagnostic_png"] = str(png)
            frames.append(entry)
        # Sort by CBD for easy scanning
        frames.sort(key=lambda f: f["cbd"] if f["cbd"] is not None else 0)
        report["echoes"][f"{st}_frames"] = frames

    report["navigation"] = {
        "nav_cbds": len(nav),
        "frames_with_nav": n_with_nav,
        "frames_total": len(merged),
    }

    # -- ASTRA (primary validation — same frames, d = 0) ------------------
    print("  Loading ASTRA picks ...")
    astra = load_astra(flt)
    if astra is not None:
        n_before = len(merged)
        merged = merged.merge(astra, left_on="cbd", right_on="CBD",
                              how="left", suffixes=("", "_astra_dup"))
        for c in merged.columns:
            if c.endswith("_astra_dup"):
                merged.drop(columns=c, inplace=True)
        n_astra = int(merged["h_ice_astra"].notna().sum())
        print(f"    {n_astra}/{n_before} frames matched to ASTRA picks")
        report["astra"] = {
            "frames_matched": n_astra,
            "frames_total": n_before,
        }
    else:
        print("    No ASTRA picks found for this flight")
        report["astra"] = None

    # -- RIGGS (independent seismic/radar, Bentley 1984) -------------------
    print("  Loading RIGGS stations ...")
    riggs_df = load_riggs_stations()
    has_coords = merged["has_nav"] & merged["LAT"].notna() & merged["LON"].notna()
    for col in ("riggs_ice", "riggs_dist", "riggs_stn_idx", "riggs_stn_name",
                "riggs_source", "spri_ice", "spri_dist"):
        merged[col] = np.nan if col not in ("riggs_stn_name", "riggs_source") else ""

    if riggs_df is not None and has_coords.any():
        lats = merged.loc[has_coords, "LAT"].values
        lons = merged.loc[has_coords, "LON"].values
        r_xy = riggs_df[["x_ps", "y_ps"]].values
        r_thk = riggs_df["h_ice"].values
        r_ice, r_dist, r_idx = match_reference(lats, lons, r_xy, r_thk,
                                                radius_m=args.radius)
        merged.loc[has_coords, "riggs_ice"] = r_ice
        merged.loc[has_coords, "riggs_dist"] = r_dist
        merged.loc[has_coords, "riggs_stn_idx"] = r_idx.astype(float)

        # Map station names and sources
        stn_names = np.where(r_idx >= 0,
                             riggs_df["Station"].values[np.clip(r_idx, 0, None)],
                             "")
        stn_sources = np.where(r_idx >= 0,
                               riggs_df["source"].values[np.clip(r_idx, 0, None)],
                               "")
        merged.loc[has_coords, "riggs_stn_name"] = stn_names
        merged.loc[has_coords, "riggs_source"] = stn_sources

        n_riggs = int(np.isfinite(r_ice).sum())
        unique_stns = set(r_idx[r_idx >= 0])
        n_riggs_stns = len(unique_stns)
        n_total_seis = int(riggs_df["source"].eq("seismic").sum())
        n_total_radar = int(riggs_df["source"].eq("radar").sum())
        n_matched_seis = sum(1 for i in unique_stns
                             if riggs_df.iloc[i]["source"] == "seismic")
        n_matched_radar = n_riggs_stns - n_matched_seis
        print(f"    {len(riggs_df)} stations loaded "
              f"({n_total_seis} seismic, {n_total_radar} radar)")
        print(f"    Matches within {args.radius/1000:.0f} km: "
              f"{n_riggs} points \u2192 {n_riggs_stns} stations "
              f"({n_matched_seis} seismic, {n_matched_radar} radar)")
        riggs_stations = []
        for idx in sorted(unique_stns):
            stn = riggs_df.iloc[idx]
            pts_mask = r_idx == idx
            md = np.median(r_dist[pts_mask]) / 1000
            print(f"      {stn['Station']:6s}  {stn['h_ice']:.0f} m "
                  f"({stn['source']})  d\u0303={md:.1f} km")
            riggs_stations.append({
                "station": stn["Station"],
                "h_ice_m": round(float(stn["h_ice"]), 0),
                "source": stn["source"],
                "median_dist_km": round(md, 1),
                "n_matched_frames": int(pts_mask.sum()),
            })
        report["riggs"] = {
            "database": {
                "total": len(riggs_df),
                "seismic": n_total_seis,
                "radar": n_total_radar,
            },
            "matched": {
                "stations": n_riggs_stns,
                "seismic": n_matched_seis,
                "radar": n_matched_radar,
                "total_frame_matches": n_riggs,
            },
            "stations": riggs_stations,
        }
    elif riggs_df is None:
        print("    RIGGS_H_Ice.xlsx not found")
        report["riggs"] = None
    else:
        print("    No coordinates available for RIGGS matching")
        report["riggs"] = None

    # -- BEDMAP1 SPRI (same-source, spatial context only) ------------------
    print("  Loading BEDMAP1 SPRI ...")
    spri_gdf_map = None
    if BEDMAP_SHP.exists() and has_coords.any():
        bedmap_gdf = load_bedmap()
        spri_gdf = bedmap_gdf[bedmap_gdf["MISSION_ID"].isin(SAME_SOURCE_MISSIONS)]
        if len(spri_gdf) > 0:
            lats = merged.loc[has_coords, "LAT"].values
            lons = merged.loc[has_coords, "LON"].values
            s_xy, s_thk = _extract_ref_arrays(spri_gdf)
            s_ice, s_dist, _ = match_reference(lats, lons, s_xy, s_thk,
                                               radius_m=args.radius)
            merged.loc[has_coords, "spri_ice"] = s_ice
            merged.loc[has_coords, "spri_dist"] = s_dist
            n_spri = int(np.isfinite(s_ice).sum())
            print(f"    BEDMAP1 SPRI matches within "
                  f"{args.radius/1000:.0f} km: {n_spri}")

            # Clip SPRI to region near flight for inset map
            flt_x, flt_y = _TRANSFORM.transform(lons, lats)
            margin = 150_000  # 150 km buffer
            spri_gdf_map = spri_gdf[
                (spri_gdf["PS_x"] >= flt_x.min() - margin)
                & (spri_gdf["PS_x"] <= flt_x.max() + margin)
                & (spri_gdf["PS_y"] >= flt_y.min() - margin)
                & (spri_gdf["PS_y"] <= flt_y.max() + margin)
            ]
            print(f"    {len(spri_gdf_map)} BEDMAP1 SPRI points in map region")
            report["bedmap1_spri"] = {
                "frame_matches": n_spri,
                "points_in_map_region": len(spri_gdf_map),
            }
        else:
            print("    No SPRI entries in BEDMAP1")
            report["bedmap1_spri"] = None
    else:
        print("    BEDMAP1 shapefile not found or no coordinates")
        report["bedmap1_spri"] = None

    # Clip RIGGS to map region
    riggs_df_map = None
    if riggs_df is not None and has_coords.any():
        lats = merged.loc[has_coords, "LAT"].values
        lons = merged.loc[has_coords, "LON"].values
        flt_x, flt_y = _TRANSFORM.transform(lons, lats)
        margin = 150_000
        riggs_df_map = riggs_df[
            (riggs_df["x_ps"] >= flt_x.min() - margin)
            & (riggs_df["x_ps"] <= flt_x.max() + margin)
            & (riggs_df["y_ps"] >= flt_y.min() - margin)
            & (riggs_df["y_ps"] <= flt_y.max() + margin)
        ]
        print(f"    {len(riggs_df_map)} RIGGS stations in map region")
        if report.get("riggs") is not None:
            report["riggs"]["stations_in_map_region"] = len(riggs_df_map)

    # -- Generate figure ---------------------------------------------------
    out_path = OUTPUT_BASE / f"F{flt}" / "validation" / f"F{flt}_validation.png"
    print("  Generating validation figure ...")
    make_validation_figure(merged, flt, out_path,
                           nav=nav, riggs_df=riggs_df_map,
                           spri_gdf=spri_gdf_map)

    # -- Comparison statistics ---------------------------------------------
    good = merged[merged["echo_status"] == "good"]

    # ASTRA comparison (primary validation)
    if "h_ice_astra" in merged.columns:
        both_astra = good[good["h_ice_astra"].notna() & good["h_ice_m"].notna()]
        if len(both_astra) > 0:
            diff = both_astra["h_ice_m"].values - both_astra["h_ice_astra"].values
            print(f"\n  LYRA vs ASTRA (n={len(both_astra)} good frames, d=0):")
            print(f"    Mean difference: {np.mean(diff):+.1f} m")
            print(f"    Std difference:  {np.std(diff):.1f} m")
            print(f"    RMSE:            {np.sqrt(np.mean(diff**2)):.1f} m")
            print(f"    Median abs diff: {np.median(np.abs(diff)):.1f} m")
            if len(both_astra) >= 3:
                r = np.corrcoef(both_astra["h_ice_astra"].values,
                                both_astra["h_ice_m"].values)[0, 1]
                print(f"    Correlation r:   {r:.3f}")
            report["astra"]["comparison"] = _comparison_stats(
                both_astra["h_ice_m"].values, both_astra["h_ice_astra"].values)

    # Spatial comparisons
    for ref_col, ref_name, report_key in [
        ("riggs_ice", "RIGGS (Bentley 1984)", "riggs"),
        ("spri_ice", "BEDMAP1 SPRI (same-source)", "bedmap1_spri"),
    ]:
        if ref_col in merged.columns:
            both = good[good[ref_col].notna() & good["h_ice_m"].notna()]
            if len(both) > 0:
                diff = both["h_ice_m"].values - both[ref_col].values
                print(f"\n  LYRA vs {ref_name} (n={len(both)} good frames):")
                print(f"    Mean difference: {np.mean(diff):+.0f} m")
                print(f"    Std difference:  {np.std(diff):.0f} m")
                print(f"    RMSE:            {np.sqrt(np.mean(diff**2)):.0f} m")
                print(f"    Median abs diff: {np.median(np.abs(diff)):.0f} m")
                if report.get(report_key) is not None:
                    report[report_key]["comparison"] = _comparison_stats(
                        both["h_ice_m"].values, both[ref_col].values)

    # -- Write JSON report -------------------------------------------------
    json_path = OUTPUT_BASE / f"F{flt}" / "validation" / f"F{flt}_validation.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report \u2192 {json_path}")

    print("Done.")


if __name__ == "__main__":
    main()
