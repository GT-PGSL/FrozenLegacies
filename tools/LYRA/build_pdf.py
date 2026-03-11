#!/usr/bin/env python3
"""
LYRA A-scope Data Product PDF generator.

Produces a CReSIS OPR-style PDF for each TIFF, with:
  - Page 1: Cover map (flight track, TIFF segment, start point)
  - Pages 2+: One per complete frame (location map + calibrated A-scope)

Usage:
    python tools/LYRA/build_pdf.py <tiff>
    python tools/LYRA/build_pdf.py 128/0          # shorthand

Requires phases 1-4 complete for the TIFF.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

# -- Resolve project root and add tools/ to path ------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "tools" / "LYRA"))
sys.path.insert(0, str(ROOT / "tools"))

from lyra import (tiff_id, resolve_tiff_arg, load_stanford_nav,
                   get_nav_positions, load_alignment)
from plot_flight_tracks import (load_basemap_ps, _to_ps, _graticule_line,
                                 _meridian_line)

# -- Parse args ---------------------------------------------------------------
if len(sys.argv) < 2:
    sys.exit("Usage: python build_pdf.py <tiff_or_shorthand>")

TIFF = resolve_tiff_arg(sys.argv[1], ROOT)
FLT  = int(TIFF.parent.name)
TID  = tiff_id(TIFF)

OUT_DIR    = ROOT / f"tools/LYRA/output/F{FLT}"
PHASE1_DIR = OUT_DIR / "phase1"
PDF_DIR    = OUT_DIR / "pdf"
PDF_DIR.mkdir(parents=True, exist_ok=True)

# -- Load data ----------------------------------------------------------------
frame_idx_df = pd.read_csv(PHASE1_DIR / f"F{FLT}_frame_index.csv")
cal_df       = pd.read_csv(OUT_DIR / "phase3" / f"F{FLT}_cal.csv")
echoes_df    = pd.read_csv(OUT_DIR / "phase4" / f"F{FLT}_echoes.csv")

# Cal picks JSON (for X-grid phase / D_anchor)
cal_picks_path = PHASE1_DIR / f"F{FLT}_cal_picks.json"
cal_picks = json.load(open(cal_picks_path)) if cal_picks_path.exists() else {}

# Filter to this TIFF
tiff_name = TIFF.name
fi_tiff   = frame_idx_df[frame_idx_df["tiff"] == tiff_name].copy()
cal_tiff  = cal_df[cal_df["tiff"] == tiff_name].copy()
echo_tiff = echoes_df[echoes_df["tiff"] == tiff_name].copy()

# Navigation
nav = load_stanford_nav(FLT)
alignment = load_alignment(FLT, OUT_DIR)
nav_offset = alignment.offset if alignment else 0

# Full flight track for map background
all_cbds = echoes_df["cbd"].dropna().values.astype(int)
flight_lats, flight_lons = get_nav_positions(all_cbds, FLT, offset=nav_offset)
flight_x, flight_y = _to_ps(flight_lons, flight_lats)

# This TIFF's CBDs
tiff_lats = echo_tiff["lat"].values
tiff_lons = echo_tiff["lon"].values
tiff_x, tiff_y = _to_ps(tiff_lons, tiff_lats)

# Load basemap (land + ice shelves)
basemap = load_basemap_ps()
ice_shelf_path = ROOT / "Data" / "ne_10m_antarctic_ice_shelves_polys.shp"
ice_shelves = None
if ice_shelf_path.exists():
    ice_shelves = gpd.read_file(ice_shelf_path).to_crs(epsg=3031)

# Load TIFF image
Image.MAX_IMAGE_PIXELS = None
img_raw = np.array(Image.open(TIFF), dtype=np.float32)
img_norm = (img_raw - img_raw.min()) / (img_raw.max() - img_raw.min() + 1e-9)
IMG_H, IMG_W = img_norm.shape

# Vertical crop: remove sprocket holes but keep CBD number text (~row 2000-2100)
# Top: film border ends ~row 140, CRT display starts ~150
# Bottom: CRT display ends ~row 2160, sprocket holes start ~row 2200
CRT_TOP    = 150
CRT_BOTTOM = 2170
CRT_H      = CRT_BOTTOM - CRT_TOP

# Compute D_anchor (X-grid phase) from cal_picks
# D_anchor = distance from mb_x to the first graticule line to its right
tiff_d_anchor = None
for fkey, picks in cal_picks.items():
    if "mb" in picks and "x_grid" in picks and len(picks["x_grid"]) >= 1:
        tiff_d_anchor = float(min(picks["x_grid"])) - float(picks["mb"])
        break

print(f"\nLYRA PDF -- F{FLT} TIFF {TID}")
print(f"  Frames: {len(fi_tiff)} total, {len(echo_tiff)} with echoes")
print(f"  D_anchor: {tiff_d_anchor:.1f} px" if tiff_d_anchor else "  D_anchor: (none)")
if ice_shelves is not None:
    print(f"  Ice shelves: {len(ice_shelves)} polygons loaded")

# -- Style constants ----------------------------------------------------------
Y_REF_DB     = -60.0   # dB at y_ref_px (noise floor reference)
CLR_SURFACE  = "#2166ac"
CLR_BED      = "#b2182b"
CLR_MB       = "#4daf4a"
CLR_GRID     = "#666666"
CLR_TRACK    = "#aaaaaa"
CLR_TIFF_SEG = "#4393c3"
CLR_FRAME    = "#d6604d"
CLR_SHELF    = "#d4e6f1"    # light blue for ice shelves
CLR_LAND     = "#e8e8e8"
CLR_OCEAN    = "#f7fbff"    # very faint blue for ocean
STATUS_CLR   = {"good": "#1b9e77", "weak_bed": "#d95f02",
                "no_bed": "#d62728", "no_surface": "#7570b3"}
FONT_LABEL   = 8
FONT_TITLE   = 10
FONT_SMALL   = 7

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": FONT_LABEL,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
})


# -- Helper: draw polar stereographic map ------------------------------------
def draw_map(ax, center_x, center_y, radius_m=100_000,
             show_flight=True, show_tiff=True, highlight_xy=None):
    """Draw a polar stereographic map centered on (center_x, center_y)."""
    ax.set_facecolor(CLR_OCEAN)
    ax.set_xlim(center_x - radius_m, center_x + radius_m)
    ax.set_ylim(center_y - radius_m, center_y + radius_m)

    # Ice shelves (below land)
    if ice_shelves is not None:
        ice_shelves.plot(ax=ax, color=CLR_SHELF, edgecolor="#b0c4de",
                         linewidth=0.3, zorder=1)

    # Land basemap
    basemap.plot(ax=ax, color=CLR_LAND, edgecolor="#bbbbbb", linewidth=0.3,
                 zorder=2)

    # Graticule
    for lat in range(-90, -60, 2):
        gx, gy = _graticule_line(np.linspace(0, 360, 500), lat)
        ax.plot(gx, gy, color="#cccccc", lw=0.25, zorder=3)
    for lon in range(0, 360, 15):
        mx, my = _meridian_line(np.linspace(-90, -60, 200), lon)
        ax.plot(mx, my, color="#cccccc", lw=0.25, zorder=3)

    # Full flight track
    if show_flight:
        valid = np.isfinite(flight_x) & np.isfinite(flight_y)
        ax.plot(flight_x[valid], flight_y[valid], color=CLR_TRACK,
                lw=0.8, zorder=4, alpha=0.6)

    # This TIFF's segment
    if show_tiff:
        valid_t = np.isfinite(tiff_x) & np.isfinite(tiff_y)
        ax.plot(tiff_x[valid_t], tiff_y[valid_t], color=CLR_TIFF_SEG,
                lw=2.0, zorder=5)

    # Highlight point (current frame)
    if highlight_xy is not None:
        hx, hy = highlight_xy
        ax.plot(hx, hy, "o", color=CLR_FRAME, ms=6, zorder=7,
                markeredgecolor="white", markeredgewidth=0.6)

    ax.set_aspect("equal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Easting (km)", fontsize=FONT_SMALL)
    ax.set_ylabel("Northing (km)", fontsize=FONT_SMALL)
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v/1e3:.0f}"))
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v/1e3:.0f}"))
    ax.tick_params(labelsize=FONT_SMALL)


# -- Helper: draw calibrated A-scope frame -----------------------------------
def draw_ascope(ax, frame_row, cal_row, echo_row):
    """Draw calibrated A-scope CRT image with grid overlays and echo markers."""
    left_px  = int(frame_row["left_px"])
    right_px = int(frame_row["right_px"])
    mb_x     = int(float(cal_row["mb_x"]))
    y_ref_px = float(cal_row["y_ref_px"])
    db_px    = float(cal_row["db_per_px"])
    us_px    = float(cal_row["us_per_px"])
    x_sp     = float(cal_row["x_spacing_px"])
    hgrid_sp = float(cal_row["hgrid_spacing"])

    frame_w  = right_px - left_px

    # -- Horizontal trim: center the CRT signal region --
    # Signal extends from MB to approximately MB + 17 us (max sweep)
    signal_end_px = mb_x + int(17.0 / us_px)
    signal_end_px = min(signal_end_px, frame_w)
    signal_span   = signal_end_px - mb_x

    # Add padding: 15% of signal span on left of MB, 10% on right
    pad_left  = max(50, int(signal_span * 0.15))
    pad_right = max(50, int(signal_span * 0.10))
    trim_left  = max(0, mb_x - pad_left)
    trim_right = min(frame_w, signal_end_px + pad_right)

    # Extract cropped frame (vertical: CRT region only)
    frame_crop = img_norm[CRT_TOP:CRT_BOTTOM,
                          left_px + trim_left : left_px + trim_right]

    # Display image
    ax.imshow(frame_crop, cmap="gray", aspect="auto", origin="upper",
              extent=[trim_left, trim_right, CRT_BOTTOM, CRT_TOP])

    # -- X grid lines: aligned to oscilloscope graticule --
    # Use D_anchor to find true graticule phase
    if tiff_d_anchor is not None:
        # First grid line right of MB
        first_xg = mb_x + tiff_d_anchor
        # Extrapolate in both directions
        xg = first_xg
        while xg > trim_left:
            xg -= x_sp
        xg += x_sp  # first visible grid line
        while xg < trim_right:
            if xg >= trim_left:
                ax.axvline(xg, color=CLR_GRID, lw=0.5, ls=":", alpha=0.45)
            xg += x_sp
    else:
        # Fallback: step from MB
        xg = mb_x
        while xg < trim_right:
            if xg >= trim_left:
                ax.axvline(xg, color=CLR_GRID, lw=0.5, ls=":", alpha=0.45)
            xg += x_sp

    # -- Y grid lines: anchored to y_ref_px, step by hgrid_spacing --
    yg = y_ref_px
    while yg > CRT_TOP:
        yg -= hgrid_sp
    while yg < CRT_BOTTOM:
        if yg >= CRT_TOP:
            ax.axhline(yg, color=CLR_GRID, lw=0.5, ls=":", alpha=0.45)
        yg += hgrid_sp

    # Label style: white background box so text is readable over the image/line
    _label_box = dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor="none", alpha=0.85)

    # -- Main bang marker --
    if trim_left <= mb_x <= trim_right:
        ax.axvline(mb_x, color=CLR_MB, lw=0.8, ls="--", alpha=0.6)
        ax.text(mb_x, CRT_TOP + 5, " MB ", fontsize=8, color=CLR_MB,
                fontweight="bold", ha="center", va="top", bbox=_label_box)

    # -- Echo markers --
    status = str(echo_row["echo_status"]) if echo_row is not None else "no_data"

    if echo_row is not None:
        # Surface
        srf_twt = echo_row.get("surface_twt_us")
        if pd.notna(srf_twt):
            srf_x = mb_x + float(srf_twt) / us_px
            if trim_left <= srf_x <= trim_right:
                ax.axvline(srf_x, color=CLR_SURFACE, lw=1.2, ls="-", alpha=0.7)
                ax.text(srf_x, CRT_TOP + 5, " S ", fontsize=9,
                        color=CLR_SURFACE, fontweight="bold",
                        ha="center", va="top", bbox=_label_box)

        # Bed
        bed_twt = echo_row.get("bed_twt_us")
        if pd.notna(bed_twt):
            bed_x = mb_x + float(bed_twt) / us_px
            if trim_left <= bed_x <= trim_right:
                ax.axvline(bed_x, color=CLR_BED, lw=1.2, ls="-", alpha=0.7)
                ax.text(bed_x, CRT_TOP + 5, " B ", fontsize=9,
                        color=CLR_BED, fontweight="bold",
                        ha="center", va="top", bbox=_label_box)

    # -- Axes --
    ax.set_xlim(trim_left, trim_right)
    ax.set_ylim(CRT_BOTTOM, CRT_TOP)

    # Primary X: column px
    ax.set_xlabel("Column (px, frame-relative)", fontsize=FONT_SMALL)
    ax.tick_params(labelsize=FONT_SMALL)

    # Secondary X: travel time from MB
    ax2_x = ax.twiny()
    ax2_x.set_xlim((trim_left - mb_x) * us_px, (trim_right - mb_x) * us_px)
    ax2_x.set_xlabel("Two-way travel time from MB (us)", fontsize=FONT_SMALL)
    ax2_x.tick_params(labelsize=FONT_SMALL)
    ax2_x.spines["top"].set_linewidth(0.4)

    # Primary Y: row px
    ax.set_ylabel("Row (px)", fontsize=FONT_SMALL)

    # Secondary Y: power dB
    ax2_y = ax.twinx()
    db_top    = Y_REF_DB + (y_ref_px - CRT_TOP) * db_px
    db_bottom = Y_REF_DB + (y_ref_px - CRT_BOTTOM) * db_px
    ax2_y.set_ylim(db_bottom, db_top)
    ax2_y.set_ylabel("Received power (dB)", fontsize=FONT_SMALL)
    ax2_y.tick_params(labelsize=FONT_SMALL)
    ax2_y.spines["right"].set_linewidth(0.4)

    # Spine visibility
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # -- Quality badge --
    badge_clr = STATUS_CLR.get(status, "#999999")
    badge_txt = status.replace("_", " ")
    ax.text(0.98, 0.97, badge_txt, transform=ax.transAxes,
            fontsize=FONT_SMALL, fontweight="bold", color="white",
            ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=badge_clr,
                      edgecolor="none", alpha=0.85))

    return status


# -- Build PDF ----------------------------------------------------------------
pdf_path = PDF_DIR / f"F{FLT}_{TID}_ascope.pdf"
print(f"  Output: {pdf_path.relative_to(ROOT)}")

with PdfPages(str(pdf_path)) as pdf:
    # ========== PAGE 1: COVER MAP ============================================
    fig_cover = plt.figure(figsize=(11, 8.5))

    fig_cover.suptitle(
        f"SPRI/TUD/NSF 60 MHz Airborne Radar  --  Flight F{FLT}, TIFF {TID}",
        fontsize=12, fontweight="bold", y=0.97)

    # Right panel: zoomed map centered on this TIFF's segment
    ax_main = fig_cover.add_axes([0.35, 0.08, 0.60, 0.82])
    valid_t = np.isfinite(tiff_x) & np.isfinite(tiff_y)
    if valid_t.any():
        cx = np.nanmean(tiff_x[valid_t])
        cy = np.nanmean(tiff_y[valid_t])
        dx = np.nanmax(tiff_x[valid_t]) - np.nanmin(tiff_x[valid_t])
        dy = np.nanmax(tiff_y[valid_t]) - np.nanmin(tiff_y[valid_t])
        radius = max(dx, dy) * 0.7 + 50_000
    else:
        cx, cy, radius = 0, 0, 500_000

    draw_map(ax_main, cx, cy, radius_m=radius)
    # Start marker
    if valid_t.any():
        i0 = np.where(valid_t)[0][0]
        ax_main.plot(tiff_x[i0], tiff_y[i0], "s", color="#1b9e77", ms=8,
                     zorder=6, markeredgecolor="white", markeredgewidth=0.8,
                     label="Start")
        ax_main.legend(loc="lower right", fontsize=FONT_SMALL, framealpha=0.9)

    # Left panel: Antarctica inset
    ax_inset = fig_cover.add_axes([0.04, 0.45, 0.28, 0.45])
    ax_inset.set_facecolor(CLR_OCEAN)
    if ice_shelves is not None:
        ice_shelves.plot(ax=ax_inset, color=CLR_SHELF, edgecolor="none",
                         linewidth=0)
    basemap.plot(ax=ax_inset, color=CLR_LAND, edgecolor="#aaaaaa", linewidth=0.3)
    valid_f = np.isfinite(flight_x) & np.isfinite(flight_y)
    ax_inset.plot(flight_x[valid_f], flight_y[valid_f], color=CLR_TRACK,
                  lw=0.5, alpha=0.5)
    ax_inset.plot(tiff_x[valid_t], tiff_y[valid_t], color=CLR_TIFF_SEG, lw=1.5)
    ax_inset.set_xlim(-3e6, 3e6)
    ax_inset.set_ylim(-3e6, 3e6)
    ax_inset.set_aspect("equal")
    ax_inset.set_xticks([])
    ax_inset.set_yticks([])
    ax_inset.set_title("Antarctica", fontsize=FONT_SMALL)
    for sp in ax_inset.spines.values():
        sp.set_linewidth(0.4)

    # Metadata text block
    n_complete = len(fi_tiff[fi_tiff["frame_type"] == "complete"])
    n_good     = len(echo_tiff[echo_tiff["echo_status"] == "good"])
    cbd_min    = echo_tiff["cbd"].min() if len(echo_tiff) else "?"
    cbd_max    = echo_tiff["cbd"].max() if len(echo_tiff) else "?"

    meta_lines = [
        f"Flight: F{FLT}",
        f"TIFF: {tiff_name}",
        f"TIFF ID: {TID}",
        f"CBD range: {cbd_min} -- {cbd_max}",
        f"Frames: {n_complete} complete, {n_good} good",
        f"Projection: EPSG:3031 (NSIDC Polar Stereo.)",
        f"Radar: 60 MHz, 1 kW, 125 ns pulse",
        f"",
        f"LYRA A-scope Data Product",
        f"FrozenLegacies Project",
    ]
    fig_cover.text(0.04, 0.35, "\n".join(meta_lines),
                   fontsize=FONT_LABEL, verticalalignment="top",
                   fontfamily="sans-serif", linespacing=1.6)

    pdf.savefig(fig_cover, dpi=150)
    plt.close(fig_cover)
    print(f"  Cover page done")

    # ========== PAGES 2+: ONE PER COMPLETE FRAME =============================
    complete_fi = fi_tiff[fi_tiff["frame_type"] == "complete"].copy()
    n_pages = len(complete_fi)

    for page_i, (_, fi_row) in enumerate(complete_fi.iterrows()):
        fidx = int(fi_row["frame_idx"])
        cbd  = fi_row["cbd"]
        cbd_str = f"{int(float(cbd)):04d}" if pd.notna(cbd) else "????"

        # Match cal and echo rows
        cal_match = cal_tiff[cal_tiff["frame_idx"] == fidx]
        if len(cal_match) == 0:
            continue
        cal_row = cal_match.iloc[0]

        # Skip excluded frames
        if pd.notna(cal_row.get("exclude_reason")) and str(cal_row.get("exclude_reason")).strip():
            continue

        echo_match = echo_tiff[echo_tiff["frame_idx"] == fidx]
        echo_row = echo_match.iloc[0] if len(echo_match) else None

        # Get position
        lat = float(echo_row["lat"]) if echo_row is not None and pd.notna(echo_row.get("lat")) else None
        lon = float(echo_row["lon"]) if echo_row is not None and pd.notna(echo_row.get("lon")) else None

        # Create figure
        fig = plt.figure(figsize=(11, 8.5))

        status_str = str(echo_row["echo_status"]) if echo_row is not None else "no data"
        fig.suptitle(
            f"F{FLT} / TIFF {TID} / CBD {cbd_str}   ({page_i+1}/{n_pages})",
            fontsize=FONT_TITLE, fontweight="bold", y=0.97)

        # -- Left panel: location map (30% width) --
        ax_map = fig.add_axes([0.04, 0.12, 0.26, 0.78])

        if lat is not None and lon is not None:
            fx, fy = _to_ps(np.array([lon]), np.array([lat]))
            draw_map(ax_map, fx[0], fy[0], radius_m=80_000,
                     highlight_xy=(fx[0], fy[0]))
        else:
            ax_map.text(0.5, 0.5, "No position", transform=ax_map.transAxes,
                        ha="center", va="center", fontsize=FONT_LABEL,
                        color="#999999")
            ax_map.set_xticks([])
            ax_map.set_yticks([])

        # Frame metadata as title (above map)
        meta = f"CBD {cbd_str}"
        if echo_row is not None:
            h_air = echo_row.get("h_air_m")
            h_ice = echo_row.get("h_ice_m")
            if pd.notna(h_air):
                meta += f"  |  h_air {h_air:.0f} m"
            if pd.notna(h_ice):
                meta += f"  |  h_ice {h_ice:.0f} m"
        ax_map.set_title(meta, fontsize=FONT_SMALL, pad=6)

        # Lat/lon below map (inside axes area to avoid overlap)
        if lat is not None and lon is not None:
            coord_str = (f"{abs(lat):.3f}{'S' if lat < 0 else 'N'}, "
                         f"{abs(lon):.3f}{'E' if lon >= 0 else 'W'}")
            ax_map.text(0.5, 0.02, coord_str, transform=ax_map.transAxes,
                        fontsize=6, ha="center", va="bottom", color="#555555",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                  edgecolor="none", alpha=0.8))

        # -- Right panel: calibrated A-scope (62% width) --
        ax_asc = fig.add_axes([0.38, 0.12, 0.58, 0.76])
        draw_ascope(ax_asc, fi_row, cal_row, echo_row)

        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        if (page_i + 1) % 50 == 0 or page_i == 0:
            print(f"    Page {page_i+1}/{n_pages}: CBD {cbd_str} [{status_str}]")

    print(f"  {n_pages} frame pages written")

print(f"\nDone. PDF: {pdf_path}")
