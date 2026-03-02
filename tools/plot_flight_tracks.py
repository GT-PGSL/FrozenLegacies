"""
Plot all SPRI/TUD/NSF 60 MHz survey flight tracks in NSIDC Polar Stereographic South.

Output: docs/figures/flight_tracks_PS.png
Purpose: Survey flight coverage map to guide selection of next flight to process.

Projection: EPSG:3031 (NSIDC Polar Stereographic South, true scale at 71 S).
Data: Navigation_Files/<FFF>.csv — columns CBD, LAT, LON, THK, SRF.
"""

from __future__ import annotations

import glob
import os
import warnings

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.ticker import FuncFormatter
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
from geodatasets import get_path
from pyproj import Transformer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAV_DIR = os.path.join(ROOT, "Navigation_Files")
OUT_DIR = os.path.join(ROOT, "docs", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# EPSG:3031 transformer from geographic WGS84
# ---------------------------------------------------------------------------
_TRANSFORM = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)


def _to_ps(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project lon/lat (degrees) to NSIDC Polar Stereo South (metres)."""
    return _TRANSFORM.transform(lon, lat)


# ---------------------------------------------------------------------------
# Publication rc
# ---------------------------------------------------------------------------
def _pub_rc() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.linewidth": 0.8,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.spines.bottom": True,
        "axes.spines.left": True,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


# ---------------------------------------------------------------------------
# Load & project all navigation files
# ---------------------------------------------------------------------------
def load_all_flights(nav_dir: str) -> pd.DataFrame:
    """Load every <FFF>.csv in nav_dir and project to EPSG:3031."""
    files = sorted(glob.glob(os.path.join(nav_dir, "*.csv")))
    files = [f for f in files if "AllFlights" not in os.path.basename(f)]

    frames: list[pd.DataFrame] = []
    for fpath in files:
        fn = int(os.path.basename(fpath).replace(".csv", ""))
        df = pd.read_csv(fpath)
        # Normalise longitude to -180..+180
        df["LON"] = ((df["LON"] + 180) % 360) - 180
        # Filter obviously bad values (keep Antarctic latitudes only)
        df = df[(df["LAT"] < -60) & (df["LAT"].notna()) & (df["LON"].notna())]
        if df.empty:
            continue
        x, y = _to_ps(df["LON"].values, df["LAT"].values)
        df = df.copy()
        df["x_m"] = x
        df["y_m"] = y
        df["FlightNumber"] = fn
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Load basemap from Natural Earth (EPSG:3031)
# ---------------------------------------------------------------------------
def load_basemap_ps() -> gpd.GeoDataFrame:
    """Return Antarctic land GeoDataFrame in EPSG:3031."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        land_path = get_path("naturalearth.land")
        land = gpd.read_file(land_path).to_crs("EPSG:3031")
    return land


# ---------------------------------------------------------------------------
# Graticule helpers
# ---------------------------------------------------------------------------
def _graticule_line(lons: np.ndarray, lat: float) -> tuple[np.ndarray, np.ndarray]:
    x, y = _to_ps(lons, np.full_like(lons, lat))
    return x, y


def _meridian_line(lats: np.ndarray, lon: float) -> tuple[np.ndarray, np.ndarray]:
    x, y = _to_ps(np.full_like(lats, lon), lats)
    return x, y


# ---------------------------------------------------------------------------
# Label placement: midpoint of each track's longest segment
# ---------------------------------------------------------------------------
def _label_pos(grp: pd.DataFrame) -> tuple[float, float]:
    mid = len(grp) // 2
    return float(grp["x_m"].iloc[mid]), float(grp["y_m"].iloc[mid])


# ---------------------------------------------------------------------------
# Split a flight's CBD series at large spatial gaps
# ---------------------------------------------------------------------------
def _split_track(grp: pd.DataFrame, gap_km: float = 100.0
                 ) -> list[tuple[np.ndarray, np.ndarray, pd.DataFrame]]:
    """Return list of (x_arr, y_arr, sub_df) for each continuous segment."""
    grp = grp.sort_values("CBD")
    x_vals = grp["x_m"].values
    y_vals = grp["y_m"].values
    if len(x_vals) < 2:
        return [(x_vals, y_vals, grp)]
    dist = np.sqrt(np.diff(x_vals)**2 + np.diff(y_vals)**2)
    brk = list(np.where(dist > gap_km * 1_000)[0] + 1)
    idx = [0] + brk + [len(grp)]
    return [
        (x_vals[idx[i]:idx[i+1]], y_vals[idx[i]:idx[i+1]], grp.iloc[idx[i]:idx[i+1]])
        for i in range(len(idx) - 1)
    ]


# ---------------------------------------------------------------------------
# Geographic annotation helper
# ---------------------------------------------------------------------------
def _geo_label(ax: plt.Axes, lon: float, lat: float, text: str,
               fontsize: float = 7, color: str = "#555555",
               ha: str = "center", va: str = "center",
               x_off: float = 0, y_off: float = 0) -> None:
    """Place a geographic label at (lon, lat)."""
    x, y = _to_ps(np.array([lon]), np.array([lat]))
    ax.text(x[0] + x_off, y[0] + y_off, text,
            fontsize=fontsize, color=color, ha=ha, va=va,
            fontstyle="italic", zorder=7,
            path_effects=[pe.withStroke(linewidth=1.5, foreground="white")])


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------
def make_flight_track_map(df: pd.DataFrame, land: gpd.GeoDataFrame) -> plt.Figure:
    _pub_rc()

    fig, ax = plt.subplots(figsize=(10, 9))

    # --- Basemap -----------------------------------------------------------
    land.plot(ax=ax, color="#e4eef5", edgecolor="#7aadbb", linewidth=0.6, zorder=1)

    # --- Map extent --------------------------------------------------------
    x_lim = (-1_600_000, 1_100_000)
    y_lim = (-1_900_000, 100_000)
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)

    # --- Graticules --------------------------------------------------------
    lon_arr  = np.linspace(-180, 180, 721)
    lat_arr  = np.linspace(-90, -60, 200)
    lat_lines = [-85, -80, -75, -70, -65]
    lon_lines = list(range(-180, 181, 30))

    for lat in lat_lines:
        lx, ly = _graticule_line(lon_arr, lat)
        ax.plot(lx, ly, color="#cccccc", linewidth=0.4, zorder=0, linestyle="--")
        # Label at west side
        lx0, ly0 = _to_ps(np.array([-175.0]), np.array([lat]))
        if x_lim[0] < lx0[0] < x_lim[1] and y_lim[0] < ly0[0] < y_lim[1]:
            ax.text(lx0[0], ly0[0], f"{abs(lat)}°S",
                    fontsize=5.5, color="#aaaaaa", ha="left", va="center")

    for lon in lon_lines:
        lx, ly = _meridian_line(lat_arr, lon)
        ax.plot(lx, ly, color="#cccccc", linewidth=0.4, zorder=0, linestyle=":")
        lx0, ly0 = _to_ps(np.array([float(lon)]), np.array([-64.5]))
        if x_lim[0] < lx0[0] < x_lim[1] and y_lim[0] < ly0[0] < y_lim[1]:
            if lon == 0:
                lbl = "0°"
            elif lon == 180 or lon == -180:
                lbl = "180°"
            elif lon > 0:
                lbl = f"{lon}°E"
            else:
                lbl = f"{abs(lon)}°W"
            ax.text(lx0[0], ly0[0], lbl, fontsize=5.5, color="#aaaaaa",
                    ha="center", va="bottom")

    # --- Flight tracks colour scheme ---------------------------------------
    # early flights: 001–040 (dark blue shades)
    # main survey:   101–149 (teal→yellow via viridis)
    # F125:          vivid red
    flight_nums = sorted(df["FlightNumber"].unique())
    early_list = [f for f in flight_nums if f < 100]
    main_list  = [f for f in flight_nums if 100 <= f <= 149 and f != 125]

    n_e = len(early_list)
    n_m = len(main_list)
    early_pal = {f: plt.cm.Blues(0.35 + 0.5 * i / max(n_e - 1, 1))
                 for i, f in enumerate(early_list)}
    main_pal  = {f: plt.cm.viridis(0.10 + 0.78 * i / max(n_m - 1, 1))
                 for i, f in enumerate(main_list)}

    colour_of = {**early_pal, **main_pal, 125: "#d62728"}

    label_dy = 28_000

    for fn in flight_nums:
        grp = df[df["FlightNumber"] == fn]
        segs = _split_track(grp)

        colour = colour_of.get(fn, "#888888")
        if fn == 125:
            lw, zo, alpha = 2.2, 5, 1.0
        elif fn < 100:
            lw, zo, alpha = 0.70, 2, 0.60
        else:
            lw, zo, alpha = 0.85, 3, 0.72

        for (sx, sy, _sub) in segs:
            if len(sx) < 2:
                continue
            ax.plot(sx, sy, color=colour, linewidth=lw, alpha=alpha, zorder=zo,
                    solid_capstyle="round")

        # Label at midpoint of longest segment
        if segs:
            sx_l, sy_l, sub_l = max(segs, key=lambda t: len(t[0]))
            if len(sub_l) >= 2:
                xm, ym = _label_pos(sub_l)
                if x_lim[0] < xm < x_lim[1] and y_lim[0] < ym < y_lim[1]:
                    ax.text(xm, ym + label_dy, str(fn),
                            fontsize=7 if fn == 125 else 5,
                            fontweight="bold" if fn == 125 else "normal",
                            color=colour, ha="center", va="bottom", zorder=zo + 1,
                            path_effects=[pe.withStroke(
                                linewidth=2.0 if fn == 125 else 1.5,
                                foreground="white")])

    # --- Geographic labels -------------------------------------------------
    _geo_label(ax, -155,  -79.5, "Ross Ice Shelf",    fontsize=8.5,
               color="#2255aa", va="center")
    _geo_label(ax, -130,  -80.5, "Siple Coast",       fontsize=7.5,
               color="#225522")
    _geo_label(ax, -100,  -80.5, "West Antarctica",   fontsize=7.0,
               color="#444444")
    _geo_label(ax,  170,  -78.5, "Ross Sea",          fontsize=7.0,
               color="#557799", x_off=-50_000)
    _geo_label(ax, -168,  -83.5, "MBL",               fontsize=6.5,
               color="#777777")
    _geo_label(ax,  150,  -72.5, "Victoria Land",     fontsize=7.0,
               color="#444444")

    # --- Legend ------------------------------------------------------------
    lh = [
        mlines.Line2D([], [], color="#d62728", linewidth=2.0,
                      label="F125 (processed)"),
        mlines.Line2D([], [], color=plt.cm.viridis(0.5), linewidth=1.0,
                      label="Main survey (101–149)"),
        mlines.Line2D([], [], color=plt.cm.Blues(0.55), linewidth=0.8,
                      label="Early season (001–040)", linestyle="-"),
    ]
    ax.legend(handles=lh, loc="upper right", fontsize=7,
              framealpha=0.9, edgecolor="#cccccc", borderpad=0.7)

    # --- Info box ----------------------------------------------------------
    n_f = df["FlightNumber"].nunique()
    n_c = len(df)
    info = (
        f"SPRI/TUD/NSF 1974–75  ·  60 MHz\n"
        f"{n_f} flights  |  {n_c:,} CBDs\n"
        f"EPSG:3031 (NSIDC PS South)"
    )
    ax.text(0.015, 0.015, info, transform=ax.transAxes, fontsize=6,
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="#aaaaaa", alpha=0.88))

    # --- Scale bar (500 km) ------------------------------------------------
    sb_x0 = x_lim[0] + 0.05 * (x_lim[1] - x_lim[0])
    sb_y0 = y_lim[0] + 0.05 * (y_lim[1] - y_lim[0])
    sb_len = 500_000
    ax.plot([sb_x0, sb_x0 + sb_len], [sb_y0, sb_y0],
            color="k", linewidth=2.0, solid_capstyle="butt", zorder=6)
    for xv in [sb_x0, sb_x0 + sb_len]:
        ax.plot([xv, xv], [sb_y0 - 15_000, sb_y0 + 15_000],
                color="k", linewidth=1.2, zorder=6)
    ax.text(sb_x0 + sb_len / 2, sb_y0 + 25_000, "500 km",
            fontsize=6.5, ha="center", va="bottom")

    # --- Axes --------------------------------------------------------------
    ax.set_aspect("equal")
    ax.set_xlabel("Polar Stereo Easting (km)", fontsize=8)
    ax.set_ylabel("Polar Stereo Northing (km)", fontsize=8)

    def _m2km(x: float, _pos=None) -> str:
        return f"{x/1e3:.0f}"

    ax.xaxis.set_major_formatter(FuncFormatter(_m2km))
    ax.yaxis.set_major_formatter(FuncFormatter(_m2km))
    ax.xaxis.set_major_locator(plt.MultipleLocator(500_000))
    ax.yaxis.set_major_locator(plt.MultipleLocator(500_000))
    ax.tick_params(labelsize=7)

    ax.set_title("SPRI/TUD/NSF 1974–75 Antarctic Survey — Flight Track Coverage",
                 fontsize=10.5, fontweight="bold", pad=10)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading navigation files …")
    df = load_all_flights(NAV_DIR)
    print(f"  {len(df):,} CBDs across {df['FlightNumber'].nunique()} flights")
    print(f"  Flights: {sorted(df['FlightNumber'].unique())}")

    print("Loading basemap …")
    land = load_basemap_ps()

    print("Generating figure …")
    fig = make_flight_track_map(df, land)

    out_path = os.path.join(OUT_DIR, "flight_tracks_PS.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close(fig)
