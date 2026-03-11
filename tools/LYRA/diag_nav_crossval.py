#!/usr/bin/env python3
"""
Diagnostic: cross-validate navigation sources for a given flight.

Compares Stanford nav, Bingham nav (DBF + corrected TXT), BEDMAP1 SPRI_7475,
RIGGS stations, and LYRA echoes to assess CBD-to-position mapping quality.

Usage:
    python tools/LYRA/diag_nav_crossval.py 137
    python tools/LYRA/diag_nav_crossval.py 137 --match-radius 30
"""
import sys, os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from pathlib import Path
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]

# Import basemap helpers from plot_flight_tracks (EPSG:3031)
sys.path.insert(0, str(ROOT / "tools"))
from plot_flight_tracks import (load_basemap_ps, _to_ps,
                                _graticule_line, _meridian_line, _geo_label)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def dms_to_dd(s):
    """Convert DMS string like '82 32'19"S' or '166 00'48"W' to decimal degrees."""
    s = s.strip().strip('"').strip("'")
    m = re.match(r"""(\d+)\D+(\d+)\D+(\d+)[\"']*\s*([NSEW])""", s)
    if not m:
        return float('nan')
    deg, mn, sec, hem = int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4)
    dd = deg + mn / 60.0 + sec / 3600.0
    if hem in ('S', 'W'):
        dd = -dd
    return dd


def haversine_km(lat1, lon1, lat2, lon2):
    """Haversine distance in km between two points (scalar or array)."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def polar_stereo(lat, lon):
    """Convert lat/lon to EPSG:3031 polar stereographic (x, y) in meters."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    return _to_ps(lon, lat)   # _to_ps takes (lon, lat)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_stanford(flt):
    """Load Stanford nav CSV. Returns DataFrame with CBD, LAT, LON, THK, SRF."""
    p = ROOT / "Navigation_Files" / "Stanford" / f"{flt}.csv"
    if not p.exists():
        p = ROOT / "Navigation_Files" / f"{flt}.csv"
    if not p.exists():
        print(f"  [!] Stanford nav not found for F{flt}")
        return None
    df = pd.read_csv(p)
    df.rename(columns={"CBD": "cbd_stanford"}, inplace=True)
    return df


def load_bingham_dbf(flt):
    """Load Bingham inherited DBF. Returns DataFrame."""
    try:
        import dbfread
    except ImportError:
        print("  [!] dbfread not installed; skipping Bingham DBF")
        return None
    p = ROOT / "Navigation_Files" / "Bingham" / "001_nav_data_rgb_inherited" / f"{flt}_nocbd.dbf"
    if not p.exists():
        print(f"  [!] Bingham DBF not found for F{flt}")
        return None
    db = dbfread.DBF(str(p))
    records = [dict(r) for r in db]
    df = pd.DataFrame(records)
    return df


def load_bingham_txt(flt):
    """Load Bingham corrected TXT with PRED_CBD. Returns DataFrame."""
    p = ROOT / "Navigation_Files" / "Bingham" / "002_nav_data_rgb_corrected" / f"sprinsftud_1974_{flt}_nav.txt"
    if not p.exists():
        print(f"  [!] Bingham TXT not found for F{flt}")
        return None
    df = pd.read_csv(p, sep="\t")
    return df


def load_riggs():
    """Load RIGGS stations. Returns DataFrame with lat/lon in decimal degrees."""
    p = ROOT / "Data" / "RIGGS" / "riggs_stations.csv"
    if not p.exists():
        print("  [!] RIGGS stations file not found")
        return None
    df = pd.read_csv(p)
    # Parse DMS coordinates
    lats, lons = [], []
    for _, row in df.iterrows():
        lats.append(dms_to_dd(str(row["Latitude"])))
        lons.append(dms_to_dd(str(row["Longitude"])))
    df["lat_dd"] = lats
    df["lon_dd"] = lons
    return df


def load_bedmap1_spri():
    """Load BEDMAP1 SPRI_7475 points. Returns DataFrame."""
    try:
        import geopandas as gpd
    except ImportError:
        print("  [!] geopandas not installed; skipping BEDMAP1")
        return None
    p = ROOT / "Data" / "BEDMAP" / "BedMap1" / "bedmap1_clip.shp"
    if not p.exists():
        print("  [!] BEDMAP1 shapefile not found")
        return None
    gdf = gpd.read_file(p)
    spri = gdf[gdf["MISSION_ID"] == "SPRI_7475"].copy()
    return spri


def load_lyra_echoes(flt):
    """Load LYRA phase4 echoes CSV."""
    p = ROOT / "tools" / "LYRA" / "output" / f"F{flt}" / "phase4" / f"F{flt}_echoes.csv"
    if not p.exists():
        print(f"  [!] LYRA echoes not found for F{flt}")
        return None
    return pd.read_csv(p)


# ---------------------------------------------------------------------------
# Cross-comparisons
# ---------------------------------------------------------------------------

def compare_stanford_vs_bingham(stanford, bingham_dbf, bingham_txt, flt):
    """Compare Stanford and Bingham nav tracks spatially."""
    print("\n" + "=" * 70)
    print("  1. STANFORD vs BINGHAM NAV TRACKS")
    print("=" * 70)

    if stanford is None:
        print("  Stanford nav not available.")
        return

    st_lat = stanford["LAT"].values
    st_lon = stanford["LON"].values
    print(f"  Stanford: {len(stanford)} points, CBD {stanford['cbd_stanford'].min()}-{stanford['cbd_stanford'].max()}")

    # --- DBF comparison ---
    if bingham_dbf is not None:
        db_lat = bingham_dbf["LATITUDE"].values
        db_lon = bingham_dbf["LONGITUDE"].values
        print(f"  Bingham DBF: {len(bingham_dbf)} points, NUMBER {bingham_dbf['NUMBER'].min()}-{bingham_dbf['NUMBER'].max()}")

        # Match by NUMBER = Stanford row index?
        # Build KDTree on Stanford and find nearest Bingham DBF point
        st_xy = np.column_stack(polar_stereo(st_lat, st_lon))
        db_xy = np.column_stack(polar_stereo(db_lat, db_lon))
        tree = cKDTree(st_xy)
        dists, idxs = tree.query(db_xy)
        dists_km = dists / 1000.0
        print(f"  DBF -> nearest Stanford point: median {np.median(dists_km):.2f} km, "
              f"mean {np.mean(dists_km):.2f} km, max {np.max(dists_km):.2f} km")
        pct_close = 100.0 * np.mean(dists_km < 5.0)
        print(f"  DBF points within 5 km of Stanford track: {pct_close:.1f}%")
    else:
        print("  Bingham DBF not available.")

    # --- TXT comparison ---
    if bingham_txt is not None:
        tx_lat = bingham_txt["LATITUDE"].values
        tx_lon = bingham_txt["LONGITUDE"].values
        pred_cbd = bingham_txt["PRED_CBD"].values
        print(f"  Bingham TXT: {len(bingham_txt)} points, PRED_CBD {np.nanmin(pred_cbd):.0f}-{np.nanmax(pred_cbd):.0f}")

        st_xy = np.column_stack(polar_stereo(st_lat, st_lon))
        tx_xy = np.column_stack(polar_stereo(tx_lat, tx_lon))
        tree = cKDTree(st_xy)
        dists, idxs = tree.query(tx_xy)
        dists_km = dists / 1000.0
        print(f"  TXT -> nearest Stanford point: median {np.median(dists_km):.2f} km, "
              f"mean {np.mean(dists_km):.2f} km, max {np.max(dists_km):.2f} km")

        # Compare PRED_CBD vs matched Stanford row index
        matched_st_cbd = stanford["cbd_stanford"].values[idxs]
        cbd_diff = pred_cbd - matched_st_cbd
        valid = ~np.isnan(cbd_diff) & (dists_km < 5.0)
        if valid.any():
            d = cbd_diff[valid]
            print(f"  PRED_CBD vs Stanford row index (within 5 km):")
            print(f"    n={valid.sum()}, mean diff={np.mean(d):.1f}, "
                  f"std={np.std(d):.1f}, range=[{np.min(d):.0f}, {np.max(d):.0f}]")
    else:
        print("  Bingham TXT not available.")

    # --- DBF NUMBER vs TXT NUMBER linkage ---
    if bingham_dbf is not None and bingham_txt is not None:
        dbf_nums = set(bingham_dbf["NUMBER"].values)
        txt_nums = set(bingham_txt["NUMBER"].values)
        common = dbf_nums & txt_nums
        print(f"\n  NUMBER field linkage:")
        print(f"    DBF unique NUMBERs: {len(dbf_nums)}")
        print(f"    TXT unique NUMBERs: {len(txt_nums)}")
        print(f"    Common NUMBERs: {len(common)}")

        # For common NUMBERs, compare positions
        if common:
            dbf_by_num = bingham_dbf.set_index("NUMBER")
            txt_by_num = bingham_txt.set_index("NUMBER")
            common_sorted = sorted(common)[:500]  # sample
            dists = []
            for n in common_sorted:
                lat1, lon1 = dbf_by_num.loc[n, "LATITUDE"], dbf_by_num.loc[n, "LONGITUDE"]
                lat2, lon2 = txt_by_num.loc[n, "LATITUDE"], txt_by_num.loc[n, "LONGITUDE"]
                if isinstance(lat1, pd.Series):
                    lat1, lon1 = lat1.iloc[0], lon1.iloc[0]
                if isinstance(lat2, pd.Series):
                    lat2, lon2 = lat2.iloc[0], lon2.iloc[0]
                dists.append(haversine_km(lat1, lon1, lat2, lon2))
            dists = np.array(dists)
            print(f"    Position diff for common NUMBERs (n={len(dists)}):")
            print(f"      median={np.median(dists):.3f} km, mean={np.mean(dists):.3f} km, "
                  f"max={np.max(dists):.3f} km")


def compare_nav_vs_bedmap1(stanford, bingham_txt, bedmap1, flt, match_radius_km=10):
    """Compare nav tracks against BEDMAP1 SPRI_7475."""
    print("\n" + "=" * 70)
    print("  2. NAV TRACKS vs BEDMAP1 SPRI_7475")
    print("=" * 70)

    if bedmap1 is None or len(bedmap1) == 0:
        print("  BEDMAP1 not available.")
        return

    # Filter BEDMAP1 to F137 spatial region
    if stanford is not None:
        lat_min = stanford["LAT"].min() - 1
        lat_max = stanford["LAT"].max() + 1
        bm_near = bedmap1[(bedmap1["Latitude"] >= lat_min) & (bedmap1["Latitude"] <= lat_max)].copy()
    else:
        bm_near = bedmap1.copy()

    print(f"  BEDMAP1 SPRI_7475 points in flight lat range: {len(bm_near)}")

    if len(bm_near) == 0:
        return

    bm_lat = bm_near["Latitude"].values
    bm_lon = bm_near["Longitude"].values
    bm_xy = np.column_stack(polar_stereo(bm_lat, bm_lon))

    # Stanford vs BEDMAP1
    if stanford is not None:
        st_lat = stanford["LAT"].values
        st_lon = stanford["LON"].values
        st_xy = np.column_stack(polar_stereo(st_lat, st_lon))
        tree = cKDTree(st_xy)
        dists, idxs = tree.query(bm_xy)
        dists_km = dists / 1000.0
        nearby = dists_km < match_radius_km
        print(f"\n  BEDMAP1 -> Stanford (within {match_radius_km} km): {nearby.sum()} / {len(bm_near)} points")
        if nearby.any():
            d = dists_km[nearby]
            print(f"    distance: median={np.median(d):.2f} km, mean={np.mean(d):.2f} km")

            # Compare ice thickness where available
            bm_thk = bm_near["Ice_thickn"].values[nearby]
            st_thk = stanford["THK"].values[idxs[nearby]]
            valid_thk = (bm_thk > 0) & (bm_thk < 9000) & (st_thk > 0) & (st_thk < 9000)
            if valid_thk.any():
                diff = st_thk[valid_thk] - bm_thk[valid_thk]
                print(f"    Stanford THK vs BEDMAP1 Ice_thickn (n={valid_thk.sum()}):")
                print(f"      mean diff={np.mean(diff):.1f} m, std={np.std(diff):.1f} m, "
                      f"range=[{np.min(diff):.0f}, {np.max(diff):.0f}]")

    # Bingham TXT vs BEDMAP1
    if bingham_txt is not None:
        tx_lat = bingham_txt["LATITUDE"].values
        tx_lon = bingham_txt["LONGITUDE"].values
        tx_xy = np.column_stack(polar_stereo(tx_lat, tx_lon))
        tree_tx = cKDTree(tx_xy)
        dists, idxs = tree_tx.query(bm_xy)
        dists_km = dists / 1000.0
        nearby = dists_km < match_radius_km
        print(f"\n  BEDMAP1 -> Bingham TXT (within {match_radius_km} km): {nearby.sum()} / {len(bm_near)} points")
        if nearby.any():
            d = dists_km[nearby]
            print(f"    distance: median={np.median(d):.2f} km, mean={np.mean(d):.2f} km")


def compare_nav_vs_riggs(stanford, bingham_txt, riggs, flt, match_radius_km=50):
    """Compare nav tracks against RIGGS station positions."""
    print("\n" + "=" * 70)
    print("  3. NAV TRACKS vs RIGGS STATIONS")
    print("=" * 70)

    if riggs is None:
        print("  RIGGS not available.")
        return

    valid_riggs = riggs.dropna(subset=["lat_dd", "lon_dd"])
    print(f"  RIGGS stations with valid coords: {len(valid_riggs)}")

    r_lat = valid_riggs["lat_dd"].values
    r_lon = valid_riggs["lon_dd"].values
    r_xy = np.column_stack(polar_stereo(r_lat, r_lon))

    for label, nav_df, lat_col, lon_col in [
        ("Stanford", stanford, "LAT", "LON"),
        ("Bingham TXT", bingham_txt, "LATITUDE", "LONGITUDE"),
    ]:
        if nav_df is None:
            continue

        n_lat = nav_df[lat_col].values
        n_lon = nav_df[lon_col].values
        n_xy = np.column_stack(polar_stereo(n_lat, n_lon))
        tree = cKDTree(n_xy)

        dists, idxs = tree.query(r_xy)
        dists_km = dists / 1000.0
        nearby = dists_km < match_radius_km

        print(f"\n  RIGGS -> {label} (within {match_radius_km} km): {nearby.sum()} stations")
        if nearby.any():
            for i in np.where(nearby)[0]:
                stn = valid_riggs.iloc[i]["Station"]
                dist = dists_km[i]
                nav_idx = idxs[i]
                nav_lat = n_lat[nav_idx]
                nav_lon = n_lon[nav_idx]
                h_radar = valid_riggs.iloc[i].get("h_i (radar), m", "")
                h_seis = valid_riggs.iloc[i].get("h_i (seismics), m", "")
                print(f"    {stn}: RIGGS ({r_lat[i]:.3f}, {r_lon[i]:.3f}) -> "
                      f"nav ({nav_lat:.3f}, {nav_lon:.3f}), dist={dist:.1f} km, "
                      f"h_radar={h_radar}, h_seis={h_seis}")


def compare_lyra_vs_nav(stanford, bingham_txt, lyra, flt):
    """Match LYRA echoes to nav positions via CBD and compare ice thickness."""
    print("\n" + "=" * 70)
    print("  4. LYRA ECHOES vs NAV ICE THICKNESS")
    print("=" * 70)

    if lyra is None:
        print("  LYRA echoes not available.")
        return

    good = lyra[lyra["echo_status"] == "good"].copy()
    print(f"  LYRA: {len(lyra)} total frames, {len(good)} good")

    cbd_min, cbd_max = good["cbd"].min(), good["cbd"].max()
    print(f"  LYRA CBD range: {cbd_min}-{cbd_max}")

    # Stanford: CBD is row index (0-based)
    if stanford is not None:
        matched = 0
        diffs = []
        for _, row in good.iterrows():
            cbd = int(row["cbd"])
            if cbd < len(stanford):
                st_thk = stanford.iloc[cbd]["THK"]
                if st_thk > 0 and st_thk < 9000 and not np.isnan(row["h_ice_m"]):
                    diffs.append(row["h_ice_m"] - st_thk)
                    matched += 1
        if matched > 0:
            diffs = np.array(diffs)
            print(f"\n  LYRA h_ice vs Stanford THK (CBD=row index, n={matched}):")
            print(f"    mean diff={np.mean(diffs):.1f} m, std={np.std(diffs):.1f} m, "
                  f"median={np.median(diffs):.1f} m")
        else:
            print("  No valid Stanford THK matches (all 9999?).")

    # Bingham TXT: PRED_CBD
    if bingham_txt is not None and "PRED_CBD" in bingham_txt.columns:
        txt_by_cbd = bingham_txt.dropna(subset=["PRED_CBD"])
        txt_by_cbd = txt_by_cbd[txt_by_cbd["ICE_THICKN"].notna() & (txt_by_cbd["ICE_THICKN"] > 0)]
        if len(txt_by_cbd) > 0:
            # Build dict PRED_CBD -> ICE_THICKN
            cbd_thk = {}
            for _, r in txt_by_cbd.iterrows():
                cbd_thk[int(r["PRED_CBD"])] = r["ICE_THICKN"]

            matched = 0
            diffs = []
            for _, row in good.iterrows():
                cbd = int(row["cbd"])
                if cbd in cbd_thk and not np.isnan(row["h_ice_m"]):
                    diffs.append(row["h_ice_m"] - cbd_thk[cbd])
                    matched += 1

            if matched > 0:
                diffs = np.array(diffs)
                print(f"\n  LYRA h_ice vs Bingham TXT ICE_THICKN (PRED_CBD match, n={matched}):")
                print(f"    mean diff={np.mean(diffs):.1f} m, std={np.std(diffs):.1f} m, "
                      f"median={np.median(diffs):.1f} m")
            else:
                print("  No valid Bingham TXT ICE_THICKN matches.")
        else:
            print("  Bingham TXT has no valid ICE_THICKN values.")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_tracks(stanford, bingham_dbf, bingham_txt, bedmap1, riggs, lyra, flt,
                match_radius_km=50):
    """Plot nav tracks on polar stereographic map + ice thickness panel."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # -- Left: EPSG:3031 polar stereographic map --------------------------------
    ax = axes[0]
    ax.set_title(f"F{flt} Navigation Cross-Validation (EPSG:3031)", fontsize=10)
    ax.set_aspect("equal")

    # Basemap
    land = load_basemap_ps()
    land.plot(ax=ax, color="#e4eef5", edgecolor="#7aadbb", linewidth=0.3, zorder=1)

    # Collect all x, y to set axis limits later
    all_x, all_y = [], []

    if stanford is not None:
        sx, sy = _to_ps(stanford["LON"].values, stanford["LAT"].values)
        ax.plot(sx, sy, "b-", lw=0.8, alpha=0.7, label="Stanford nav", zorder=3)
        all_x.extend(sx); all_y.extend(sy)

    if bingham_dbf is not None:
        dx, dy = _to_ps(bingham_dbf["LONGITUDE"].values, bingham_dbf["LATITUDE"].values)
        ax.plot(dx, dy, "g.", ms=1, alpha=0.3, label="Bingham DBF", zorder=2)
        all_x.extend(dx); all_y.extend(dy)

    if bingham_txt is not None:
        tx, ty = _to_ps(bingham_txt["LONGITUDE"].values, bingham_txt["LATITUDE"].values)
        ax.plot(tx, ty, "r.", ms=1, alpha=0.3, label="Bingham TXT", zorder=2)
        all_x.extend(tx); all_y.extend(ty)

    if bedmap1 is not None and stanford is not None:
        lat_min = stanford["LAT"].min() - 1
        lat_max = stanford["LAT"].max() + 1
        bm_near = bedmap1[(bedmap1["Latitude"] >= lat_min) & (bedmap1["Latitude"] <= lat_max)]
        if len(bm_near) > 0:
            bx, by = _to_ps(bm_near["Longitude"].values, bm_near["Latitude"].values)
            ax.plot(bx, by, "k.", ms=1, alpha=0.15,
                    label=f"BEDMAP1 SPRI (n={len(bm_near)})", zorder=2)

    if riggs is not None:
        valid_riggs = riggs.dropna(subset=["lat_dd", "lon_dd"])
        if stanford is not None:
            st_xy = np.column_stack(polar_stereo(stanford["LAT"].values, stanford["LON"].values))
            r_xy = np.column_stack(polar_stereo(valid_riggs["lat_dd"].values, valid_riggs["lon_dd"].values))
            tree = cKDTree(st_xy)
            dists, _ = tree.query(r_xy)
            nearby = dists / 1000.0 < match_radius_km
            near_riggs = valid_riggs[nearby]
        else:
            near_riggs = valid_riggs
        if len(near_riggs) > 0:
            rx, ry = _to_ps(near_riggs["lon_dd"].values, near_riggs["lat_dd"].values)
            ax.plot(rx, ry, "r^", ms=8, label="RIGGS", zorder=5)
            for i, (_, r) in enumerate(near_riggs.iterrows()):
                ax.annotate(r["Station"], (rx[i], ry[i]),
                            fontsize=6, ha="left", va="bottom",
                            path_effects=[pe.withStroke(linewidth=1.5, foreground="white")])

    if lyra is not None and stanford is not None:
        good = lyra[lyra["echo_status"] == "good"]
        lyra_lats, lyra_lons = [], []
        for _, row in good.iterrows():
            cbd = int(row["cbd"])
            if cbd < len(stanford):
                lyra_lats.append(stanford.iloc[cbd]["LAT"])
                lyra_lons.append(stanford.iloc[cbd]["LON"])
        if lyra_lats:
            lx, ly = _to_ps(np.array(lyra_lons), np.array(lyra_lats))
            ax.scatter(lx, ly, c="magenta", s=3, alpha=0.5,
                       label="LYRA good frames", zorder=4)

    # Axis limits with 10% padding
    if all_x:
        xarr, yarr = np.array(all_x), np.array(all_y)
        dx = xarr.max() - xarr.min()
        dy = yarr.max() - yarr.min()
        pad = max(dx, dy) * 0.10
        ax.set_xlim(xarr.min() - pad, xarr.max() + pad)
        ax.set_ylim(yarr.min() - pad, yarr.max() + pad)

    # Graticules
    lon_arr = np.linspace(-180, 180, 720)
    lat_arr_mer = np.linspace(-89, -70, 200)
    for lat in range(-85, -74):
        gx, gy = _graticule_line(lon_arr, lat)
        ax.plot(gx, gy, color="#cccccc", linewidth=0.3, zorder=0)
        # Label at 180 meridian
        lx, ly = _to_ps(np.array([180.0]), np.array([float(lat)]))
        ax.text(lx[0], ly[0], f"{abs(lat)}S", fontsize=5, color="#999999",
                ha="left", va="center")
    for lon in range(-180, 180, 15):
        mx, my = _meridian_line(lat_arr_mer, lon)
        ax.plot(mx, my, color="#cccccc", linewidth=0.2, zorder=0)

    # Km-scale axis labels
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1e3:.0f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1e3:.0f}"))
    ax.set_xlabel("Easting (km)")
    ax.set_ylabel("Northing (km)")
    ax.legend(fontsize=7, loc="best")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    # -- Right: ice thickness along-track ----------------------------------------
    ax2 = axes[1]
    ax2.set_title(f"F{flt} Ice Thickness Comparison", fontsize=10)

    if stanford is not None:
        thk = stanford["THK"].values.copy().astype(float)
        thk[thk >= 9999] = np.nan
        cbd_s = stanford["cbd_stanford"].values
        valid = ~np.isnan(thk)
        if valid.any():
            ax2.plot(cbd_s[valid], thk[valid], "b-", lw=0.5, alpha=0.7, label="Stanford THK")

    if bingham_txt is not None:
        valid_tx = bingham_txt["ICE_THICKN"].notna() & (bingham_txt["ICE_THICKN"] > 0)
        if valid_tx.any():
            ax2.plot(bingham_txt.loc[valid_tx, "PRED_CBD"],
                     bingham_txt.loc[valid_tx, "ICE_THICKN"],
                     "g-", lw=0.5, alpha=0.7, label="Bingham TXT ICE_THICKN")

    if lyra is not None:
        good = lyra[lyra["echo_status"] == "good"]
        ax2.scatter(good["cbd"], good["h_ice_m"], c="magenta", s=5, alpha=0.5,
                    label="LYRA h_ice", zorder=5)

    ax2.set_xlabel("CBD")
    ax2.set_ylabel("Ice thickness (m)")
    ax2.legend(fontsize=7, loc="best")
    ax2.grid(True, alpha=0.3)
    for sp in ["top", "right"]:
        ax2.spines[sp].set_visible(False)

    plt.tight_layout()
    out_dir = ROOT / "tools" / "LYRA" / "output" / f"F{flt}" / "validation"
    os.makedirs(out_dir, exist_ok=True)
    out_path = out_dir / f"F{flt}_nav_crossval.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Cross-validate navigation sources")
    parser.add_argument("flight", type=str, help="Flight number (e.g. 137)")
    parser.add_argument("--match-radius", type=float, default=50,
                        help="RIGGS/BEDMAP match radius in km (default 50)")
    args = parser.parse_args()

    flt = args.flight.lstrip("Ff")
    match_radius_km = args.match_radius

    print(f"Navigation cross-validation for F{flt}")
    print(f"Match radius: {match_radius_km} km")

    # Load all sources
    stanford = load_stanford(flt)
    bingham_dbf = load_bingham_dbf(flt)
    bingham_txt = load_bingham_txt(flt)
    riggs = load_riggs()
    bedmap1 = load_bedmap1_spri()
    lyra = load_lyra_echoes(flt)

    # Run comparisons
    compare_stanford_vs_bingham(stanford, bingham_dbf, bingham_txt, flt)
    compare_nav_vs_bedmap1(stanford, bingham_txt, bedmap1, flt, match_radius_km)
    compare_nav_vs_riggs(stanford, bingham_txt, riggs, flt, match_radius_km)
    compare_lyra_vs_nav(stanford, bingham_txt, lyra, flt)

    # Plot
    plot_tracks(stanford, bingham_dbf, bingham_txt, bedmap1, riggs, lyra, flt,
                match_radius_km)


if __name__ == "__main__":
    main()
