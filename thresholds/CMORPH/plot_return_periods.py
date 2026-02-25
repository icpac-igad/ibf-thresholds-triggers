# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "xarray",
#     "netcdf4",
#     "matplotlib",
#     "cartopy",
# ]
# ///
"""Plot return period maps from CMORPH East Africa analysis."""

import json
import pathlib

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.collections import LineCollection

# ── paths ────────────────────────────────────────────────────────────────
HERE = pathlib.Path(__file__).parent
NC_PATH = pathlib.Path("/data/data-nodelete/cmorph_return_periods/cmorph_ea_return_periods.nc")
GEOJSON_PATH = pathlib.Path("/data/08-2023/working_notes_jupyter/ignore_nka_gitrepos/grib-index-kerchunk/gefs/ea_ghcf_simple.geojson")
OUT_DIR = pathlib.Path("/data/data-nodelete/cmorph_return_periods")

# ── duration labels for filenames and titles ─────────────────────────────
DURATION_FNAME = {
    "30min": "30min",
    "1hr": "1hr",
    "3hr": "3hr",
    "6hr": "6hr",
    "12hr": "12hr",
    "24hr": "24hr",
    "48hr": "48hr",
    "72hr": "72hr",
    "7day": "7day",
}

# ── helpers ──────────────────────────────────────────────────────────────

def load_geojson_lines(path: pathlib.Path) -> list[np.ndarray]:
    """Load GeoJSON and return a list of (N,2) coordinate arrays."""
    with open(path) as f:
        gj = json.load(f)
    lines: list[np.ndarray] = []
    for feat in gj["features"]:
        geom = feat["geometry"]
        if geom is None:
            continue
        gtype = geom["type"]
        if gtype == "Polygon":
            for ring in geom["coordinates"]:
                lines.append(np.asarray(ring))
        elif gtype == "MultiPolygon":
            for poly in geom["coordinates"]:
                for ring in poly:
                    lines.append(np.asarray(ring))
        elif gtype == "LineString":
            lines.append(np.asarray(geom["coordinates"]))
        elif gtype == "MultiLineString":
            for part in geom["coordinates"]:
                lines.append(np.asarray(part))
    return lines


def add_boundaries(ax, lines: list[np.ndarray], **kwargs):
    """Draw country boundaries on a cartopy GeoAxes."""
    segs = [ln[:, :2] for ln in lines]  # (lon, lat) pairs
    lc = LineCollection(segs, transform=ccrs.PlateCarree(), **kwargs)
    ax.add_collection(lc)


def sensible_vmax(da: xr.DataArray) -> float:
    """Return a rounded vmax based on the 99th percentile."""
    p99 = float(np.nanpercentile(da.values, 99))
    # round up to a nice number
    if p99 < 10:
        return np.ceil(p99)
    elif p99 < 100:
        return np.ceil(p99 / 5) * 5
    elif p99 < 500:
        return np.ceil(p99 / 10) * 10
    else:
        return np.ceil(p99 / 50) * 50


# ── main ─────────────────────────────────────────────────────────────────

def main():
    ds = xr.open_dataset(NC_PATH)
    rp_precip = ds["return_period_precip"]  # (duration, return_period, lat, lon)
    durations = ds["duration"].values
    return_periods = ds["return_period"].values

    lines = load_geojson_lines(GEOJSON_PATH)
    proj = ccrs.PlateCarree()

    lon = ds["lon"].values
    lat = ds["lat"].values
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]

    # ── per-duration PNGs (2×3 grid: 6 return periods) ──────────────────
    for di, dur in enumerate(durations):
        dur_str = str(dur)
        fname_tag = DURATION_FNAME.get(dur_str, dur_str)

        da_dur = rp_precip.isel(duration=di)  # (return_period, lat, lon)
        vmax = sensible_vmax(da_dur)

        fig, axes = plt.subplots(
            2, 3,
            figsize=(14, 8),
            subplot_kw={"projection": proj},
        )
        fig.suptitle(f"CMORPH Return Period Precipitation — {dur_str} duration", fontsize=14, y=0.97)

        for ri, (ax, rp) in enumerate(zip(axes.flat, return_periods)):
            da = da_dur.isel(return_period=ri)
            im = ax.pcolormesh(
                lon, lat, da.values,
                transform=proj,
                cmap="YlGnBu",
                vmin=0,
                vmax=vmax,
                shading="auto",
            )
            add_boundaries(ax, lines, colors="black", linewidths=0.5)
            ax.set_extent(extent, crs=proj)
            ax.set_title(f"{int(rp)}-yr return period", fontsize=10)

        fig.subplots_adjust(bottom=0.08, top=0.90, left=0.04, right=0.96, wspace=0.08, hspace=0.15)
        cbar_ax = fig.add_axes([0.15, 0.03, 0.7, 0.02])
        fig.colorbar(im, cax=cbar_ax, orientation="horizontal", label="Precipitation (mm)")

        out = OUT_DIR / f"rp_{fname_tag}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out.name}")

    # ── stamp plot: 3×3 grid of durations for a single return period ─────
    sel_rp = 100
    rp_idx = int(np.where(return_periods == sel_rp)[0][0])

    fig, axes = plt.subplots(
        3, 3,
        figsize=(14, 14),
        subplot_kw={"projection": proj},
    )
    fig.suptitle(f"CMORPH {sel_rp}-yr Return Period Precipitation by Duration", fontsize=14, y=0.97)

    for di, (ax, dur) in enumerate(zip(axes.flat, durations)):
        dur_str = str(dur)
        da = rp_precip.isel(duration=di, return_period=rp_idx)
        vmax = sensible_vmax(da)
        im = ax.pcolormesh(
            lon, lat, da.values,
            transform=proj,
            cmap="YlGnBu",
            vmin=0,
            vmax=vmax,
            shading="auto",
        )
        add_boundaries(ax, lines, colors="black", linewidths=0.5)
        ax.set_extent(extent, crs=proj)
        ax.set_title(dur_str, fontsize=11)
        fig.colorbar(im, ax=ax, orientation="horizontal", shrink=0.8, pad=0.04, label="mm")

    fig.subplots_adjust(top=0.93, bottom=0.03, left=0.03, right=0.97, wspace=0.10, hspace=0.20)

    out = OUT_DIR / f"rp_stamp_{sel_rp}yr.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name}")

    ds.close()
    print("Done — 10 PNGs written.")


if __name__ == "__main__":
    main()
