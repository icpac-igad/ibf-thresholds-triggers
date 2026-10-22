"""Shared settings and store openers for the exercises. Change TOWN, LAT, LON and BOX here, once, for your area.

Three anonymous, cloud-native (ARCO) stores, nothing to download in advance:
  ERA5 SPI, pencil chunks   source.coop e4drr-project/observations/era5_ecmwf_pencil   (time 1033, 5, 5)
  ERA5 SPI thresholds, pan  source.coop e4drr-project/observations/era5_ecmwf_rp_icechunk (1 whole map per chunk)
  SEAS5 history, monthly    source.coop e4drr-project/forecasts/seas51_tp_icechunk_v2 (Icechunk, 1 deg, East Africa only)
"""
import os
import pathlib

# Stray AWS_* variables silently redirect anonymous reads; every store here is public, so drop them first.
for k in [k for k in os.environ if k.startswith("AWS_")]:
    os.environ.pop(k)

import icechunk
import json
import numpy as np
import xarray as xr
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import Rectangle

TOWN, LAT, LON = "Garissa", -0.45, 39.65                  # a town where October-December (OND) is the main rainy season
BOX = dict(lat=slice(12, -5), lon=slice(34, 52))          # eastern Horn: Kenya east, Somalia, south Ethiopia (lat descends)
SEASON_END = 12                                           # SPI-3 labelled December = October + November + December
FC_INIT = "2026-09-01"                                    # the newest SEAS5 start
CAL_YEARS = range(1991, 2021)                             # September starts 1991-2020: the model climatology (calibration)
CAL_MEMBERS = slice(0, 24)                                # the 25 members that exist in every year (51 only from 2017)
OUT = pathlib.Path("out")
OUT.mkdir(exist_ok=True)
ADMIN = pathlib.Path(__file__).resolve().parents[1] / "data" / "icpac_adm1v3.geojson"   # admin-1, ICPAC countries

SOURCE_COOP = {"anon": True, "client_kwargs": {"endpoint_url": "https://data.source.coop"}}


def open_spi_pencil() -> xr.Dataset:
    """ERA5 SPI-1..SPI-48, monthly 1940-01..2026-01, 0.25 deg. One chunk = the whole 86-year series of 5x5 pixels."""
    return xr.open_zarr("s3://e4drr-project/observations/era5_ecmwf_pencil", consolidated=False,
                        storage_options=SOURCE_COOP)


def open_spi_thresholds() -> xr.Dataset:
    """Pre-computed SPI drought thresholds per pixel for return periods 3-50 years. One chunk = one whole map."""
    storage = icechunk.s3_storage(bucket="e4drr-project", prefix="observations/era5_ecmwf_rp_icechunk",
                                  endpoint_url="https://data.source.coop", region="us-east-1",
                                  force_path_style=True, anonymous=True, from_env=False)
    session = icechunk.Repository.open(storage).readonly_session("main")
    return xr.open_zarr(session.store, consolidated=False)


def open_seas5_history() -> xr.Dataset:
    """SEAS5 monthly mean rainfall rate for East Africa only, every start 1981-01..2026-09, 6 forecast months, 1 deg.
    Starts 2026-05..2026-09 were added from Planette's daily SEAS5 (tools/append_planette_monthly.py).
    Small and regional: the model's own climatology for a start month costs a few small chunks."""
    storage = icechunk.s3_storage(bucket="e4drr-project", prefix="forecasts/seas51_tp_icechunk_v2",
                                  endpoint_url="https://data.source.coop", region="us-east-1",
                                  force_path_style=True, anonymous=True, from_env=False)
    session = icechunk.Repository.open(storage).readonly_session("main")
    return xr.open_zarr(session.store, consolidated=False)


def admin_rings() -> list:
    """Every ring of every admin-1 polygon in ADMIN, as (lon, lat) arrays: boundaries to draw, no geopandas needed."""
    rings = []
    for f in json.loads(ADMIN.read_text())["features"]:
        g = f["geometry"]
        for poly in ([g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]):
            rings += [np.asarray(r)[:, :2] for r in poly]
    return rings


def add_admin(ax, inset: bool = True) -> None:
    """Draw admin-1 boundaries on a map; with inset, add a small East Africa map with BOX in red (lower right)."""
    rings = admin_rings()
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.add_collection(LineCollection(rings, colors="0.25", linewidths=0.4))
    ax.set_xlim(xlim); ax.set_ylim(ylim)                                  # keep the map on BOX
    if inset:
        ins = ax.inset_axes([0.70, 0.02, 0.29, 0.40])
        ins.add_collection(PolyCollection(rings, facecolors="0.85", edgecolors="0.2", linewidths=0.35))
        lat, lon = BOX["lat"], BOX["lon"]
        ins.add_patch(Rectangle((lon.start, lat.stop), lon.stop - lon.start, lat.start - lat.stop,
                                fill=False, edgecolor="r", linewidth=1.5))
        ins.set_xlim(21, 52); ins.set_ylim(-12, 24); ins.set_aspect("equal")
        ins.set_xticks([]); ins.set_yticks([]); ins.set_facecolor("white")
