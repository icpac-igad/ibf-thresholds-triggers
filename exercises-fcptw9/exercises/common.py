"""Shared settings and store openers for the exercises. Change TOWN, LAT, LON and BOX here, once, for your area.

Four anonymous, cloud-native (ARCO) stores, nothing to download in advance:
  ERA5 SPI, pencil chunks   source.coop e4drr-project/observations/era5_ecmwf_pencil   (time 1033, 5, 5)
  ERA5 SPI thresholds, pan  source.coop e4drr-project/observations/era5_ecmwf_rp_icechunk (1 whole map per chunk)
  SEAS5 forecast, daily     Planette C3S, s3://planette-c3s-seasonal-forecasts/seas5_ic/forecast (Icechunk, 1 deg, global)
  SEAS5 history, monthly    source.coop e4drr-project/forecasts/seas51_tp_icechunk_v2 (Icechunk, 1 deg, East Africa only)
"""
import os
import pathlib

# Stray AWS_* variables silently redirect anonymous reads; every store here is public, so drop them first.
for k in [k for k in os.environ if k.startswith("AWS_")]:
    os.environ.pop(k)

import icechunk
import xarray as xr

TOWN, LAT, LON = "Garissa", -0.45, 39.65                  # a town where October-December (OND) is the main rainy season
BOX = dict(lat=slice(12, -5), lon=slice(34, 52))          # eastern Horn: Kenya east, Somalia, south Ethiopia (lat descends)
SEASON_END = 12                                           # SPI-3 labelled December = October + November + December
FC_INIT = "2026-09-01"                                    # the newest SEAS5 start on Planette
CAL_YEARS = range(1991, 2021)                             # September starts 1991-2020: the model climatology (calibration)
CAL_MEMBERS = slice(0, 24)                                # the 25 members that exist in every year (51 only from 2017)
OUT = pathlib.Path("out")
OUT.mkdir(exist_ok=True)

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


def open_seas5() -> xr.Dataset:
    """SEAS5 daily forecasts, 1 deg global: number 51, init_time monthly 2017-01.., lead 1..215 days.
    One pr chunk = 10 members x 1 start x 46 days x the WHOLE globe (about 120 MB): a small box still fetches it all."""
    storage = icechunk.s3_storage(bucket="planette-c3s-seasonal-forecasts", prefix="seas5_ic/forecast",
                                  region="us-east-2", anonymous=True, from_env=False)
    session = icechunk.Repository.open(storage).readonly_session("main")
    return xr.open_zarr(session.store, group="single/1platx1plon", consolidated=False)


def open_seas5_history() -> xr.Dataset:
    """SEAS5 monthly mean rainfall rate for East Africa only, every start 1981-01..2026-04, 6 forecast months, 1 deg.
    Small and regional: the model's own climatology for a start month costs a few small chunks."""
    storage = icechunk.s3_storage(bucket="e4drr-project", prefix="forecasts/seas51_tp_icechunk_v2",
                                  endpoint_url="https://data.source.coop", region="us-east-1",
                                  force_path_style=True, anonymous=True, from_env=False)
    session = icechunk.Repository.open(storage).readonly_session("main")
    return xr.open_zarr(session.store, consolidated=False)
