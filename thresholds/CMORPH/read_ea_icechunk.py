#!/usr/bin/env python3
"""Read the CMORPH East Africa Icechunk store as an xarray Dataset.

Opens the materialized EA subset store from GCS and demonstrates
basic access patterns: inspect metadata, load time series at a point,
load spatial snapshot at a timestep, and check fill progress.

Usage:
    # Quick inspection (metadata + fill progress)
    micromamba run -n aifs-etl python read_ea_icechunk.py

    # Load a spatial snapshot for a given date
    micromamba run -n aifs-etl python read_ea_icechunk.py --snapshot 2005-06-15T12:00

    # Load a time series at a point (Nairobi)
    micromamba run -n aifs-etl python read_ea_icechunk.py --timeseries --lat -1.29 --lon 36.82

    # Use a local store instead of GCS
    micromamba run -n aifs-etl python read_ea_icechunk.py --local /path/to/store
"""

import argparse
import sys

import icechunk
import numpy as np
import xarray as xr

SERVICE_ACCOUNT_FILE = "coiled-data-e4drr_202505.json"
GCS_BUCKET = "cpc_awc"
GCS_PREFIX = "cmorph_ea_subset"
FILL_VALUE = np.float32(9.969209968386869e+36)


def open_ea_store(args):
    """Open the EA Icechunk store and return an xarray Dataset."""
    if args.local:
        storage = icechunk.local_filesystem_storage(path=args.local)
    else:
        storage = icechunk.gcs_storage(
            bucket=args.gcs_bucket,
            prefix=args.gcs_prefix,
            service_account_file=args.service_account,
        )

    repo = icechunk.Repository.open(
        storage, config=icechunk.RepositoryConfig.default(),
    )
    session = repo.readonly_session("main")
    ds = xr.open_zarr(session.store, consolidated=False)
    return ds, repo


def inspect(ds, repo):
    """Print dataset metadata and fill progress."""
    print("=" * 60)
    print("CMORPH East Africa Icechunk Store")
    print("=" * 60)
    print(f"\n{ds}\n")
    print(f"Dimensions: {dict(ds.sizes)}")
    print(f"  time:  {ds.sizes['time']} steps "
          f"[{str(ds.time.values[0])[:19]} .. {str(ds.time.values[-1])[:19]}]")
    print(f"  lat:   {ds.sizes['lat']} pts "
          f"[{float(ds.lat.values[0]):.2f} .. {float(ds.lat.values[-1]):.2f}]")
    print(f"  lon:   {ds.sizes['lon']} pts "
          f"[{float(ds.lon.values[0]):.2f} .. {float(ds.lon.values[-1]):.2f}]")

    if "cmorph" in ds.data_vars:
        da = ds["cmorph"]
        print(f"\nVariable 'cmorph':")
        print(f"  dtype:  {da.dtype}")
        print(f"  shape:  {da.shape}")
        if hasattr(da, "encoding") and "chunks" in da.encoding:
            print(f"  chunks: {da.encoding['chunks']}")

    # Check fill progress by sampling timesteps
    print("\nFill progress (sampling every 1000th timestep):")
    n_time = ds.sizes["time"]
    sample_indices = list(range(0, n_time, max(1, n_time // 10)))
    filled_count = 0
    for idx in sample_indices:
        sample = ds["cmorph"].isel(time=idx, lat=slice(200, 210), lon=slice(200, 210)).values
        is_filled = np.any((sample != 0) & (~np.isnan(sample)) & (sample != FILL_VALUE))
        status = "filled" if is_filled else "empty"
        t = str(ds.time.values[idx])[:19]
        if is_filled:
            filled_count += 1
        print(f"  t[{idx:>6d}] {t}: {status}")

    print(f"\n  {filled_count}/{len(sample_indices)} sampled timesteps have data")

    # Commit history summary
    try:
        commits = list(repo.ancestry(branch="main"))
        fill_commits = [c for c in commits if c.message.startswith("fill days ")]
        print(f"\nCommit history: {len(commits)} total, {len(fill_commits)} fill commits")
        if fill_commits:
            print(f"  Latest: {fill_commits[0].message}")
            print(f"  Oldest: {fill_commits[-1].message}")
    except Exception:
        pass


def snapshot(ds, time_str):
    """Load and display a spatial snapshot at a given time."""
    import pandas as pd

    print(f"\nSpatial snapshot at {time_str}:")
    try:
        # Use isel with searchsorted to avoid reindexing issues with
        # non-unique time coords (can happen during fill if two files
        # share a timestamp boundary)
        target = np.datetime64(pd.Timestamp(time_str))
        time_vals = ds.time.values
        idx = int(np.searchsorted(time_vals, target))
        idx = min(idx, len(time_vals) - 1)
        data = ds["cmorph"].isel(time=idx).load()
    except Exception as e:
        print(f"  Error: {e}")
        return

    actual_time = str(ds.time.values[idx])[:19]
    vals = data.values
    valid = (~np.isnan(vals)) & (vals != FILL_VALUE) & (vals != 0)

    print(f"  Nearest time: {actual_time}")
    print(f"  Shape: {vals.shape}")
    print(f"  Min:   {np.nanmin(vals):.4f} mm/hr")
    print(f"  Max:   {np.nanmax(vals):.4f} mm/hr")
    print(f"  Mean:  {np.nanmean(vals):.4f} mm/hr")
    print(f"  Valid (non-zero, non-NaN): {valid.sum()} / {vals.size} "
          f"({100 * valid.sum() / vals.size:.1f}%)")


def timeseries(ds, lat, lon, time_start=None, time_end=None):
    """Load and display a time series at a given point.

    NOTE: Before pencil rechunking, loading a full 27-year time series
    requires reading ~10K chunks (one per day) and is very slow.  Use
    --time-start / --time-end to bound the query, or wait until the
    rechunked pencil-chunk store is available.
    """
    print(f"\nTime series at lat={lat}, lon={lon}:")

    da = ds["cmorph"].sel(lat=lat, lon=lon, method="nearest")

    # Bound the time range to avoid loading 473K timesteps across 10K chunks
    if time_start or time_end:
        sel = {}
        if time_start and time_end:
            sel["time"] = slice(time_start, time_end)
        elif time_start:
            sel["time"] = slice(time_start, None)
        else:
            sel["time"] = slice(None, time_end)
        da = da.sel(**sel)
    else:
        # Default: load 1 year (17,520 timesteps = 365 chunks) as a safe demo
        n_time = ds.sizes["time"]
        one_year = min(365 * 48, n_time)
        da = da.isel(time=slice(0, one_year))
        print(f"  (Loading first year only — use --time-start/--time-end for custom range)")
        print(f"  (Full time series needs pencil rechunking for fast access)")

    try:
        ts = da.load()
    except Exception as e:
        print(f"  Error: {e}")
        return

    actual_lat = float(ts.lat.values)
    actual_lon = float(ts.lon.values)
    vals = ts.values
    valid = (~np.isnan(vals)) & (vals != FILL_VALUE)
    nonzero = valid & (vals != 0)

    print(f"  Nearest point: lat={actual_lat:.2f}, lon={actual_lon:.2f}")
    print(f"  Timesteps: {len(vals)}")
    print(f"  Time range: {str(ts.time.values[0])[:19]} .. {str(ts.time.values[-1])[:19]}")
    print(f"  Valid values: {valid.sum()} / {len(vals)} ({100 * valid.sum() / len(vals):.1f}%)")
    print(f"  Rainy steps:  {nonzero.sum()} ({100 * nonzero.sum() / max(1, valid.sum()):.1f}% of valid)")

    if valid.sum() > 0:
        valid_vals = vals[valid]
        print(f"  Min:   {valid_vals.min():.4f} mm/hr")
        print(f"  Max:   {valid_vals.max():.4f} mm/hr")
        print(f"  Mean:  {valid_vals.mean():.4f} mm/hr")


def main():
    parser = argparse.ArgumentParser(
        description="Read CMORPH East Africa Icechunk store",
    )
    parser.add_argument("--gcs-bucket", default=GCS_BUCKET)
    parser.add_argument("--gcs-prefix", default=GCS_PREFIX)
    parser.add_argument("--service-account", default=SERVICE_ACCOUNT_FILE)
    parser.add_argument("--local", default=None, help="Local store path (overrides GCS)")

    parser.add_argument("--snapshot", default=None, metavar="TIME",
                        help="Load spatial snapshot at this time (e.g. 2005-06-15T12:00)")
    parser.add_argument("--timeseries", action="store_true",
                        help="Load time series at --lat/--lon")
    parser.add_argument("--lat", type=float, default=-1.29, help="Latitude for time series")
    parser.add_argument("--lon", type=float, default=36.82, help="Longitude for time series")
    parser.add_argument("--time-start", default=None, metavar="TIME",
                        help="Start time for time series (e.g. 2005-01-01)")
    parser.add_argument("--time-end", default=None, metavar="TIME",
                        help="End time for time series (e.g. 2005-12-31)")

    args = parser.parse_args()

    ds, repo = open_ea_store(args)
    inspect(ds, repo)

    if args.snapshot:
        snapshot(ds, args.snapshot)

    if args.timeseries:
        timeseries(ds, args.lat, args.lon, args.time_start, args.time_end)


if __name__ == "__main__":
    main()
