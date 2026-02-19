#!/usr/bin/env python3
"""Local test for the fill worker pipeline (no Coiled needed).

Tests the exact same obstore → tempfile → netcdf4 → EA subset → numpy
pipeline that Coiled workers will run, but locally on a single day.

Usage:
    micromamba run -n aifs-etl python test_fill_local.py
    micromamba run -n aifs-etl python test_fill_local.py --year 2000 --month 6 --day 15
    micromamba run -n aifs-etl python test_fill_local.py --catalog cmorph_vds_catalog/catalog.parquet
"""

import argparse
import base64
import json
import os
import tempfile
import time

import numpy as np
import obstore as obs
import pandas as pd
import xarray as xr
from obstore.store import from_url

# ─── Constants (same as cmorph_east_africa_icechunk.py) ──────────────────────

S3_BUCKET_URL = "s3://noaa-cdr-precip-cmorph-pds/"
S3_BUCKET_PREFIX = "s3://noaa-cdr-precip-cmorph-pds/"
EA_LAT_MIN, EA_LAT_MAX = -12.0, 23.0
EA_LON_MIN, EA_LON_MAX = 21.0, 53.0


def decode_ea_indices(catalog_path):
    """Decode global coords from catalog and compute EA slice indices."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(catalog_path)
    for batch in pf.iter_batches(batch_size=1, columns=["kerchunk_refs", "status"]):
        row = batch.to_pydict()
        if row["status"][0] == "success" and row["kerchunk_refs"][0]:
            refs = json.loads(row["kerchunk_refs"][0])
            break
    else:
        raise ValueError("No successful rows in catalog")

    lat_zarray = json.loads(refs["refs"]["lat/.zarray"])
    lat_data = base64.b64decode(refs["refs"]["lat/0"].replace("base64:", ""))
    lat_vals = np.frombuffer(lat_data, dtype=lat_zarray["dtype"])

    lon_zarray = json.loads(refs["refs"]["lon/.zarray"])
    lon_data = base64.b64decode(refs["refs"]["lon/0"].replace("base64:", ""))
    lon_vals = np.frombuffer(lon_data, dtype=lon_zarray["dtype"])

    lat_idx = np.where((lat_vals >= EA_LAT_MIN) & (lat_vals <= EA_LAT_MAX))[0]
    lon_idx = np.where((lon_vals >= EA_LON_MIN) & (lon_vals <= EA_LON_MAX))[0]

    lat_s, lat_e = int(lat_idx[0]), int(lat_idx[-1]) + 1
    lon_s, lon_e = int(lon_idx[0]), int(lon_idx[-1]) + 1

    return lat_s, lat_e, lon_s, lon_e, lat_vals[lat_s:lat_e], lon_vals[lon_s:lon_e]


def read_day_ea_subset(df_day, s3_store, lat_s, lat_e, lon_s, lon_e):
    """Read one day's EA subset — same logic as the Coiled worker function."""
    subsets = []
    for i, (_, row) in enumerate(df_day.iterrows()):
        s3_key = row["s3_url"].replace(S3_BUCKET_PREFIX, "")

        result = obs.get(s3_store, s3_key)
        nc_bytes = result.bytes()

        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
            tmp.write(nc_bytes)
            tmp_path = tmp.name

        try:
            ds = xr.open_dataset(tmp_path, engine="netcdf4")
            data = ds["cmorph"].isel(
                lat=slice(lat_s, lat_e),
                lon=slice(lon_s, lon_e),
            ).values
            ds.close()
        finally:
            os.unlink(tmp_path)

        subsets.append(data)
        fname = row["s3_url"].split("/")[-1]
        print(
            f"  [{i+1}/{len(df_day)}] {fname}: "
            f"shape={data.shape}, "
            f"min={np.nanmin(data):.3f}, max={np.nanmax(data):.3f}"
        )

    return np.concatenate(subsets, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Local test of fill worker pipeline")
    parser.add_argument(
        "--catalog", default="cmorph_vds_catalog/catalog.parquet",
        help="Path to local Parquet catalog",
    )
    parser.add_argument("--year", type=int, default=1998)
    parser.add_argument("--month", type=int, default=1)
    parser.add_argument("--day", type=int, default=17)
    args = parser.parse_args()

    print("=" * 60)
    print(f"LOCAL FILL TEST: {args.year}-{args.month:02d}-{args.day:02d}")
    print("=" * 60)

    # Step 1: Decode EA indices from catalog
    print("\n1. Decoding EA indices from catalog...")
    t0 = time.time()
    lat_s, lat_e, lon_s, lon_e, lat_ea, lon_ea = decode_ea_indices(args.catalog)
    print(f"   lat slice [{lat_s}:{lat_e}] = {len(lat_ea)} pts "
          f"[{lat_ea[0]:.2f} .. {lat_ea[-1]:.2f}]")
    print(f"   lon slice [{lon_s}:{lon_e}] = {len(lon_ea)} pts "
          f"[{lon_ea[0]:.2f} .. {lon_ea[-1]:.2f}]")

    # Step 2: Read day's S3 URLs from catalog
    print(f"\n2. Reading catalog for {args.year}-{args.month:02d}-{args.day:02d}...")
    df_day = pd.read_parquet(
        args.catalog,
        filters=[
            ("year", "==", args.year),
            ("month", "==", args.month),
            ("day", "==", args.day),
            ("status", "==", "success"),
        ],
        columns=["datetime", "s3_url"],
    ).sort_values("datetime")
    print(f"   Found {len(df_day)} files")

    if len(df_day) == 0:
        print("   ERROR: No files found for this day!")
        return

    # Step 3: Download + read + subset via obstore
    print(f"\n3. Downloading and reading EA subsets via obstore...")
    s3_store = from_url(S3_BUCKET_URL, region="us-east-1", skip_signature=True)
    t1 = time.time()
    result = read_day_ea_subset(df_day, s3_store, lat_s, lat_e, lon_s, lon_e)
    elapsed = time.time() - t1

    # Step 4: Validate result
    expected_timesteps = len(df_day) * 2
    print(f"\n4. Result validation:")
    print(f"   Shape:    {result.shape}")
    print(f"   Expected: ({expected_timesteps}, {len(lat_ea)}, {len(lon_ea)})")
    print(f"   dtype:    {result.dtype}")
    print(f"   Min:      {np.nanmin(result):.4f}")
    print(f"   Max:      {np.nanmax(result):.4f}")
    print(f"   Mean:     {np.nanmean(result):.4f}")
    n_nan = np.count_nonzero(np.isnan(result))
    print(f"   NaN:      {n_nan} / {result.size} ({100*n_nan/result.size:.1f}%)")
    print(f"   Time:     {elapsed:.1f}s ({elapsed/len(df_day):.2f}s per file)")

    shape_ok = result.shape == (expected_timesteps, len(lat_ea), len(lon_ea))
    data_ok = np.nanmax(result) > 0 and result.dtype == np.float32

    total = time.time() - t0
    print(f"\n{'=' * 60}")
    if shape_ok and data_ok:
        print(f"PASS — all checks passed ({total:.1f}s total)")
    else:
        print("FAIL —", end=" ")
        if not shape_ok:
            print(f"shape mismatch", end=" ")
        if not data_ok:
            print(f"data issue", end=" ")
        print()
    print("=" * 60)


if __name__ == "__main__":
    main()
