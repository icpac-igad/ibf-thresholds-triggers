#!/usr/bin/env python3
"""Subprocess worker: write one month to Icechunk from Parquet catalog.

Called by cmorph_parquet_vds_catalog.py's icechunk subcommand. Each invocation
is a fresh Python process, guaranteeing full memory reclamation after each
month. This avoids OOM from manifest growth during to_icechunk(append_dim).

Usage:
    python icechunk_month_worker.py \
        --catalog PATH --month-key 2010-11 --month-idx 154 \
        --coords-npz /tmp/coords.npz \
        --gcs-bucket cpc_awc --gcs-prefix cmorph_1998_2024_catalog \
        --service-account creds.json --group cmorph

Prints JSON result to stdout: {"ok": true, "n_files": 720} or
{"ok": false, "error": "message"}
"""

import argparse
import base64
import json
import sys
import warnings

warnings.filterwarnings("ignore", message="Numcodecs codecs are not in the Zarr")

S3_BUCKET = "s3://noaa-cdr-precip-cmorph-pds/"
S3_REGION = "us-east-1"


def process_month(args):
    import icechunk
    import numpy as np
    import pandas as pd
    import xarray as xr
    from obstore.store import from_url
    from obspec_utils.registry import ObjectStoreRegistry
    from virtualizarr.manifests import ManifestStore
    from virtualizarr.parsers.kerchunk.translator import manifestgroup_from_kerchunk_refs

    # Load coords
    npz = np.load(args.coords_npz)
    lat_vals = npz["lat"]
    lon_vals = npz["lon"]

    # Set up obstore registry
    s3_store = from_url(S3_BUCKET, region=S3_REGION, skip_signature=True)
    registry = ObjectStoreRegistry({S3_BUCKET: s3_store})

    # Set up icechunk
    config = icechunk.RepositoryConfig.default()
    config.set_virtual_chunk_container(
        icechunk.VirtualChunkContainer(
            S3_BUCKET, store=icechunk.s3_store(region=S3_REGION, anonymous=True),
        )
    )
    if args.local:
        storage = icechunk.local_filesystem_storage(path=args.local)
    else:
        storage = icechunk.gcs_storage(
            bucket=args.gcs_bucket, prefix=args.gcs_prefix,
            service_account_file=args.service_account,
        )

    try:
        repo = icechunk.Repository.open(storage, config=config)
    except Exception:
        repo = icechunk.Repository.create(storage, config=config)

    # Read month's refs
    df_month = pd.read_parquet(
        args.catalog,
        filters=[("month_key", "==", args.month_key), ("status", "==", "success")],
        columns=["s3_url", "datetime", "kerchunk_refs"],
    )
    df_month = df_month.sort_values("datetime").reset_index(drop=True)
    n_files = len(df_month)

    # Reconstruct VDS
    virtual_datasets = []
    for _, row in df_month.iterrows():
        refs = json.loads(row["kerchunk_refs"])
        mg = manifestgroup_from_kerchunk_refs(
            refs, skip_variables=["time", "lat", "lon", "nv"],
        )
        ms = ManifestStore(group=mg, registry=registry)
        ds = ms.to_virtual_dataset(decode_times=False)[["cmorph"]]
        t0 = np.datetime64(row["datetime"])
        t1 = t0 + np.timedelta64(30, "m")
        ds = ds.assign_coords(time=("time", [t0, t1]))
        virtual_datasets.append(ds)

    combined = xr.concat(
        virtual_datasets, dim="time",
        coords="minimal", compat="override",
    )
    combined = combined.assign_coords(
        lat=("lat", lat_vals), lon=("lon", lon_vals),
    )

    # Write to icechunk
    session = repo.writable_session("main")
    if args.month_idx == 0:
        combined.virtualize.to_icechunk(session.store, group=args.group)
    else:
        combined.virtualize.to_icechunk(
            session.store, group=args.group, append_dim="time",
        )
    session.commit(message=f"{args.month_key}: {n_files} files")

    return {"ok": True, "n_files": n_files}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--month-key", required=True)
    parser.add_argument("--month-idx", type=int, required=True)
    parser.add_argument("--coords-npz", required=True)
    parser.add_argument("--gcs-bucket", default="cpc_awc")
    parser.add_argument("--gcs-prefix", default="cmorph_1998_2024_catalog")
    parser.add_argument("--service-account", default="coiled-data-e4drr_202505.json")
    parser.add_argument("--local", default=None)
    parser.add_argument("--group", default="cmorph")
    args = parser.parse_args()

    try:
        result = process_month(args)
    except Exception as e:
        result = {"ok": False, "error": str(e)}

    # Print result as JSON to stdout (parent reads this)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
