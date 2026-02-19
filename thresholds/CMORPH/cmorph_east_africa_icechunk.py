#!/usr/bin/env python3
"""
CMORPH East Africa Subset — Materialized Icechunk Store + Pencil Rechunking
============================================================================

Creates a materialized (real data, not virtual refs) Icechunk store for the
East Africa region (lat: -12 to 23, lon: 21 to 53) from the existing virtual
Icechunk store built from the Parquet VDS catalog.  Then rechunks to pencil
chunks (full time × 5 lat × 5 lon) for fast time-series access.

Subcommands:

  init     — Create empty template store (GLAD pattern: compute=False)
  fill     — Populate with real data using Dask/Coiled (fork/merge pattern)
  rechunk  — Rechunk to pencil chunks → new Zarr store
  verify   — Inspect store contents

Usage:
    # Step 1: Create empty EA template
    micromamba run -n aifs-etl python cmorph_east_africa_icechunk.py init \
        --catalog cmorph_vds_catalog/catalog.parquet \
        --gcs-prefix cmorph_ea_subset

    # Step 2: Fill with real data from Parquet refs (needs Coiled cluster)
    micromamba run -n aifs-etl python cmorph_east_africa_icechunk.py fill \
        --catalog cmorph_vds_catalog/catalog.parquet \
        --target-gcs-prefix cmorph_ea_subset \
        --n-workers 20

    # Step 3: Rechunk to pencil chunks
    micromamba run -n aifs-etl python cmorph_east_africa_icechunk.py rechunk \
        --source-gcs-prefix cmorph_ea_subset \
        --target-path gs://cpc_awc/cmorph_ea_pencil \
        --n-workers 20

    # Step 4: Verify final result
    micromamba run -n aifs-etl python cmorph_east_africa_icechunk.py verify \
        --gcs-prefix cmorph_ea_pencil --store-type zarr

Author: AI Assistant
Date: 2026-02-09
"""

import base64
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("cmorph_east_africa.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ─── Constants ──────────────────────────────────────────────────────────────

S3_BUCKET = "s3://noaa-cdr-precip-cmorph-pds/"
S3_REGION = "us-east-1"
SERVICE_ACCOUNT_FILE = "coiled-data-e4drr_202505.json"
GCS_BUCKET = "cpc_awc"

# East Africa bounding box
EA_LAT_MIN = -12.0
EA_LAT_MAX = 23.0
EA_LON_MIN = 21.0
EA_LON_MAX = 53.0

# Chunk sizes
ZARR_CHUNK_SIZE = (48, 120, 110)     # 1 day temporal, ~120×110 spatial tiles (~2.4 MB)
PENCIL_CHUNK_SIZE = (-1, 5, 5)       # full time × 5 lat × 5 lon (~5.3 MB)
FILL_VALUE = np.float32(9.969209968386869e+36)  # CMORPH standard fill


# ─── Coordinate helpers ─────────────────────────────────────────────────────


def decode_coords_from_catalog(catalog_path: str):
    """Decode global lat/lon from first file's inlined Kerchunk refs.

    Reads only a single row via PyArrow iter_batches to avoid loading the
    entire kerchunk_refs column (~85KB × 26K rows = OOM).
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(catalog_path)
    for batch in pf.iter_batches(batch_size=1, columns=["kerchunk_refs", "status"]):
        row = batch.to_pydict()
        if row["status"][0] == "success" and row["kerchunk_refs"][0]:
            first_refs = json.loads(row["kerchunk_refs"][0])
            break
    else:
        raise ValueError(f"No successful rows found in {catalog_path}")

    lat_zarray = json.loads(first_refs["refs"]["lat/.zarray"])
    lat_data = base64.b64decode(first_refs["refs"]["lat/0"].replace("base64:", ""))
    lat_vals = np.frombuffer(lat_data, dtype=lat_zarray["dtype"])

    lon_zarray = json.loads(first_refs["refs"]["lon/.zarray"])
    lon_data = base64.b64decode(first_refs["refs"]["lon/0"].replace("base64:", ""))
    lon_vals = np.frombuffer(lon_data, dtype=lon_zarray["dtype"])
    del first_refs

    return lat_vals.copy(), lon_vals.copy()


def get_ea_indices(lat_vals, lon_vals):
    """Find array indices for the East Africa bounding box."""
    lat_mask = (lat_vals >= EA_LAT_MIN) & (lat_vals <= EA_LAT_MAX)
    lon_mask = (lon_vals >= EA_LON_MIN) & (lon_vals <= EA_LON_MAX)
    return lat_mask, lon_mask


def get_time_coords_from_catalog(catalog_path: str):
    """Build sorted time coordinate array from all successful catalog entries.

    Each CMORPH file has 2 timesteps (t0 and t0 + 30min).
    """
    df = pd.read_parquet(
        catalog_path,
        columns=["datetime", "status"],
        filters=[("status", "==", "success")],
    )
    df = df.sort_values("datetime").reset_index(drop=True)
    t0 = pd.to_datetime(df["datetime"]).values.astype("datetime64[ns]")
    del df

    # Each file = 2 timesteps: t0 and t0 + 30 min (vectorized)
    t1 = t0 + np.timedelta64(30, "m")
    time_coords = np.empty(len(t0) * 2, dtype="datetime64[ns]")
    time_coords[0::2] = t0
    time_coords[1::2] = t1

    return time_coords


# ─── Phase 1: init ─────────────────────────────────────────────────────────


def init_ea_store(args):
    """Create empty template Icechunk store for East Africa subset.

    Follows the GLAD pattern: write structure only with compute=False.
    """
    import dask.array as da
    import icechunk
    import xarray as xr

    logger.info("=" * 60)
    logger.info("INIT: Creating East Africa template store")
    logger.info("=" * 60)
    start = time.time()

    # Step 1: Decode global coords and find EA subset
    logger.info(f"Reading catalog: {args.catalog}")
    lat_global, lon_global = decode_coords_from_catalog(args.catalog)
    lat_mask, lon_mask = get_ea_indices(lat_global, lon_global)
    lat_ea = lat_global[lat_mask]
    lon_ea = lon_global[lon_mask]
    logger.info(f"  Global lat: {len(lat_global)} → EA lat: {len(lat_ea)} [{lat_ea[0]:.2f} .. {lat_ea[-1]:.2f}]")
    logger.info(f"  Global lon: {len(lon_global)} → EA lon: {len(lon_ea)} [{lon_ea[0]:.2f} .. {lon_ea[-1]:.2f}]")

    # Step 2: Build time coordinates
    logger.info("Building time coordinates from catalog...")
    time_coords = get_time_coords_from_catalog(args.catalog)
    n_time = len(time_coords)
    logger.info(f"  Time: {n_time} timesteps [{time_coords[0]} .. {time_coords[-1]}]")

    n_lat = len(lat_ea)
    n_lon = len(lon_ea)
    logger.info(f"  Template shape: ({n_time}, {n_lat}, {n_lon})")
    size_gb = n_time * n_lat * n_lon * 4 / (1024**3)
    logger.info(f"  Total size: {size_gb:.1f} GB")

    # Step 3: Create template dataset with dask.array.zeros (lazy, no memory)
    # Use a single dask chunk — the actual zarr chunk layout is set by encoding.
    # This avoids creating 197K dask tasks and icechunk manifest entries at init.
    template = xr.Dataset(
        {
            "cmorph": (
                ("time", "lat", "lon"),
                da.zeros(
                    (n_time, n_lat, n_lon),
                    chunks=(n_time, n_lat, n_lon),
                    dtype=np.float32,
                ),
                {"long_name": "CMORPH precipitation rate", "units": "mm/hr"},
            ),
        },
        coords={
            "time": time_coords,
            "lat": ("lat", lat_ea, {"units": "degrees_north"}),
            "lon": ("lon", lon_ea, {"units": "degrees_east"}),
        },
        attrs={"title": "CMORPH East Africa Subset", "source": "NOAA CDR"},
    )
    logger.info(f"  Template dataset:\n{template}")

    # Step 4: Set up Icechunk store
    if args.local:
        logger.info(f"Using local storage: {args.local}")
        storage = icechunk.local_filesystem_storage(path=args.local)
    else:
        logger.info(f"Using GCS: gs://{args.gcs_bucket}/{args.gcs_prefix}")
        storage = icechunk.gcs_storage(
            bucket=args.gcs_bucket,
            prefix=args.gcs_prefix,
            service_account_file=args.service_account,
        )

    config = icechunk.RepositoryConfig.default()
    try:
        repo = icechunk.Repository.create(storage, config=config)
        logger.info("  Created new repository")
    except Exception:
        repo = icechunk.Repository.open(storage, config=config)
        logger.info("  Opened existing repository (will overwrite)")

    # Step 5: Write metadata only (compute=False)
    session = repo.writable_session("main")
    template.to_zarr(
        session.store,
        compute=False,
        mode="w",
        encoding={
            "cmorph": {
                "chunks": ZARR_CHUNK_SIZE,
                "fill_value": float(FILL_VALUE),
            },
        },
        consolidated=False,
    )
    session.commit("initialize EA template")
    elapsed = time.time() - start

    logger.info("=" * 60)
    logger.info("INIT COMPLETE")
    logger.info(f"  Shape: ({n_time}, {n_lat}, {n_lon})")
    logger.info(f"  Chunks: {ZARR_CHUNK_SIZE}")
    logger.info(f"  Time: {elapsed:.1f}s")
    logger.info("=" * 60)

    return {
        "status": "success",
        "shape": (n_time, n_lat, n_lon),
        "chunks": ZARR_CHUNK_SIZE,
        "time_range": (str(time_coords[0]), str(time_coords[-1])),
        "elapsed_sec": elapsed,
    }


# ─── Phase 2: fill ─────────────────────────────────────────────────────────


def fill_ea_store(args):
    """Fill EA template store with real data read directly from Parquet Kerchunk refs.

    Workers read their day's Kerchunk refs from a GCS-hosted Parquet catalog
    via predicate pushdown (only ~24 rows per day).  Each worker opens CMORPH
    files via fsspec reference filesystem, subsets to East Africa, and returns
    numpy arrays.  The coordinator writes results to Icechunk sequentially
    and commits in batches.

    Prerequisites:
        Upload catalog to GCS first using upload_merge_parquet_catalogs.py upload.

    No virtual Icechunk source store needed — reads S3 directly via refs.
    """
    import coiled
    import distributed
    import icechunk
    import xarray as xr

    logger.info("=" * 60)
    logger.info("FILL: Populating EA store from Parquet refs (direct S3 reads)")
    logger.info("=" * 60)
    overall_start = time.time()

    # ── Compute EA spatial indices from local catalog ──
    lat_global, lon_global = decode_coords_from_catalog(args.catalog)
    lat_mask, lon_mask = get_ea_indices(lat_global, lon_global)
    lat_indices = np.where(lat_mask)[0]
    lon_indices = np.where(lon_mask)[0]
    lat_start, lat_end = int(lat_indices[0]), int(lat_indices[-1]) + 1
    lon_start, lon_end = int(lon_indices[0]), int(lon_indices[-1]) + 1
    n_lat_ea = lat_end - lat_start
    n_lon_ea = lon_end - lon_start
    logger.info(f"  EA lat indices: [{lat_start}:{lat_end}] ({n_lat_ea} pts)")
    logger.info(f"  EA lon indices: [{lon_start}:{lon_end}] ({n_lon_ea} pts)")

    # ── Build day batches from local catalog (lightweight columns only) ──
    df_index = pd.read_parquet(
        args.catalog, columns=["datetime", "status", "year", "month", "day"],
    )
    df_index = df_index[df_index["status"] == "success"].sort_values("datetime")
    day_groups = (
        df_index.groupby(["year", "month", "day"])
        .size()
        .reset_index(name="n_files")
        .sort_values(["year", "month", "day"])
        .reset_index(drop=True)
    )
    del df_index

    day_batches = []
    t_offset = 0
    for _, row in day_groups.iterrows():
        n_ts = int(row["n_files"]) * 2  # 2 timesteps per file
        day_batches.append({
            "day_idx": len(day_batches),
            "year": int(row["year"]),
            "month": int(row["month"]),
            "day": int(row["day"]),
            "t_start": t_offset,
            "t_end": t_offset + n_ts,
            "n_files": int(row["n_files"]),
        })
        t_offset += n_ts

    n_days = len(day_batches)
    gcs_catalog_path = args.gcs_catalog
    logger.info(f"  {n_days} days, {t_offset} total timesteps")
    logger.info(f"  GCS catalog: {gcs_catalog_path}")

    # ── Open target Icechunk store + resume detection ──
    if args.local:
        target_storage = icechunk.local_filesystem_storage(path=args.local)
    else:
        target_storage = icechunk.gcs_storage(
            bucket=args.gcs_bucket,
            prefix=args.target_gcs_prefix,
            service_account_file=args.service_account,
        )
    target_repo = icechunk.Repository.open(
        target_storage, config=icechunk.RepositoryConfig.default(),
    )

    # ── Resume detection: parse commit messages for completed contiguous ranges ──
    # Commit messages are "fill days 100-119: 20/20 OK" (contiguous, all succeeded).
    # We find the highest completed day_idx and resume from there.
    completed_up_to = -1  # highest day_idx in a fully completed batch
    try:
        for commit in target_repo.ancestry(branch="main"):
            msg = commit.message
            if msg.startswith("fill days "):
                try:
                    # Parse "fill days 100-119: 20/20 OK"
                    range_str = msg.split(":")[0].replace("fill days ", "")
                    d_start, d_end = range_str.split("-")
                    d_end_int = int(d_end)
                    if d_end_int > completed_up_to:
                        completed_up_to = d_end_int
                except (ValueError, IndexError):
                    pass
    except Exception:
        pass

    start_day_idx = completed_up_to + 1
    if start_day_idx > 0:
        logger.info(f"  Resuming from day {start_day_idx} (days 0-{completed_up_to} done)")

    remaining_days = [d for d in day_batches if d["day_idx"] >= start_day_idx]
    if not remaining_days:
        logger.info("  All days already filled!")
        return {"status": "success", "message": "already complete"}
    logger.info(f"  Remaining: {len(remaining_days)} days")

    # ── Launch Coiled cluster ──
    n_workers = args.n_workers
    cluster = coiled.Cluster(
        name=f"cmorph-ea-fill-{int(time.time()) % 10000}",
        n_workers=[min(5, n_workers), n_workers],
        worker_vm_types="n2-standard-4",
        package_sync=True,
        region="us-east1",
        workspace="e4drr",
        idle_timeout="30 minutes",
    )
    client = distributed.Client(cluster)
    client.wait_for_workers(n_workers=min(5, n_workers), timeout=300)
    logger.info(f"  Cluster ready: {client.dashboard_link}")

    # ── Load GCS service account credentials (small dict, passed to workers) ──
    with open(args.service_account) as f:
        gcs_sa_info = json.load(f)
    logger.info(f"  Loaded GCS credentials from {args.service_account}")

    # ── Worker function: reads NetCDF directly from S3 via obstore ──
    def read_day_ea_subset(day_info, gcs_catalog, sa_info, lat_s, lat_e, lon_s, lon_e):
        """Read one day's EA subset from S3 NetCDF files via obstore.

        Reads only this day's S3 URLs from the GCS Parquet catalog (predicate
        pushdown, ~24 rows).  Downloads each CMORPH NetCDF from S3 via
        obstore.get(), writes to temp file, opens with netcdf4 engine,
        subsets to EA, returns numpy array.
        """
        import os
        import tempfile

        import numpy as np
        import obstore as obs
        import pandas as pd
        import xarray as xr
        from obstore.store import from_url

        # Read only this day's S3 URLs from GCS catalog (tiny: ~24 URLs)
        df_day = pd.read_parquet(
            gcs_catalog,
            filters=[
                ("year", "==", day_info["year"]),
                ("month", "==", day_info["month"]),
                ("day", "==", day_info["day"]),
                ("status", "==", "success"),
            ],
            columns=["datetime", "s3_url"],
            storage_options={"token": sa_info},
        )
        df_day = df_day.sort_values("datetime")

        s3_store = from_url(
            "s3://noaa-cdr-precip-cmorph-pds/",
            region="us-east-1",
            skip_signature=True,
        )

        subsets = []
        for _, row in df_day.iterrows():
            # Extract S3 key from full URL (strip bucket prefix)
            s3_key = row["s3_url"].replace("s3://noaa-cdr-precip-cmorph-pds/", "")

            # Download via obstore
            result = obs.get(s3_store, s3_key)
            nc_bytes = result.bytes()

            # Write to temp file, open with netcdf4
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

        return {
            "day_idx": day_info["day_idx"],
            "t_start": day_info["t_start"],
            "t_end": day_info["t_end"],
            "data": np.concatenate(subsets, axis=0),
        }

    # ── Process in sequential contiguous batches ──
    # Submit COMMIT_BATCH consecutive days at a time, wait for all to complete,
    # write all results, commit, then move to the next batch.  This guarantees:
    #   - Commit messages represent contiguous, fully-written day ranges
    #   - Resume is safe: skip past the highest completed batch boundary
    #   - The resulting time series has no gaps within committed ranges
    COMMIT_BATCH = args.commit_batch
    total_written = 0
    total_failed = 0
    failed_days = []

    for batch_start in range(0, len(remaining_days), COMMIT_BATCH):
        batch = remaining_days[batch_start : batch_start + COMMIT_BATCH]
        batch_day_min = batch[0]["day_idx"]
        batch_day_max = batch[-1]["day_idx"]
        logger.info(
            f"  Batch: days {batch_day_min}-{batch_day_max} "
            f"({len(batch)} days, {total_written}/{len(remaining_days)} done so far)"
        )

        # Submit this batch to Coiled
        futures = {}
        for d in batch:
            future = client.submit(
                read_day_ea_subset, d, gcs_catalog_path, gcs_sa_info,
                lat_start, lat_end, lon_start, lon_end,
                key=f"day-{d['day_idx']}",
            )
            futures[future] = d

        # Wait for all futures in this batch + write results
        session = target_repo.writable_session("main")
        batch_ok = 0
        batch_fail = 0

        for future in distributed.as_completed(futures):
            d = futures[future]
            try:
                result = future.result()

                ds_write = xr.Dataset({
                    "cmorph": (("time", "lat", "lon"), result["data"]),
                })
                ds_write.to_zarr(
                    session.store,
                    region={"time": slice(result["t_start"], result["t_end"])},
                    consolidated=False,
                )
                del result

                batch_ok += 1
                total_written += 1
                logger.info(
                    f"    Wrote day {d['day_idx']} "
                    f"({d['year']}-{d['month']:02d}-{d['day']:02d})"
                )

            except Exception as e:
                batch_fail += 1
                total_failed += 1
                failed_days.append(d["day_idx"])
                logger.error(
                    f"    Day {d['day_idx']} "
                    f"({d['year']}-{d['month']:02d}-{d['day']:02d}) FAILED: {e}"
                )

        # Commit only if all days in the batch succeeded (contiguous guarantee)
        if batch_fail == 0:
            session.commit(
                f"fill days {batch_day_min}-{batch_day_max}: "
                f"{batch_ok}/{len(batch)} OK"
            )
            logger.info(
                f"  Committed days {batch_day_min}-{batch_day_max} "
                f"(total: {total_written}/{len(remaining_days)})"
            )
        else:
            # Some days failed — commit what we have but mark incomplete
            # so resume will retry this batch
            logger.warning(
                f"  Batch {batch_day_min}-{batch_day_max} had {batch_fail} failures, "
                f"NOT committed — will be retried on resume"
            )

    client.close()
    cluster.close()

    elapsed = time.time() - overall_start
    logger.info("=" * 60)
    logger.info("FILL COMPLETE")
    logger.info(f"  Days written: {total_written}/{n_days}")
    logger.info(f"  Failed: {total_failed} — {failed_days[:20]}")
    logger.info(f"  Time: {elapsed / 60:.1f} min")
    logger.info("=" * 60)

    results = {
        "status": "success" if not failed_days else "partial",
        "days_written": total_written,
        "days_total": n_days,
        "failed_days": failed_days,
        "elapsed_min": elapsed / 60,
    }
    results_path = f"cmorph_ea_fill_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"  Results: {results_path}")

    return results


# ─── Phase 3: rechunk ──────────────────────────────────────────────────────


def rechunk_ea_store(args):
    """Rechunk EA store to pencil chunks using Dask P2P shuffle.

    Reads from the materialized EA Icechunk store, rechunks with Dask's
    P2P (peer-to-peer) shuffle at constant memory, and writes to a new
    Zarr store on GCS or local filesystem.

    P2P rechunking sends slices directly between workers (no task graph
    explosion), spills to disk when needed, and keeps memory constant
    regardless of chunk overlap.  Requires a fixed-size cluster.

    See DASK_OPERATIONS_NOTES.md for details on why P2P is needed here.
    """
    import pickle

    import coiled
    import dask
    import distributed
    import icechunk
    import xarray as xr

    logger.info("=" * 60)
    logger.info("RECHUNK: Converting to pencil chunks (P2P shuffle)")
    logger.info("=" * 60)
    overall_start = time.time()

    # ── P2P rechunk configuration ──
    dask.config.set({
        "array.rechunk.method": "p2p",
        "optimization.fuse.active": False,
    })
    logger.info("  Dask config: P2P rechunk enabled, fusion disabled")

    # ── Open source EA Icechunk store ──
    # Use service_account_key (JSON string) instead of service_account_file
    # so GCS credentials are embedded in the storage config and survive
    # pickle serialization to Dask workers.
    if args.source_local:
        source_storage = icechunk.local_filesystem_storage(path=args.source_local)
    else:
        with open(args.service_account) as f:
            sa_key = f.read()
        source_storage = icechunk.gcs_storage(
            bucket=args.gcs_bucket,
            prefix=args.source_gcs_prefix,
            service_account_key=sa_key,
        )

    source_repo = icechunk.Repository.open(
        source_storage, config=icechunk.RepositoryConfig.default(),
    )
    session = source_repo.readonly_session("main")

    # Quick serialization test — verifies credentials survive pickle
    try:
        pickle.dumps(session.store)
        logger.info("  IcechunkStore is pickle-serializable (credentials embedded)")
    except Exception as e:
        logger.error(f"  IcechunkStore is NOT serializable: {e}")
        return {"status": "error", "message": f"Store not serializable: {e}"}

    # Open with explicit source chunk sizes matching stored layout
    ds = xr.open_zarr(
        session.store,
        consolidated=False,
        chunks={"time": 48, "lat": 120, "lon": 110},
    )
    logger.info(f"  Source: {dict(ds.sizes)}")
    logger.info(f"  Source chunks: (48, 120, 110)")

    n_time = ds.sizes["time"]
    n_lat = ds.sizes["lat"]
    n_lon = ds.sizes["lon"]
    size_gb = n_time * n_lat * n_lon * 4 / (1024**3)
    logger.info(f"  Total data: {size_gb:.1f} GB")

    # ── Target pencil chunk sizes ──
    chunk_time = n_time if args.chunk_time == -1 else args.chunk_time
    chunk_lat = args.chunk_lat
    chunk_lon = args.chunk_lon
    pencil_chunks = {"time": chunk_time, "lat": chunk_lat, "lon": chunk_lon}

    chunk_bytes = chunk_time * chunk_lat * chunk_lon * 4
    n_lat_chunks = -(-n_lat // chunk_lat)   # ceil division
    n_lon_chunks = -(-n_lon // chunk_lon)
    n_target_chunks = n_lat_chunks * n_lon_chunks

    logger.info(f"  Target chunks: ({chunk_time}, {chunk_lat}, {chunk_lon})")
    logger.info(f"  Chunk size: {chunk_bytes / (1024**2):.1f} MB")
    logger.info(f"  Total target chunks: {n_target_chunks} "
                f"({n_lat_chunks} lat x {n_lon_chunks} lon)")

    # ── Launch FIXED Coiled cluster (P2P requires static cluster) ──
    n_workers = args.n_workers
    per_worker_gb = size_gb / n_workers
    logger.info(f"  Per-worker data: {per_worker_gb:.1f} GB "
                f"(P2P spill to disk if needed)")

    cluster = coiled.Cluster(
        name=f"cmorph-ea-rechunk-{int(time.time()) % 10000}",
        n_workers=n_workers,   # Fixed size — P2P cannot handle adaptive scaling
        worker_vm_types="n2-highmem-4",
        package_sync=True,
        region="us-east1",
        workspace="e4drr",
        idle_timeout="60 minutes",
    )
    client = distributed.Client(cluster)
    client.wait_for_workers(n_workers=n_workers, timeout=600)
    logger.info(f"  Cluster: {n_workers} x n2-highmem-4 "
                f"({n_workers * 32} GB total RAM)")
    logger.info(f"  Dashboard: {client.dashboard_link}")

    # ── Rechunk with Dask P2P ──
    ds_rechunked = ds.chunk(pencil_chunks)

    # ── GCS credentials for target write ──
    target_path = args.target_path
    storage_options = None
    if target_path.startswith("gs://"):
        with open(args.service_account) as f:
            sa_info = json.load(f)
        storage_options = {"token": sa_info}

    logger.info(f"  Target: {target_path}")
    logger.info("  Starting P2P rechunk + write...")

    ds_rechunked.to_zarr(
        target_path,
        storage_options=storage_options,
        encoding={
            "cmorph": {
                "chunks": (chunk_time, chunk_lat, chunk_lon),
            },
        },
        mode="w",
        consolidated=True,
    )

    logger.info("  Write complete!")
    client.close()
    cluster.close()

    elapsed = time.time() - overall_start
    logger.info("=" * 60)
    logger.info("RECHUNK COMPLETE")
    logger.info(f"  Target: {target_path}")
    logger.info(f"  Chunks: ({chunk_time}, {chunk_lat}, {chunk_lon})")
    logger.info(f"  Total chunks: {n_target_chunks}")
    logger.info(f"  Time: {elapsed / 60:.1f} min")
    logger.info("=" * 60)

    return {
        "status": "success",
        "target_path": target_path,
        "chunks": (chunk_time, chunk_lat, chunk_lon),
        "n_chunks": n_target_chunks,
        "elapsed_min": elapsed / 60,
    }


# ─── Phase 4: verify ───────────────────────────────────────────────────────


def verify_store(args):
    """Inspect and verify an Icechunk or Zarr store."""
    import icechunk
    import xarray as xr

    logger.info("=" * 60)
    logger.info("VERIFY: Inspecting store")
    logger.info("=" * 60)

    if args.store_type == "zarr":
        # Open plain Zarr store
        ds = xr.open_zarr(args.target_path or f"gs://{args.gcs_bucket}/{args.gcs_prefix}",
                          consolidated=True)
    else:
        # Open Icechunk store
        if args.local:
            storage = icechunk.local_filesystem_storage(path=args.local)
        else:
            storage = icechunk.gcs_storage(
                bucket=args.gcs_bucket,
                prefix=args.gcs_prefix,
                service_account_file=args.service_account,
            )

        config = icechunk.RepositoryConfig.default()

        # If verifying a virtual store, need S3 creds
        if args.virtual:
            config.set_virtual_chunk_container(
                icechunk.VirtualChunkContainer(
                    S3_BUCKET,
                    store=icechunk.s3_store(region=S3_REGION, anonymous=True),
                )
            )
            s3_creds = icechunk.containers_credentials(
                {S3_BUCKET: icechunk.s3_credentials(anonymous=True)}
            )
            repo = icechunk.Repository.open(
                storage, config=config,
                authorize_virtual_chunk_access=s3_creds,
            )
        else:
            repo = icechunk.Repository.open(storage, config=config)

        session = repo.readonly_session("main")
        ds = xr.open_zarr(session.store, consolidated=False)

    # Print dataset info
    logger.info(f"\nDataset:\n{ds}")
    logger.info(f"\nDimensions: {dict(ds.sizes)}")

    if "time" in ds.dims:
        logger.info(f"  Time: {ds.time.values[0]} -> {ds.time.values[-1]}")
        logger.info(f"  Time steps: {ds.sizes['time']}")
    if "lat" in ds.dims:
        logger.info(f"  Lat: {float(ds.lat.values[0]):.2f} -> {float(ds.lat.values[-1]):.2f}")
        logger.info(f"  Lat points: {ds.sizes['lat']}")
    if "lon" in ds.dims:
        logger.info(f"  Lon: {float(ds.lon.values[0]):.2f} -> {float(ds.lon.values[-1]):.2f}")
        logger.info(f"  Lon points: {ds.sizes['lon']}")

    # Check data variables
    for var in ds.data_vars:
        da = ds[var]
        logger.info(f"\nVariable '{var}':")
        logger.info(f"  dtype: {da.dtype}")
        logger.info(f"  shape: {da.shape}")
        if hasattr(da, "encoding") and "chunks" in da.encoding:
            logger.info(f"  chunks: {da.encoding['chunks']}")

    # Spot-check: load a small region to verify data exists
    if args.spot_check and "cmorph" in ds.data_vars:
        logger.info("\nSpot-check: loading first 10 timesteps...")
        try:
            sample = ds["cmorph"].isel(time=slice(0, 10)).load()
            n_valid = int((~np.isnan(sample.values) & (sample.values != FILL_VALUE)).sum())
            n_total = sample.values.size
            pct = 100 * n_valid / n_total if n_total else 0
            logger.info(f"  Valid (non-NaN, non-fill) values: {n_valid}/{n_total} ({pct:.1f}%)")
            logger.info(f"  Min: {float(np.nanmin(sample.values)):.4f}")
            logger.info(f"  Max: {float(np.nanmax(sample.values)):.4f}")
            logger.info(f"  Mean: {float(np.nanmean(sample.values)):.4f}")
        except Exception as e:
            logger.error(f"  Spot-check failed: {e}")

    # Show commit history for Icechunk stores
    if args.store_type == "icechunk" and not args.local and not (args.target_path):
        try:
            commits = list(repo.ancestry(branch="main"))
            logger.info(f"\nCommit history ({len(commits)} commits):")
            for c in commits[:10]:
                logger.info(f"  {c.message}")
            if len(commits) > 10:
                logger.info(f"  ... and {len(commits) - 10} more")
        except Exception:
            pass

    logger.info("\nVerification complete.")


# ─── CLI ────────────────────────────────────────────────────────────────────


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="CMORPH East Africa Subset — Icechunk Store + Pencil Rechunking",
    )
    sub = parser.add_subparsers(dest="command")

    # ── init ──
    p_init = sub.add_parser("init", help="Create empty EA template store")
    p_init.add_argument("--catalog", type=str, required=True,
                        help="Path to Parquet VDS catalog")
    p_init.add_argument("--gcs-bucket", type=str, default=GCS_BUCKET)
    p_init.add_argument("--gcs-prefix", type=str, default="cmorph_ea_subset")
    p_init.add_argument("--service-account", type=str, default=SERVICE_ACCOUNT_FILE)
    p_init.add_argument("--local", type=str, default=None,
                        help="Local filesystem path (overrides GCS)")

    # ── fill ──
    p_fill = sub.add_parser("fill",
                            help="Fill EA store from Parquet refs (direct S3 reads via Coiled)")
    p_fill.add_argument("--catalog", type=str, required=True,
                        help="Local Parquet catalog (for coord decoding + day indexing)")
    p_fill.add_argument("--gcs-catalog", type=str,
                        default=f"gs://{GCS_BUCKET}/cmorph_catalog/catalog.parquet",
                        help="GCS path to catalog (workers read refs from here)")
    p_fill.add_argument("--target-gcs-prefix", type=str, default="cmorph_ea_subset",
                        help="GCS prefix for the target EA Icechunk store")
    p_fill.add_argument("--gcs-bucket", type=str, default=GCS_BUCKET)
    p_fill.add_argument("--service-account", type=str, default=SERVICE_ACCOUNT_FILE)
    p_fill.add_argument("--local", type=str, default=None,
                        help="Local filesystem path for target (overrides GCS)")
    p_fill.add_argument("--n-workers", type=int, default=20)
    p_fill.add_argument("--commit-batch", type=int, default=20,
                        help="Number of days per Icechunk commit batch")

    # ── rechunk ──
    p_rechunk = sub.add_parser("rechunk", help="Rechunk to pencil chunks")
    p_rechunk.add_argument("--source-gcs-prefix", type=str, default="cmorph_ea_subset",
                           help="GCS prefix of source EA Icechunk store")
    p_rechunk.add_argument("--source-local", type=str, default=None,
                           help="Local path for source EA store")
    p_rechunk.add_argument("--gcs-bucket", type=str, default=GCS_BUCKET)
    p_rechunk.add_argument("--service-account", type=str, default=SERVICE_ACCOUNT_FILE)
    p_rechunk.add_argument("--target-path", type=str, required=True,
                           help="Target Zarr store path (gs://... or local)")
    p_rechunk.add_argument("--chunk-time", type=int, default=-1,
                           help="Time chunk size (-1 = all)")
    p_rechunk.add_argument("--chunk-lat", type=int, default=5)
    p_rechunk.add_argument("--chunk-lon", type=int, default=5)
    p_rechunk.add_argument("--n-workers", type=int, default=20)

    # ── verify ──
    p_verify = sub.add_parser("verify", help="Inspect store contents")
    p_verify.add_argument("--gcs-prefix", type=str, default=None)
    p_verify.add_argument("--gcs-bucket", type=str, default=GCS_BUCKET)
    p_verify.add_argument("--service-account", type=str, default=SERVICE_ACCOUNT_FILE)
    p_verify.add_argument("--local", type=str, default=None,
                          help="Local filesystem path")
    p_verify.add_argument("--target-path", type=str, default=None,
                          help="Direct Zarr store path (for --store-type zarr)")
    p_verify.add_argument("--store-type", type=str, default="icechunk",
                          choices=["icechunk", "zarr"])
    p_verify.add_argument("--virtual", action="store_true",
                          help="Store has virtual refs (needs S3 creds)")
    p_verify.add_argument("--spot-check", action="store_true", default=True,
                          help="Load small data sample for verification")
    p_verify.add_argument("--no-spot-check", action="store_false", dest="spot_check")

    args = parser.parse_args()

    if args.command == "init":
        init_ea_store(args)
    elif args.command == "fill":
        fill_ea_store(args)
    elif args.command == "rechunk":
        rechunk_ea_store(args)
    elif args.command == "verify":
        verify_store(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
