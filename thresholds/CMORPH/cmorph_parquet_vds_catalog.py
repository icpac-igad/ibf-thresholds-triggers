#!/usr/bin/env python3
"""
CMORPH Parquet VDS Catalog
==========================

Builds a single Parquet catalog of CMORPH virtual dataset references.
Each row is one NetCDF file with: s3_url, filename, datetime, and the
Kerchunk JSON refs — everything needed for the next Icechunk concat stage.

Coiled workers virtualize files and return complete rows.  The coordinator
streams each batch into ONE Parquet file via PyArrow ParquetWriter.
Nothing accumulates in RAM — the writer flushes each batch to disk.

Subcommands:

  catalog   — Discover S3 files, virtualize on Coiled, write single Parquet.
  info      — Show catalog statistics.

Usage:
    # Build catalog 1998-2002 with 10 Coiled workers
    micromamba run -n aifs-etl python cmorph_parquet_vds_catalog.py catalog \
        --start-year 1998 --end-year 2002 --n-workers 10

    # Test with 200 files
    micromamba run -n aifs-etl python cmorph_parquet_vds_catalog.py catalog \
        --start-year 2020 --end-year 2020 --max-files 200 --n-workers 5

    # Listing-only (no Coiled)
    micromamba run -n aifs-etl python cmorph_parquet_vds_catalog.py catalog \
        --start-year 1998 --end-year 2024 --lite

Author: AI Assistant
Date: 2026-02-08
"""

import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fsspec
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("cmorph_parquet_catalog.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

S3_BUCKET = "s3://noaa-cdr-precip-cmorph-pds/"
S3_REGION = "us-east-1"
SERVICE_ACCOUNT_FILE = "coiled-data-e4drr_202505.json"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CATALOG_DIR = str(SCRIPT_DIR / "cmorph_vds_catalog")

# Schema for the single Parquet catalog file
CATALOG_SCHEMA = pa.schema([
    ("s3_url", pa.string()),
    ("filename", pa.string()),
    ("datetime", pa.timestamp("us")),
    ("year", pa.int32()),
    ("month", pa.int32()),
    ("day", pa.int32()),
    ("hour", pa.int32()),
    ("minute", pa.int32()),
    ("month_key", pa.string()),
    ("status", pa.string()),
    ("kerchunk_refs", pa.string()),
])


# ─── Datetime parsing ────────────────────────────────────────────────────────


def parse_cmorph_datetime(s3_url: str):
    """
    Parse datetime from CMORPH filename.

    Format: CMORPH_V1.0_ADJ_8km-30min_YYYYMMDDSS.nc
    SS = half-hour slot 00-47 → hour = SS // 2, minute = (SS % 2) * 30
    """
    match = re.search(r"(\d{10})\.nc$", s3_url)
    if not match:
        return None
    ts = match.group(1)
    year, month, day = int(ts[0:4]), int(ts[4:6]), int(ts[6:8])
    slot = int(ts[8:10])
    hour, minute = slot // 2, (slot % 2) * 30
    try:
        return datetime(year, month, day, hour, minute)
    except ValueError:
        return None


# ─── File discovery ───────────────────────────────────────────────────────────


def discover_files(
    start_year: int, end_year: int,
    start_month: int = 1, end_month: int = 12,
    max_files: Optional[int] = None,
) -> List[str]:
    """List all CMORPH S3 URLs for the date range, sorted."""
    logger.info(f"Discovering files: {start_year}-{start_month:02d} to {end_year}-{end_month:02d}")
    fs = fsspec.filesystem("s3", anon=True)
    all_urls = []

    y, m = start_year, start_month
    while (y < end_year) or (y == end_year and m <= end_month):
        pattern = f"noaa-cdr-precip-cmorph-pds/data/30min/8km/{y}/{m:02d}/**/*.nc"
        try:
            files = fs.glob(pattern)
            urls = [f"s3://{f}" for f in files]
            all_urls.extend(urls)
            logger.info(f"  {y}-{m:02d}: {len(urls)} files")
        except Exception as e:
            logger.warning(f"  {y}-{m:02d}: Error - {e}")
        m += 1
        if m > 12:
            m, y = 1, y + 1

    # Sort by the 10-digit timestamp in filename
    all_urls.sort(key=lambda u: re.search(r"(\d{10})\.nc$", u).group(1)
                  if re.search(r"(\d{10})\.nc$", u) else u)

    if max_files and len(all_urls) > max_files:
        all_urls = all_urls[:max_files]
        logger.info(f"  Limited to first {max_files} files")

    logger.info(f"Discovered {len(all_urls)} files total")
    return all_urls


# ─── Coiled worker function ──────────────────────────────────────────────────


def virtualize_batch(args: Tuple) -> Dict[str, Any]:
    """
    Coiled worker: virtualize a batch of NetCDF files.

    Returns COMPLETE rows ready for Parquet — s3_url, filename, datetime
    components, status, and kerchunk_refs JSON.  The coordinator writes
    these straight to disk with zero processing.
    """
    batch_idx, urls, bucket, region = args
    import time as _time
    import re as _re
    from pathlib import PurePosixPath
    from datetime import datetime as _dt

    try:
        start = _time.time()

        from virtualizarr import open_virtual_dataset
        from virtualizarr.parsers import HDFParser
        from virtualizarr.registry import ObjectStoreRegistry
        from obstore.store import from_url
        import json as _json

        store = from_url(bucket, region=region, skip_signature=True)
        registry = ObjectStoreRegistry({bucket: store})
        parser = HDFParser()

        rows = []
        for url in urls:
            # Parse datetime from filename
            fname = PurePosixPath(url).name
            match = _re.search(r"(\d{10})\.nc$", url)
            if match:
                ts = match.group(1)
                yr, mo, dy = int(ts[0:4]), int(ts[4:6]), int(ts[6:8])
                slot = int(ts[8:10])
                hr, mn = slot // 2, (slot % 2) * 30
                dt_str = _dt(yr, mo, dy, hr, mn).isoformat()
            else:
                yr = mo = dy = hr = mn = 0
                dt_str = None

            try:
                vds = open_virtual_dataset(url=url, parser=parser, registry=registry)
                refs = vds.virtualize.to_kerchunk(format="dict")
                rows.append({
                    "s3_url": url,
                    "filename": fname,
                    "datetime": dt_str,
                    "year": yr, "month": mo, "day": dy,
                    "hour": hr, "minute": mn,
                    "month_key": f"{yr}-{mo:02d}",
                    "status": "success",
                    "kerchunk_refs": _json.dumps(refs),
                })
            except Exception as e:
                rows.append({
                    "s3_url": url,
                    "filename": fname,
                    "datetime": dt_str,
                    "year": yr, "month": mo, "day": dy,
                    "hour": hr, "minute": mn,
                    "month_key": f"{yr}-{mo:02d}",
                    "status": f"error: {e}",
                    "kerchunk_refs": None,
                })

        elapsed = _time.time() - start
        return {"batch_idx": batch_idx, "status": "success",
                "rows": rows, "elapsed_sec": elapsed}

    except Exception as e:
        import traceback
        return {"batch_idx": batch_idx, "status": "error",
                "error": str(e), "traceback": traceback.format_exc(), "rows": []}


# ─── Phase: Build catalog ────────────────────────────────────────────────────


def build_catalog(
    start_year: int, end_year: int,
    start_month: int = 1, end_month: int = 12,
    n_workers: int = 10, batch_size: int = 100,
    max_files: Optional[int] = None,
    lite: bool = False,
    catalog_dir: str = DEFAULT_CATALOG_DIR,
) -> str:
    """
    Build a single Parquet VDS catalog.

    Workers return complete rows.  Coordinator streams each batch into ONE
    Parquet file via PyArrow ParquetWriter — nothing stays in RAM.
    """
    catalog_path = Path(catalog_dir)
    catalog_path.mkdir(parents=True, exist_ok=True)
    out_path = str(catalog_path / "catalog.parquet")
    overall_start = time.time()

    # Step 1: Discover files
    urls = discover_files(start_year, end_year, start_month, end_month, max_files)
    if not urls:
        logger.error("No files found!")
        return None

    # ── Lite mode: just write URLs + datetime, no Coiled ──
    if lite:
        rows = []
        for url in urls:
            dt = parse_cmorph_datetime(url)
            if dt:
                rows.append({
                    "s3_url": url, "filename": Path(url).name,
                    "datetime": dt, "year": dt.year, "month": dt.month,
                    "day": dt.day, "hour": dt.hour, "minute": dt.minute,
                    "month_key": f"{dt.year}-{dt.month:02d}",
                    "status": "pending", "kerchunk_refs": None,
                })
        df = pd.DataFrame(rows).sort_values("datetime").reset_index(drop=True)
        df.to_parquet(out_path, engine="pyarrow", compression="zstd")
        logger.info(f"Lite catalog saved: {out_path} ({len(df)} rows)")
        return out_path

    # ── Full mode: Coiled virtualization → stream to single Parquet ──
    import coiled
    from dask.distributed import Client, as_completed

    batches = [urls[i : i + batch_size] for i in range(0, len(urls), batch_size)]
    logger.info(f"Submitting {len(batches)} batches ({batch_size} files/batch) to {n_workers} workers")

    cluster = coiled.Cluster(
        name=f"cmorph-catalog-{int(time.time()) % 10000}",
        n_workers=n_workers, worker_vm_types="n2-standard-2",
        package_sync=True, region="us-east1",
        workspace="e4drr", idle_timeout="30 minutes",
    )
    client = Client(cluster)
    client.wait_for_workers(n_workers=n_workers, timeout=300)
    logger.info(f"Cluster ready: {client.dashboard_link}")

    task_args = [(i, batch, S3_BUCKET, S3_REGION) for i, batch in enumerate(batches)]
    futures = client.map(virtualize_batch, task_args)

    # Open ONE ParquetWriter — stream every batch into it
    writer = pq.ParquetWriter(out_path, CATALOG_SCHEMA, compression="zstd")
    completed = 0
    total_rows = 0
    n_success = 0
    n_error = 0

    for future in as_completed(futures):
        completed += 1
        try:
            result = future.result()
            rows = result.get("rows", [])
            if rows:
                # Convert to PyArrow table and write immediately
                table = pa.table({
                    "s3_url": [r["s3_url"] for r in rows],
                    "filename": [r["filename"] for r in rows],
                    "datetime": [pd.Timestamp(r["datetime"]) if r["datetime"] else None for r in rows],
                    "year": [r["year"] for r in rows],
                    "month": [r["month"] for r in rows],
                    "day": [r["day"] for r in rows],
                    "hour": [r["hour"] for r in rows],
                    "minute": [r["minute"] for r in rows],
                    "month_key": [r["month_key"] for r in rows],
                    "status": [r["status"] for r in rows],
                    "kerchunk_refs": [r["kerchunk_refs"] for r in rows],
                }, schema=CATALOG_SCHEMA)

                writer.write_table(table)
                total_rows += len(rows)

                ok = sum(1 for r in rows if r["status"] == "success")
                n_success += ok
                n_error += len(rows) - ok

                # Free memory
                del table, rows, result

                logger.info(
                    f"  [{completed}/{len(batches)}] {ok}/{batch_size} ok — "
                    f"total {total_rows} rows written"
                )
            else:
                logger.error(
                    f"  [{completed}/{len(batches)}] FAILED — "
                    f"{result.get('error', 'no rows')}"
                )
        except Exception as e:
            logger.error(f"  [{completed}/{len(batches)}] Future error: {e}")

    writer.close()
    logger.info("Shutting down Coiled cluster...")
    client.close()
    cluster.close()

    total_time = time.time() - overall_start
    logger.info("=" * 60)
    logger.info("CATALOG BUILD COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  File:        {out_path}")
    logger.info(f"  Total rows:  {total_rows}")
    logger.info(f"  Success:     {n_success}")
    logger.info(f"  Errors:      {n_error}")
    logger.info(f"  Time:        {total_time / 60:.1f} min")

    # Summary JSON
    summary = {
        "catalog_path": out_path,
        "total_rows": total_rows, "success": n_success, "errors": n_error,
        "build_time_min": total_time / 60,
        "timestamp": datetime.now().isoformat(),
    }
    summary_path = str(catalog_path / "catalog_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"  Summary:     {summary_path}")

    return out_path


# ─── CLI ──────────────────────────────────────────────────────────────────────


def main():
    import argparse

    parser = argparse.ArgumentParser(description="CMORPH Parquet VDS Catalog")
    sub = parser.add_subparsers(dest="command")

    cat = sub.add_parser("catalog", help="Build Parquet catalog from S3")
    cat.add_argument("--start-year", type=int, default=1998)
    cat.add_argument("--end-year", type=int, default=2024)
    cat.add_argument("--start-month", type=int, default=1)
    cat.add_argument("--end-month", type=int, default=12)
    cat.add_argument("--n-workers", type=int, default=10)
    cat.add_argument("--batch-size", type=int, default=100)
    cat.add_argument("--max-files", type=int, default=None)
    cat.add_argument("--lite", action="store_true")
    cat.add_argument("--catalog-dir", type=str, default=DEFAULT_CATALOG_DIR)

    info = sub.add_parser("info", help="Show catalog stats")
    info.add_argument("--catalog", type=str, required=True)

    args = parser.parse_args()

    if args.command == "catalog":
        build_catalog(
            start_year=args.start_year, end_year=args.end_year,
            start_month=args.start_month, end_month=args.end_month,
            n_workers=args.n_workers, batch_size=args.batch_size,
            max_files=args.max_files, lite=args.lite,
            catalog_dir=args.catalog_dir,
        )
    elif args.command == "info":
        # Read only lightweight columns for info
        df = pd.read_parquet(args.catalog, columns=[
            "s3_url", "datetime", "year", "month_key", "status",
        ])
        print(f"Catalog: {args.catalog}")
        print(f"  Rows:       {len(df)}")
        print(f"  Time range: {df['datetime'].min()} to {df['datetime'].max()}")
        print(f"  Years:      {sorted(df['year'].unique())}")
        print(f"  Months:     {df['month_key'].nunique()} unique")
        print(f"  Status:     {df['status'].value_counts().to_dict()}")
        # Check file size on disk
        import os
        size_mb = os.path.getsize(args.catalog) / (1024 * 1024)
        print(f"  File size:  {size_mb:.1f} MB")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
