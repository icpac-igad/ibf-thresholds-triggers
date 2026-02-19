#!/usr/bin/env python3
"""Merge and upload CMORPH Parquet VDS catalogs.

Subcommands:

  merge    — Merge multiple Parquet catalogs into one (ZSTD compression).
  upload   — Upload a local Parquet catalog to GCS.
  verify   — Verify catalog completeness (year/month file counts).

Usage:
    # Merge two partial catalogs
    micromamba run -n aifs-etl python upload_merge_parquet_catalogs.py merge \
        --inputs catalog_1998_2000.parquet catalog_2001_2024.parquet \
        --output catalog.parquet

    # Upload catalog to GCS (for Coiled workers to read)
    micromamba run -n aifs-etl python upload_merge_parquet_catalogs.py upload \
        --input cmorph_vds_catalog/catalog.parquet \
        --gcs-path gs://cpc_awc/cmorph_catalog/catalog.parquet

    # Verify catalog
    micromamba run -n aifs-etl python upload_merge_parquet_catalogs.py verify \
        --input cmorph_vds_catalog/catalog.parquet
"""

import argparse
import calendar
import os
import sys
from collections import Counter

import pyarrow.parquet as pq

SERVICE_ACCOUNT_FILE = "coiled-data-e4drr_202505.json"
GCS_BUCKET = "cpc_awc"
DEFAULT_GCS_CATALOG = f"gs://{GCS_BUCKET}/cmorph_catalog/catalog.parquet"


def merge_catalogs(input_paths: list[str], output_path: str) -> dict:
    """Merge multiple Parquet catalogs into one with ZSTD compression."""
    for p in input_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Input not found: {p}")

    pf_first = pq.ParquetFile(input_paths[0])
    schema = pf_first.schema_arrow

    writer = pq.ParquetWriter(output_path, schema, compression="zstd")
    total_rows = 0

    for path in input_paths:
        pf = pq.ParquetFile(path)
        n_rg = pf.metadata.num_row_groups
        n_rows = pf.metadata.num_rows
        print(f"  {os.path.basename(path)}: {n_rows:,} rows, {n_rg} row groups")

        for i in range(n_rg):
            writer.write_table(pf.read_row_group(i))

        total_rows += n_rows

    writer.close()
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"\nMerged: {output_path}")
    print(f"  Rows: {total_rows:,}")
    print(f"  Size: {size_mb:.1f} MB")
    return {"total_rows": total_rows, "size_mb": size_mb}


def upload_catalog(
    local_path: str,
    gcs_path: str,
    service_account: str = SERVICE_ACCOUNT_FILE,
) -> str:
    """Upload a local Parquet catalog to GCS."""
    import gcsfs

    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Local file not found: {local_path}")

    size_mb = os.path.getsize(local_path) / 1024 / 1024
    print(f"Uploading {local_path} ({size_mb:.1f} MB) -> {gcs_path}")

    fs = gcsfs.GCSFileSystem(token=service_account)
    fs.put(local_path, gcs_path)

    # Verify upload
    info = fs.info(gcs_path)
    remote_mb = info["size"] / 1024 / 1024
    print(f"  Uploaded: {remote_mb:.1f} MB on GCS")
    print(f"  Path: {gcs_path}")
    return gcs_path


def verify_catalog(path: str):
    """Verify catalog completeness."""
    storage_options = {}
    if path.startswith("gs://"):
        storage_options = {"token": SERVICE_ACCOUNT_FILE}

    if path.startswith("gs://"):
        import pandas as pd
        df = pd.read_parquet(
            path,
            columns=["datetime", "year", "month_key", "status"],
            storage_options=storage_options,
        )
        years = df["year"].tolist()
        statuses = df["status"].tolist()
        datetimes = df["datetime"].tolist()
        month_keys = df["month_key"].tolist()
    else:
        pf = pq.ParquetFile(path)
        t = pf.read(columns=["datetime", "year", "month_key", "status"])
        years = t.column("year").to_pylist()
        statuses = t.column("status").to_pylist()
        datetimes = t.column("datetime").to_pylist()
        month_keys = t.column("month_key").to_pylist()

    print(f"\n=== Verification: {path} ===")
    print(f"Total rows: {len(years):,}")
    print(f"All success: {all(s == 'success' for s in statuses)}")
    print(f"Years: {min(years)}-{max(years)} ({len(set(years))} years)")
    print(f"Months: {len(set(month_keys))}")
    print(f"Time range: {min(datetimes)} -> {max(datetimes)}")

    year_counts = Counter(years)
    all_ok = True
    for yr in sorted(year_counts.keys()):
        days = 366 if calendar.isleap(yr) else 365
        expected = days * 24
        actual = year_counts[yr]
        if actual != expected:
            print(f"  {yr}: {actual:,} files (expected {expected:,}) *** MISMATCH")
            all_ok = False

    if all_ok:
        print(f"All {len(year_counts)} years: file counts match expected")
    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="Merge and upload CMORPH Parquet VDS catalogs"
    )
    sub = parser.add_subparsers(dest="command")

    # merge
    p_merge = sub.add_parser("merge", help="Merge multiple Parquet catalogs")
    p_merge.add_argument("--inputs", nargs="+", required=True)
    p_merge.add_argument("--output", required=True)

    # upload
    p_upload = sub.add_parser("upload", help="Upload catalog to GCS")
    p_upload.add_argument("--input", required=True, help="Local Parquet file")
    p_upload.add_argument(
        "--gcs-path", default=DEFAULT_GCS_CATALOG,
        help=f"GCS destination (default: {DEFAULT_GCS_CATALOG})",
    )
    p_upload.add_argument("--service-account", default=SERVICE_ACCOUNT_FILE)

    # verify
    p_verify = sub.add_parser("verify", help="Verify catalog completeness")
    p_verify.add_argument("--input", required=True, help="Parquet file (local or gs://)")

    args = parser.parse_args()

    if args.command == "merge":
        print("Merging catalogs...")
        merge_catalogs(args.inputs, args.output)
        ok = verify_catalog(args.output)
        sys.exit(0 if ok else 1)
    elif args.command == "upload":
        upload_catalog(args.input, args.gcs_path, args.service_account)
    elif args.command == "verify":
        ok = verify_catalog(args.input)
        sys.exit(0 if ok else 1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
