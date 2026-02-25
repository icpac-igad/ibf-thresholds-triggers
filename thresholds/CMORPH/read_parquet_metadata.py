#!/usr/bin/env python3
"""
Read Parquet Catalog Metadata (zero-copy, low memory)
=====================================================

Reads ONLY the Parquet footer from a local or GCS catalog file to extract
row count, column count, row-group count, schema, and column-level stats
— without loading any actual row data into memory.

How it works
------------
A Parquet file stores its metadata in a **footer** at the end of the file.
The footer contains:

  - Schema (column names, types)
  - Row-group metadata (row count, byte offsets, min/max stats per column)
  - Total row count (sum of all row-group counts)

``pyarrow.parquet.ParquetFile`` reads only this footer on open — typically
a few KB regardless of how large the file is.  For a 224 MB catalog with
236K rows, the metadata read uses ~7 MB of memory and takes ~7 seconds
(dominated by GCS network latency, not computation).

This is the standard way to inspect Parquet files without loading data:

  - ``pf.metadata.num_rows``       — total rows across all row groups
  - ``pf.metadata.num_columns``    — number of columns
  - ``pf.metadata.num_row_groups`` — number of row groups
  - ``pf.schema_arrow``            — full Arrow schema with types

No row data is deserialized, so memory stays constant regardless of file
size.  This makes it safe to inspect multi-GB Parquet files on machines
with limited RAM.

Usage
-----
    # Inspect the GCS catalog (default)
    micromamba run -n zarrv3 python read_parquet_metadata.py

    # Inspect a local parquet file
    micromamba run -n zarrv3 python read_parquet_metadata.py \
        --input /path/to/catalog.parquet

    # Inspect with full schema details
    micromamba run -n zarrv3 python read_parquet_metadata.py --schema

    # Inspect with per-row-group stats
    micromamba run -n zarrv3 python read_parquet_metadata.py --row-groups
"""

import argparse
import time
import tracemalloc

import pyarrow.parquet as pq


# Default GCS path from the CMORPH pipeline
DEFAULT_GCS_PATH = "cpc_awc/cmorph_catalog/catalog.parquet"


def open_parquet_file(path: str) -> pq.ParquetFile:
    """Open a ParquetFile from a local path or GCS URI.

    For GCS paths, uses gcsfs to open a file handle.  Only the Parquet
    footer is read — no row data is loaded.
    """
    if path.startswith("gs://") or not path.startswith("/"):
        # GCS path — strip gs:// prefix if present
        import gcsfs

        gcs_path = path.removeprefix("gs://")
        fs = gcsfs.GCSFileSystem()
        return pq.ParquetFile(fs.open(gcs_path))
    else:
        # Local file
        return pq.ParquetFile(path)


def print_metadata(pf: pq.ParquetFile, show_schema: bool, show_row_groups: bool):
    """Print catalog metadata from the Parquet footer."""
    meta = pf.metadata

    print(f"Rows:        {meta.num_rows:,}")
    print(f"Columns:     {meta.num_columns}")
    print(f"Row groups:  {meta.num_row_groups}")
    print(f"Format ver:  {meta.format_version}")

    if show_schema:
        print(f"\nSchema ({meta.num_columns} columns):")
        schema = pf.schema_arrow
        for i, field in enumerate(schema):
            print(f"  {i:2d}. {field.name:30s}  {field.type}")

    if show_row_groups:
        print(f"\nRow groups ({meta.num_row_groups} total):")
        for i in range(min(meta.num_row_groups, 10)):
            rg = meta.row_group(i)
            print(f"  [{i:4d}] {rg.num_rows:,} rows, "
                  f"{rg.total_byte_size / 1024 / 1024:.1f} MB")
        if meta.num_row_groups > 10:
            print(f"  ... ({meta.num_row_groups - 10} more row groups)")


def main():
    parser = argparse.ArgumentParser(
        description="Read Parquet metadata without loading row data"
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_GCS_PATH,
        help="Path to parquet file (local or gs://). "
        f"Default: {DEFAULT_GCS_PATH}",
    )
    parser.add_argument(
        "--schema",
        action="store_true",
        help="Print full column schema with types",
    )
    parser.add_argument(
        "--row-groups",
        action="store_true",
        help="Print per-row-group stats (first 10)",
    )
    args = parser.parse_args()

    tracemalloc.start()
    t0 = time.time()

    pf = open_parquet_file(args.input)
    print_metadata(pf, args.schema, args.row_groups)

    elapsed = time.time() - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"\nTime:        {elapsed:.3f}s")
    print(f"Peak memory: {peak / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
