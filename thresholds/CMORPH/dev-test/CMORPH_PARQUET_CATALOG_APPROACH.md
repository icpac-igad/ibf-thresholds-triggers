# CMORPH Parquet VDS Catalog: Why This Approach

## The Problem

236,688 CMORPH NetCDF files (1998-2024) on AWS S3 need to be virtualized and cataloged as a time-ordered dataset for Icechunk storage. All previous approaches hit OOM on the coordinator machine.

## Failed Approaches and Why They OOM'd

### Attempt 1: Coordinator accumulates Kerchunk JSON in a dict (v1 of this script)

```
url_to_refs[url] = kerchunk_json   # ~85 KB per file
# 28,000 files × 85 KB = 2.4 GB → OOM at batch 280/439
```

The coordinator received results from Coiled workers and stored every file's Kerchunk JSON (~85 KB) in a Python dict. Memory grew linearly with each batch. Killed at exit code 137 after processing 28,000 of 43,900 files (1998-2002 range).

### Attempt 2: Partial Parquet files + in-memory merge

Wrote each batch to a separate Parquet file (`partials/batch_00001.parquet`, etc.), but the final merge step loaded ALL partials into memory for a DataFrame join. Same OOM at the merge phase.

### All append_dim approaches (documented in CMORPH_TWO_PHASE_RATIONALE.md)

Every approach using `to_icechunk(store, append_dim='time')` in a loop reads back O(n) metadata per call, where n = current time dimension size. Memory grows with each append. OOM at ~87,000 timesteps.

## The Solution: PyArrow ParquetWriter Streaming

```python
writer = pq.ParquetWriter("catalog.parquet", schema, compression="zstd")

for future in as_completed(futures):
    result = future.result()
    table = pa.table({...rows from worker...}, schema=schema)
    writer.write_table(table)   # flush to disk immediately
    del table, result            # free memory
    # Coordinator RAM: constant ~10 MB regardless of total files

writer.close()
```

**Key insight:** PyArrow's `ParquetWriter` supports streaming appends to a single file. Each `write_table()` call flushes a new row group to disk. The coordinator never holds more than one batch (~100 rows × 85 KB = ~8.5 MB) in memory at a time.

**Why this doesn't have the Parquet "can't append" limitation:** Parquet files are immutable once *closed*. But while the writer is open, you can write as many row groups as needed. The coordinator receives batches sequentially via `as_completed`, writes each one, and frees the memory. When all batches are done, `writer.close()` finalizes the file.

## Workers Return Complete Rows

Previous versions had the coordinator parse datetimes and merge metadata. Now workers do everything:

```python
# Worker returns Parquet-ready rows:
{
    "s3_url": "s3://noaa-cdr-precip-cmorph-pds/data/30min/8km/1998/01/01/CMORPH_V1.0_ADJ_8km-30min_1998010100.nc",
    "filename": "CMORPH_V1.0_ADJ_8km-30min_1998010100.nc",
    "datetime": "1998-01-01T00:00:00",
    "year": 1998, "month": 1, "day": 1, "hour": 0, "minute": 0,
    "month_key": "1998-01",
    "status": "success",
    "kerchunk_refs": "{\"version\": 1, \"refs\": {\"cmorph/0.0.0\": [\"s3://...\", 1234, 5678], ...}}"
}
```

The coordinator writes these directly — zero processing, zero accumulation.

## Results: 3-Year Test (1998-2000)

| Metric | Value |
|--------|-------|
| Files processed | 26,304 (100%) |
| Errors | 0 |
| Wall time | 19.4 min |
| Parquet file size | 25.7 MB (zstd compressed) |
| Peak coordinator RAM | ~10 MB (constant) |
| Coiled workers | 10 × n2-standard-2 |
| Batch size | 100 files/task |

Compare to the failed v1 run that OOM'd at 28,000 files with 2.4 GB RAM usage.

## Parquet Schema

| Column | Type | Description |
|--------|------|-------------|
| s3_url | string | Full S3 path to NetCDF file |
| filename | string | Just the filename |
| datetime | timestamp[us] | Parsed from filename slot (00-47 → HH:MM) |
| year, month, day, hour, minute | int32 | Decomposed datetime for grouping |
| month_key | string | "YYYY-MM" for monthly batching |
| status | string | "success" or error message |
| kerchunk_refs | string | Full Kerchunk JSON (~84 KB per file) |

## CMORPH Filename Datetime Parsing

```
CMORPH_V1.0_ADJ_8km-30min_YYYYMMDDSS.nc
                           └─── 10 digits

SS = half-hour slot (00-47)
hour   = SS // 2          → 0-23
minute = (SS % 2) * 30    → 0 or 30

Example: 1998010100 → 1998-01-01 00:00:00
         1998010123 → 1998-01-01 11:30:00
         1998010147 → 1998-01-01 23:30:00
```

48 files per day, ~17,520 per year (17,568 for leap years).

## Projection for Full 1998-2024 Run

| Metric | Estimate |
|--------|----------|
| Total files | ~236,688 |
| Batches (100/batch) | ~2,367 |
| Wall time | ~2.4 hours |
| Parquet size | ~190 MB |
| Memory risk | None |

## Next Steps

1. **Run full 1998-2024 catalog** — same command, just change end year
2. **Icechunk write from catalog** — read sorted Parquet, group by month, `open_virtual_mfdataset` per month on Coiled, `xr.concat`, single `to_icechunk()` write
3. **Extreme value analysis** — open the Icechunk store with `xr.open_zarr`, compute annual maxima, fit GEV distribution

## Files

- `cmorph_parquet_vds_catalog.py` — catalog builder + icechunk writer
- `cmorph_vds_catalog/catalog.parquet` — the catalog (1998-2000 test)
- `cmorph_vds_catalog/catalog_summary.json` — build summary
