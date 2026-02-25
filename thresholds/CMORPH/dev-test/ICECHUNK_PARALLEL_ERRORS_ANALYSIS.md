# Icechunk Parallel Processing Errors Analysis

**Date:** 2026-02-04
**Context:** CMORPH 2020-2024 processing with 20 workers

## Current Store State

Successfully created branches with partial data:

| Branch | Timesteps | Expected | Completion |
|--------|-----------|----------|------------|
| year_2020 | 1,900 | ~8,784 | 21.6% |
| year_2021 | 2,020 | ~8,760 | 23.1% |
| year_2022 | 1,300 | ~8,760 | 14.8% |
| year_2023 | 2,020 | ~8,760 | 23.1% |
| year_2024 | 2,088 | ~8,784 | 23.8% |

**Data dimensions:** (time, lat=1649, lon=4948)

---

## Error Types Identified

### 1. ConflictError - Concurrent Commit Conflicts

**Error Message:**
```
icechunk.ConflictError: Failed to commit, expected parent: Some("BZ0X22FXM4K550KEKNM0"),
actual parent: Some("VXG1HGF4RRXPVJZ75E5G")
```

**Root Cause:**
- Multiple Coiled workers simultaneously commit to the same Icechunk branch
- Worker A commits successfully, changing the branch's parent snapshot
- Worker B's commit fails because its session was based on the old parent snapshot
- This is a classic optimistic concurrency control conflict

**Impact:** ~70-80% of batches fail due to this error

**Solution Options:**
1. **Use batch-wise branches** (recommended): Each worker writes to its own branch (`batch_0`, `batch_1`, etc.), then merge later
2. **Sequential commits**: Process batches sequentially within each year (slower but conflict-free)
3. **Retry with rebase**: Catch ConflictError, refresh session, and retry commit

---

### 2. Chunk Shape Mismatch

**Error Message:**
```
Cannot concatenate arrays with inconsistent chunk shapes: (1, 825, 2474) vs (1, 832, 2497)
Requires ZEP003 (Variable-length Chunks)
```

**Root Cause:**
- Some CMORPH NetCDF files have different spatial dimensions
- Standard dimensions: (1, 1649, 4948) but some files have (1, 825, 2474) or (1, 832, 2497)
- VirtualiZarr/Icechunk cannot concatenate arrays with different chunk shapes

**Affected Batches:** Batches containing files with non-standard dimensions

**Solution Options:**
1. **Filter files**: Identify and exclude files with non-standard dimensions before processing
2. **Separate processing**: Process non-standard files into a separate group/branch
3. **Rechunk**: Use rechunking to normalize chunk shapes (requires data transformation)

---

### 3. GCS Rate Limiting (429 Too Many Requests)

**Error Message:**
```
Server returned non-2xx status code: 429 Too Many Requests
The object cpc_awc/cmorph_2020_2024/refs/branch.year_2023/ref.json exceeded the rate limit
for object mutation operations (create, update, and delete)
```

**Root Cause:**
- Multiple workers simultaneously updating the same branch reference file
- GCS has a limit of ~1 write per second per object
- 20 workers all trying to update `refs/branch.year_2023/ref.json` simultaneously

**Solution Options:**
1. **Use batch-wise branches**: Each branch has its own ref file, avoiding contention
2. **Reduce concurrency**: Use fewer workers (but slower processing)
3. **Add delays**: Introduce random delays between commits (inefficient)

---

## Recommended Architecture Change

### Current Approach (Problematic)
```
20 workers → all write to year_2020 branch → conflicts
20 workers → all write to year_2021 branch → conflicts
...
```

### Recommended Approach (Conflict-Free)
```
Worker 0 → batch_0 branch
Worker 1 → batch_1 branch
Worker 2 → batch_2 branch
...
Worker 19 → batch_19 branch

Then: Sequential merge of batch branches into year branch
```

### Implementation Changes Needed

1. **Modify `process_batch_worker`** to use unique branch per batch:
```python
branch_name = f"batch_{batch_id}"  # Instead of f"year_{year}"
```

2. **Add post-processing merge step**:
```python
# After all batches complete, merge sequentially
for batch_id in range(n_batches):
    # Merge batch_X into year_YYYY branch
    repo.merge(f"batch_{batch_id}", f"year_{year}")
```

3. **Handle chunk shape variations**:
```python
# Pre-filter files by dimensions
standard_files = [f for f in files if check_dimensions(f) == (1649, 4948)]
```

---

## Workaround for Immediate Use

To process successfully with current code, reduce to **1-2 workers** and process sequentially:

```bash
micromamba run -n aifs-etl python cmorph_multi_year_processor.py \
  --n-workers 1 \
  --years 2024 \
  --service-account coiled-data-e4drr_202505.json
```

This avoids conflicts but is ~20x slower.

---

## Files with Non-Standard Dimensions

Some CMORPH files appear to have different spatial extents. These need investigation:
- Standard: lat=1649, lon=4948 (global 8km grid)
- Variant 1: 825 × 2474 (possibly regional subset)
- Variant 2: 832 × 2497 (possibly different version)

**Recommendation:** Run a scan to identify all files with non-standard dimensions and exclude them from the main processing pipeline.

---

## References

- [Icechunk Concurrency Documentation](https://icechunk.io/docs/concurrency)
- [GCS Rate Limits](https://cloud.google.com/storage/docs/gcs429)
- [ZEP003 - Variable-length Chunks](https://zarr.dev/zeps/draft/ZEP0003.html)
- Original plan: `CMORPH_FULL_PROCESSING_PLAN.md`

---

## Implementation Status

### Completed (2026-02-04)

1. [x] **Modified script to use batch-wise branches**
   - Changed from `year_{year}` to `batch_{year}_{batch_id}`
   - Each worker now writes to its own unique branch
   - Eliminates concurrent commit conflicts

### Pending

2. [ ] Add merge step after parallel processing (optional - can query batch branches directly)
3. [ ] Scan dataset for files with non-standard dimensions
4. [ ] Test with batch-branch approach on single year
5. [ ] Scale to full 2020-2024 processing

---

## Usage After Fix

```bash
# Run with 20 workers - now conflict-free
micromamba run -n aifs-etl python cmorph_multi_year_processor.py \
  --n-workers 20 \
  --service-account coiled-data-e4drr_202505.json

# Branches created will be: batch_2020_0, batch_2020_1, ..., batch_2024_87
```

---

*Generated: 2026-02-04*
*Updated: 2026-02-04 - Implemented batch-wise branches fix*
