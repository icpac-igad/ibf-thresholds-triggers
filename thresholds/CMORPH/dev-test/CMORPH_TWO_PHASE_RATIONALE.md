# Why Two-Phase Concat: Rationale and Failure Analysis

## The Core Problem

236,688 CMORPH NetCDF files (1998-2024) on AWS S3 need to be represented as a single continuous virtual dataset in an Icechunk store on GCS. Virtual references only — no data copying.

## Failed Attempts and Root Causes

### Attempt 1: Individual File Writes

**Script:** `cmorph_s3_to_gcs_icechunk_versioned.py` (line 280)
**Store:** `gs://cpc_awc/cmorph_20260123`

```python
# Line 280 — each file overwrites the previous
virtual_ds.virtualize.to_icechunk(session.store)  # no concat_dim, no append_dim
```

**Why it failed:** Without `concat_dim` or `append_dim`, each write overwrites the dataset structure. The store ends up with only the last file's metadata. Time slices fail because there is no time concatenation.

**Outcome:** Single timestep works, time slices fail with "rust future panicked".

---

### Attempt 2: Parallel Batch Branches (Coiled, 20 workers)

**Script:** `cmorph_multi_year_processor.py` (lines 164-202)
**Store:** `gs://cpc_awc/cmorph_2020_2024` (2,378 branches)

```python
# Line 164-165 — each worker concatenates 100 files
combined_ds = xr.concat(virtual_datasets, dim='time')

# Line 202 — writes to its own branch to avoid conflicts
combined_ds.virtualize.to_icechunk(session.store, group=group_name)
```

**Why it's stuck:** Each batch writes to a separate branch (`batch_1998_0`, `batch_2021_52`, etc.). The within-batch concat works fine. But:

- Icechunk **does not support branch-to-branch merging** (only `session.fork()` merging for cooperative writes)
- 2,378 isolated branches cannot be combined into one continuous dataset
- Opening branches with `xr.open_zarr()` gives dask arrays, not VirtualiZarr ManifestArrays — can't extract virtual references back out

**Outcome:** 2,378 branches exist but are unusable as a single dataset.

---

### Attempt 3: Sequential Month-by-Month Append (No Coiled)

**Script:** `cmorph_concatenated_icechunk.py` (v1, lines 276-285)
**Store:** `gs://cpc_awc/cmorph_1998_2024_virtual`

```python
# Line 262-268 — create monthly virtual dataset
combined_vds = open_virtual_mfdataset(urls, combine="nested", concat_dim="time")

# Line 281-285 — append to existing store
combined_vds.virtualize.to_icechunk(session.store, group='cmorph', append_dim='time')
```

**Why it's slow:** `open_virtual_mfdataset` reads S3 file headers serially for ~720 files per month. Takes ~6 minutes per month × 324 months = ~35 hours.

**Outcome:** Successfully built 2021 (12 months, 17,520 timesteps). Time slices work. But impractical for 27 years.

---

### Attempt 4: Coiled parallel='dask' + Sequential Append

**Script:** `cmorph_concatenated_icechunk.py` (v2, lines 282-291)
**Store:** `gs://cpc_awc/cmorph_1998_2024_virtual_v3`

```python
# Line 287 — parallel metadata reads via Coiled
combined_vds = open_virtual_mfdataset(urls, ..., parallel="dask")

# Lines 304-309 — still sequential append
combined_vds.virtualize.to_icechunk(session.store, group='cmorph', append_dim='time')
```

**Why it OOM'd:** The metadata reads are now fast (~45s vs 6 min). But `append_dim='time'` must read the existing store metadata to know the current time dimension size. As the store grows:

```
Month   1: write  8s   (store has    1,488 timesteps)
Month  60: write 20s   (store has   43,000 timesteps)
Month 100: write 34s   (store has   73,000 timesteps)
Month 120: write 36s   (store has   87,000 timesteps)
Month 121: OOM killed   (exit code 137)
```

**Root cause:** Each append operation loads O(n) metadata where n = current time dimension size. Memory grows linearly with each append.

**Outcome:** 1998-2007 complete (120 months, 87,648 files). OOM at Jan 2008.

---

### Attempt 5: Daily Batch Append (test_concatenate_existing_icechunk.py)

**Script:** `test_concatenate_existing_icechunk.py` (lines 319-331)

```python
# Same append_dim pattern, just with smaller batches (48 files = 1 day)
if batch_idx == 0:
    combined_vds.virtualize.to_icechunk(session.store, group='cmorph')
else:
    combined_vds.virtualize.to_icechunk(session.store, group='cmorph', append_dim='time')
```

**Why it would be worse:** Daily batches mean ~9,855 append operations instead of 324 monthly ones. Each append reads the full growing metadata. Would OOM even sooner.

---

## The Common Failure Pattern

Every failed attempt shares the same root cause:

```
┌──────────────────────────────────────────────────────────────┐
│  append_dim='time' is O(n) in memory per call               │
│                                                              │
│  Call 1:   read 0 metadata      → write batch 1    (fast)   │
│  Call 2:   read batch 1 meta    → write batch 2    (fast)   │
│  Call 50:  read 50 batches meta → write batch 51   (slower) │
│  Call 120: read 120 batches meta → write batch 121 (OOM)    │
│                                                              │
│  The metadata reader grows linearly with each append.        │
│  No batching strategy can fix this — only avoiding           │
│  append entirely solves it.                                  │
└──────────────────────────────────────────────────────────────┘
```

## Why Two-Phase Concat Will NOT OOM

### Phase 1: Create Virtual Datasets (Coiled Workers)

Each Coiled worker creates a monthly virtual dataset from S3 file headers. A virtual dataset is just metadata — ManifestArray objects containing byte-range pointers.

```
1 month VDS ≈ 720 files × ~100 bytes per manifest entry ≈ ~70 KB
324 months × 70 KB = ~23 MB total
```

Workers return these lightweight objects to the coordinator. No Icechunk involved.

### Phase 2: Single Concat + Single Write (Local)

```python
# Concat 324 VDS objects — pure metadata operation, ~23 MB
full_vds = xr.concat(monthly_vds_list, dim="time")

# ONE write to Icechunk — no append, no growing metadata
full_vds.virtualize.to_icechunk(session.store, group='cmorph')
session.commit(message="Full 1998-2024")
```

**Why this is different from all previous attempts:**

| Property | Append approach | Two-phase approach |
|----------|----------------|-------------------|
| Icechunk writes | 120-324 sequential appends | **1 single write** |
| Metadata read-back per write | O(n) growing | **O(1) — nothing to read back** |
| Memory at write time | Grows with store size | **Constant — just the VDS** |
| Peak memory | ~87k timesteps metadata → OOM | **~23 MB of manifests** |

The key: `to_icechunk()` without `append_dim` writes the entire dataset structure once. It doesn't need to read any existing metadata because there is nothing to append to — it's a fresh write of the complete dataset.

### Memory Budget

```
Phase 1 (per worker):
  - open_virtual_mfdataset for ~720 files: ~200 MB peak (S3 header parsing)
  - Return VDS to coordinator: ~70 KB

Phase 2 (coordinator):
  - 324 monthly VDS in memory: ~23 MB
  - xr.concat of 324 VDS: ~50 MB peak (metadata manipulation)
  - to_icechunk single write: ~100 MB peak (writing all manifests)
  - Total: ~200 MB — well within any machine's capacity

Compare to append approach:
  - Month 120: reading back metadata for 87,000+ timesteps → multiple GB → OOM
```

## Implementation

The two-phase script is `cmorph_two_phase_concat.py`. See `CMORPH_CONCATENATION_GUIDE.md` for full usage commands.
