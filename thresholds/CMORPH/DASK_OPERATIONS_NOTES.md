# Dask Operations Notes — CMORPH Pipeline

Lessons learned from running the CMORPH East Africa fill and rechunk pipeline
on Coiled clusters.

---

## 1. Why Workers Sit Idle During Sequential-Batch Fill

### What We Observe

In the Coiled dashboard, some workers show as "idle" even while the fill is
running.  With 5 workers and `--commit-batch 5`, the dashboard often shows
2-3 workers idle at any given moment.

### Root Cause: Sequential Batch Barrier

Our fill uses **sequential batch processing** for data integrity:

```
Batch 1: submit days [0,1,2,3,4] → wait ALL 5 → write → commit
Batch 2: submit days [5,6,7,8,9] → wait ALL 5 → write → commit
...
```

The problem is the **barrier between batches**:

```
Time ──────────────────────────────────────────────────────►

Worker A: ██████████░░░░░░░░░░░░░░████████████░░░░░░░░░░░░░
Worker B: ████████████████░░░░░░░░████████░░░░░░░░░░░░░░░░░
Worker C: ██████████████░░░░░░░░░░██████████████░░░░░░░░░░░░
Worker D: ████████░░░░░░░░░░░░░░░░████████████████░░░░░░░░░░
Worker E: ██████████████████░░░░░░██████████░░░░░░░░░░░░░░░░
                           ↑ barrier          ↑ barrier
                           │                  │
                    Coordinator writes  Coordinator writes
                    + commits to        + commits to
                    Icechunk            Icechunk
```

Each worker finishes at a different time because:
- **Network variability**: S3 download speeds vary per file/region
- **Data density**: Some CMORPH files have more valid data (denser
  precipitation regions take fractionally longer to process)
- **GCS catalog reads**: First read per worker is slower (cold cache)

The **fast workers wait** for the slowest worker in the batch.  Then the
coordinator sequentially writes all results to Icechunk and commits.  During
this coordinator phase, **all workers are idle**.

### Why We Accept This

We chose sequential batches over `as_completed` because:

1. **Contiguous commit guarantee**: Each commit message `"fill days X-Y"`
   guarantees every day in [X,Y] has real data.  This makes resume trivial
   and gap detection automatic.

2. **Resume correctness**: With `as_completed`, a batch containing days
   [5, 200, 3, 100, 50] would create commit `"fill days 3-200"` — resume
   would skip 195 unwritten days.

3. **Time-series integrity**: The resulting Icechunk store has no silent
   gaps (zeros masquerading as data within committed ranges).

### How to Reduce Idle Time

| Approach | Trade-off |
|----------|-----------|
| **Larger batches** (`--commit-batch 50`) | Less frequent barriers, but larger blast radius on failure |
| **More workers** (`--n-workers 20`) | Same batch size processes faster, but more idle workers per barrier |
| **Overlapped I/O**: Submit batch N+1 while writing batch N | Complex coordinator logic, risk of Icechunk session conflicts |
| **Pipeline approach**: Workers write directly to store | Requires distributed Icechunk writes (fork/merge), more complex |

**Recommendation for production**: Use `--commit-batch 50 --n-workers 20`.
Each batch takes ~10s of compute (20 workers × 50 days, each worker handles
2-3 days).  Coordinator write+commit takes ~5s.  Idle fraction drops to
~15% vs ~40% with batch=5.

### Dask Concepts Illustrated

- **Task graph**: Each `client.submit(read_day_ea_subset, ...)` creates a
  single task node — no dependencies between days within a batch.

- **`as_completed` vs batch barrier**: `as_completed` maximizes throughput
  (workers never idle) but sacrifices ordering.  Batch barriers sacrifice
  throughput for correctness guarantees.

- **Data locality**: Workers download from S3 (us-east-1) and return numpy
  arrays to the coordinator.  The ~40 MB per day result is small enough
  that serialization overhead is negligible.

- **No Dask graph for writes**: The coordinator writes to Icechunk
  sequentially — this is intentional.  Icechunk sessions are not
  thread-safe, so distributed writes require fork/merge (Phase 2
  alternative).

---

## 2. P2P Rechunking for Phase 3 (Pencil Chunks)

### The Rechunking Problem

Our fill creates chunks of `(48, 120, 110)` — optimized for writing day
batches.  For time-series access (e.g., "give me all precipitation at
Nairobi for 27 years"), we need **pencil chunks**: `(all_time, 5, 5)`.

This is a **full data shuffle** — every input chunk contributes to many
output chunks, and every output chunk reads from many input chunks:

```
Input chunks (48, 120, 110):       Output chunks (473376, 5, 5):

┌──────────────────────┐           ┌─┬─┬─┬─┬─┐
│  48 time steps       │           │ │ │ │ │ │  All 473K timesteps
│  120 lat × 110 lon   │    →     │ │ │ │ │ │  but only 5×5 spatial
│  ~2.4 MB each        │   full   │ │ │ │ │ │  ~5 MB each
│  ~197K chunks        │  shuffle │ │ │ │ │ │
└──────────────────────┘           └─┴─┴─┴─┴─┘
                                    ~8,400 chunks
```

### Traditional Task-Based Rechunking

Dask's default `method="tasks"` creates intermediate chunks:

```
Input chunk → split → intermediate chunks → merge → output chunk
```

For our data this creates a graph with **135K+ nodes** and requires holding
many intermediate chunks in memory simultaneously.  Memory usage scales
with the number of chunks that overlap between input and output — for a
time×space → space×time reshape, this can be enormous.

**Risk**: OOM on workers, graph serialization overhead, long scheduling
delays.

### P2P (Peer-to-Peer) Rechunking

Reference: [Pangeo Discourse: Rechunking large data at constant memory](https://discourse.pangeo.io/t/rechunking-large-data-at-constant-memory-in-dask-experimental/3266)

P2P rechunking uses **worker-to-worker communication** instead of task
dependencies:

```
Traditional (tasks):
  read input chunk → create N split tasks → schedule N merge tasks → write

P2P (shuffle):
  read input chunk → send slices directly to workers owning output chunks
                   → each worker accumulates its output chunk on disk
                   → write when complete
```

Key properties:

| Property | Tasks | P2P |
|----------|-------|-----|
| Memory | Proportional to overlap | **Constant** |
| Graph size | 135K+ nodes | **13K nodes** |
| Scheduling | Long delay before compute starts | Compute starts immediately |
| Disk usage | None | Spills to worker local disk |
| Worker communication | Via scheduler | **Direct worker-to-worker** |

### How to Use P2P for Our Rechunk Phase

```python
import dask

with dask.config.set({"array.rechunk.method": "p2p"}):
    ds_rechunked = ds.chunk({"time": -1, "lat": 5, "lon": 5})
    ds_rechunked.to_zarr(target_path, mode="w")
```

Or globally:
```python
dask.config.set({"array.rechunk.method": "p2p"})
```

### Important Caveats for Our Pipeline

1. **`to_zarr` compatibility**: Early versions of P2P could not be combined
   with `da.store()` / `to_zarr()` in the same compute call.  As of Dask
   2024.x+ this is resolved, but test before relying on it.

   **Workaround if needed**: Rechunk to a Dask array, then do a separate
   `to_zarr` call:
   ```python
   rechunked = ds.chunk({"time": -1, "lat": 5, "lon": 5}).compute()
   rechunked.to_zarr(target_path, mode="w")
   ```
   (Only works if the dataset fits in coordinator memory — ~44 GB for
   3-year EA, too large.  Use chunked `to_zarr` instead.)

2. **Static cluster sizing**: P2P locks participating workers at startup.
   Adaptive scaling causes new workers to sit idle while original workers
   are overloaded.  Use fixed `n_workers` (not a range):
   ```python
   cluster = coiled.Cluster(n_workers=20, ...)  # NOT n_workers=[5, 20]
   ```

3. **Disk space on workers**: P2P spills intermediate data to worker local
   disk.  Total disk needed ≈ dataset size spread across workers.  For
   44 GB EA dataset with 20 workers ≈ 2.2 GB per worker — well within
   typical VM disk sizes.

4. **Disable fusion**: P2P currently requires:
   ```python
   dask.config.set({"optimization.fuse.active": False})
   ```

### Rechunk Phase Plan (Updated)

```python
# Phase 3: Rechunk EA store to pencil chunks using P2P
import dask

cluster = coiled.Cluster(
    n_workers=20,              # Fixed, not adaptive
    worker_vm_types="n2-highmem-4",  # 32 GB RAM per worker
    worker_disk_size=10,       # GB local disk for P2P spill
)
client = distributed.Client(cluster)

dask.config.set({
    "array.rechunk.method": "p2p",
    "optimization.fuse.active": False,
})

ds = xr.open_zarr(source_store, consolidated=False)
ds_pencil = ds.chunk({"time": -1, "lat": 5, "lon": 5})

ds_pencil.to_zarr(
    "gs://cpc_awc/cmorph_ea_pencil/",
    encoding={"cmorph": {"chunks": (n_time, 5, 5)}},
    mode="w",
)
```

### Memory Estimate for Rechunk

| Parameter | Value |
|-----------|-------|
| Dataset size (3yr EA) | ~44 GB |
| Input chunks | (48, 120, 110) = 2.4 MB × 197K |
| Output chunks | (52608, 5, 5) = 5.3 MB × 8,400 |
| Workers | 20 |
| Per-worker data | ~2.2 GB |
| Worker RAM | 32 GB (n2-highmem-4) |
| P2P disk spill | ~2.2 GB per worker |
| Estimated time | 15-30 min |

---

## 3. Fill vs Rechunk: Different Dask Patterns

| Aspect | Fill (Phase 2) | Rechunk (Phase 3) |
|--------|---------------|-------------------|
| Pattern | Embarrassingly parallel | Full shuffle |
| Workers do | Independent S3 reads | Exchange data with each other |
| Coordinator | Writes to Icechunk | Passive (Dask manages) |
| Communication | Worker → coordinator only | Worker ↔ worker (P2P) |
| Memory scaling | Constant (one day per worker) | Constant (P2P) or linear (tasks) |
| Failure mode | Retry individual days | Restart entire rechunk |
| Idle workers | Batch barriers | P2P initialization phase |

---

## References

- [Pangeo: Rechunking large data at constant memory (P2P)](https://discourse.pangeo.io/t/rechunking-large-data-at-constant-memory-in-dask-experimental/3266)
- [dask.array.rechunk API](https://docs.dask.org/en/latest/generated/dask.array.rechunk.html)
- [Rechunker library](https://rechunker.readthedocs.io/en/latest/tutorial.html)
- [xarray.Dataset.to_zarr](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.to_zarr.html)
- [Dask distributed changelog](https://distributed.dask.org/en/stable/changelog.html)
