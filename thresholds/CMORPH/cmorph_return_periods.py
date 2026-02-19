#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "xarray",
#     "zarr",
#     "coiled",
#     "distributed",
#     "gcsfs",
#     "netcdf4",
# ]
# ///
"""
CMORPH East Africa — Precipitation Return Period Analysis
==========================================================

Computes precipitation return periods for every pixel in the East Africa
pencil-chunked Zarr store (gs://cpc_awc/cmorph_ea_pencil).

The pencil store has chunks (473376, 5, 5) — all 27 years at 25 pixels per
chunk — perfect for per-pixel time-series analysis.  Each Coiled worker
reads exactly one chunk (~45 MB), computes rolling accumulations for 9
durations, extracts annual maxima across 27 calendar years, fits a Gumbel
distribution via method-of-moments (pure numpy, no scipy), and returns
~25 KB of results.

Subcommands:

  compute  — Run distributed return period analysis on Coiled
  verify   — Validate output NetCDF for physical reasonableness

Usage:
    micromamba run -n aifs-etl python cmorph_return_periods.py compute \
        --source gs://cpc_awc/cmorph_ea_pencil \
        --output cmorph_ea_return_periods.nc \
        --n-workers 20

    micromamba run -n aifs-etl python cmorph_return_periods.py verify \
        --input cmorph_ea_return_periods.nc

Author: AI Assistant
Date: 2026-02-09
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("cmorph_return_periods.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ─── Constants ──────────────────────────────────────────────────────────────

SERVICE_ACCOUNT_FILE = "coiled-data-e4drr_202505.json"
GCS_BUCKET = "cpc_awc"
FILL_VALUE = np.float32(9.969209968386869e+36)

# Accumulation durations: name → number of 30-minute timesteps
DURATIONS = {
    "30min": 1,
    "1hr": 2,
    "3hr": 6,
    "6hr": 12,
    "12hr": 24,
    "24hr": 48,
    "48hr": 96,
    "72hr": 144,
    "7day": 336,
}

# Return periods in years
RETURN_PERIODS = [2, 5, 10, 20, 50, 100]

# Pencil chunk shape in source store
PENCIL_LAT = 5
PENCIL_LON = 5

# Euler-Mascheroni constant
EULER_MASCHERONI = 0.5772156649015329


# ─── Worker function ────────────────────────────────────────────────────────


def process_chunk(
    source_path,
    sa_info,
    lat_start,
    lat_end,
    lon_start,
    lon_end,
    chunk_id,
    include_annual_maxima=False,
):
    """Process one pencil chunk: rolling sums → annual maxima → Gumbel fit.

    Runs on a Coiled worker.  Reads exactly one pencil chunk (~45 MB),
    computes all 9 durations, extracts annual maxima for 27 calendar years,
    and fits Gumbel distribution via method-of-moments (pure numpy).

    Parameters
    ----------
    source_path : str
        GCS Zarr store path (e.g. gs://cpc_awc/cmorph_ea_pencil).
    sa_info : dict
        GCS service account credentials dict.
    lat_start, lat_end : int
        Latitude index slice [lat_start:lat_end].
    lon_start, lon_end : int
        Longitude index slice [lon_start:lon_end].
    chunk_id : int
        Chunk identifier for logging/tracking.
    include_annual_maxima : bool
        If True, include the full annual_maxima array in the result.

    Returns
    -------
    dict with keys:
        chunk_id, lat_start, lat_end, lon_start, lon_end,
        rp_precip (n_dur × n_rp × n_lat × n_lon),
        dist_location (n_dur × n_lat × n_lon),
        dist_scale (n_dur × n_lat × n_lon),
        lat_vals, lon_vals,
        [annual_maxima (n_dur × n_years × n_lat × n_lon) if requested],
        status, elapsed_sec
    """
    import time as _time

    import numpy as np
    import xarray as xr

    t0 = _time.time()

    durations = {
        "30min": 1, "1hr": 2, "3hr": 6, "6hr": 12, "12hr": 24,
        "24hr": 48, "48hr": 96, "72hr": 144, "7day": 336,
    }
    return_periods = [2, 5, 10, 20, 50, 100]
    euler = 0.5772156649015329
    fill_val = np.float32(9.969209968386869e+36)

    try:
        # ── Open store and load exactly one chunk ──
        ds = xr.open_zarr(
            source_path,
            storage_options={"token": sa_info},
            consolidated=True,
        )
        subset = ds["cmorph"].isel(
            lat=slice(lat_start, lat_end),
            lon=slice(lon_start, lon_end),
        ).load()

        data = subset.values.astype(np.float32)  # (n_time, n_lat, n_lon)
        time_vals = subset.time.values
        lat_vals = subset.lat.values.tolist()
        lon_vals = subset.lon.values.tolist()
        ds.close()
        del subset

        n_time, n_lat, n_lon = data.shape

        # ── Mask fill values and negatives → NaN ──
        data[(data >= fill_val * 0.99) | (data < 0)] = np.nan

        # ── Build calendar year index ──
        # time_vals is datetime64[ns]; extract year for each timestep
        years = (time_vals.astype("datetime64[Y]").astype(int) + 1970).astype(int)
        unique_years = np.unique(years)
        n_years = len(unique_years)
        year_list = unique_years.tolist()

        # Pre-compute year boundaries: year_bounds[y] = (start_idx, end_idx)
        year_bounds = {}
        for y in unique_years:
            indices = np.where(years == y)[0]
            year_bounds[y] = (int(indices[0]), int(indices[-1]) + 1)

        # ── Compute cumsum once (NaN → 0 for cumsum) ──
        data_clean = np.where(np.isnan(data), 0.0, data)
        cumsum = np.cumsum(data_clean, axis=0)  # (n_time, n_lat, n_lon)
        del data_clean

        # Prepend a row of zeros for the rolling-sum subtraction
        cs_padded = np.concatenate(
            [np.zeros((1, n_lat, n_lon), dtype=np.float32), cumsum],
            axis=0,
        )  # (n_time+1, n_lat, n_lon)
        del cumsum

        # ── Results arrays ──
        dur_names = list(durations.keys())
        n_dur = len(dur_names)
        n_rp = len(return_periods)

        rp_precip = np.full((n_dur, n_rp, n_lat, n_lon), np.nan, dtype=np.float32)
        dist_location = np.full((n_dur, n_lat, n_lon), np.nan, dtype=np.float32)
        dist_scale = np.full((n_dur, n_lat, n_lon), np.nan, dtype=np.float32)

        if include_annual_maxima:
            annual_maxima_all = np.full(
                (n_dur, n_years, n_lat, n_lon), np.nan, dtype=np.float32
            )

        # ── Process each duration ──
        for d_idx, dur_name in enumerate(dur_names):
            w = durations[dur_name]

            # Vectorized rolling sum: rolling[i] = cs_padded[i+w] - cs_padded[i]
            if w == 1:
                rolling = data.copy()
            else:
                rolling = cs_padded[w:] - cs_padded[:-w]
                # rolling shape: (n_time - w + 1, n_lat, n_lon)

            # Convert from mm/hr (rate × 30min = 0.5 hr) to mm accumulation
            # Each timestep is 30 min, rolling sum of w timesteps = w × 0.5 hr
            # But the data is already a rate (mm/hr), so accumulation = sum(rate × 0.5hr)
            # The rolling sum already sums the rates; multiply by 0.5 to get mm
            rolling = rolling * 0.5

            # ── Extract annual maxima ──
            am = np.full((n_years, n_lat, n_lon), np.nan, dtype=np.float32)

            for y_idx, y in enumerate(unique_years):
                yb_start, yb_end = year_bounds[y]

                # rolling[i] = sum of raw timesteps [i .. i+w-1].
                # Assign rolling[i] to the year of raw timestep i (window start).
                # For w>1, rolling is shorter than data, so clip at len(rolling).
                r_start = max(0, yb_start)
                r_end = min(len(rolling), yb_end)

                if r_start >= r_end:
                    continue

                chunk = rolling[r_start:r_end]
                with np.errstate(invalid="ignore"):
                    am[y_idx] = np.nanmax(chunk, axis=0)

            if include_annual_maxima:
                annual_maxima_all[d_idx] = am

            # ── Gumbel method-of-moments fit (vectorized over lat/lon) ──
            # For each pixel: mu_gumbel = mean(am) - euler * beta
            #                  beta = std(am) * sqrt(6) / pi
            #                  x_T = mu - beta * ln(-ln(1 - 1/T))
            with np.errstate(invalid="ignore"):
                valid_count = np.sum(~np.isnan(am), axis=0)  # (n_lat, n_lon)
                am_mean = np.nanmean(am, axis=0)
                am_std = np.nanstd(am, axis=0, ddof=1)

            beta = am_std * np.sqrt(6.0) / np.pi
            mu = am_mean - euler * beta

            dist_location[d_idx] = mu
            dist_scale[d_idx] = beta

            # Compute return period values
            for r_idx, T in enumerate(return_periods):
                # Reduced variate for Gumbel
                y_T = -np.log(-np.log(1.0 - 1.0 / T))
                rp_val = mu + beta * y_T

                # Mask pixels with too few valid years (< 5)
                rp_val[valid_count < 5] = np.nan
                rp_precip[d_idx, r_idx] = rp_val

            del rolling, am

        result = {
            "status": "success",
            "chunk_id": chunk_id,
            "lat_start": lat_start,
            "lat_end": lat_end,
            "lon_start": lon_start,
            "lon_end": lon_end,
            "rp_precip": rp_precip,
            "dist_location": dist_location,
            "dist_scale": dist_scale,
            "lat_vals": lat_vals,
            "lon_vals": lon_vals,
            "n_years": n_years,
            "year_list": year_list,
            "elapsed_sec": _time.time() - t0,
        }
        if include_annual_maxima:
            result["annual_maxima"] = annual_maxima_all

        return result

    except Exception as e:
        import traceback
        return {
            "status": "error",
            "chunk_id": chunk_id,
            "lat_start": lat_start,
            "lat_end": lat_end,
            "lon_start": lon_start,
            "lon_end": lon_end,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "elapsed_sec": _time.time() - t0,
        }


# ─── compute subcommand ────────────────────────────────────────────────────


def run_compute(args):
    """Run distributed return period analysis on Coiled."""
    import coiled
    import distributed
    import xarray as xr

    logger.info("=" * 70)
    logger.info("COMPUTE: Precipitation Return Period Analysis")
    logger.info("=" * 70)
    overall_start = time.time()

    source_path = args.source

    # ── Load GCS credentials ──
    with open(args.service_account) as f:
        sa_info = json.load(f)
    logger.info(f"Loaded GCS credentials from {args.service_account}")

    # ── Open source store metadata (lazy, no data) ──
    logger.info(f"Opening source: {source_path}")
    ds = xr.open_zarr(
        source_path,
        storage_options={"token": sa_info},
        consolidated=True,
    )
    n_time = ds.sizes["time"]
    n_lat = ds.sizes["lat"]
    n_lon = ds.sizes["lon"]
    lat_vals = ds.lat.values
    lon_vals = ds.lon.values
    time_vals = ds.time.values
    ds.close()

    logger.info(f"  Shape: time={n_time}, lat={n_lat}, lon={n_lon}")
    logger.info(f"  Lat: {float(lat_vals[0]):.2f} .. {float(lat_vals[-1]):.2f}")
    logger.info(f"  Lon: {float(lon_vals[0]):.2f} .. {float(lon_vals[-1]):.2f}")

    # ── Enumerate chunk blocks ──
    chunk_lat = PENCIL_LAT
    chunk_lon = PENCIL_LON
    n_lat_blocks = -(-n_lat // chunk_lat)  # ceil division
    n_lon_blocks = -(-n_lon // chunk_lon)
    n_chunks = n_lat_blocks * n_lon_blocks
    logger.info(f"  Chunk grid: {n_lat_blocks} lat x {n_lon_blocks} lon = {n_chunks} chunks")

    chunks = []
    chunk_id = 0
    for i_lat in range(n_lat_blocks):
        lat_s = i_lat * chunk_lat
        lat_e = min(lat_s + chunk_lat, n_lat)
        for i_lon in range(n_lon_blocks):
            lon_s = i_lon * chunk_lon
            lon_e = min(lon_s + chunk_lon, n_lon)
            chunks.append({
                "chunk_id": chunk_id,
                "lat_start": lat_s,
                "lat_end": lat_e,
                "lon_start": lon_s,
                "lon_end": lon_e,
            })
            chunk_id += 1

    # ── Launch Coiled cluster ──
    n_workers = args.n_workers
    logger.info(f"Launching Coiled cluster with {n_workers} workers...")

    cluster = coiled.Cluster(
        name=f"cmorph-rp-{int(time.time()) % 10000}",
        n_workers=n_workers,
        worker_vm_types="n2-standard-4",
        package_sync=True,
        region="us-east1",
        workspace="e4drr",
        idle_timeout="30 minutes",
    )
    client = distributed.Client(cluster)
    client.wait_for_workers(n_workers=min(5, n_workers), timeout=300)
    logger.info(f"Cluster ready: {client.dashboard_link}")

    # ── Submit all tasks ──
    include_am = not args.no_annual_maxima
    logger.info(f"Submitting {n_chunks} tasks (include_annual_maxima={include_am})...")

    futures = {}
    for c in chunks:
        future = client.submit(
            process_chunk,
            source_path,
            sa_info,
            c["lat_start"],
            c["lat_end"],
            c["lon_start"],
            c["lon_end"],
            c["chunk_id"],
            include_am,
            key=f"chunk-{c['chunk_id']}",
        )
        futures[future] = c

    # ── Collect results ──
    logger.info("Collecting results...")
    results = []
    n_success = 0
    n_fail = 0
    t_collect_start = time.time()

    for future in distributed.as_completed(futures):
        c = futures[future]
        try:
            result = future.result()
            results.append(result)

            if result["status"] == "success":
                n_success += 1
            else:
                n_fail += 1
                logger.error(
                    f"  Chunk {result['chunk_id']} FAILED: {result.get('error', 'unknown')}"
                )

            total_done = n_success + n_fail
            if total_done % 100 == 0 or total_done == n_chunks:
                elapsed = time.time() - t_collect_start
                rate = total_done / elapsed if elapsed > 0 else 0
                eta_min = (n_chunks - total_done) / rate / 60 if rate > 0 else 0
                logger.info(
                    f"  Progress: {total_done}/{n_chunks} "
                    f"({n_success} OK, {n_fail} fail) "
                    f"rate={rate:.1f}/s ETA={eta_min:.1f}min"
                )

        except Exception as e:
            n_fail += 1
            logger.error(f"  Chunk {c['chunk_id']} exception: {e}")
            results.append({
                "status": "error",
                "chunk_id": c["chunk_id"],
                "error": str(e),
            })

    client.close()
    cluster.close()
    logger.info(f"Cluster shut down. {n_success}/{n_chunks} chunks successful, {n_fail} failed.")

    if n_success == 0:
        logger.error("All chunks failed!")
        return

    # ── Assemble results into full arrays ──
    logger.info("Assembling results into output arrays...")

    dur_names = list(DURATIONS.keys())
    n_dur = len(dur_names)
    n_rp = len(RETURN_PERIODS)

    # Get year list from first successful result
    first_ok = next(r for r in results if r["status"] == "success")
    year_list = first_ok["year_list"]
    n_years = first_ok["n_years"]

    rp_full = np.full((n_dur, n_rp, n_lat, n_lon), np.nan, dtype=np.float32)
    loc_full = np.full((n_dur, n_lat, n_lon), np.nan, dtype=np.float32)
    scale_full = np.full((n_dur, n_lat, n_lon), np.nan, dtype=np.float32)

    if include_am:
        am_full = np.full((n_dur, n_years, n_lat, n_lon), np.nan, dtype=np.float32)

    for r in results:
        if r["status"] != "success":
            continue
        ls, le = r["lat_start"], r["lat_end"]
        os_, oe = r["lon_start"], r["lon_end"]

        rp_full[:, :, ls:le, os_:oe] = r["rp_precip"]
        loc_full[:, ls:le, os_:oe] = r["dist_location"]
        scale_full[:, ls:le, os_:oe] = r["dist_scale"]

        if include_am and "annual_maxima" in r:
            am_full[:, :, ls:le, os_:oe] = r["annual_maxima"]

    # ── Build xarray Dataset ──
    logger.info("Building xarray Dataset...")

    data_vars = {
        "return_period_precip": (
            ["duration", "return_period", "lat", "lon"],
            rp_full,
            {
                "long_name": "Precipitation depth for return period",
                "units": "mm",
                "description": (
                    "Precipitation accumulation expected to be exceeded "
                    "once per return period (Gumbel method-of-moments)"
                ),
            },
        ),
        "dist_location": (
            ["duration", "lat", "lon"],
            loc_full,
            {
                "long_name": "Gumbel location parameter (mu)",
                "units": "mm",
            },
        ),
        "dist_scale": (
            ["duration", "lat", "lon"],
            scale_full,
            {
                "long_name": "Gumbel scale parameter (beta)",
                "units": "mm",
            },
        ),
    }

    if include_am:
        data_vars["annual_maxima"] = (
            ["duration", "year", "lat", "lon"],
            am_full,
            {
                "long_name": "Annual maximum precipitation accumulation",
                "units": "mm",
            },
        )

    coords = {
        "duration": ("duration", dur_names, {"long_name": "Accumulation duration"}),
        "return_period": (
            "return_period",
            RETURN_PERIODS,
            {"units": "years", "long_name": "Return period"},
        ),
        "lat": ("lat", lat_vals, {"units": "degrees_north", "long_name": "Latitude"}),
        "lon": ("lon", lon_vals, {"units": "degrees_east", "long_name": "Longitude"}),
    }
    if include_am:
        coords["year"] = ("year", year_list, {"long_name": "Calendar year"})

    ds_out = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs={
            "title": "CMORPH East Africa Precipitation Return Period Analysis",
            "source": "NOAA CDR CMORPH v1.0",
            "source_store": source_path,
            "institution": "Generated using Coiled + pencil-chunked Zarr",
            "history": f"Created {datetime.now().isoformat()}",
            "distribution": "Gumbel (method of moments)",
            "Conventions": "CF-1.8",
            "n_years": n_years,
            "year_range": f"{year_list[0]}-{year_list[-1]}",
            "chunks_total": n_chunks,
            "chunks_successful": n_success,
            "chunks_failed": n_fail,
        },
    )

    # ── Write output ──
    output_path = args.output
    logger.info(f"Writing output to {output_path}...")

    if output_path.startswith("gs://"):
        # Write to temp file, then upload via gcsfs
        import gcsfs
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
            tmp_path = tmp.name
        ds_out.to_netcdf(tmp_path)
        fs = gcsfs.GCSFileSystem(token=sa_info)
        fs.put(tmp_path, output_path)
        Path(tmp_path).unlink()
        logger.info(f"Uploaded to {output_path}")
    else:
        ds_out.to_netcdf(output_path)

    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024) if not output_path.startswith("gs://") else 0
    elapsed = time.time() - overall_start

    logger.info("=" * 70)
    logger.info("COMPUTE COMPLETE")
    logger.info(f"  Output: {output_path}")
    if file_size_mb > 0:
        logger.info(f"  Size: {file_size_mb:.1f} MB")
    logger.info(f"  Dimensions: {dict(ds_out.sizes)}")
    logger.info(f"  Chunks OK/fail: {n_success}/{n_fail}")
    logger.info(f"  Time: {elapsed / 60:.1f} min")
    logger.info("=" * 70)


# ─── verify subcommand ─────────────────────────────────────────────────────


def run_verify(args):
    """Validate output NetCDF for physical reasonableness."""
    import xarray as xr

    logger.info("=" * 70)
    logger.info("VERIFY: Checking return period output")
    logger.info("=" * 70)

    input_path = args.input

    if input_path.startswith("gs://"):
        with open(args.service_account) as f:
            sa_info = json.load(f)
        import gcsfs
        fs = gcsfs.GCSFileSystem(token=sa_info)
        ds = xr.open_dataset(fs.open(input_path))
    else:
        ds = xr.open_dataset(input_path)

    logger.info(f"Dataset:\n{ds}")
    logger.info(f"\nDimensions: {dict(ds.sizes)}")

    rp = ds["return_period_precip"]
    loc = ds["dist_location"]
    scale = ds["dist_scale"]

    n_lat = ds.sizes["lat"]
    n_lon = ds.sizes["lon"]
    total_pixels = n_lat * n_lon

    # ── NaN fraction ──
    logger.info("\n--- NaN Fraction ---")
    for d_idx, dur in enumerate(ds.duration.values):
        nan_frac = float(np.isnan(rp.values[d_idx]).mean())
        logger.info(f"  {dur}: {nan_frac * 100:.1f}% NaN ({int(nan_frac * total_pixels)}/{total_pixels} pixels)")

    # ── Monotonicity check: higher return period → higher precip ──
    logger.info("\n--- Monotonicity Check (RP ordering) ---")
    rp_vals = rp.values  # (n_dur, n_rp, n_lat, n_lon)
    all_monotonic = True
    for d_idx, dur in enumerate(ds.duration.values):
        layer = rp_vals[d_idx]  # (n_rp, n_lat, n_lon)
        violations = 0
        valid_pixels = 0
        for r_idx in range(1, layer.shape[0]):
            mask = ~np.isnan(layer[r_idx]) & ~np.isnan(layer[r_idx - 1])
            valid_pixels += int(mask.sum())
            violations += int((layer[r_idx][mask] < layer[r_idx - 1][mask]).sum())

        status = "PASS" if violations == 0 else "FAIL"
        if violations > 0:
            all_monotonic = False
        logger.info(f"  {dur}: {status} ({violations} violations out of {valid_pixels} comparisons)")

    # ── Duration ordering: longer duration → higher precip at same RP ──
    logger.info("\n--- Duration Ordering Check ---")
    dur_order = ["30min", "1hr", "3hr", "6hr", "12hr", "24hr", "48hr", "72hr", "7day"]
    dur_to_idx = {str(d): i for i, d in enumerate(ds.duration.values)}
    all_dur_ok = True

    for r_idx, rp_val in enumerate(ds.return_period.values):
        violations = 0
        valid_pixels = 0
        for i in range(1, len(dur_order)):
            d_curr = dur_to_idx.get(dur_order[i])
            d_prev = dur_to_idx.get(dur_order[i - 1])
            if d_curr is None or d_prev is None:
                continue
            curr = rp_vals[d_curr, r_idx]
            prev = rp_vals[d_prev, r_idx]
            mask = ~np.isnan(curr) & ~np.isnan(prev)
            valid_pixels += int(mask.sum())
            violations += int((curr[mask] < prev[mask]).sum())

        status = "PASS" if violations == 0 else f"WARN ({violations} violations)"
        if violations > 0:
            all_dur_ok = False
        logger.info(f"  RP={rp_val}yr: {status} (out of {valid_pixels} comparisons)")

    # ── Nairobi spot check (lat ~-1.3, lon ~36.8) ──
    logger.info("\n--- Nairobi Spot Check (lat≈-1.3, lon≈36.8) ---")
    try:
        nairobi = rp.sel(lat=-1.3, lon=36.8, method="nearest")
        logger.info(f"  Nearest pixel: lat={float(nairobi.lat):.2f}, lon={float(nairobi.lon):.2f}")
        for d_idx, dur in enumerate(ds.duration.values):
            vals = [f"{float(nairobi.values[d_idx, r]):.1f}" for r in range(len(ds.return_period))]
            rp_labels = [str(int(x)) for x in ds.return_period.values]
            pairs = ", ".join(f"{l}yr={v}mm" for l, v in zip(rp_labels, vals))
            logger.info(f"  {dur}: {pairs}")

        # Highlight 24hr 100yr
        val_24h_100yr = float(nairobi.sel(duration="24hr", return_period=100).values)
        if 30 <= val_24h_100yr <= 300:
            logger.info(f"  24hr 100yr = {val_24h_100yr:.1f} mm — REASONABLE")
        else:
            logger.info(f"  24hr 100yr = {val_24h_100yr:.1f} mm — UNEXPECTED (expected 30-300 mm)")
    except Exception as e:
        logger.warning(f"  Nairobi spot check failed: {e}")

    # ── Summary statistics ──
    logger.info("\n--- Summary Statistics ---")
    for d_idx, dur in enumerate(ds.duration.values):
        layer = rp_vals[d_idx]
        with np.errstate(invalid="ignore"):
            logger.info(
                f"  {dur}: min={np.nanmin(layer):.1f}, "
                f"median={np.nanmedian(layer):.1f}, "
                f"max={np.nanmax(layer):.1f} mm"
            )

    # ── Overall verdict ──
    logger.info("\n" + "=" * 70)
    if all_monotonic:
        logger.info("MONOTONICITY: PASS (all return periods correctly ordered)")
    else:
        logger.info("MONOTONICITY: FAIL (some pixels have non-monotonic return periods)")

    if all_dur_ok:
        logger.info("DURATION ORDER: PASS (longer durations yield higher precip)")
    else:
        logger.info("DURATION ORDER: WARN (some pixels violate duration ordering)")
    logger.info("=" * 70)


# ─── CLI ────────────────────────────────────────────────────────────────────


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="CMORPH East Africa — Precipitation Return Period Analysis",
    )
    sub = parser.add_subparsers(dest="command")

    # ── compute ──
    p_compute = sub.add_parser("compute", help="Run distributed return period analysis")
    p_compute.add_argument(
        "--source", type=str, default="gs://cpc_awc/cmorph_ea_pencil",
        help="Source pencil-chunked Zarr store path",
    )
    p_compute.add_argument(
        "--output", type=str, default="cmorph_ea_return_periods.nc",
        help="Output NetCDF path (local or gs://)",
    )
    p_compute.add_argument("--n-workers", type=int, default=20)
    p_compute.add_argument(
        "--service-account", type=str, default=SERVICE_ACCOUNT_FILE,
    )
    p_compute.add_argument(
        "--no-annual-maxima", action="store_true",
        help="Skip writing annual_maxima variable (~920 MB savings)",
    )

    # ── verify ──
    p_verify = sub.add_parser("verify", help="Validate output NetCDF")
    p_verify.add_argument(
        "--input", type=str, default="cmorph_ea_return_periods.nc",
        help="Input NetCDF path (local or gs://)",
    )
    p_verify.add_argument(
        "--service-account", type=str, default=SERVICE_ACCOUNT_FILE,
    )

    args = parser.parse_args()

    if args.command == "compute":
        run_compute(args)
    elif args.command == "verify":
        run_verify(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
