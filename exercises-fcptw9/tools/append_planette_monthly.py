"""Instructor tool, run once before the session (not part of the 60 minutes).
Adds SEAS5 starts to the East-Africa-only monthly store on source.coop (e4drr-project/forecasts/seas51_tp_icechunk_v2),
using Planette's daily SEAS5 as the source instead of a CDS download. Each start: 51 members, daily pr averaged to a
monthly mean rate (m/s) for forecast months 1-6, on the store's own 1 deg East Africa grid.

1. Reads every start in STARTS from Planette, one member chunk at a time, with retries (works on a slow line);
   keeps each converted start in out/planette_monthly_<start>.nc, so a rerun resumes.
2. Checks CHECK_START (in both stores) against the store: stops unless they agree.
3. Only with --write: appends the starts that are not yet in the store, as one icechunk commit.
   Credentials come from `source-coop creds` (run `source-coop login` yourself first, in your own terminal);
   they are passed to icechunk directly and never printed or saved.

Run:  uv run python tools/append_planette_monthly.py            # read, convert, check; no write
      uv run python tools/append_planette_monthly.py --write    # then append to source.coop
"""
import argparse
import datetime as dt
import json
import pathlib
import subprocess
import sys
import time

import dask
import icechunk
import numpy as np
import pandas as pd
import xarray as xr
from icechunk.xarray import to_icechunk

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "exercises"))
from common import open_seas5_history  # noqa: E402  (also drops stray AWS_* variables)

PREFIX = "forecasts/seas51_tp_icechunk_v2"
STARTS = ["2026-05-01", "2026-06-01", "2026-07-01", "2026-08-01", "2026-09-01"]
CHECK_START = "2026-04-01"               # already in the store: the Planette conversion must reproduce it
OUT = pathlib.Path("out")
OUT.mkdir(exist_ok=True)

SLOW_LINE = icechunk.RepositoryConfig(
    max_concurrent_requests=2,
    storage=icechunk.StorageSettings(
        retries=icechunk.StorageRetriesSettings(max_tries=12, initial_backoff_ms=2000, max_backoff_ms=60000),
        timeouts=icechunk.StorageTimeoutSettings(connect_timeout_ms=30000, read_timeout_ms=300000)))


def planette() -> xr.DataArray:
    storage = icechunk.s3_storage(bucket="planette-c3s-seasonal-forecasts", prefix="seas5_ic/forecast",
                                  region="us-east-2", anonymous=True, from_env=False,
                                  network_stream_timeout_seconds=300)
    session = icechunk.Repository.open(storage, config=SLOW_LINE).readonly_session("main")
    return xr.open_zarr(session.store, group="single/1platx1plon", consolidated=False).pr


def monthly(pr: xr.DataArray, start: str, lat, lon, members=slice(0, 50)) -> xr.DataArray:
    """Daily mean rate (kg m-2 s-1 = mm/s) -> monthly mean rate in m/s for forecast months 1-6 (1 = the start month)."""
    t0 = pd.Timestamp(start)
    ends = [t0 + pd.DateOffset(months=k) for k in range(7)]
    last_lead = (ends[6] - t0).days                         # lead 1 = the start day, so lead (ends[6]-t0).days = last day
    sub = pr.sel(init_time=start, number=members, lat=lat, lon=lon, lead=slice(1, last_lead))
    parts = []
    for m0 in range(0, sub.sizes["number"], 10):            # one member chunk at a time: a few requests, retried
        t = time.time()
        with dask.config.set(scheduler="synchronous"):
            parts.append(sub.isel(number=slice(m0, m0 + 10)).load())
        print(f"    {start} members {m0}-{min(m0 + 9, sub.sizes['number'] - 1)}: {time.time() - t:.0f} s", flush=True)
    daily = xr.concat(parts, "number")
    valid = t0 + pd.to_timedelta(daily.lead.values - 1, unit="D")
    month = np.searchsorted(np.array(ends[1:], dtype="datetime64[ns]"), valid.values, side="right") + 1
    out = (daily.assign_coords(forecastMonth=("lead", month)).groupby("forecastMonth").mean("lead") / 1000.0)
    return (out.drop_vars(["init_time"], errors="ignore").rename(lat="latitude", lon="longitude")
            .transpose("number", "forecastMonth", "latitude", "longitude").astype("float32")
            .expand_dims(time=[np.datetime64(start, "ns")], axis=2))


def source_coop_credentials() -> icechunk.S3StaticCredentials:
    """Temporary keys from `source-coop creds` (after `source-coop login`). Never printed."""
    c = json.loads(subprocess.run(["source-coop", "creds"], capture_output=True, text=True, check=True).stdout)
    exp = c.get("Expiration")
    return icechunk.S3StaticCredentials(access_key_id=c["AccessKeyId"], secret_access_key=c["SecretAccessKey"],
                                        session_token=c.get("SessionToken"),
                                        expires_after=dt.datetime.fromisoformat(exp.replace("Z", "+00:00")) if exp else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="append to source.coop (needs `source-coop login` first)")
    args = ap.parse_args()

    store = open_seas5_history()
    lat, lon = store.latitude.values, store.longitude.values
    have = set(pd.to_datetime(store.time.values).strftime("%Y-%m-%d"))
    print(f"store: {store.sizes['time']} starts, last {str(store.time.values[-1])[:10]}; grid {len(lat)} x {len(lon)}")

    pr = None
    conv = {}
    for start in [CHECK_START] + STARTS:
        f = OUT / f"planette_monthly_{start}.nc"
        if f.exists():
            conv[start] = xr.open_dataarray(f).load()
            print(f"  {start}: kept from an earlier run ({f})")
            continue
        pr = pr if pr is not None else planette()
        print(f"  {start}: reading Planette", flush=True)
        # the check needs only one member chunk (10 members); new starts need all 51
        da = monthly(pr, start, lat, lon, slice(0, 9) if start == CHECK_START else slice(0, 50)).rename("tprate")
        da.to_netcdf(f)
        conv[start] = da

    # Check: the Planette conversion must reproduce a start that is already in the store.
    new = conv[CHECK_START].isel(time=0)
    ref = store.tprate.sel(time=CHECK_START, number=new.number).load()
    r = float(xr.corr(ref, new))
    rel = float(abs(new - ref).mean() / abs(ref).mean())
    print(f"check {CHECK_START}: correlation {r:.4f}, mean relative difference {rel:.2%}")
    if r < 0.999 or rel > 0.01:
        sys.exit("check FAILED: the conversion does not match the store; nothing written")

    todo = [s for s in STARTS if s not in have]
    print(f"starts to append: {todo or 'none'}")
    if not args.write or not todo:
        print("no write (run with --write to append)" if todo else "nothing to do")
        return

    ds = xr.concat([conv[s] for s in todo], "time").to_dataset(name="tprate")
    ds = ds.assign_coords(surface=store.surface.values) if "surface" in store.coords else ds
    for v in ds.variables.values():
        v.encoding = {}
    ds.tprate.attrs = store.tprate.attrs
    storage = icechunk.s3_storage(bucket="e4drr-project", prefix=PREFIX, endpoint_url="https://data.source.coop",
                                  region="us-east-1", force_path_style=True,
                                  get_credentials=source_coop_credentials, scatter_initial_credentials=True)
    session = icechunk.Repository.open(storage).writable_session("main")
    to_icechunk(ds, session, append_dim="time")
    snap = session.commit(f"Append SEAS5 starts {todo[0][:7]}..{todo[-1][:7]} from Planette C3S daily "
                          f"(monthly mean rate, 51 members); check {CHECK_START}: r={r:.4f}, rel diff {rel:.2%}")
    print(f"appended {len(todo)} starts, commit {snap}")
    after = open_seas5_history()
    print(f"store now: {after.sizes['time']} starts, last {str(after.time.values[-1])[:10]}")


if __name__ == "__main__":
    main()
