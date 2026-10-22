"""Exercise 2 (Threshold): how dry is "extreme" for October-December, pixel by pixel?
Reads 45 OND SPI-3 values (1981-2025) for every pixel of BOX from the pencil store, then derives the 1-in-5 and
1-in-10 year dry thresholds two ways: empirical (a quantile of the 45 values) and fitted (a normal distribution
fitted with xclim, then its 1/T quantile). Compares them with the all-months thresholds of the pan store.
Run:  uv run python exercises/02_ond_thresholds.py
"""
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from xclim.indices import stats

from common import BOX, LAT, LON, OUT, SEASON_END, TOWN, add_admin, open_spi_pencil, open_spi_thresholds

warnings.filterwarnings("ignore", "All-NaN slice")                 # sea pixels have no SPI
PERIODS = [5, 10]                                          # return periods in years: 1-in-5 and 1-in-10 dry seasons

spi = open_spi_pencil()
sub = spi.SPI3.sel(**BOX)
n_chunks = int(np.ceil(sub.sizes["lat"] / 5) * np.ceil(sub.sizes["lon"] / 5)) + 1
print(f"BOX = {sub.sizes['lat']} x {sub.sizes['lon']} pixels: about {n_chunks} pencil chunks (a map from pencils is costly)")

t0 = time.time()
ond = sub.sel(time=(sub.time.dt.month == SEASON_END)).sel(time=slice("1981", "2025")).load()
print(f"read {ond.sizes['time']} OND seasons x {ond.sizes['lat'] * ond.sizes['lon']} pixels in {time.time() - t0:.0f} s")

# Where October-December is not a rainy season (e.g. Afar, Djibouti), the OND series has no dry tail: one season in
# five is not even below normal, so a "dry threshold" has no meaning there. Mask those pixels.
land = ond.notnull().any("time")
rainy = ond.quantile(0.2, dim="time") < -0.3
ond = ond.where(rainy)
print(f"kept {int(rainy.sum())} of {int(land.sum())} land pixels; masked {int((land & ~rainy).sum())} where OND is not a "
      f"rainy season (no dry tail)")

# Empirical: the SPI value that 1 season in T is drier than.
emp = ond.quantile([1 / T for T in PERIODS], dim="time").assign_coords(quantile=PERIODS).rename(quantile="return_period")

# Fitted: a normal distribution per pixel (SPI is meant to be standard normal), then its 1/T quantile. Watch out:
# a few seasons with no rain at all get extreme SPI values (down to -8) in this product and stretch the fit.
params = stats.fit(ond, dist="norm", dim="time")
fit = xr.concat([stats.dist_method("ppf", params, 1 / T) for T in PERIODS], dim="return_period")
fit = fit.assign_coords(return_period=PERIODS)

allmonths = (open_spi_thresholds().empirical_threshold.sel(spi_period="SPI3", return_period=PERIODS)
             .sel(**BOX).load().where(rainy))

obs = ond.assign_coords(year=ond.time.dt.year).swap_dims(time="year").drop_vars("time")  # observed OND SPI-3, for exercise 4
out = xr.Dataset({"ond_empirical": emp, "ond_fitted": fit, "allmonths_empirical": allmonths, "ond_spi3": obs})
out.attrs["description"] = "SPI-3 dry thresholds and OND SPI-3 (Dec label) 1981-2025 from the ERA5 pencil store; all-months thresholds from the pan store"
out.to_netcdf(OUT / "02_ond_thresholds.nc")

p = dict(lat=LAT, lon=LON, method="nearest")
print(f"\n{TOWN}: SPI-3 dry thresholds")
for T in PERIODS:
    print(f"  1-in-{T:>2} yr: OND empirical {float(emp.sel(return_period=T).sel(**p)):+.2f} | "
          f"OND fitted {float(fit.sel(return_period=T).sel(**p)):+.2f} | "
          f"all months {float(allmonths.sel(return_period=T).sel(**p)):+.2f}")
d = (emp - allmonths).sel(return_period=5)
print(f"BOX: OND minus all-months 1-in-5 threshold, median {float(d.median()):+.2f}, 5-95% "
      f"{float(d.quantile(0.05)):+.2f} to {float(d.quantile(0.95)):+.2f}")

fig, axs = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True)
panels = [(emp.sel(return_period=5), "OND, empirical, 1-in-5", "YlOrBr_r", (-1.4, 0)),
          (allmonths.sel(return_period=5), "All months (pan store), 1-in-5", "YlOrBr_r", (-1.4, 0)),
          (d, "OND minus all months", "PuOr", (-0.6, 0.6))]
for ax, (da, title, cmap, (lo, hi)) in zip(axs, panels):
    m = ax.pcolormesh(da.lon, da.lat, da, cmap=cmap, vmin=lo, vmax=hi, shading="nearest")
    ax.plot(LON, LAT, "k^", ms=6)
    add_admin(ax, inset=ax is axs[0])
    ax.set_title(title, fontsize=10)
    fig.colorbar(m, ax=ax, shrink=0.8, label="SPI-3")
fig.suptitle("SPI-3 dry thresholds (lower = drier needed), ERA5 1981-2025", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "02_ond_thresholds.png", dpi=130)
print(f"\nplot: {OUT / '02_ond_thresholds.png'}; thresholds saved for exercise 4: {OUT / '02_ond_thresholds.nc'}")
