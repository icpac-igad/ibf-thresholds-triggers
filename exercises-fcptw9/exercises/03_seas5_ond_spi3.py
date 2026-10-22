"""Exercise 3 (Forecast): SPI-3 for October-December 2026 from the SEAS5 September 2026 forecast, with xclim.
1. Forecast: daily rainfall of the 1 September 2026 start (51 members) from Planette, added up over October-December
   (lead days 31-122). The store's chunks cover the whole globe, so the East Africa box still reads 18 chunks.
2. Climatology: the same model's September starts 1991-2020 (25 members), from a small East-Africa-only monthly
   SEAS5 store on source.coop. SPI needs it: "dry" means dry compared with what this model usually gives.
3. SPI-3 with xclim: fit a gamma distribution to the 750 climatology totals at each pixel, then turn each 2026
   member's total into SPI-3: its gamma probability, expressed as a standard-normal value (the SPI definition).
The small result is kept in out/03_seas5_ond_totals.nc, so a second run reads nothing.
Run:  uv run python exercises/03_seas5_ond_spi3.py
"""
import time


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.stats import norm
from xclim.indices import stats

from common import BOX, CAL_MEMBERS, CAL_YEARS, FC_INIT, LAT, LON, OUT, TOWN, open_seas5, open_seas5_history

LEADS = slice(31, 122)                  # 1 Sep start: lead 1 = 1 Sep, so lead 31 = 1 Oct and lead 122 = 31 Dec
MIN_OND_MM = 50                         # below this the model has no real OND rainy season: SPI is meaningless
CACHE = OUT / "03_seas5_ond_totals.nc"


def ond_total(pr: xr.DataArray) -> xr.DataArray:
    """Daily mean rate (kg m-2 s-1 = mm/s) -> mm per day -> October-December total in mm."""
    return (pr.sel(lead=LEADS) * 86400).sum("lead").assign_attrs(units="mm", long_name="Oct-Dec total rainfall")


if CACHE.exists():
    tot = xr.open_dataset(CACHE)
    print(f"using the totals kept from the last run: {CACHE} (delete it to read the stores again)")
else:
    fc = open_seas5().pr.sel(**BOX)
    print(f"Planette pr chunks {fc.encoding['chunks']} (member, start, lead, lat, lon): the box still reads whole-globe chunks")
    t0 = time.time()
    new = ond_total(fc.sel(init_time=FC_INIT)).load()
    print(f"  forecast {FC_INIT}, 51 members: {time.time() - t0:.0f} s")

    # Monthly mean rate in m/s for the September starts; forecastMonth 1 = September, so 2-4 = October-December.
    hist = open_seas5_history().tprate.rename(latitude="lat", longitude="lon").sel(**BOX)
    sept = hist.sel(time=hist.time.dt.month == 9).sel(time=slice(str(CAL_YEARS[0]), str(CAL_YEARS[-1])))
    days = xr.DataArray([31, 30, 31], dims="forecastMonth", coords={"forecastMonth": [2, 3, 4]})
    t0 = time.time()
    cal = (sept.sel(forecastMonth=[2, 3, 4], number=CAL_MEMBERS) * 1000 * 86400 * days).sum("forecastMonth").load()
    cal = cal.assign_coords(year=cal.time.dt.year).swap_dims(time="year").drop_vars(["time", "surface"], errors="ignore")
    print(f"  climatology {CAL_YEARS[0]}-{CAL_YEARS[-1]}, {cal.sizes['number']} members: {time.time() - t0:.0f} s")
    tot = xr.Dataset({"forecast": new.drop_vars("init_time"), "climatology": cal.assign_attrs(units="mm")})
    tot.to_netcdf(CACHE)

# --- SPI-3 with xclim: fit a gamma distribution (location fixed at 0) to the 30 years x 25 members at each pixel.
clim = tot.climatology.stack(sample=("year", "number")).transpose("sample", ...)
rainy = clim.mean("sample") >= MIN_OND_MM
params = stats.fit(clim.where(rainy), dist="gamma", dim="sample", method="ML", floc=0)


def to_spi(total: xr.DataArray) -> xr.DataArray:
    """SPI = the standard-normal value with the same probability as the total under the fitted gamma."""
    p = stats.dist_method("cdf", params, total).clip(1e-6, 1 - 1e-6)
    return xr.apply_ufunc(norm.ppf, p).where(rainy).rename("spi3")


spi_fc = to_spi(tot.forecast)                                   # OND 2026, 51 members
spi_cal = to_spi(tot.climatology)                               # 1991-2020, 25 members: used to choose the trigger (exercise 4)
xr.Dataset({"spi3_forecast": spi_fc, "spi3_climatology": spi_cal, "rainy": rainy}).to_netcdf(OUT / "03_seas5_ond_spi3.nc")

p = dict(lat=LAT, lon=LON, method="nearest")
town_fc, town_tot = spi_fc.sel(**p), tot.forecast.sel(**p)
print(f"\n{TOWN} (nearest 1 deg pixel {float(town_fc.lat):.1f}, {float(town_fc.lon):.1f}), OND 2026 forecast:")
print(f"  OND total, median of 51 members: {float(town_tot.median()):.0f} mm "
      f"(model climatology median {float(tot.climatology.sel(**p).median()):.0f} mm)")
print(f"  SPI-3 median {float(town_fc.median()):+.2f}; members below 0: {int((town_fc < 0).sum())} of 51; "
      f"below -1: {int((town_fc < -1).sum())} of 51")
print(f"  pixels with a model OND rainy season: {int(rainy.sum())} of {rainy.size}")

fig, axs = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw=dict(width_ratios=[1.2, 1]))
med = spi_fc.median("number")
m = axs[0].pcolormesh(med.lon, med.lat, med, cmap="BrBG", vmin=-2, vmax=2, shading="nearest")
axs[0].plot(LON, LAT, "k^", ms=6)
axs[0].set_title(f"SEAS5 start {FC_INIT}: OND 2026 SPI-3, median of 51 members", fontsize=10)
fig.colorbar(m, ax=axs[0], shrink=0.85, label="SPI-3 (dry < 0 < wet)")
cal_town = spi_cal.sel(**p)
axs[1].hist(cal_town.values.ravel(), bins=np.arange(-3, 3.01, 0.25), alpha=0.5, color="0.5",
            label="model climatology 1991-2020 (750)")
axs[1].hist(town_fc.values, bins=np.arange(-3, 3.01, 0.25), alpha=0.7, color="#a6611a", label="OND 2026 members (51)")
axs[1].set_title(f"{TOWN}: SPI-3 of each member", fontsize=10)
axs[1].set_xlabel("SPI-3")
axs[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "03_seas5_ond_spi3.png", dpi=130)
print(f"\nplot: {OUT / '03_seas5_ond_spi3.png'}; SPI-3 saved for exercise 4: {OUT / '03_seas5_ond_spi3.nc'}")
