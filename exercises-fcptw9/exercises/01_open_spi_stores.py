"""Exercise 1 (Load): open the observed SPI stores and read the October-December SPI-3 history of one town.
Two layouts of the same ERA5 data: a PENCIL store (each chunk = 86 years at 5x5 pixels, ideal for a time series)
and a PAN store of thresholds (each chunk = one whole map, ideal for maps).
Run:  uv run python exercises/01_open_spi_stores.py
"""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import LAT, LON, OUT, SEASON_END, TOWN, open_spi_pencil, open_spi_thresholds

t0 = time.time()
spi = open_spi_pencil()
print(f"pencil store opened in {time.time() - t0:.1f} s: {dict(spi.sizes)}, variables {sorted(spi.data_vars)}")
print(f"  SPI3 chunks {spi.SPI3.encoding['chunks']}: one chunk holds every month of 5 x 5 pixels")

# One point, every month since 1940: exactly ONE chunk of the pencil store.
t0 = time.time()
series = spi.SPI3.sel(lat=LAT, lon=LON, method="nearest").load()
print(f"{TOWN} SPI-3 series, {series.size} months, read in {time.time() - t0:.1f} s (1 chunk)")

# SPI-3 labelled December = the October-December total, standardised. Keep 1981-2025, one value per year.
ond = series.sel(time=series.time.dt.month == SEASON_END).sel(time=slice("1981", "2025"))
ond = ond.assign_coords(year=ond.time.dt.year).swap_dims(time="year")

t0 = time.time()
rp = open_spi_thresholds()
thr = rp.empirical_threshold.sel(spi_period="SPI3").sel(lat=LAT, lon=LON, method="nearest").load()
print(f"threshold store opened and read in {time.time() - t0:.1f} s: return periods {rp.return_period.values.tolist()} years")
print("  (these thresholds use ALL months of the year; exercise 2 makes October-December thresholds)")

print(f"\n{TOWN} ({LAT}, {LON}), OND SPI-3 1981-2025:")
print(f"  driest 5 years: {', '.join(f'{int(y)} ({v:+.2f})' for y, v in ond.sortby(ond).to_series().head(5).items())}")
print(f"  wettest year:   {int(ond.idxmax())} ({float(ond.max()):+.2f})")
for T, v in zip(thr.return_period.values, thr.values):
    print(f"  1-in-{T:>2} year dry threshold (all months): SPI-3 <= {v:+.2f}")

fig, ax = plt.subplots(figsize=(9, 3.6))
ax.bar(ond.year, ond, color=["#b2182b" if v < 0 else "#2166ac" for v in ond.values], width=0.8)
for T, v, ls in zip(thr.return_period.values, thr.values, [":", "--", "-", "-.", (0, (1, 3))]):
    if T in (5, 10):
        ax.axhline(v, color="k", lw=1, ls=ls, label=f"1-in-{T} yr ({v:+.2f})")
ax.set_title(f"{TOWN}: October-December SPI-3, ERA5 1981-2025")
ax.set_ylabel("SPI-3 (dry < 0 < wet)")
ax.legend(loc="lower left", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "01_ond_spi3_series.png", dpi=130)
print(f"\nplot: {OUT / '01_ond_spi3_series.png'}")
