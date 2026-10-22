"""Exercise 4 (Trigger): would you trigger anticipatory action for a dry October-December 2026?
Threshold (exercise 2): the observed OND SPI-3 that 1 season in 5 is drier than, per pixel.
Probability: the share of the 51 SEAS5 members whose OND 2026 SPI-3 (exercise 3) is at or below that threshold.
Trigger: the probability at which you act. We choose it by testing every trigger on 1991-2020: the forecast
probabilities of those years (25 members) against what ERA5 then observed. Then we apply it to 2026.
Needs the outputs of exercises 2 and 3 in out/ (no download).
Run:  uv run python exercises/04_trigger_probability.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from common import CAL_YEARS, FC_INIT, LAT, LON, OUT, TOWN

RETURN_PERIOD = 5                                   # act on a 1-in-5 year dry season (try 10)
TRIGGERS = np.round(np.arange(0.04, 0.61, 0.04), 2) # candidate probabilities at which to act (25 members: steps of 4%)

obs = xr.open_dataset(OUT / "02_ond_thresholds.nc")
fc = xr.open_dataset(OUT / "03_seas5_ond_spi3.nc")

# ERA5 is 0.25 deg, SEAS5 is 1 deg: put the observed threshold and observed SPI-3 on the forecast grid.
thr = obs.ond_empirical.sel(return_period=RETURN_PERIOD).interp(lat=fc.lat, lon=fc.lon)
seen = obs.ond_spi3.sel(year=list(CAL_YEARS)).interp(lat=fc.lat, lon=fc.lon)
valid = thr.notnull() & fc.rainy

# --- Choose the trigger on 1991-2020: forecast probability (25 members) vs the observed dry season.
prob_past = (fc.spi3_climatology <= thr).mean("number").where(valid)
event = (seen <= thr).where(valid)
print(f"threshold: 1-in-{RETURN_PERIOD} year OND dry season; observed in "
      f"{float(event.mean()) * 100:.0f}% of pixel-seasons 1991-2020 (about {100 / RETURN_PERIOD:.0f}% expected)")
rows = []
for t in TRIGGERS:
    warn = prob_past >= t
    hits = int((warn & (event == 1)).sum()); misses = int((~warn & (event == 1)).sum())
    false = int((warn & (event == 0)).sum()); quiet = int((~warn & (event == 0)).sum())
    hr, pofd = hits / max(hits + misses, 1), false / max(false + quiet, 1)
    rows.append(dict(trigger=t, hit_rate=hr, false_alarm_ratio=false / max(hits + false, 1),
                     false_alarm_rate=pofd, peirce=hr - pofd, warnings=hits + false))
skill = pd.DataFrame(rows).set_index("trigger")
best = float(skill.peirce.idxmax())
print("\nWhich trigger worked best on 1991-2020 (all pixels pooled):")
print(skill.round(2).to_string())
print(f"best trigger by Peirce score (hit rate minus false alarm rate): {best:.0%}")

# --- Apply to OND 2026.
prob = (fc.spi3_forecast <= thr).mean("number").where(valid)
act = prob >= best
p = dict(lat=LAT, lon=LON, method="nearest")
print(f"\nOND {FC_INIT[:4]} from the {FC_INIT} start, threshold 1-in-{RETURN_PERIOD} yr, trigger {best:.0%}:")
print(f"  pixels where action is triggered: {int(act.sum())} of {int(valid.sum())} ({float(act.sum() / valid.sum()):.0%})")
print(f"  {TOWN}: threshold SPI-3 <= {float(thr.sel(**p)):+.2f}, forecast probability {float(prob.sel(**p)):.0%} "
      f"-> {'ACT' if bool(act.sel(**p)) else 'no action'}")

fig, axs = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw=dict(width_ratios=[1.2, 1]))
m = axs[0].pcolormesh(prob.lon, prob.lat, prob, cmap="YlOrBr", vmin=0, vmax=0.8, shading="nearest")
yy, xx = np.nonzero(act.values)                     # one dot per pixel where action is triggered
axs[0].plot(act.lon.values[xx], act.lat.values[yy], "k.", ms=4)
axs[0].plot(LON, LAT, "k^", ms=6)
axs[0].set_title(f"P(OND {FC_INIT[:4]} SPI-3 <= 1-in-{RETURN_PERIOD} yr level); dots = act (>= {best:.0%})", fontsize=10)
fig.colorbar(m, ax=axs[0], shrink=0.85, label="share of 51 members")
axs[1].plot(skill.index, skill.hit_rate, "o-", label="hit rate")
axs[1].plot(skill.index, skill.false_alarm_ratio, "s-", label="false alarm ratio")
axs[1].plot(skill.index, skill.peirce, "^-", label="Peirce score")
axs[1].axvline(best, color="k", ls="--", lw=1)
axs[1].set_xlabel("trigger (forecast probability)")
axs[1].set_title("Choosing the trigger on 1991-2020", fontsize=10)
axs[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "04_trigger.png", dpi=130)
print(f"\nplot: {OUT / '04_trigger.png'}")
