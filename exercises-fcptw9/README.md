# Hands-on: drought thresholds and triggers for October-December 2026

*9th Foundational Climate Prediction Training Workshop, ICPAC · Hands-on: evaluating drought triggers and thresholds
(targeting seasonal forecasting) · 60 minutes*

Four short Python scripts take you through the threshold-and-trigger chain of the ICPAC drought anticipatory-action
system (the `kmj` branch of [ibf-thresholds-triggers](https://github.com/icpac-igad/ibf-thresholds-triggers)). They
use cloud-native (ARCO) stores, read anonymously, so nothing has to be downloaded in advance:

1. **Threshold:** how dry an October-December (OND) season has to be to count as "extreme", from 45 years of
   observed SPI-3.
2. **Forecast:** SPI-3 for OND 2026, computed with xclim from the 51 members of the SEAS5 forecast started on
   1 September 2026.
3. **Trigger:** the share of members below the threshold, and the probability at which to act, tested on 1991-2020.

The exercises work for any area of the eastern Horn. Set your own town and box once, in `exercises/common.py`.

## Data

| Store | What | Layout | Used in |
|---|---|---|---|
| `source.coop/e4drr-project/observations/era5_ecmwf_pencil` | ERA5 SPI-1 to SPI-48, monthly 1940-2026, 0.25° | Zarr, **pencil** chunks (1033 months × 5 × 5) | 1, 2 |
| `source.coop/e4drr-project/observations/era5_ecmwf_rp_icechunk` | SPI drought thresholds for 3-50 year return periods | Icechunk, **pan** chunks (one map) | 1, 2 |
| `s3://planette-c3s-seasonal-forecasts/seas5_ic/forecast` (Planette, C3S) | SEAS5 daily forecasts, 1°, 51 members, starts 2017-01 to 2026-09, 215 days: only the **2026-09 start** is read | Icechunk, chunks (10 members, 1 start, 46 days, whole globe) | 3 |
| `source.coop/e4drr-project/forecasts/seas51_tp_icechunk_v2` | SEAS5 monthly rainfall, **East Africa only**, 1°, starts 1981-01 to 2026-04: the model climatology | Icechunk, small regional chunks | 3 |

SPI-3 labelled December is the standardised October + November + December total.

## Setup

```bash
git clone -b kmj https://github.com/icpac-igad/ibf-thresholds-triggers.git
cd ibf-thresholds-triggers/exercises-fcptw9
curl -LsSf https://astral.sh/uv/install.sh | sh      # if uv is missing; no sudo needed
uv sync                                              # Python 3.12 environment from pyproject.toml
```

On Google Colab or JupyterHub, run the same commands in a terminal. `uv` builds its own environment and does not use
or change the notebook's Python packages.

## The 60 minutes

| Time | Script | Question | Look for |
|---|---|---|---|
| 0-5 | | Introduction: threshold, forecast probability, trigger | |
| 5-15 | `01_open_spi_stores.py` | Load: how dry were past OND seasons at my town? | 1 chunk for an 86-year series; the driest years (2025, 2010, 1998, 2005, 2022 at Garissa) |
| 15-25 | `02_ond_thresholds.py` | Threshold: what is a 1-in-5 and a 1-in-10 year dry OND, pixel by pixel? | Empirical vs fitted thresholds; OND vs all-months thresholds; masked dry-season pixels |
| 25-45 | `03_seas5_ond_spi3.py` | Forecast: what SPI-3 does SEAS5 give for OND 2026? | xclim gamma fit on the model's own climatology; the spread of the 51 members |
| 45-60 | `04_trigger_probability.py` | Trigger: at what probability would we act, and where does it trigger in 2026? | Hit rate vs false alarms per trigger; the hatched "act" map |

Run each script with `uv run python exercises/0N_....py`. Plots go to `out/`.

**Exercise 3 reads the most.** Everything is read in place from the cloud: no files are downloaded. Only the
2026-09 start comes from Planette, but each of its chunks covers the whole globe, so the East Africa box still reads
18 chunks (about 500 MB): seconds on Colab, minutes on a slow office line. The climatology comes from a small
East-Africa-only store. Compare with exercise 1, where an 86-year series is a single pencil chunk: the chunk layout
decides the cost. The small result is kept in `out/03_seas5_ond_totals.nc`, so later runs read nothing.

## Things to check (and discuss)

- **Units.**
  - SEAS5 `pr` is a daily mean rate (kg m⁻² s⁻¹ = mm s⁻¹): × 86400 gives mm per day.
  - SPI has no units.
- **Lead time.** For a 1 September start, lead day 1 is 1 September, so OND is lead days 31-122.
- **Latitude order.** Both stores list latitude from north to south.
- **Two different climatologies.**
  - The threshold comes from ERA5 (1981-2025).
  - The forecast SPI comes from the model's own September starts (1991-2020, 25 members).
  - SPI makes the two comparable: each is "how unusual for its own record".
- **The trigger test is in-sample.** The 30 test seasons (1991-2020, 25 members, so steps of 4%) are the same years the
  SPI was fitted on. The hindcast has 25 members and the 2026 forecast has 51.
- **Comparison with the `kmj` branch.**
  - A 1-in-5 year threshold corresponds to SPI-3 ≈ −0.84, one of the `kmj` thresholds (`--thresholds="-0.68,-0.84"`).
  - Compare the trigger chosen here with the `kmj` trigger of 15.2%.

## Going further

- Set `RETURN_PERIOD = 10` in exercise 4. What happens to the hit rate, and to the area triggered?
- Move `BOX`, `TOWN`, `LAT` and `LON` in `common.py` to your own country, and rerun exercises 1-4. Delete
  `out/03_seas5_ond_totals.nc` first.
- Average the probability over a district or county instead of using pixels (see `08-kmj-district-stats.py`).
