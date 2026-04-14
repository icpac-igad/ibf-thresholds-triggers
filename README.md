# Manual on Operational Thresholds and Triggers for Drought Anticipatory Action using SEAS5.1

---

## Introduction to Thresholds and Triggers

### What is a Threshold?

A threshold in drought anticipatory action represents a critical value derived from historical observations that indicates drought conditions. For this system, thresholds are calculated using:

- **Standardized Precipitation Index (SPI-3)**: A 3-month rolling precipitation index
- **Extreme Value Analysis**: Statistical analysis of long-term SPI3 observations (1981-present)
- **Focus**: Agricultural drought conditions affecting Karamoja region

### What is a Trigger?

A trigger is the forecast probability value that could be chosen to activate anticipatory action based on forecast verification. When the ensemble forecast probability of drought exceeds the trigger value, action is recommended.

- **Trigger Value**: Expressed as probability (0–1 scale), e.g., 0.152 = 15.2%
- **Calculation**: Proportion of ensemble members predicting drought conditions
- **Decision Rule**: If P(SPI < threshold) ≥ trigger, then ACTIVATE

### System Flow

```
Historical Data --> Threshold (SPI value)
     |
Forecast Data --> Empirical Probability --> Compare with Trigger --> Decision
     |
51 Ensemble Members --> Count below threshold / 51 = Probability
```

---

## Getting Started with GitHub

### Repository Access

The operational forecasting system is maintained at:

`https://github.com/icpac-igad/ibf-thresholds-triggers.git`

### Installation Steps

```bash
# Clone the repository
git clone https://github.com/icpac-igad/ibf-thresholds-triggers.git

# Navigate to project directory
cd ibf-thresholds-triggers

# Update to latest version (for existing installations)
git pull origin main
```

### Environment Setup

The Python environment is managed with **micromamba** using the
`devops/environment.yml` file. Full step-by-step instructions —
including installing micromamba, creating/activating the environment,
remote SSH/Jupyter access, and troubleshooting the common
`('U', 40)` GRIB/eccodes error at Step 01 — are in
[`devops/readme.md`](devops/readme.md).

Quick setup:

```bash
# Install micromamba (see devops/readme.md for details)
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)

# Create the drought_env environment
micromamba create -f devops/environment.yml

# Activate it
micromamba activate drought_env
```

### CDS API Setup (Required for data download)

1. Create an account at: <https://cds.climate.copernicus.eu/>
2. Accept license terms for "Seasonal forecast monthly statistics"
3. Create a `~/.cdsapirc` file:

   ```
   url: https://cds.climate.copernicus.eu/api/v2
   key: <your-uid>:<your-api-key>
   ```

---

## Lead Time and Seasonal Configuration

### Understanding Lead Time

Lead time represents the months between forecast initialization and the
target season's valid time. ECMWF SEAS5.1 provides 6 lead months, but
only specific initialization months are valid for each season.

### MAM Season (March–April–May)

Valid time: May (month 5)

| Init Month | Lead Index | Months Ahead | Example              |
|------------|------------|--------------|----------------------|
| December   | 4          | 5            | Dec 2025 → MAM 2026  |
| January    | 3          | 4            | Jan 2026 → MAM 2026  |
| February   | 2          | 3            | Feb 2026 → MAM 2026  |

### JJA Season (June–July–August)

Valid time: August (month 8)

| Init Month | Lead Index | Months Ahead | Example              |
|------------|------------|--------------|----------------------|
| March      | 4          | 5            | Mar 2026 → JJA 2026  |
| April      | 3          | 4            | Apr 2026 → JJA 2026  |
| May        | 2          | 3            | May 2026 → JJA 2026  |

**Key Points**

- Lead Index is 0-based for array indexing in scripts.
- December initialization targets NEXT year's MAM season.
- Scripts automatically calculate lead time from month/season combination.

---

## Running the Operational Pipeline

### Monthly Operational Forecasting

This mode processes the latest SEAS51 forecast for drought anticipatory action.

### Step 1: Download SEAS51 Data (`00-download-data.py`)

```bash
python 00-download-data.py \
    --output-dir ./run-test \
    --months 1-12 \
    --year-start 1981 \
    --year-end 2026 \
    --skip-unavailable
```

Check data availability:

```bash
python 00-download-data.py --check-availability --year 2026
```

### Step 2: Process SPI-3 (`01-run-process-spi.py`)

```bash
python 01-run-process-spi.py \
    --region-id kmj \
    --mode seas51 \
    --output-dir ./run-test \
    --use-local \
    --local-shapefile kmj_polygon.geojson \
    --seas51-main-file ./run-output/seas5_precipitation_*.grib \
    --apply-mask \
    --mask-buffer 0.25 \
    --output-year 2026 \
    --output-month 1 \
    --cleanup-intermediate
```

### Step 3: Generate Forecast Plots (`07-plot-sea51-forecast.py`)

```bash
python 07-plot-sea51-forecast.py \
    --region_id kmj \
    --season MAM \
    --lead_time 3 \
    --year 2026 \
    --month 1 \
    --threshold -0.68 \
    --trigger 0.152 \
    --use_shpfile \
    --shapefile_path kmj_polygon.geojson \
    --fct_file ./run-output/kmj_rgr_seas51_spi3_masked_2026_01.nc \
    --output_dir ./run-test
```

### Step 4: Calculate District Statistics (`08-kmj-district-stats.py`)

```bash
python 08-kmj-district-stats.py \
    --input_netcdf ./run-output/kmj_seas51_spi3_mam_eprob_2026_01_th0p68_tr15p2.nc \
    --district_shapefile karamoja_9_districts.geojson \
    --admin_level admin2
```

### Orchestrated Pipeline (`run_pipeline.py`)

Run all steps with a single command:

```bash
# MAM forecast from January 2026
python run_pipeline.py \
    --year 2026 \
    --month 1 \
    --season MAM \
    --output-dir ./run-output

# JJA forecast from April 2026
python run_pipeline.py \
    --year 2026 \
    --month 4 \
    --season JJA \
    --output-dir ./run-output

# Multiple thresholds/triggers
python run_pipeline.py \
    --year 2026 \
    --month 1 \
    --season MAM \
    --output-dir ./run-output \
    --thresholds="-0.68,-0.84" \
    --triggers="0.152,0.111"

# Dry run (preview commands)
python run_pipeline.py \
    --year 2026 \
    --month 1 \
    --season MAM \
    --output-dir ./run-output \
    --dry-run
```

### Output Files

- `seas5_precipitation_20260120_years1981-2025_months_12_months.grib` — Long-term SEAS5.1 forecast data 1981–2025
- `seas5_precipitation_20260120_year2026_months_01.grib` — Monthly forecast data available for 2026 (currently includes only January when run on 2026-01-23), obtained separately due to CDS API requirements
- `kmj_rgr_seas51_spi3_masked_2026_01.nc` — Masked SPI3 forecast
- `kmj_seas51_spi3_mam_eprob_2026_01_th0p68_tr15p2.nc` — Empirical probability
- `kmj_mam_lt3_th0p68_tr15p2.png` — Stamp plot with ensemble members
- `*_district_averages.csv` — District-level statistics

---

*Document Version 1.0 — January 2025*
