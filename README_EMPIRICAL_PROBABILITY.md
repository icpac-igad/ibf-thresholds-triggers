# Empirical Probability Calculation from SPI3 Forecasts

This document explains how to use the `03-calculate-empirical-probability.py` script to calculate empirical drought probabilities by comparing SPI3 ensemble forecasts with return period thresholds.

## Overview

The script takes two main inputs:
1. **SPI3 Ensemble Forecasts**: Generated from SEAS51 forecasts (from `02-run-spi3-from-grib.py`)
2. **Return Period Thresholds**: Pre-calculated SPI3 thresholds for different drought return periods

It then calculates three types of statistics:
1. **Empirical Probabilities**: Probability maps showing likelihood of exceeding each drought threshold
2. **Area Statistics**: Fraction of the region affected by drought
3. **Pixel Statistics**: Per-pixel ensemble statistics

## Methodology

### What is Empirical Probability?

For each forecast initialization time, lead time, and location:
- **Empirical Probability** = (Number of ensemble members with SPI3 ≤ threshold) / (Total ensemble members)

For example, if 15 out of 51 ensemble members have SPI3 values below the 10-year drought threshold at a location, the empirical probability is 15/51 = 0.29 (29% chance).

### Return Period Thresholds

The thresholds represent SPI3 values corresponding to different drought severities:
- **3-year return period**: Occurs ~33% of the time (moderate drought)
- **5-year return period**: Occurs ~20% of the time (moderate-severe drought)
- **10-year return period**: Occurs ~10% of the time (severe drought)
- **20-year return period**: Occurs ~5% of the time (extreme drought)
- **50-year return period**: Occurs ~2% of the time (exceptional drought)

Lower (more negative) SPI3 values indicate more severe drought conditions.

## Usage

### Basic Usage

```bash
python 03-calculate-empirical-probability.py \
    --spi3-file /srv/spi3_output/e401d9798000628d11a618c66a04372c_spi3.nc \
    --threshold-file /srv/spi_3_return_period_thresholds_20250805/spi_3_return_period_thresholds_20250805.nc \
    --output-dir ./empirical_probability_output
```

### Advanced Usage

```bash
python 03-calculate-empirical-probability.py \
    --spi3-file /srv/spi3_output/e401d9798000628d11a618c66a04372c_spi3.nc \
    --threshold-file /srv/spi_3_return_period_thresholds_20250805/spi_3_return_period_thresholds_20250805.nc \
    --output-dir ./empirical_probability_output \
    --return-periods 5 10 20 \
    --regrid-method bilinear \
    --log-file ./empirical_probability_output/processing.log
```

## Command-Line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--spi3-file` | Yes | - | Path to SPI3 forecast NetCDF file |
| `--threshold-file` | Yes | - | Path to return period threshold NetCDF file |
| `--output-dir` | No | `./empirical_probability_output` | Output directory |
| `--return-periods` | No | `3 5 10 20 50` | Return periods to calculate (years) |
| `--regrid-method` | No | `bilinear` | Regridding method for thresholds |
| `--log-file` | No | None | Path to save detailed logs |

## Output Files

The script generates 4 NetCDF files:

### 1. `empirical_probabilities.nc`

**Dimensions**: `(lead, init, lat, lon)`

**Variables**:
- `eprob_3yr`: Probability of 3-year drought (0-1)
- `eprob_5yr`: Probability of 5-year drought (0-1)
- `eprob_10yr`: Probability of 10-year drought (0-1)
- `eprob_20yr`: Probability of 20-year drought (0-1)
- `eprob_50yr`: Probability of 50-year drought (0-1)

**Use cases**:
- Create probability maps showing drought risk
- Identify regions with high drought probability
- Compare probabilities across different lead times
- Time series of drought probability at specific locations

### 2. `area_statistics.nc`

**Dimensions**: `(lead, member, init)` and `(lead, init)`

**Variables**:
- `area_frac_Xyr`: Fraction of area exceeding X-year threshold (per member)
- `area_frac_Xyr_ensmean`: Ensemble mean fraction of area affected

**Use cases**:
- Track spatial extent of drought over time
- Identify forecasts predicting widespread drought
- Calculate district/region-level drought coverage
- Compare ensemble spread in drought extent

### 3. `pixel_statistics.nc`

**Dimensions**: `(lead, init, lat, lon)`

**Variables**:
- `pixel_member_count_Xyr`: Number of members exceeding X-year threshold
- `pixel_member_frac_Xyr`: Fraction of members exceeding X-year threshold

**Use cases**:
- Identify locations with high ensemble agreement
- Map areas with uncertain predictions (low agreement)
- Quality control for ensemble forecasts
- Spatial patterns of forecast confidence

### 4. `thresholds_regridded.nc`

**Dimensions**: `(lat, lon)`

**Variables**:
- `spi_3_threshold_3yr`: 3-year drought threshold
- `spi_3_threshold_5yr`: 5-year drought threshold
- `spi_3_threshold_10yr`: 10-year drought threshold
- `spi_3_threshold_20yr`: 20-year drought threshold
- `spi_3_threshold_50yr`: 50-year drought threshold

**Use cases**:
- Reference for understanding local drought thresholds
- Verification that regridding was successful
- Input for other analyses

## Understanding the Outputs

### Empirical Probability Interpretation

| Probability | Interpretation |
|-------------|----------------|
| 0.0 - 0.2 | Low probability (0-20%) - Unlikely |
| 0.2 - 0.4 | Moderate probability (20-40%) - Possible |
| 0.4 - 0.6 | Medium probability (40-60%) - Likely |
| 0.6 - 0.8 | High probability (60-80%) - Very likely |
| 0.8 - 1.0 | Very high probability (80-100%) - Almost certain |

### Area Fraction Interpretation

| Area Fraction | Interpretation |
|---------------|----------------|
| 0.0 - 0.1 | < 10% of area affected - Localized |
| 0.1 - 0.3 | 10-30% affected - Scattered drought |
| 0.3 - 0.5 | 30-50% affected - Moderate coverage |
| 0.5 - 0.7 | 50-70% affected - Extensive drought |
| 0.7 - 1.0 | > 70% affected - Widespread drought |

## Example Analysis Workflow

### 1. Calculate Empirical Probabilities

```bash
# Run the script
python 03-calculate-empirical-probability.py \
    --spi3-file /srv/spi3_output/e401d9798000628d11a618c66a04372c_spi3.nc \
    --threshold-file /srv/spi_3_return_period_thresholds_20250805/spi_3_return_period_thresholds_20250805.nc \
    --output-dir ./eprob_output
```

### 2. Analyze Results with Python

```python
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

# Load empirical probabilities
eprob = xr.open_dataset('./eprob_output/empirical_probabilities.nc')

# Example 1: Plot probability map for a specific forecast
# Lead time 2 (3 months ahead), latest initialization
lead_idx = 2
init_idx = -1  # Latest forecast

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot different drought severities
eprob['eprob_5yr'].isel(lead=lead_idx, init=init_idx).plot(ax=axes[0], vmin=0, vmax=1)
axes[0].set_title('5-year Drought Probability')

eprob['eprob_10yr'].isel(lead=lead_idx, init=init_idx).plot(ax=axes[1], vmin=0, vmax=1)
axes[1].set_title('10-year Drought Probability')

eprob['eprob_20yr'].isel(lead=lead_idx, init=init_idx).plot(ax=axes[2], vmin=0, vmax=1)
axes[2].set_title('20-year Drought Probability')

plt.tight_layout()
plt.savefig('drought_probability_maps.png')

# Example 2: Time series of district-averaged probability
# Average over spatial dimensions
district_avg = eprob['eprob_10yr'].mean(dim=['lat', 'lon'])

# Plot for each lead time
fig, ax = plt.subplots(figsize=(12, 6))
for lead in range(6):
    district_avg.isel(lead=lead).plot(ax=ax, label=f'Lead {lead+1}')
ax.set_xlabel('Initialization Time')
ax.set_ylabel('Probability of 10-year Drought')
ax.set_title('District-Averaged Drought Probability Over Time')
ax.legend()
ax.grid(True)
plt.savefig('drought_probability_timeseries.png')

# Example 3: Identify high-risk periods
# Find when probability exceeds 40% for 10-year drought
threshold = 0.4
high_risk = eprob['eprob_10yr'] > threshold
high_risk_count = high_risk.sum(dim=['lat', 'lon'])

print("Number of pixels at high risk (>40% probability):")
print(high_risk_count)

# Example 4: Analyze area statistics
area = xr.open_dataset('./eprob_output/area_statistics.nc')

# Plot area fraction time series
fig, ax = plt.subplots(figsize=(12, 6))
area['area_frac_10yr_ensmean'].isel(lead=2).plot(ax=ax)
ax.set_xlabel('Initialization Time')
ax.set_ylabel('Fraction of Area Affected')
ax.set_title('10-year Drought: Fraction of Area Affected (Lead 3)')
ax.grid(True)
plt.savefig('area_fraction_timeseries.png')
```

### 3. Export to CSV for Further Analysis

```python
import pandas as pd

# Load data
eprob = xr.open_dataset('./eprob_output/empirical_probabilities.nc')
area = xr.open_dataset('./eprob_output/area_statistics.nc')

# Create summary table for latest forecast
latest_init = eprob.init[-1].values

summary_data = []
for lead in range(6):
    for rp in [5, 10, 20]:
        # Spatial average probability
        mean_prob = float(eprob[f'eprob_{rp}yr'].isel(lead=lead, init=-1).mean())

        # Area fraction
        area_frac = float(area[f'area_frac_{rp}yr_ensmean'].isel(lead=lead, init=-1))

        summary_data.append({
            'Lead_Time': lead + 1,
            'Return_Period': rp,
            'Mean_Probability': mean_prob,
            'Area_Fraction': area_frac
        })

df = pd.DataFrame(summary_data)
df.to_csv('./eprob_output/summary_table.csv', index=False)
print(df)
```

## Performance Considerations

- **Processing time**: Typically 2-10 minutes depending on data size
- **Memory usage**: ~2-8 GB RAM for typical datasets
- **Output size**: Each output file is typically 50-500 MB

## Troubleshooting

### Error: "Coordinates don't match"
- Check that both input files use the same coordinate system
- Verify lat/lon naming conventions
- The script will automatically regrid thresholds to match forecast grid

### Warning: "Threshold variable not found"
- Your threshold file may use different naming conventions
- Check variable names in the threshold file
- Adjust return periods argument if needed

### Memory Issues
- Process smaller subsets by selecting specific lead times or time periods
- Use chunking in xarray for large datasets
- Close other applications to free up RAM

## Integration with Existing Workflow

This script fits into the workflow after:
1. `01-run-process-spi.py` - Calculate SPI3 from observations and forecasts
2. `02-run-spi3-from-grib.py` - Calculate SPI3 for each ensemble member

And before:
- `04-run-bar-plot.py` - Visualization
- `05-run-heatmap.py` - Analysis
- Other verification and plotting scripts

## References

- McKee, T. B., et al. (1993). The relationship of drought frequency and duration to time scales.
- Standardized Precipitation Index (SPI): https://climatedataguide.ucar.edu/climate-data/standardized-precipitation-index-spi
- Return Period Analysis: Understanding extreme events in climatology
