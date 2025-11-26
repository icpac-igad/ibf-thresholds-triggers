# SPI-3 Calculation from GRIB Files

This document explains how to use the `02-run-spi3-from-grib.py` script to calculate SPI-3 (Standardized Precipitation Index with 3-month window) for each forecast month in a SEAS51 GRIB file.

## Overview

The script processes SEAS51 forecast data in GRIB format and calculates SPI-3 for:
- **All forecast months (lead times)**: Typically 1-6 months
- **All ensemble members**: Usually 51 members (0-50)
- **All initialization times**: Historical forecasts from 1981 onwards

## Key Features

1. **Automatic handling of ensemble members**:
   - Members 0-24: Use calibration period 1991-2018
   - Members 25-50: Use calibration period 2017-2024

2. **Quality control**:
   - Checks for excessive NaN values
   - Skips problematic members/months
   - Logs warnings and errors

3. **Flexible output**: Saves results as NetCDF with proper dimensions and metadata

## Usage

### Basic Usage

```bash
python 02-run-spi3-from-grib.py --grib-file /srv/e401d9798000628d11a618c66a04372c.grib --output-dir ./output
```

### Advanced Usage

```bash
python 02-run-spi3-from-grib.py \
    --grib-file /srv/e401d9798000628d11a618c66a04372c.grib \
    --output-dir ./output \
    --cal-start 1991-01-01 \
    --cal-end 2018-01-01 \
    --log-file spi3_processing.log
```

## Command-Line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--grib-file` | Yes | - | Path to input GRIB file |
| `--output-dir` | No | `.` (current dir) | Output directory for results |
| `--cal-start` | No | `1991-01-01` | Calibration period start date |
| `--cal-end` | No | `2018-01-01` | Calibration period end date |
| `--log-file` | No | None | Path to save detailed logs |

## Expected Input Format

The GRIB file should have the following structure:
- **Dimensions**:
  - `number`: Ensemble members (e.g., 51 members)
  - `forecastMonth`: Lead times (e.g., 1-6 months)
  - `time`: Initialization times (e.g., 1981-01 to 2024-12)
  - `latitude`: Spatial dimension
  - `longitude`: Spatial dimension
- **Variable**: `tprate` (precipitation rate)

## Output Format

The output NetCDF file will have:
- **Dimensions**:
  - `member`: Ensemble members
  - `lead`: Forecast lead times (1-6 months)
  - `init`: Initialization times
  - `lat`: Latitude
  - `lon`: Longitude
- **Variable**: `spi3` (Standardized Precipitation Index)

## Example with Your Data

For the file at `/srv/e401d9798000628d11a618c66a04372c.grib`:

```bash
# Create output directory
mkdir -p ./spi3_output

# Run the script
python 02-run-spi3-from-grib.py \
    --grib-file /srv/e401d9798000628d11a618c66a04372c.grib \
    --output-dir ./spi3_output \
    --log-file ./spi3_output/processing.log

# The output will be: ./spi3_output/e401d9798000628d11a618c66a04372c_spi3.nc
```

## Processing Time

Depending on the size of your GRIB file:
- **Small datasets** (few years): 1-5 minutes
- **Large datasets** (40+ years, 51 members, 6 lead times): 10-30 minutes

## Troubleshooting

### Error: "No valid members for forecast month X"
- Check if your GRIB file has data for all forecast months
- Verify that precipitation values are not all NaN

### Error: "No 'number' dimension found in dataset"
- Your GRIB file might not have ensemble members
- Try opening the file manually with `xarray` to inspect its structure

### Warning: "X% NaN values in input data"
- Some grid points or time periods have missing data
- The script will skip members with >90% NaN values

## Inspecting Results

To quickly inspect the output:

```python
import xarray as xr

# Open the output file
ds = xr.open_dataset('./spi3_output/e401d9798000628d11a618c66a04372c_spi3.nc')

# View structure
print(ds)

# Check dimensions
print(ds.dims)

# Look at SPI-3 values for first lead time, first member, first init time
print(ds.spi3.isel(lead=0, member=0, init=0))
```

## Notes

1. **Memory usage**: Large GRIB files may require significant RAM. Monitor your system resources.

2. **Calibration periods**: The script uses different calibration periods for different ensemble members following the logic from the original script.

3. **Units**: Precipitation is converted to mm/month before calculating SPI-3.

4. **Missing values**: The script handles NaN values gracefully and reports statistics in the logs.
