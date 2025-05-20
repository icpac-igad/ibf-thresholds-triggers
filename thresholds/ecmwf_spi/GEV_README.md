# GEV Return Period Calculator for SPI Data

This repository contains tools to calculate Generalized Extreme Value (GEV) return periods for Standardized Precipitation Index (SPI) data across different regions in East Africa.

## Overview

The main script `calculate_gev_return_periods.py` performs the following tasks:

1. Processes SPI data from NetCDF or Zarr formats
2. Creates region masks based on a GeoJSON file of administrative boundaries
3. Calculates return periods for drought and flood events using GEV distribution
4. Generates CSV output files and visualizations for different return periods

## Scripts

- `calculate_gev_return_periods.py`: Main script to calculate GEV return periods for SPI data
- `nc_to_zarr.py`: Script to convert NetCDF files to Zarr format for efficient data storage
- `upload_zarr_to_gcs.py`: Script to upload Zarr files to Google Cloud Storage

## Prerequisites

- Python 3.7+
- Required Python packages:
  - numpy
  - pandas
  - xarray
  - geopandas
  - regionmask
  - scipy
  - matplotlib
  - zarr (for Zarr file support)
  - gcsfs (for Google Cloud Storage support)

## Usage

### 1. Calculate Return Periods

```bash
python calculate_gev_return_periods.py
```

This will:
1. Load SPI data from NetCDF or Zarr format
2. Create masks for regions defined in the GeoJSON file
3. Calculate return periods for each region and month
4. Save results as CSV files in the `return_periods` directory
5. Generate visualization plots in the `return_periods/SPI*/plots` directories

### 2. Configuration

You can configure the script by modifying the following parameters at the top of the script:

```python
# Configuration
REGIONS_FILE = "icpac_regions.geojson"
SPI_DATA_DIR = "ecmwf_spi"
OUTPUT_DIR = "return_periods"
SPI_TYPES = ["SPI1", "SPI3", "SPI6", "SPI12", "SPI24", "SPI36", "SPI48"]
RETURN_PERIODS = [2, 4, 7, 10, 15, 20, 40, 100]  # in years
MONTHS = list(range(1, 13))  # 1 to 12
```

## Return Period Calculation

The script calculates return periods for both drought (negative SPI) and flood (positive SPI) events:

1. **Drought Return Periods**: For each region and month, negative SPI values are extracted, and a GEV distribution is fitted to their absolute values. Return levels are then calculated for specified return periods and converted back to negative values.

2. **Flood Return Periods**: For each region and month, positive SPI values are extracted, and a GEV distribution is fitted directly to them. Return levels are then calculated for specified return periods.

## Output Files

The script generates the following outputs:

1. **CSV Files**: For each SPI type and month, a CSV file is created containing return period values for all regions (`return_periods/SPI*/month_*.csv`).

2. **Plots**: For each region, plots are generated showing drought and flood return levels for different return periods (`return_periods/SPI*/plots/*.png`).

## Example Output

Here's an example of the CSV output:

```
region_id,region_name,month,drought_rp_2,flood_rp_2,drought_rp_4,flood_rp_4,...
1,Djibouti,1,-0.65,0.65,-1.04,1.06,...
2,Gitega,1,-0.63,0.57,-1.02,1.00,...
```

## Regional Focus

The script is designed to work with the ICPAC regions in East Africa, including:
- Djibouti
- Eritrea
- Ethiopia
- Kenya
- Somalia
- South Sudan
- Sudan
- Uganda
- Burundi
- Rwanda
- Tanzania

The regions are defined in the `icpac_regions.geojson` file.

## Data Sources

The SPI data used by this script can be derived from various precipitation datasets, such as:
- ERA5 reanalysis data
- CHIRPS (Climate Hazards Group InfraRed Precipitation with Station data)
- GPM (Global Precipitation Measurement)
- Local weather station data

## Notes

- The GEV fitting requires sufficient data points for reliable parameter estimation. A minimum of 30 samples is recommended.
- For regions with limited data, the GEV fitting may not converge, resulting in NaN values in the output.
- The script includes error handling to skip regions or months where GEV fitting fails.
