#!/usr/bin/env python3
"""
=================================================================================
SEAS51 DROUGHT FORECAST PIPELINE - Orchestration Script
=================================================================================

This script orchestrates the complete drought forecast processing pipeline for
anticipatory action decision-making. It runs four scripts in sequence:

  1. 00-download-data.py    - Download SEAS51 GRIB data from ECMWF CDS
  2. 01-run-process-spi.py  - Process GRIB to SPI3 NetCDF (regridded, masked)
  3. 07-plot-sea51-forecast.py - Generate forecast plots and probability maps
  4. 08-kmj-district-stats.py  - Calculate district-level statistics

=================================================================================
PREREQUISITES - CDS API SETUP (Required for Step 00 - Download)
=================================================================================

  To download SEAS51 data from ECMWF, you need to set up CDS API credentials:

  1. Create an account at: https://cds.climate.copernicus.eu/
  2. Accept the license terms for "Seasonal forecast monthly statistics"
  3. Get your API key from: https://cds.climate.copernicus.eu/api-how-to
  4. Create the credentials file ~/.cdsapirc with the following content:

     url: https://cds.climate.copernicus.eu/api/v2
     key: <your-uid>:<your-api-key>

  Example ~/.cdsapirc file:
     url: https://cds.climate.copernicus.eu/api/v2
     key: 12345:abcdef12-3456-7890-abcd-ef1234567890

  If the CDS API is not configured, use --skip-download and provide existing
  GRIB files in the output directory.

=================================================================================
USAGE EXAMPLES:
=================================================================================

  MAM forecast from January 2026 initialization:
  ----------------------------------------------
  python run_pipeline.py --year 2026 --month 1 --season MAM --output-dir run-test 

  MAM forecast from December 2025 (earliest lead time):
  -----------------------------------------------------
  python run_pipeline.py --year 2025 --month 12 --season MAM --output-dir run-test

  JJA forecast from March 2026 initialization:
  --------------------------------------------
  python run_pipeline.py --year 2026 --month 3 --season JJA --output-dir run-test

  Skip download if GRIB files already exist:
  ------------------------------------------
  python run_pipeline.py --year 2026 --month 1 --season MAM --output-dir run-test --skip-download

  Run only specific steps:
  ------------------------
  python run_pipeline.py --year 2026 --month 1 --season MAM --output-dir run-test --steps 1,7,8

  Multiple threshold/trigger combinations (NOTE: use = for negative values):
  --------------------------------------------------------------------------
  python run_pipeline.py --year 2026 --month 1 --season MAM --output-dir run-test --thresholds="-0.68,-0.84" --triggers="0.152,0.111"

  Dry run (show commands without executing):
  ------------------------------------------
  python run_pipeline.py --year 2026 --month 1 --season MAM --output-dir run-test --dry-run

=================================================================================
SEASON AND LEAD TIME REFERENCE:
=================================================================================

  IMPORTANT: The --month must be valid for the target --season!
  Lead time is automatically calculated based on the month/season combination.

  MAM Season (March-April-May) - Valid initialization months:
  -----------------------------------------------------------
    +-------------+------------+---------------+------------------+
    | Init Month  | Lead Index | Months Ahead  | Example          |
    +-------------+------------+---------------+------------------+
    | December    | 4          | 5             | Dec 2025 → MAM 2026
    | January     | 3          | 4             | Jan 2026 → MAM 2026
    | February    | 2          | 3             | Feb 2026 → MAM 2026
    +-------------+------------+---------------+------------------+

  JJA Season (June-July-August) - Valid initialization months:
  ------------------------------------------------------------
    +-------------+------------+---------------+------------------+
    | Init Month  | Lead Index | Months Ahead  | Example          |
    +-------------+------------+---------------+------------------+
    | March       | 4          | 5             | Mar 2026 → JJA 2026
    | April       | 3          | 4             | Apr 2026 → JJA 2026
    | May         | 2          | 3             | May 2026 → JJA 2026
    +-------------+------------+---------------+------------------+

  NOTE: For December initialization targeting MAM, the forecast year is
        automatically set to init_year + 1 (e.g., Dec 2025 → MAM 2026)

=================================================================================
VARIABLE REFERENCE:
=================================================================================

  --year          : Forecast initialization year (e.g., 2026)
                    Used in: 00, 01, 07 scripts

  --month         : Forecast initialization month (1-12)
                    Used in: 00, 01, 07 scripts
                    MUST be valid for the target season (see table above)

  --season        : Target forecast season (MAM or JJA)
                    MAM valid months: December (12), January (1), February (2)
                    JJA valid months: March (3), April (4), May (5)

  --lead-time     : Auto-calculated from month/season (override not recommended)
                    Formula: Lead Index = Months Ahead - 1

  --year-start    : Historical data start year (default: 1981)
                    Used in: 00 script for downloading historical baseline

  --year-end      : Historical data end year (default: year - 1)
                    Used in: 00 script

  --threshold     : SPI drought threshold (e.g., -0.68 for moderate drought)
                    Used in: 07 script

  --trigger       : Probability trigger for AA activation (0-1 scale)
                    Used in: 07 script

=================================================================================
OUTPUT FILES:
=================================================================================

  Step 00 - Download:
    - {output_dir}/seas5_precipitation_{date}_years{start}-{end}_months_12_months.grib
    - {output_dir}/seas5_precipitation_{date}_year{year}_months_{month}.grib

  Step 01 - SPI Processing:
    - {output_dir}/{region}_rgr_seas51_spi3_{year}_{month:02d}.nc
    - {output_dir}/{region}_rgr_seas51_spi3_{year}_{month:02d}_masked.nc

  Step 07 - Forecast Plots:
    - {output_dir}/kmj_seas51_spi3_{season}_eprob_{year}_{month}_th{th}_tr{tr}.nc
    - {output_dir}/{region}_{season}_lt{lead}_th{th}_tr{tr}.png

  Step 08 - District Stats:
    - {output_dir}/kmj_seas51_spi3_{season}_eprob_{year}_{month}_th{th}_tr{tr}_district_averages.csv

=================================================================================
"""

import os
import sys
import glob
import argparse
import subprocess
import datetime
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# SEASON AND LEAD TIME VALIDATION
# =============================================================================
#
# Based on SEAS51 forecast structure (see docs/source/5.rst):
#
# MAM Season (March-April-May) - valid_time = May (month 5)
#   +-------------+------------+---------------+
#   | Init Month  | Lead INDEX | Months Ahead  |
#   +-------------+------------+---------------+
#   | December    | 4          | 5             |
#   | January     | 3          | 4             |
#   | February    | 2          | 3             |
#   +-------------+------------+---------------+
#
# JJA Season (June-July-August) - valid_time = August (month 8)
#   +-------------+------------+---------------+
#   | Init Month  | Lead INDEX | Months Ahead  |
#   +-------------+------------+---------------+
#   | March       | 4          | 5             |
#   | April       | 3          | 4             |
#   | May         | 2          | 3             |
#   +-------------+------------+---------------+
#
# Key: Lead INDEX = Months Ahead - 1

# Valid initialization months for each season
SEASON_VALID_MONTHS = {
    "MAM": {
        12: {"lead_index": 4, "months_ahead": 5, "forecast_year_offset": 1},  # December -> next year MAM
        1:  {"lead_index": 3, "months_ahead": 4, "forecast_year_offset": 0},  # January
        2:  {"lead_index": 2, "months_ahead": 3, "forecast_year_offset": 0},  # February
    },
    "JJA": {
        3: {"lead_index": 4, "months_ahead": 5, "forecast_year_offset": 0},  # March
        4: {"lead_index": 3, "months_ahead": 4, "forecast_year_offset": 0},  # April
        5: {"lead_index": 2, "months_ahead": 3, "forecast_year_offset": 0},  # May
    },
}

MONTH_NAMES = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December"
}


def validate_month_for_season(month, season):
    """
    Validate that the initialization month is valid for the target season.

    Args:
        month (int): Initialization month (1-12)
        season (str): Target season ("MAM" or "JJA")

    Returns:
        dict: Contains 'valid', 'lead_index', 'months_ahead', 'forecast_year_offset',
              and 'error_message' if invalid

    Raises:
        ValueError: If season is not supported
    """
    season = season.upper()

    if season not in SEASON_VALID_MONTHS:
        return {
            "valid": False,
            "error_message": f"Unsupported season '{season}'. Supported: {list(SEASON_VALID_MONTHS.keys())}"
        }

    valid_months = SEASON_VALID_MONTHS[season]

    if month not in valid_months:
        valid_month_names = [f"{MONTH_NAMES[m]} ({m})" for m in valid_months.keys()]
        return {
            "valid": False,
            "error_message": (
                f"Month {month} ({MONTH_NAMES[month]}) is not valid for {season} season.\n"
                f"Valid initialization months for {season}: {', '.join(valid_month_names)}"
            )
        }

    info = valid_months[month]
    return {
        "valid": True,
        "lead_index": info["lead_index"],
        "months_ahead": info["months_ahead"],
        "forecast_year_offset": info["forecast_year_offset"],
        "error_message": None
    }


def get_season_info(month, season, year):
    """
    Get complete season information including lead time and adjusted forecast year.

    Args:
        month (int): Initialization month (1-12)
        season (str): Target season ("MAM" or "JJA")
        year (int): Initialization year

    Returns:
        dict: Contains:
            - season: Target season string
            - lead_index: Lead time index (0-based, for array indexing)
            - lead_time: Lead time value (1-based, for display/scripts)
            - months_ahead: Number of months until valid time
            - init_month: Initialization month
            - init_year: Initialization year
            - forecast_year: Year of the target season (may differ from init_year)
            - valid_month: Month when forecast is valid (end of season)

    Raises:
        ValueError: If month is not valid for the season
    """
    validation = validate_month_for_season(month, season)

    if not validation["valid"]:
        raise ValueError(validation["error_message"])

    season = season.upper()
    forecast_year = year + validation["forecast_year_offset"]

    # Calculate valid month (end of 3-month season)
    valid_months = {"MAM": 5, "JJA": 8, "OND": 12}
    valid_month = valid_months.get(season, 5)

    return {
        "season": season,
        "lead_index": validation["lead_index"],
        "lead_time": validation["lead_index"],  # 0-based lead index for 07-plot script
        "months_ahead": validation["months_ahead"],
        "init_month": month,
        "init_year": year,
        "forecast_year": forecast_year,
        "valid_month": valid_month,
    }


def print_season_info(season_info):
    """Print formatted season information for user clarity."""
    logger.info("=" * 60)
    logger.info("FORECAST TIMING INFORMATION")
    logger.info("=" * 60)
    logger.info(f"  Initialization: {MONTH_NAMES[season_info['init_month']]} {season_info['init_year']}")
    logger.info(f"  Target Season:  {season_info['season']} {season_info['forecast_year']}")
    logger.info(f"  Lead Time:      {season_info['months_ahead']} months ahead (index={season_info['lead_index']})")
    logger.info(f"  Valid Month:    {MONTH_NAMES[season_info['valid_month']]} {season_info['forecast_year']}")
    logger.info("=" * 60)


# =============================================================================
# CONFIGURATION DEFAULTS
# =============================================================================

DEFAULT_CONFIG = {
    "region_id": "kmj",
    "region_shapefile": "kmj_polygon.geojson",
    "district_shapefile": "karamoja_9_districts.geojson",
    "admin_level": "admin2",
    "season": "MAM",
    "lead_time": 3,
    "year_start": 1981,
    "mask_buffer": 0.25,
    "grid_resolution": 0.25,  # Output grid resolution in degrees
    # Default threshold/trigger combinations for AA decision-making
    "thresholds": [-0.68, -0.84],
    "triggers": [0.152, 0.111],
}


def check_directory_exists(directory, create=True):
    """
    Check if a directory exists, optionally create it.

    Args:
        directory (str): Path to directory
        create (bool): If True, create the directory if it doesn't exist

    Returns:
        bool: True if directory exists (or was created), False otherwise
    """
    if os.path.exists(directory):
        logger.info(f"Directory exists: {directory}")
        return True
    elif create:
        try:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created directory: {directory}")
            return True
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {e}")
            return False
    else:
        logger.warning(f"Directory does not exist: {directory}")
        return False


def find_grib_files(output_dir, year, month, date_str=None):
    """
    Find GRIB files in the output directory matching the expected naming pattern.

    Args:
        output_dir (str): Directory to search
        year (int): Current/target year
        month (int): Current/target month
        date_str (str): Optional date string (YYYYMMDD) to match specific files

    Returns:
        dict: Dictionary with 'main' and 'additional' keys containing file paths
    """
    result = {"main": None, "additional": None}

    # Pattern for historical data file (main file)
    # e.g., seas5_precipitation_20260120_years1981-2025_months_12_months.grib
    main_patterns = [
        f"seas5_precipitation_*_years*-{year-1}_months_*.grib",
        f"seas5_precipitation_*_years*_months_*.grib",
    ]

    # Pattern for current year file (additional file)
    # Can be single month (e.g., months_01.grib) or multiple months (e.g., months_01_02.grib)
    month_str = f"{month:02d}"
    additional_patterns = [
        f"seas5_precipitation_*_year{year}_months_{month_str}.grib",  # Single month exact match
        f"seas5_precipitation_*_year{year}_months_*{month_str}*.grib",  # Multi-month file containing target month
        f"*_year{year}_months_{month_str}.grib",  # Generic single month
        f"*_year{year}_months_*.grib",  # Any months file for the target year (most flexible)
    ]

    # Search for main file
    for pattern in main_patterns:
        matches = glob.glob(os.path.join(output_dir, pattern))
        if matches:
            # Sort by modification time, get most recent
            matches.sort(key=os.path.getmtime, reverse=True)
            result["main"] = matches[0]
            logger.info(f"Found main GRIB file: {result['main']}")
            break

    # Search for additional file
    for pattern in additional_patterns:
        matches = glob.glob(os.path.join(output_dir, pattern))
        if matches:
            matches.sort(key=os.path.getmtime, reverse=True)
            result["additional"] = matches[0]
            logger.info(f"Found additional GRIB file: {result['additional']}")
            break

    return result


def check_grib_files_exist(output_dir, year, month):
    """
    Check if required GRIB files already exist.

    Args:
        output_dir (str): Directory to check
        year (int): Target year
        month (int): Target month

    Returns:
        tuple: (bool, dict) - Whether files exist and the file paths
    """
    grib_files = find_grib_files(output_dir, year, month)

    if grib_files["main"] and grib_files["additional"]:
        logger.info("All required GRIB files found")
        return True, grib_files
    elif grib_files["main"]:
        logger.warning(f"Main GRIB file found, but additional file for year {year} month {month} is missing")
        return False, grib_files
    else:
        logger.info("GRIB files not found - download required")
        return False, grib_files


def run_command(cmd, description, dry_run=False):
    """
    Run a shell command with logging.

    Args:
        cmd (list): Command and arguments as list
        description (str): Description for logging
        dry_run (bool): If True, only print the command without executing

    Returns:
        int: Return code (0 for success in dry_run mode)
    """
    cmd_str = " ".join(cmd)

    logger.info(f"{'[DRY RUN] ' if dry_run else ''}Running: {description}")
    logger.info(f"Command: {cmd_str}")

    if dry_run:
        print(f"\n{'='*70}")
        print(f"[DRY RUN] {description}")
        print(f"{'='*70}")
        print(cmd_str)
        print()
        return 0

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        logger.info(f"Completed: {description}")
        return result.returncode
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed: {description}")
        logger.error(f"Return code: {e.returncode}")
        return e.returncode
    except Exception as e:
        logger.error(f"Error running command: {e}")
        return 1


def step_00_download_data(args, config, dry_run=False):
    """
    Step 00: Download SEAS51 GRIB data from ECMWF CDS.

    This step:
    - Creates the output directory if it doesn't exist
    - Checks if GRIB files already exist (skips download if found)
    - Downloads historical data (1981 to year-1)
    - Downloads current year data (only available months)
    """
    logger.info("=" * 70)
    logger.info("STEP 00: Download SEAS51 Data")
    logger.info("=" * 70)

    # Check if directory exists
    check_directory_exists(args.output_dir, create=True)

    # Check if GRIB files already exist
    files_exist, grib_files = check_grib_files_exist(args.output_dir, args.year, args.month)

    if files_exist and not args.force_download:
        logger.info("GRIB files already exist. Skipping download.")
        logger.info(f"  Main file: {grib_files['main']}")
        logger.info(f"  Additional file: {grib_files['additional']}")
        return 0, grib_files

    # Build download command
    cmd = [
        "python", "00-download-data.py",
        "--output-dir", args.output_dir,
        "--months", "1-12",
        "--year-start", str(args.year_start),
        "--year-end", str(args.year),
        "--skip-unavailable",
    ]

    result = run_command(cmd, "Download SEAS51 data", dry_run)

    if result == 0 and not dry_run:
        # Find the downloaded files
        _, grib_files = check_grib_files_exist(args.output_dir, args.year, args.month)

    return result, grib_files


def step_01_process_spi(args, config, grib_files, dry_run=False):
    """
    Step 01: Process GRIB files to SPI3 NetCDF.

    This step:
    - Processes SEAS51 GRIB data using the local shapefile
    - Calculates SPI3 for the region
    - Applies region mask
    - Grid is derived from shapefile extent (no obs_file required)
    - Output filename includes year/month for downstream process compatibility
    """
    logger.info("=" * 70)
    logger.info("STEP 01: Process SPI3")
    logger.info("=" * 70)

    if not grib_files.get("main"):
        logger.error("Main GRIB file not found. Cannot proceed with SPI processing.")
        return 1

    # Build SPI processing command
    # Using 01-geojson-run-process-spi.py which doesn't require obs_file
    cmd = [
        "python", "01-run-process-spi.py",
        "--region-id", config["region_id"],
        "--mode", "seas51",
        "--output-dir", args.output_dir,
        "--use-local",
        "--local-shapefile", config["region_shapefile"],
        "--seas51-main-file", grib_files["main"],
        "--grid-resolution", str(config.get("grid_resolution", 0.25)),
        "--apply-mask",
        "--mask-buffer", str(config["mask_buffer"]),
        "--output-year", str(args.year),
        "--output-month", str(args.month),
        "--cleanup-intermediate",  # Remove intermediate files to save disk space
    ]

    # Add additional file if present
    if grib_files.get("additional"):
        cmd.extend(["--seas51-additional-files", grib_files["additional"]])

    return run_command(cmd, "Process SPI3 from GRIB data", dry_run)


def get_spi_output_filename(output_dir, region_id, year, month, masked=True):
    """
    Get the expected SPI output filename.

    Output filename now includes year/month for downstream process compatibility.
    Example: kmj_rgr_seas51_spi3_masked_2026_01.nc

    Args:
        output_dir (str): Output directory
        region_id (str): Region identifier
        year (int): Year to include in filename
        month (int): Month to include in filename
        masked (bool): Whether to get masked or unmasked filename

    Returns:
        str: Expected output filename
    """
    # Naming convention with year/month included
    if masked:
        return os.path.join(
            output_dir,
            f"{region_id}_rgr_seas51_spi3_masked_{year}_{month:02d}.nc"
        )
    else:
        return os.path.join(
            output_dir,
            f"{region_id}_rgr_seas51_spi3_{year}_{month:02d}.nc"
        )


def step_07_plot_forecast(args, config, spi_file, threshold, trigger, dry_run=False):
    """
    Step 07: Generate forecast plots and probability maps.

    This step:
    - Loads SPI3 forecast data
    - Calculates empirical probability for the given threshold
    - Creates binary trigger map
    - Generates stamp plot with ensemble members
    - Saves probability data to NetCDF
    """
    logger.info("=" * 70)
    logger.info(f"STEP 07: Generate Forecast Plot (threshold={threshold}, trigger={trigger})")
    logger.info("=" * 70)

    cmd = [
        "python", "07-plot-sea51-forecast.py",
        "--region_id", config["region_id"],
        "--season", config["season"],
        "--lead_time", str(config["lead_time"]),
        "--year", str(args.year),
        "--month", str(args.month),
        "--threshold", str(threshold),
        "--trigger", str(trigger),
        "--use_shpfile",
        "--shapefile_path", config["region_shapefile"],
        "--fct_file", spi_file,
        "--output_dir", args.output_dir,
    ]

    return run_command(cmd, f"Generate forecast plot (th={threshold}, tr={trigger})", dry_run)


def get_eprob_filename(output_dir, region_id, season, year, month, threshold, trigger):
    """
    Get the expected empirical probability NetCDF filename.

    Args:
        output_dir (str): Output directory
        region_id (str): Region identifier
        season (str): Season string (e.g., 'mam')
        year (int): Year
        month (int): Month
        threshold (float): SPI threshold
        trigger (float): Trigger probability

    Returns:
        str: Expected filename
    """
    threshold_str = f"{abs(threshold):.2f}".replace('.', 'p')
    trigger_str = f"{trigger*100:.1f}".replace('.', 'p')
    return os.path.join(
        output_dir,
        f"kmj_seas51_spi3_{season.lower()}_eprob_{year}_{month:02d}_th{threshold_str}_tr{trigger_str}.nc"
    )


def step_08_district_stats(args, config, eprob_file, dry_run=False):
    """
    Step 08: Calculate district-level statistics.

    This step:
    - Loads empirical probability NetCDF
    - Regrids to 1km resolution
    - Calculates district-level averages
    - Outputs CSV with district statistics
    """
    logger.info("=" * 70)
    logger.info(f"STEP 08: Calculate District Statistics")
    logger.info("=" * 70)

    if not os.path.exists(eprob_file) and not dry_run:
        logger.error(f"Empirical probability file not found: {eprob_file}")
        return 1

    cmd = [
        "python", "08-kmj-district-stats.py",
        "--input_netcdf", eprob_file,
        "--district_shapefile", config["district_shapefile"],
        "--admin_level", config["admin_level"],
    ]

    return run_command(cmd, "Calculate district statistics", dry_run)


def parse_list_arg(value, value_type=float):
    """
    Parse a comma-separated list argument.

    Args:
        value (str): Comma-separated string
        value_type: Type to convert values to

    Returns:
        list: List of converted values
    """
    if isinstance(value, list):
        return value
    return [value_type(x.strip()) for x in value.split(",")]


def main():
    """Main function to orchestrate the pipeline."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Required arguments
    parser.add_argument("--year", type=int, required=True,
                        help="Forecast initialization year (e.g., 2026)")
    parser.add_argument("--month", type=int, required=True,
                        help="Forecast initialization month (1-12)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for all generated files")

    # Optional year range for historical data
    parser.add_argument("--year-start", type=int, default=DEFAULT_CONFIG["year_start"],
                        help=f"Historical data start year (default: {DEFAULT_CONFIG['year_start']})")

    # Season and lead time
    # NOTE: lead_time is automatically calculated based on month and season
    parser.add_argument("--season", type=str, default=DEFAULT_CONFIG["season"],
                        help=(f"Target forecast season (default: {DEFAULT_CONFIG['season']}). "
                              "Valid months: MAM=Dec,Jan,Feb; JJA=Mar,Apr,May"))
    parser.add_argument("--lead-time", type=int, default=DEFAULT_CONFIG["lead_time"],
                        help=f"Lead time (auto-calculated from month/season, override not recommended)")

    # Threshold and trigger options
    # NOTE: For negative values, use = sign: --thresholds="-0.68,-0.84"
    parser.add_argument("--thresholds", type=str,
                        default=",".join(str(t) for t in DEFAULT_CONFIG["thresholds"]),
                        help="Comma-separated SPI thresholds. Use = for negatives: --thresholds=\"-0.68,-0.84\" (default: -0.68,-0.84)")
    parser.add_argument("--triggers", type=str,
                        default=",".join(str(t) for t in DEFAULT_CONFIG["triggers"]),
                        help="Comma-separated trigger probabilities (default: 0.152,0.111)")

    # Region configuration
    parser.add_argument("--region-id", type=str, default=DEFAULT_CONFIG["region_id"],
                        help=f"Region identifier (default: {DEFAULT_CONFIG['region_id']})")
    parser.add_argument("--region-shapefile", type=str, default=DEFAULT_CONFIG["region_shapefile"],
                        help=f"Path to region shapefile/geojson (default: {DEFAULT_CONFIG['region_shapefile']})")
    parser.add_argument("--district-shapefile", type=str, default=DEFAULT_CONFIG["district_shapefile"],
                        help=f"Path to district shapefile/geojson (default: {DEFAULT_CONFIG['district_shapefile']})")
    parser.add_argument("--admin-level", type=str, default=DEFAULT_CONFIG["admin_level"],
                        help=f"Admin level for district stats (default: {DEFAULT_CONFIG['admin_level']})")

    # Execution control
    parser.add_argument("--steps", type=str, default="0,1,7,8",
                        help="Comma-separated step numbers to run (default: 0,1,7,8)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip step 00 (download) even if files don't exist")
    parser.add_argument("--force-download", action="store_true",
                        help="Force re-download even if GRIB files exist")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")

    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Parse list arguments
    thresholds = parse_list_arg(args.thresholds, float)
    triggers = parse_list_arg(args.triggers, float)
    steps = parse_list_arg(args.steps, int)

    # Validate month/season combination and get correct lead time
    try:
        season_info = get_season_info(args.month, args.season, args.year)
    except ValueError as e:
        logger.error(f"Invalid month/season combination: {e}")
        logger.error("")
        logger.error("Quick Reference - Valid initialization months:")
        logger.error("  MAM season: December (12), January (1), February (2)")
        logger.error("  JJA season: March (3), April (4), May (5)")
        sys.exit(1)

    # Use calculated lead_time from season_info (override any user-provided value)
    calculated_lead_time = season_info["lead_time"]
    if args.lead_time != DEFAULT_CONFIG["lead_time"] and args.lead_time != calculated_lead_time:
        logger.warning(f"Overriding user-provided lead_time ({args.lead_time}) with "
                      f"calculated value ({calculated_lead_time}) based on month/season")

    # Build configuration
    config = DEFAULT_CONFIG.copy()
    config.update({
        "region_id": args.region_id,
        "region_shapefile": args.region_shapefile,
        "district_shapefile": args.district_shapefile,
        "admin_level": args.admin_level,
        "season": season_info["season"],
        "lead_time": calculated_lead_time,
        "lead_index": season_info["lead_index"],
        "forecast_year": season_info["forecast_year"],
        "thresholds": thresholds,
        "triggers": triggers,
    })

    logger.info("=" * 70)
    logger.info("SEAS51 DROUGHT FORECAST PIPELINE")
    logger.info("=" * 70)
    print_season_info(season_info)
    logger.info(f"Output Directory: {args.output_dir}")
    logger.info(f"Steps to run: {steps}")
    logger.info(f"Thresholds: {thresholds}")
    logger.info(f"Triggers: {triggers}")
    logger.info("=" * 70)

    # Track results
    results = {}
    grib_files = {"main": None, "additional": None}

    # Step 0: Download data
    if 0 in steps and not args.skip_download:
        result, grib_files = step_00_download_data(args, config, args.dry_run)
        results[0] = result
        if result != 0 and not args.dry_run:
            logger.error("Step 00 failed. Aborting pipeline.")
            sys.exit(1)
    else:
        # Still need to find existing GRIB files
        _, grib_files = check_grib_files_exist(args.output_dir, args.year, args.month)
        if not grib_files.get("main") and 1 in steps:
            logger.error("GRIB files not found and download was skipped. Cannot continue.")
            sys.exit(1)

    # Step 1: Process SPI
    spi_file = get_spi_output_filename(args.output_dir, config["region_id"], args.year, args.month, masked=True)

    if 1 in steps:
        result = step_01_process_spi(args, config, grib_files, args.dry_run)
        results[1] = result
        if result != 0 and not args.dry_run:
            logger.error("Step 01 failed. Aborting pipeline.")
            sys.exit(1)
    else:
        # Check if SPI file exists
        # Try both new naming (with year/month) and old naming (without)
        if not os.path.exists(spi_file):
            old_spi_file = os.path.join(args.output_dir, f"{config['region_id']}_rgr_seas51_spi3_masked.nc")
            if os.path.exists(old_spi_file):
                logger.info(f"Using existing SPI file (old naming): {old_spi_file}")
                spi_file = old_spi_file

    # Steps 7 & 8: Run for each threshold/trigger combination
    if len(thresholds) != len(triggers):
        logger.error(f"Number of thresholds ({len(thresholds)}) must match triggers ({len(triggers)})")
        sys.exit(1)

    for threshold, trigger in zip(thresholds, triggers):
        # Step 7: Plot forecast
        if 7 in steps:
            result = step_07_plot_forecast(args, config, spi_file, threshold, trigger, args.dry_run)
            results[(7, threshold, trigger)] = result
            if result != 0 and not args.dry_run:
                logger.warning(f"Step 07 failed for threshold={threshold}, trigger={trigger}")

        # Step 8: District stats
        if 8 in steps:
            eprob_file = get_eprob_filename(
                args.output_dir, config["region_id"], config["season"],
                args.year, args.month, threshold, trigger
            )
            result = step_08_district_stats(args, config, eprob_file, args.dry_run)
            results[(8, threshold, trigger)] = result
            if result != 0 and not args.dry_run:
                logger.warning(f"Step 08 failed for threshold={threshold}, trigger={trigger}")

    # Summary
    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)

    if args.dry_run:
        logger.info("[DRY RUN] No commands were executed")
    else:
        for step_key, result in results.items():
            status = "SUCCESS" if result == 0 else "FAILED"
            logger.info(f"Step {step_key}: {status}")

    # Return overall success/failure
    failed = [k for k, v in results.items() if v != 0]
    if failed and not args.dry_run:
        logger.error(f"Failed steps: {failed}")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
