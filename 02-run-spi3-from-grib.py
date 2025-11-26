#!/usr/bin/env python3
"""
Script to calculate SPI-3 for each forecast month in a SEAS51 GRIB file.

This script:
1. Loads a SEAS51 GRIB file with precipitation forecasts
2. Calculates SPI-3 for each forecast month (lead time 1-6)
3. Processes all ensemble members
4. Saves the results to a NetCDF file

Usage:
    python 02-run-spi3-from-grib.py --grib-file /path/to/file.grib --output-dir ./output
"""

import sys
import argparse
import logging
import time
import os
import xarray as xr
import numpy as np
from xclim.indices import standardized_precipitation_index

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def calculate_spi3_for_ensemble_member(precip_data, member_num, cal_start, cal_end):
    """
    Calculate SPI-3 for a single ensemble member.

    Args:
        precip_data (xarray.DataArray): Precipitation data for one member
        member_num (int): Ensemble member number
        cal_start (str): Calibration start date (e.g., '1991-01-01')
        cal_end (str): Calibration end date (e.g., '2018-01-01')

    Returns:
        xarray.DataArray: SPI-3 values or None if processing fails
    """
    try:
        # Set units
        precip_data.attrs['units'] = 'mm/month'

        # Check data validity
        nan_count = np.isnan(precip_data.values).sum()
        if nan_count > 0:
            nan_percent = (nan_count / precip_data.size) * 100
            logger.warning(f"Member {member_num}: {nan_percent:.1f}% NaN values in input data")
            if nan_percent > 90:
                logger.warning(f"Skipping member {member_num} due to excessive NaNs")
                return None

        # Calculate SPI-3
        spi_3 = standardized_precipitation_index(
            precip_data,
            freq="MS",
            window=3,
            dist="gamma",
            method="APP",
            cal_start=cal_start,
            cal_end=cal_end,
            fitkwargs={"floc": 0}
        )

        # Compute the result
        spi_computed = spi_3.compute()

        # Check output validity
        spi_nan_count = np.isnan(spi_computed.values).sum()
        if spi_nan_count > 0:
            spi_nan_percent = (spi_nan_count / spi_computed.size) * 100
            logger.warning(f"Member {member_num}: {spi_nan_percent:.1f}% NaN values in SPI output")

            if spi_nan_percent > 95:
                logger.warning(f"Skipping member {member_num} due to excessive NaNs in output")
                return None

        logger.info(f"Successfully processed member {member_num}")
        return spi_computed

    except Exception as e:
        logger.error(f"Error processing member {member_num}: {e}")
        return None


def process_forecast_month(dataset, forecast_month, cal_start='1991-01-01', cal_end='2018-01-01'):
    """
    Process a single forecast month (lead time) for all ensemble members.

    Args:
        dataset (xarray.Dataset): The SEAS51 dataset
        forecast_month (int): Forecast month (lead time) to process
        cal_start (str): Calibration start date
        cal_end (str): Calibration end date

    Returns:
        xarray.DataArray: SPI-3 for all members at this lead time, or None if failed
    """
    logger.info(f"Processing forecast month {forecast_month}...")
    start_time = time.time()

    # Select data for this forecast month
    fm_data = dataset.sel(forecastMonth=forecast_month)

    # List to store SPI-3 for each ensemble member
    member_spi_list = []

    # Get number of ensemble members
    n_members = len(fm_data.number.values)
    logger.info(f"Processing {n_members} ensemble members...")

    # Process each ensemble member
    for member_num in fm_data.number.values:
        # Select data for this member
        member_data = fm_data.sel(number=member_num)
        precip = member_data.tprate

        # Use different calibration periods based on member number
        # (following the logic from the original script)
        if member_num < 25:
            member_cal_start = cal_start
            member_cal_end = cal_end
        else:
            # Use more recent period for members 25+
            member_cal_start = '2017-01-01'
            member_cal_end = '2024-01-01'

        # Calculate SPI-3
        spi_result = calculate_spi3_for_ensemble_member(
            precip, member_num, member_cal_start, member_cal_end
        )

        if spi_result is not None:
            member_spi_list.append(spi_result)

    if not member_spi_list:
        logger.error(f"No valid members for forecast month {forecast_month}")
        return None

    # Concatenate all members
    try:
        combined_spi = xr.concat(member_spi_list, dim='member')
        elapsed = time.time() - start_time
        logger.info(f"Forecast month {forecast_month} completed in {elapsed:.2f}s "
                   f"({len(member_spi_list)}/{n_members} members)")
        return combined_spi
    except Exception as e:
        logger.error(f"Error combining members for forecast month {forecast_month}: {e}")
        return None


def process_grib_file(grib_file, output_dir='.', cal_start='1991-01-01', cal_end='2018-01-01'):
    """
    Process a SEAS51 GRIB file and calculate SPI-3 for all forecast months.

    Args:
        grib_file (str): Path to GRIB file
        output_dir (str): Output directory for results
        cal_start (str): Calibration start date
        cal_end (str): Calibration end date

    Returns:
        str: Path to output NetCDF file
    """
    logger.info(f"Loading GRIB file: {grib_file}")

    # Load the GRIB file
    try:
        ds = xr.open_dataset(
            grib_file,
            engine='cfgrib',
            backend_kwargs=dict(time_dims=('forecastMonth', 'time'))
        )
        logger.info(f"Dataset loaded successfully")
        logger.info(f"Dimensions: {dict(ds.dims)}")
        logger.info(f"Forecast months: {ds.forecastMonth.values}")
        logger.info(f"Time range: {ds.time.values[0]} to {ds.time.values[-1]}")
        logger.info(f"Number of ensemble members: {len(ds.number.values)}")
    except Exception as e:
        logger.error(f"Error loading GRIB file: {e}")
        raise

    # Process each forecast month
    all_forecast_months = []

    for fm in ds.forecastMonth.values:
        fm_result = process_forecast_month(ds, int(fm), cal_start, cal_end)

        if fm_result is not None:
            all_forecast_months.append(fm_result)
        else:
            logger.warning(f"Skipping forecast month {fm} due to processing errors")

    if not all_forecast_months:
        raise ValueError("No forecast months were successfully processed")

    # Combine all forecast months
    logger.info("Combining all forecast months...")
    try:
        combined_dataset = xr.concat(all_forecast_months, dim='lead')

        # Convert to dataset with proper naming
        if isinstance(combined_dataset, xr.DataArray):
            combined_dataset = combined_dataset.to_dataset(name='spi3')

        # Add attributes
        combined_dataset['lead'].attrs['units'] = 'months'
        combined_dataset['lead'].attrs['long_name'] = 'Forecast lead time'
        combined_dataset['spi3'].attrs['long_name'] = 'Standardized Precipitation Index (3-month)'
        combined_dataset['spi3'].attrs['calibration_period'] = f"{cal_start} to {cal_end}"

        # Rename coordinates if needed
        if 'longitude' in combined_dataset.dims:
            combined_dataset = combined_dataset.rename({'longitude': 'lon', 'latitude': 'lat'})

        # Rename time to init for clarity
        if 'time' in combined_dataset.dims or 'time' in combined_dataset.coords:
            combined_dataset = combined_dataset.rename({'time': 'init'})

        logger.info(f"Final dataset dimensions: {dict(combined_dataset.dims)}")

    except Exception as e:
        logger.error(f"Error combining forecast months: {e}")
        raise

    # Save to NetCDF
    os.makedirs(output_dir, exist_ok=True)

    # Create output filename based on input filename
    base_name = os.path.splitext(os.path.basename(grib_file))[0]
    output_file = os.path.join(output_dir, f'{base_name}_spi3.nc')

    logger.info(f"Saving results to {output_file}...")
    combined_dataset.to_netcdf(output_file)
    logger.info(f"Successfully saved SPI-3 data to {output_file}")

    return output_file


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Calculate SPI-3 for each forecast month in a SEAS51 GRIB file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--grib-file",
        type=str,
        required=True,
        help="Path to input GRIB file"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory for results"
    )

    parser.add_argument(
        "--cal-start",
        type=str,
        default="1991-01-01",
        help="Calibration period start date (YYYY-MM-DD)"
    )

    parser.add_argument(
        "--cal-end",
        type=str,
        default="2018-01-01",
        help="Calibration period end date (YYYY-MM-DD)"
    )

    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log file (optional)"
    )

    return parser.parse_args()


def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()

    # Set up logging to file if requested
    if args.log_file:
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(file_handler)

    logger.info("="*60)
    logger.info("Starting SPI-3 calculation from GRIB file")
    logger.info("="*60)
    logger.info(f"Input file: {args.grib_file}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Calibration period: {args.cal_start} to {args.cal_end}")

    # Check if input file exists
    if not os.path.exists(args.grib_file):
        logger.error(f"Input file does not exist: {args.grib_file}")
        sys.exit(1)

    # Process the file
    try:
        start_time = time.time()
        output_file = process_grib_file(
            args.grib_file,
            args.output_dir,
            args.cal_start,
            args.cal_end
        )
        elapsed = time.time() - start_time

        logger.info("="*60)
        logger.info("Processing completed successfully!")
        logger.info(f"Total time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
        logger.info(f"Output file: {output_file}")
        logger.info("="*60)

        print("\n" + "="*60)
        print("SPI-3 Calculation Complete")
        print("="*60)
        print(f"Input:  {args.grib_file}")
        print(f"Output: {output_file}")
        print(f"Time:   {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
        print("="*60 + "\n")

    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
