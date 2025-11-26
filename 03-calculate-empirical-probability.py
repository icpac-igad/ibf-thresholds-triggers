#!/usr/bin/env python3
"""
Script to calculate empirical probabilities by comparing SPI3 forecasts with return period thresholds.

This script:
1. Loads SPI3 ensemble forecast data
2. Loads return period threshold data
3. Regrids thresholds to match forecast grid
4. Calculates empirical probabilities for each return period
5. Calculates area-based statistics (fraction of pixels exceeding thresholds)
6. Saves results as NetCDF files

Usage:
    python 03-calculate-empirical-probability.py \
        --spi3-file /srv/spi3_output/e401d9798000628d11a618c66a04372c_spi3.nc \
        --threshold-file /srv/spi_3_return_period_thresholds_20250805/spi_3_return_period_thresholds_20250805.nc \
        --output-dir ./empirical_probability_output
"""

import sys
import argparse
import logging
import time
import os
import xarray as xr
import numpy as np
import xesmf as xe

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def regrid_thresholds_to_forecast_grid(threshold_ds, forecast_ds):
    """
    Regrid threshold data to match the forecast grid.

    Args:
        threshold_ds (xarray.Dataset): Dataset with return period thresholds
        forecast_ds (xarray.Dataset): Forecast dataset with target grid

    Returns:
        xarray.Dataset: Regridded thresholds
    """
    logger.info("Regridding thresholds to forecast grid...")

    # Rename coordinates if needed
    if 'latitude' in threshold_ds.dims:
        threshold_ds = threshold_ds.rename({'latitude': 'lat', 'longitude': 'lon'})

    # Create output grid matching forecast
    ds_out = xr.Dataset({
        "lat": (["lat"], forecast_ds['lat'].values),
        "lon": (["lon"], forecast_ds['lon'].values),
    })

    # Create regridder
    regridder = xe.Regridder(threshold_ds, ds_out, "bilinear", periodic=False)

    # Regrid each threshold variable
    regridded_vars = {}
    threshold_vars = [v for v in threshold_ds.data_vars if 'threshold' in v]

    for var in threshold_vars:
        logger.info(f"  Regridding {var}...")
        regridded_vars[var] = regridder(threshold_ds[var], keep_attrs=True)

    # Create new dataset with regridded thresholds
    regridded_ds = xr.Dataset(regridded_vars)

    logger.info("Regridding complete")
    return regridded_ds


def calculate_empirical_probabilities(spi3_data, thresholds_ds, return_periods=None):
    """
    Calculate empirical probabilities for each return period.

    For each lead time, initialization time, and return period:
    - Calculate the fraction of ensemble members where SPI3 <= threshold
    - This gives the probability of exceeding each drought threshold

    Args:
        spi3_data (xarray.DataArray): SPI3 ensemble data (lead, member, init, lat, lon)
        thresholds_ds (xarray.Dataset): Regridded thresholds (lat, lon)
        return_periods (list): List of return periods to calculate (e.g., [3, 5, 10, 20, 50])

    Returns:
        xarray.Dataset: Dataset with empirical probabilities for each return period
    """
    if return_periods is None:
        return_periods = [3, 5, 10, 20, 50]

    logger.info(f"Calculating empirical probabilities for return periods: {return_periods}")

    prob_dict = {}

    for rp in return_periods:
        var_name = f'spi_3_threshold_{rp}yr'

        if var_name not in thresholds_ds:
            logger.warning(f"Threshold variable {var_name} not found, skipping...")
            continue

        logger.info(f"  Processing {rp}-year return period...")

        threshold = thresholds_ds[var_name]

        # Broadcast threshold to match SPI3 dimensions
        # SPI3: (lead, member, init, lat, lon)
        # Threshold: (lat, lon)
        # We need to expand threshold to (1, 1, 1, lat, lon) and broadcast
        threshold_expanded = threshold.expand_dims(
            lead=spi3_data.lead,
            member=spi3_data.member,
            init=spi3_data.init
        )

        # Calculate where SPI3 is less than or equal to threshold (drought condition)
        # This creates a boolean array
        exceeds_threshold = spi3_data <= threshold_expanded

        # Calculate empirical probability as fraction of ensemble members
        # exceeding threshold at each (lead, init, lat, lon)
        empirical_prob = exceeds_threshold.mean(dim='member')

        # Store with descriptive name
        prob_name = f'eprob_{rp}yr'
        prob_dict[prob_name] = empirical_prob
        prob_dict[prob_name].attrs = {
            'long_name': f'Empirical probability of {rp}-year drought',
            'units': 'probability (0-1)',
            'description': f'Fraction of ensemble members with SPI3 <= {rp}-year threshold',
            'return_period_years': rp
        }

        logger.info(f"    Probability range: {float(empirical_prob.min()):.4f} to {float(empirical_prob.max()):.4f}")

    # Create dataset with all probabilities
    prob_ds = xr.Dataset(prob_dict)

    logger.info("Empirical probability calculation complete")
    return prob_ds


def calculate_area_statistics(spi3_data, thresholds_ds, return_periods=None):
    """
    Calculate area-based statistics: fraction of pixels exceeding thresholds.

    For each lead time, initialization time, ensemble member, and return period:
    - Calculate the fraction of pixels where SPI3 <= threshold
    - This gives the spatial extent of drought

    Args:
        spi3_data (xarray.DataArray): SPI3 ensemble data (lead, member, init, lat, lon)
        thresholds_ds (xarray.Dataset): Regridded thresholds (lat, lon)
        return_periods (list): List of return periods

    Returns:
        xarray.Dataset: Dataset with area fraction statistics
    """
    if return_periods is None:
        return_periods = [3, 5, 10, 20, 50]

    logger.info(f"Calculating area statistics for return periods: {return_periods}")

    area_stats = {}

    for rp in return_periods:
        var_name = f'spi_3_threshold_{rp}yr'

        if var_name not in thresholds_ds:
            logger.warning(f"Threshold variable {var_name} not found, skipping...")
            continue

        logger.info(f"  Processing {rp}-year return period...")

        threshold = thresholds_ds[var_name]

        # Expand threshold to match SPI3 dimensions
        threshold_expanded = threshold.expand_dims(
            lead=spi3_data.lead,
            member=spi3_data.member,
            init=spi3_data.init
        )

        # Calculate where SPI3 <= threshold
        exceeds_threshold = spi3_data <= threshold_expanded

        # Calculate fraction of pixels exceeding threshold
        # Average over spatial dimensions (lat, lon)
        area_fraction = exceeds_threshold.mean(dim=['lat', 'lon'])

        # Store results
        area_name = f'area_frac_{rp}yr'
        area_stats[area_name] = area_fraction
        area_stats[area_name].attrs = {
            'long_name': f'Fraction of area exceeding {rp}-year drought threshold',
            'units': 'fraction (0-1)',
            'description': f'Spatial fraction where SPI3 <= {rp}-year threshold',
            'return_period_years': rp
        }

        # Also calculate the ensemble mean area fraction
        area_mean_name = f'area_frac_{rp}yr_ensmean'
        area_stats[area_mean_name] = area_fraction.mean(dim='member')
        area_stats[area_mean_name].attrs = {
            'long_name': f'Ensemble mean fraction of area exceeding {rp}-year drought threshold',
            'units': 'fraction (0-1)',
            'description': f'Ensemble mean of spatial fraction where SPI3 <= {rp}-year threshold',
            'return_period_years': rp
        }

        logger.info(f"    Area fraction range: {float(area_fraction.min()):.4f} to {float(area_fraction.max()):.4f}")

    # Create dataset
    area_ds = xr.Dataset(area_stats)

    logger.info("Area statistics calculation complete")
    return area_ds


def calculate_pixel_statistics(spi3_data, thresholds_ds, return_periods=None):
    """
    Calculate pixel-level statistics: number/fraction of pixels exceeding thresholds.

    For each lead time and initialization time:
    - Count how many pixels exceed each threshold across all ensemble members
    - This is useful for understanding spatial patterns

    Args:
        spi3_data (xarray.DataArray): SPI3 ensemble data
        thresholds_ds (xarray.Dataset): Regridded thresholds
        return_periods (list): List of return periods

    Returns:
        xarray.Dataset: Dataset with pixel-level statistics
    """
    if return_periods is None:
        return_periods = [3, 5, 10, 20, 50]

    logger.info(f"Calculating pixel-level statistics for return periods: {return_periods}")

    pixel_stats = {}

    for rp in return_periods:
        var_name = f'spi_3_threshold_{rp}yr'

        if var_name not in thresholds_ds:
            continue

        logger.info(f"  Processing {rp}-year return period...")

        threshold = thresholds_ds[var_name]

        # Expand threshold
        threshold_expanded = threshold.expand_dims(
            lead=spi3_data.lead,
            member=spi3_data.member,
            init=spi3_data.init
        )

        # Calculate where SPI3 <= threshold
        exceeds_threshold = spi3_data <= threshold_expanded

        # Count ensemble members exceeding threshold at each pixel
        member_count = exceeds_threshold.sum(dim='member')
        pixel_stats[f'pixel_member_count_{rp}yr'] = member_count
        pixel_stats[f'pixel_member_count_{rp}yr'].attrs = {
            'long_name': f'Number of members exceeding {rp}-year threshold at each pixel',
            'units': 'count',
            'description': f'Count of ensemble members where SPI3 <= {rp}-year threshold'
        }

        # Calculate ensemble fraction at each pixel
        member_fraction = exceeds_threshold.mean(dim='member')
        pixel_stats[f'pixel_member_frac_{rp}yr'] = member_fraction
        pixel_stats[f'pixel_member_frac_{rp}yr'].attrs = {
            'long_name': f'Fraction of members exceeding {rp}-year threshold at each pixel',
            'units': 'fraction (0-1)',
            'description': f'Fraction of ensemble members where SPI3 <= {rp}-year threshold'
        }

    pixel_ds = xr.Dataset(pixel_stats)

    logger.info("Pixel-level statistics calculation complete")
    return pixel_ds


def process_empirical_probability(spi3_file, threshold_file, output_dir='.',
                                  return_periods=None, regrid_method='bilinear'):
    """
    Main processing function to calculate all empirical probabilities and statistics.

    Args:
        spi3_file (str): Path to SPI3 forecast NetCDF file
        threshold_file (str): Path to return period threshold NetCDF file
        output_dir (str): Output directory
        return_periods (list): List of return periods to process
        regrid_method (str): Regridding method (bilinear, conservative, etc.)

    Returns:
        dict: Paths to output files
    """
    if return_periods is None:
        return_periods = [3, 5, 10, 20, 50]

    logger.info("="*60)
    logger.info("Starting empirical probability calculation")
    logger.info("="*60)
    logger.info(f"SPI3 file: {spi3_file}")
    logger.info(f"Threshold file: {threshold_file}")
    logger.info(f"Return periods: {return_periods}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load datasets
    logger.info("Loading datasets...")
    spi3_ds = xr.open_dataset(spi3_file)
    threshold_ds = xr.open_dataset(threshold_file)

    logger.info(f"SPI3 dimensions: {dict(spi3_ds.dims)}")
    logger.info(f"Threshold dimensions: {dict(threshold_ds.dims)}")

    # Regrid thresholds to forecast grid
    thresholds_regridded = regrid_thresholds_to_forecast_grid(threshold_ds, spi3_ds)

    # Get SPI3 data array
    spi3_data = spi3_ds['spi3']

    output_files = {}

    # 1. Calculate empirical probabilities (probability maps)
    logger.info("\n" + "="*60)
    logger.info("STEP 1: Calculating empirical probabilities")
    logger.info("="*60)
    prob_ds = calculate_empirical_probabilities(spi3_data, thresholds_regridded, return_periods)

    # Add metadata
    prob_ds.attrs['title'] = 'Empirical drought probabilities from SPI3 ensemble forecasts'
    prob_ds.attrs['description'] = 'Fraction of ensemble members exceeding drought thresholds'
    prob_ds.attrs['source_spi3'] = spi3_file
    prob_ds.attrs['source_thresholds'] = threshold_file
    prob_ds.attrs['creation_date'] = time.strftime('%Y-%m-%d %H:%M:%S')

    # Save empirical probabilities
    prob_file = os.path.join(output_dir, 'empirical_probabilities.nc')
    logger.info(f"Saving empirical probabilities to {prob_file}")
    prob_ds.to_netcdf(prob_file)
    output_files['probabilities'] = prob_file

    # 2. Calculate area statistics
    logger.info("\n" + "="*60)
    logger.info("STEP 2: Calculating area statistics")
    logger.info("="*60)
    area_ds = calculate_area_statistics(spi3_data, thresholds_regridded, return_periods)

    area_ds.attrs['title'] = 'Area-based drought statistics'
    area_ds.attrs['description'] = 'Fraction of pixels exceeding drought thresholds'
    area_ds.attrs['source_spi3'] = spi3_file
    area_ds.attrs['source_thresholds'] = threshold_file
    area_ds.attrs['creation_date'] = time.strftime('%Y-%m-%d %H:%M:%S')

    area_file = os.path.join(output_dir, 'area_statistics.nc')
    logger.info(f"Saving area statistics to {area_file}")
    area_ds.to_netcdf(area_file)
    output_files['area_stats'] = area_file

    # 3. Calculate pixel-level statistics
    logger.info("\n" + "="*60)
    logger.info("STEP 3: Calculating pixel-level statistics")
    logger.info("="*60)
    pixel_ds = calculate_pixel_statistics(spi3_data, thresholds_regridded, return_periods)

    pixel_ds.attrs['title'] = 'Pixel-level drought statistics'
    pixel_ds.attrs['description'] = 'Per-pixel ensemble statistics for drought thresholds'
    pixel_ds.attrs['source_spi3'] = spi3_file
    pixel_ds.attrs['source_thresholds'] = threshold_file
    pixel_ds.attrs['creation_date'] = time.strftime('%Y-%m-%d %H:%M:%S')

    pixel_file = os.path.join(output_dir, 'pixel_statistics.nc')
    logger.info(f"Saving pixel statistics to {pixel_file}")
    pixel_ds.to_netcdf(pixel_file)
    output_files['pixel_stats'] = pixel_file

    # 4. Save regridded thresholds for reference
    threshold_regrid_file = os.path.join(output_dir, 'thresholds_regridded.nc')
    logger.info(f"Saving regridded thresholds to {threshold_regrid_file}")
    thresholds_regridded.to_netcdf(threshold_regrid_file)
    output_files['thresholds_regridded'] = threshold_regrid_file

    logger.info("\n" + "="*60)
    logger.info("Processing complete!")
    logger.info("="*60)

    return output_files


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Calculate empirical probabilities from SPI3 forecasts and return period thresholds",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--spi3-file",
        type=str,
        required=True,
        help="Path to SPI3 forecast NetCDF file"
    )

    parser.add_argument(
        "--threshold-file",
        type=str,
        required=True,
        help="Path to return period threshold NetCDF file"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./empirical_probability_output",
        help="Output directory for results"
    )

    parser.add_argument(
        "--return-periods",
        type=int,
        nargs="+",
        default=[3, 5, 10, 20, 50],
        help="Return periods to calculate (years)"
    )

    parser.add_argument(
        "--regrid-method",
        type=str,
        default="bilinear",
        choices=["bilinear", "conservative", "nearest_s2d", "nearest_d2s", "patch"],
        help="Regridding method"
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
    args = parse_arguments()

    # Set up logging to file if requested
    if args.log_file:
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(file_handler)

    # Check if input files exist
    if not os.path.exists(args.spi3_file):
        logger.error(f"SPI3 file does not exist: {args.spi3_file}")
        sys.exit(1)

    if not os.path.exists(args.threshold_file):
        logger.error(f"Threshold file does not exist: {args.threshold_file}")
        sys.exit(1)

    # Process
    try:
        start_time = time.time()

        output_files = process_empirical_probability(
            args.spi3_file,
            args.threshold_file,
            args.output_dir,
            args.return_periods,
            args.regrid_method
        )

        elapsed = time.time() - start_time

        logger.info("\n" + "="*60)
        logger.info("SUMMARY")
        logger.info("="*60)
        logger.info(f"Total processing time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
        logger.info("\nOutput files:")
        for key, path in output_files.items():
            logger.info(f"  {key}: {path}")
        logger.info("="*60)

        print("\n" + "="*60)
        print("Empirical Probability Calculation Complete")
        print("="*60)
        print(f"Processing time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
        print("\nOutput files:")
        for key, path in output_files.items():
            print(f"  {key}:")
            print(f"    {path}")
        print("="*60 + "\n")

    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
