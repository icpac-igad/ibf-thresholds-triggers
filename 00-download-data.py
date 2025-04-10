#!/usr/bin/env python3
"""
ECMWF 
MWF SEAS5 and CHIRPS Data Downloader

This script downloads:
1. SEAS5 seasonal forecast data from ECMWF CDS API
2. CHIRPS precipitation data (optional)

The script cleans up old SEAS5 files when new data is downloaded.

#Flexible downloading options:

Download both datasets (default)
Download only SEAS5 data with --seas5-only
Download only CHIRPS data with --chirps-only
Download only specific month SEAS5 data with --only-current-month-seas5 [month] and optional --year [year]


#SEAS5 file management:

Downloads SEAS5 data with the correct parameters
Automatically names files with date stamps (e.g., seas5_precipitation_20250402.grib)
Cleans up old SEAS5 files by default, keeping only the latest one
Option to keep all SEAS5 files with --keep-all-seas5


#CHIRPS handling:

Downloads CHIRPS data only once (checks if file already exists)
Uses wget to retrieve the data as specified


# Download both datasets (run monthly)
python ecmwf_downloader.py

# Download only SEAS5 (typical monthly update)
python ecmwf_downloader.py --seas5-only

# Download only CHIRPS (rarely needed)
python ecmwf_downloader.py --chirps-only

# Specify a different output directory
python ecmwf_downloader.py --output-dir /path/to/data

# Download SEAS5 data for a specific month only (1-12)
python ecmwf_downloader.py --only-current-month-seas5 4

# Download SEAS5 data for a specific month and year
python ecmwf_downloader.py --only-current-month-seas5 4 --year 2023
"""

import os
import argparse
import glob
import datetime
import subprocess
import cdsapi

def download_seas5(output_dir="./data", filename_prefix="seas5_precipitation_"):
    """
    Download SEAS5 dataset from ECMWF CDS API
    
    Args:
        output_dir: Directory to save the downloaded data
        filename_prefix: Prefix for the output filename
    
    Returns:
        str: Path to the downloaded file
    """
    print("Downloading SEAS5 data from ECMWF CDS...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Current date to use in the filename
    current_date = datetime.datetime.now().strftime("%Y%m%d")
    output_file = os.path.join(output_dir, f'{filename_prefix}{current_date}.grib')
    
    # Define the SEAS5 dataset and request parameters
    dataset = "seasonal-monthly-single-levels"
    request = {
        "originating_centre": "ecmwf",
        "system": "51",
        "variable": ["total_precipitation"],
        "year": [
            "1981", "1982", "1983", "1984", "1985", "1986", "1987", "1988", "1989",
            "1990", "1991", "1992", "1993", "1994", "1995", "1996", "1997", "1998",
            "1999", "2000", "2001", "2002", "2003", "2004", "2005", "2006", "2007",
            "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016",
            "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025"
        ],
        "month": [
            "01", "02", "03", "04", "05", "06", 
            "07", "08", "09", "10", "11", "12"
        ],
        "leadtime_month": ["1", "2", "3", "4", "5", "6"],
        "data_format": "grib",
        "product_type": ["monthly_mean"],
        "area": [23, 21, -12, 53]
    }
    
    try:
        client = cdsapi.Client()
        client.retrieve(dataset, request, output_file)
        print(f"SEAS5 data downloaded successfully to: {output_file}")
        return output_file
    except Exception as e:
        print(f"Error downloading SEAS5 data: {e}")
        return None


def download_current_month_seas5(output_dir="./data", filename_prefix="seas5_precipitation_", month=None, year=None):
    """
    Download SEAS5 dataset from ECMWF CDS API for a specific month and optionally a specific year
    
    Args:
        output_dir: Directory to save the downloaded data
        filename_prefix: Prefix for the output filename
        month: Month to download data for (1-12 integer)
        year: Year to download data for (4-digit integer, defaults to current year if None)
    
    Returns:
        str: Path to the downloaded file
    """
    # If year is not provided, use current year
    if year is None:
        year = datetime.datetime.now().year
    
    year_str = str(year)
    
    print(f"Downloading SEAS5 data from ECMWF CDS for month {month}, year {year_str}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Format month as two-digit string (e.g., "01", "02", etc.)
    month_str = f"{month:02d}"
    
    # Current date to use in the filename
    current_date = datetime.datetime.now().strftime("%Y%m%d")
    output_file = os.path.join(output_dir, f'{filename_prefix}{current_date}_year{year_str}_month{month_str}.grib')
    
    # Define the SEAS5 dataset and request parameters
    dataset = "seasonal-monthly-single-levels"
    request = {
        "originating_centre": "ecmwf",
        "system": "51",
        "variable": ["total_precipitation"],
        "year": [
            year_str
        ],
        "month": [
            month_str
        ],
        "leadtime_month": ["1", "2", "3", "4", "5", "6"],
        "data_format": "grib",
        "product_type": ["monthly_mean"],
        "area": [23, 21, -12, 53]
    }
    
    try:
        client = cdsapi.Client()
        client.retrieve(dataset, request, output_file)
        print(f"SEAS5 data for month {month_str}, year {year_str} downloaded successfully to: {output_file}")
        return output_file
    except Exception as e:
        print(f"Error downloading SEAS5 data for month {month_str}, year {year_str}: {e}")
        return None


def download_chirps(output_dir="./data", filename="chirps-v3.0.monthly.nc"):
    """
    Download CHIRPS precipitation data using wget
    
    Args:
        output_dir: Directory to save the downloaded data
        filename: Name for the output file
    
    Returns:
        str: Path to the downloaded file
    """
    print("Downloading CHIRPS precipitation data...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, filename)
    url = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_monthly/netcdf/chirps-v2.0.monthly.nc"
    
    try:
        # Check if file already exists
        if os.path.exists(output_file):
            print(f"CHIRPS data already exists at: {output_file}")
            return output_file
            
        # Use wget to download the file
        subprocess.run(["wget", url, "-O", output_file], check=True)
        print(f"CHIRPS data downloaded successfully to: {output_file}")
        return output_file
    except subprocess.CalledProcessError as e:
        print(f"Error downloading CHIRPS data: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

def cleanup_old_seas5_files(output_dir="./data", filename_prefix="seas5_precipitation_", keep_latest=True):
    """
    Remove old SEAS5 files, optionally keeping the latest one
    
    Args:
        output_dir: Directory containing the SEAS5 files
        filename_prefix: Prefix of the SEAS5 files
        keep_latest: Whether to keep the latest file
    """
    print("Cleaning up old SEAS5 files...")
    
    # Get all SEAS5 files
    pattern = os.path.join(output_dir, f"{filename_prefix}*.grib")
    files = glob.glob(pattern)
    
    if not files:
        print("No SEAS5 files found for cleanup.")
        return
    
    if keep_latest and len(files) > 1:
        # Sort files by modification time (newest last)
        files.sort(key=os.path.getmtime)
        # Remove the latest file from the list
        latest_file = files.pop()
        print(f"Keeping latest file: {os.path.basename(latest_file)}")
    
    # Remove remaining files
    for file in files:
        try:
            os.remove(file)
            print(f"Removed: {os.path.basename(file)}")
        except Exception as e:
            print(f"Error removing {file}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Download ECMWF SEAS5 and CHIRPS data")
    parser.add_argument("--output-dir", default="./data", help="Directory to save downloaded data")
    parser.add_argument("--seas5-only", action="store_true", help="Download only SEAS5 data")
    parser.add_argument("--chirps-only", action="store_true", help="Download only CHIRPS data")
    parser.add_argument("--keep-all-seas5", action="store_true", help="Keep all SEAS5 files (don't clean up)")
    parser.add_argument("--only-current-month-seas5", type=int, choices=range(1, 13), 
                        help="Download SEAS5 data for a specific month only (1-12)")
    parser.add_argument("--year", type=int, 
                        help="Year to download data for when using --only-current-month-seas5 (defaults to current year)")
    
    args = parser.parse_args()
    
    # Handle mutually exclusive options
    if sum([args.seas5_only, args.chirps_only, args.only_current_month_seas5 is not None]) > 1:
        print("Error: Cannot specify multiple download options together")
        return
    
    # Validate that --year is only used with --only-current-month-seas5
    if args.year is not None and args.only_current_month_seas5 is None:
        print("Error: --year can only be used with --only-current-month-seas5")
        return
    
    # Download SEAS5 data for specific month if requested
    if args.only_current_month_seas5 is not None:
        seas5_file = download_current_month_seas5(args.output_dir, month=args.only_current_month_seas5, year=args.year)
        if not args.keep_all_seas5:
            cleanup_old_seas5_files(args.output_dir)
    # Download full SEAS5 data if requested or if no specific option is specified
    elif args.seas5_only or not args.chirps_only:
        seas5_file = download_seas5(args.output_dir)
        if not args.keep_all_seas5:
            cleanup_old_seas5_files(args.output_dir)
    
    # Download CHIRPS data if requested or if neither option is specified
    if args.chirps_only or (not args.seas5_only and args.only_current_month_seas5 is None):
        chirps_file = download_chirps(args.output_dir)
    
    print("Data download process complete")

if __name__ == "__main__":
    main()
