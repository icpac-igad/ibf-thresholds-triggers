#!/usr/bin/env python3
"""
ECMWF SEAS5 and CHIRPS Data Downloader

Downloads SEAS5 seasonal forecast data from ECMWF CDS API and CHIRPS precipitation data.
Run with --help for usage examples and data availability information.
"""

import os
import argparse
import datetime
import subprocess
import cdsapi


# =============================================================================
# SEAS5 DATA AVAILABILITY CONFIGURATION
# =============================================================================
# Historical years have all months available. Current/future years may have
# limited months depending on when the forecasts are released.
# Update CURRENT_YEAR_AVAILABLE_MONTHS as new forecasts become available.

# Full historical data is available for these years (all 12 months)
SEAS5_FULL_YEARS = list(range(1981, 2026))  # 1981-2025 have all months

# For the current year (2026), specify available months
# ECMWF releases forecasts around the 13th of each month
# Update this as new months become available
CURRENT_YEAR = 2026
CURRENT_YEAR_AVAILABLE_MONTHS = [1]  # As of January 2026, only month 1 is available

# Future years have no data available
FUTURE_YEARS_START = 2027


def get_available_months_for_year(year):
    """
    Get list of available months for a given year.

    Args:
        year: Year to check (integer)

    Returns:
        list: List of available month numbers (1-12), or empty list if year not available
    """
    year = int(year)

    if year < 1981:
        return []
    elif year in SEAS5_FULL_YEARS:
        return list(range(1, 13))  # All months 1-12
    elif year == CURRENT_YEAR:
        return CURRENT_YEAR_AVAILABLE_MONTHS.copy()
    else:
        return []  # Future years


def validate_year_month_availability(year, months):
    """
    Validate that requested months are available for the specified year.

    Args:
        year: Year to download (integer)
        months: List of month strings (e.g., ["01", "02", "03"])

    Returns:
        tuple: (is_valid, available_months, unavailable_months)
    """
    available = get_available_months_for_year(year)
    available_set = set(available)

    requested_months = [int(m) for m in months]
    unavailable = [m for m in requested_months if m not in available_set]
    valid_months = [m for m in requested_months if m in available_set]

    return (len(unavailable) == 0, valid_months, unavailable)


def print_availability_info(year=None):
    """
    Print information about SEAS5 data availability.

    Args:
        year: Optional specific year to check. If None, prints general info.
    """
    print("\n" + "=" * 70)
    print("SEAS5 DATA AVAILABILITY INFORMATION")
    print("=" * 70)

    if year is not None:
        year = int(year)
        available = get_available_months_for_year(year)

        if not available:
            if year < 1981:
                print(f"\nYear {year}: NO DATA AVAILABLE (SEAS5 starts from 1981)")
            else:
                print(f"\nYear {year}: NO DATA AVAILABLE YET (future year)")
        else:
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            available_names = [month_names[m-1] for m in available]
            print(f"\nYear {year}: {len(available)} months available")
            print(f"Available months: {', '.join(available_names)}")
            print(f"Month numbers: {', '.join(str(m) for m in available)}")

            if year == CURRENT_YEAR and len(available) < 12:
                print(f"\nNote: This is the current year. More months will become")
                print(f"available as ECMWF releases new forecasts (~13th of each month).")
    else:
        print(f"\nHistorical data (1981-{max(SEAS5_FULL_YEARS)}): All 12 months available")
        print(f"Current year ({CURRENT_YEAR}): Months {CURRENT_YEAR_AVAILABLE_MONTHS} available")
        print(f"Future years ({FUTURE_YEARS_START}+): No data available yet")

    print("\n" + "-" * 70)
    print("IMPORTANT: When downloading data for the current year, ensure you only")
    print("request months that are available. Use --check-availability --year YYYY")
    print("to verify availability before downloading.")
    print("=" * 70 + "\n")

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


def parse_month_input(month_input):
    """
    Parse month input in various formats (single month, comma-separated, or range)

    Args:
        month_input: Input string like "3", "1,2,3,4", "1-6", or "1-12"

    Returns:
        list: List of month strings formatted as two digits (e.g., ["01", "02"])
    """
    months = []

    # Check if input is a range (e.g., "1-6")
    if isinstance(month_input, str) and "-" in month_input:
        start, end = map(int, month_input.split("-"))
        if 1 <= start <= 12 and 1 <= end <= 12:
            months = [f"{m:02d}" for m in range(start, end + 1)]
        else:
            raise ValueError("Month range must be between 1-12")

    # Check if input is comma-separated (e.g., "1,3,5")
    elif isinstance(month_input, str) and "," in month_input:
        for m in month_input.split(","):
            month_num = int(m.strip())
            if 1 <= month_num <= 12:
                months.append(f"{month_num:02d}")
            else:
                raise ValueError(f"Invalid month: {month_num}. Must be between 1-12")

    # Single string that can be converted to integer (e.g., "8")
    elif isinstance(month_input, str):
        try:
            month_num = int(month_input.strip())
            if 1 <= month_num <= 12:
                months.append(f"{month_num:02d}")
            else:
                raise ValueError(f"Invalid month: {month_num}. Must be between 1-12")
        except ValueError:
            raise ValueError(f"Invalid month format: {month_input}. Expected a number, comma-separated numbers, or range (e.g., 1-6)")

    # Single integer or list of integers
    elif isinstance(month_input, int) or isinstance(month_input, list):
        if isinstance(month_input, int):
            month_input = [month_input]
        for m in month_input:
            if 1 <= m <= 12:
                months.append(f"{m:02d}")
            else:
                raise ValueError(f"Invalid month: {m}. Must be between 1-12")

    if not months:
        raise ValueError("No valid months provided")

    return months

def download_current_month_seas5(output_dir="./data", filename_prefix="seas5_precipitation_",
                                  month_input=None, year=None, year_start=None, year_end=None,
                                  validate_availability=False, skip_unavailable=False):
    """
    Download SEAS5 dataset from ECMWF CDS API for specific months and year(s)

    Args:
        output_dir: Directory to save the downloaded data
        filename_prefix: Prefix for the output filename
        month_input: Month(s) to download data for - can be:
                    - Single integer (1-12)
                    - String range like "1-6"
                    - Comma-separated string like "1,3,5"
                    - List of integers [1, 3, 5]
        year: Single year to download (4-digit integer, defaults to current year if None)
        year_start: Start year for year range (use with year_end for multi-year download)
        year_end: End year for year range (use with year_start for multi-year download)
        validate_availability: If True, validate months against known availability before download
        skip_unavailable: If True, skip unavailable months/years instead of failing

    Returns:
        str: Path to the downloaded file, or None if download failed

    Raises:
        ValueError: If requested months are not available and skip_unavailable is False

    Note:
        SEAS5 data availability varies by year:
        - Historical years (1981-2025): All months (1-12) available
        - Current year (2026): Only released months available (updated monthly ~13th)
        - Future years: No data available

        Use --check-availability --year YYYY to verify before downloading.

    Examples:
        # Single year download
        download_current_month_seas5(month_input="1-12", year=2025)

        # Year range download (1981-2025)
        download_current_month_seas5(month_input="1-12", year_start=1981, year_end=2025)
    """
    # Determine years to download
    if year_start is not None and year_end is not None:
        # Year range mode
        if year_start > year_end:
            print(f"ERROR: --year-start ({year_start}) must be <= --year-end ({year_end})")
            return None
        years = list(range(year_start, year_end + 1))
        year_str_list = [str(y) for y in years]
        year_display = f"{year_start}-{year_end}"
    elif year is not None:
        # Single year mode
        years = [year]
        year_str_list = [str(year)]
        year_display = str(year)
    else:
        # Default to current year
        year = datetime.datetime.now().year
        years = [year]
        year_str_list = [str(year)]
        year_display = str(year)

    # Parse month input to get list of months in proper format
    months = parse_month_input(month_input)

    # Validate availability for each year if requested or if any year >= CURRENT_YEAR
    needs_validation = validate_availability or any(y >= CURRENT_YEAR for y in years)

    if needs_validation:
        valid_years = []
        skipped_years = []

        for y in years:
            is_valid, valid_months_for_year, unavailable = validate_year_month_availability(y, months)

            if is_valid:
                valid_years.append(y)
            else:
                available = get_available_months_for_year(y)
                available_str = ', '.join(str(m) for m in available) if available else 'None'
                unavailable_str = ', '.join(str(m) for m in unavailable)

                if y >= CURRENT_YEAR:
                    if skip_unavailable:
                        print(f"WARNING: Year {y} - months {unavailable_str} not available. Skipping year.")
                        skipped_years.append(y)
                    else:
                        print(f"\nERROR: Requested months {unavailable_str} are not available for year {y}.")
                        print(f"Available months for {y}: {available_str}")
                        print(f"\nTo check availability: python 00-download-data.py --check-availability --year {y}")
                        print(f"To skip unavailable years: add --skip-unavailable flag")

                        if y == CURRENT_YEAR:
                            print(f"\nNote: {y} is the current year. ECMWF releases new forecasts")
                            print(f"around the 13th of each month. Update CURRENT_YEAR_AVAILABLE_MONTHS")
                            print(f"in this script when new months become available.")

                        return None
                else:
                    valid_years.append(y)  # Historical years should have all months

        if skipped_years:
            print(f"Skipped years due to unavailable months: {', '.join(str(y) for y in skipped_years)}")

        if not valid_years:
            print("ERROR: No valid years to download after validation.")
            return None

        years = valid_years
        year_str_list = [str(y) for y in years]
        if len(years) == 1:
            year_display = str(years[0])
        else:
            year_display = f"{min(years)}-{max(years)}"

    print(f"Downloading SEAS5 data from ECMWF CDS for months {', '.join(months)}, years {year_display}...")
    print(f"Total years: {len(years)}, Total months per year: {len(months)}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create filename with year and month info
    if len(years) == 1:
        year_info = f"year{years[0]}"
    else:
        year_info = f"years{min(years)}-{max(years)}"

    month_info = "months_" + "_".join(months) if len(months) <= 4 else f"months_{len(months)}_months"
    current_date = datetime.datetime.now().strftime("%Y%m%d")
    output_file = os.path.join(output_dir, f'{filename_prefix}{current_date}_{year_info}_{month_info}.grib')

    # Define the SEAS5 dataset and request parameters
    dataset = "seasonal-monthly-single-levels"
    request = {
        "originating_centre": "ecmwf",
        "system": "51",
        "variable": ["total_precipitation"],
        "year": year_str_list,
        "month": months,
        "leadtime_month": ["1", "2", "3", "4", "5", "6"],
        "data_format": "grib",
        "product_type": ["monthly_mean"],
        "area": [23, 21, -12, 53]
    }

    try:
        client = cdsapi.Client()
        client.retrieve(dataset, request, output_file)
        print(f"SEAS5 data for months {', '.join(months)}, years {year_display} downloaded successfully to: {output_file}")
        return output_file
    except Exception as e:
        print(f"Error downloading SEAS5 data for months {', '.join(months)}, years {year_display}: {e}")
        return None

def check_grib_file(grib_file_path):
    """
    Check a GRIB file to display available time information
    
    Args:
        grib_file_path: Path to the GRIB file to check
    """
    print(f"Checking GRIB file: {grib_file_path}")
    
    try:
        import xarray as xr
        
        # Open the dataset with specific backend settings
        ds = xr.open_dataset(grib_file_path, engine='cfgrib',
                            backend_kwargs=dict(time_dims=('forecastMonth', 'time')))
        
        # Extract and display time information
        print("\nGRIB file contents summary:")
        print("-------------------------")
        
        # Display dataset dimensions
        print(f"Dimensions: {dict(ds.dims)}")
        
        # Print time-related coordinates
        for time_dim in ['time', 'forecastMonth', 'valid_time']:
            if time_dim in ds.coords:
                time_values = ds[time_dim].values
                if len(time_values) > 0:
                    print(f"\n{time_dim}:")
                    print(f"  - Start: {time_values[0]}")
                    print(f"  - End: {time_values[-1]}")
                    print(f"  - Total values: {len(time_values)}")
        
        # Show variables in the dataset
        print("\nVariables in dataset:")
        for var_name, var in ds.variables.items():
            if var_name not in ds.dims and var_name not in ds.coords:
                print(f"  - {var_name}: {var.dims}")
        
        ds.close()
        
    except Exception as e:
        print(f"Error checking GRIB file: {e}")

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

def main():
    # Create epilog with data availability information
    epilog_text = """
DATA AVAILABILITY NOTES:
------------------------
SEAS5 data availability varies by year:
  - Historical years (1981-2025): All months (1-12) are available
  - Current year (2026): Only months with released forecasts are available
    (ECMWF releases new forecasts around the 13th of each month)
  - Future years: No data available

IMPORTANT: When downloading data for the current year (2026), you must only
request months that have been released. Requesting unavailable months will
cause the CDS API to fail.

To check which months are available:
  python 00-download-data.py --check-availability --year 2026

To automatically validate and skip unavailable months:
  python 00-download-data.py --only-current-month-seas5 1-6 --year 2026 --validate-availability --skip-unavailable

EXAMPLES:
---------
# Download single year historical data (all months available):
  python 00-download-data.py --output-dir ../data --only-current-month-seas5 1-12 --year 2025

# Download full historical dataset (1981-2025, all months):
  python 00-download-data.py --output-dir ../data --only-current-month-seas5 1-12 --year-start 1981 --year-end 2025

# Download partial historical range:
  python 00-download-data.py --output-dir ../data --only-current-month-seas5 1-12 --year-start 2000 --year-end 2025

# Download current year (only available months):
  python 00-download-data.py --output-dir ../data --only-current-month-seas5 1 --year 2026

# Download historical + current year (skip unavailable):
  python 00-download-data.py --output-dir ../data --only-current-month-seas5 1-12 --year-start 1981 --year-end 2026 --skip-unavailable

# Check data availability:
  python 00-download-data.py --check-availability --year 2026

# Validate before downloading:
  python 00-download-data.py --only-current-month-seas5 1-3 --year 2026 --validate-availability
"""

    parser = argparse.ArgumentParser(
        description="Download ECMWF SEAS5 and CHIRPS data",
        epilog=epilog_text,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Output options
    parser.add_argument("--output-dir", default="./data",
                        help="Directory to save downloaded data (default: ./data)")

    # Data source selection
    source_group = parser.add_argument_group('Data Source Selection')
    source_group.add_argument("--seas5-only", action="store_true",
                              help="Download only SEAS5 data (no CHIRPS)")
    source_group.add_argument("--chirps-only", action="store_true",
                              help="Download only CHIRPS data (no SEAS5)")

    # SEAS5 specific options
    seas5_group = parser.add_argument_group('SEAS5 Options')
    seas5_group.add_argument("--only-current-month-seas5", type=str, metavar="MONTHS",
                             help="Download SEAS5 data for specific months. Formats: "
                                  "single (3), comma-separated (1,2,3), or range (1-12)")
    seas5_group.add_argument("--year", type=int,
                             help="Single year for SEAS5 data. IMPORTANT: For 2026+, only released "
                                  "months are available. Use --check-availability to verify. "
                                  "Cannot be used with --year-start/--year-end.")
    seas5_group.add_argument("--year-start", type=int, metavar="YEAR",
                             help="Start year for multi-year download (e.g., 1981). "
                                  "Use with --year-end for downloading year ranges.")
    seas5_group.add_argument("--year-end", type=int, metavar="YEAR",
                             help="End year for multi-year download (e.g., 2025). "
                                  "Use with --year-start for downloading year ranges.")

    # Data availability options
    avail_group = parser.add_argument_group('Data Availability',
                                            'Options for checking and validating data availability')
    avail_group.add_argument("--check-availability", action="store_true",
                             help="Check and display SEAS5 data availability for a year "
                                  "(use with --year to check a specific year)")
    avail_group.add_argument("--validate-availability", action="store_true",
                             help="Validate requested months against known availability before "
                                  "downloading (automatically enabled for year >= 2026)")
    avail_group.add_argument("--skip-unavailable", action="store_true",
                             help="Skip unavailable months/years instead of failing "
                                  "(use with --validate-availability or year ranges including 2026+)")

    # Utility options
    util_group = parser.add_argument_group('Utility Options')
    util_group.add_argument("--check-available-grib", type=str, metavar="GRIB_FILE_PATH",
                            help="Check and display time information for a specified GRIB file")

    args = parser.parse_args()

    # If check-availability option is provided, show availability info and exit
    if args.check_availability:
        print_availability_info(args.year)
        return

    # If check-available-grib option is provided, do that and exit
    if args.check_available_grib:
        check_grib_file(args.check_available_grib)
        return

    # Handle mutually exclusive options
    if sum([args.seas5_only, args.chirps_only, args.only_current_month_seas5 is not None]) > 1:
        print("Error: Cannot specify multiple download options together")
        print("Use --seas5-only OR --chirps-only OR --only-current-month-seas5")
        return

    # Validate year arguments
    has_year = args.year is not None
    has_year_range = args.year_start is not None or args.year_end is not None

    if has_year and has_year_range:
        print("Error: Cannot use --year together with --year-start/--year-end")
        print("Use either --year for single year OR --year-start and --year-end for year range")
        return

    if (args.year_start is not None) != (args.year_end is not None):
        print("Error: --year-start and --year-end must be used together")
        return

    if args.year_start is not None and args.year_end is not None:
        if args.year_start > args.year_end:
            print(f"Error: --year-start ({args.year_start}) must be <= --year-end ({args.year_end})")
            return
        if args.year_start < 1981:
            print(f"Error: --year-start ({args.year_start}) cannot be before 1981 (SEAS5 data starts from 1981)")
            return

    # Validate that year options are only used with appropriate download options
    if (has_year or has_year_range) and args.only_current_month_seas5 is None and not args.check_availability:
        print("Error: Year options can only be used with --only-current-month-seas5 or --check-availability")
        return

    # Validate --skip-unavailable requires --validate-availability
    if args.skip_unavailable and not args.validate_availability:
        print("Warning: --skip-unavailable requires --validate-availability. Enabling validation.")
        args.validate_availability = True

    # Download SEAS5 data for specific month if requested
    if args.only_current_month_seas5 is not None:
        seas5_file = download_current_month_seas5(
            args.output_dir,
            month_input=args.only_current_month_seas5,
            year=args.year,
            year_start=args.year_start,
            year_end=args.year_end,
            validate_availability=args.validate_availability,
            skip_unavailable=args.skip_unavailable
        )
        if seas5_file is None:
            print("\nDownload failed. Please check the error messages above.")
            return
    # Download full SEAS5 data if requested or if no specific option is specified
    elif args.seas5_only or not args.chirps_only:
        seas5_file = download_seas5(args.output_dir)
    
    # Download CHIRPS data if requested or if neither option is specified
    if args.chirps_only or (not args.seas5_only and args.only_current_month_seas5 is None):
        chirps_file = download_chirps(args.output_dir)
    
    print("Data download process complete")

if __name__ == "__main__":
    main()
