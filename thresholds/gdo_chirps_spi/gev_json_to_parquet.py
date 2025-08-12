#!/usr/bin/env python3
"""
Convert GEV JSON files to long table format and upload to GCS.

This script processes JSON files containing GEV (Generalized Extreme Value) 
return period data for different SPI (Standardized Precipitation Index) 
timescales and converts them into a single parquet file for upload to GCS.

python gev_json_to_parquet.py --json-folder gdo-chirps-spi-gev --bucket-name gev_e4drr --gcs-path gdo_chirps_spi --upload --service-account-json coiled-data-e4drr_202505.json --csv
"""

import json
import pandas as pd
import os
from pathlib import Path
import argparse
from google.cloud import storage
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_spi_from_filename(filename):
    """Extract SPI timescale from filename (e.g., spi3, spi6, etc.)"""
    parts = filename.split('_')
    for part in parts:
        if part.startswith('spi') and part[3:].isdigit():
            return part
    return None


def json_to_long_table(json_file_path):
    """Convert a single JSON file to long table format"""
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # Extract SPI name from filename
    spi_name = extract_spi_from_filename(os.path.basename(json_file_path))

    # Extract metadata
    metadata = data.get('metadata', {})
    session_id = metadata.get('session_id', '')
    created_at = metadata.get('created_at', '')
    script_version = metadata.get('script_version', '')

    # Process results
    rows = []
    for result in data.get('results', []):
        region_id = result.get('region_id')
        region_name = result.get('region_name')
        return_periods = result.get('return_periods', [])
        return_levels = result.get('return_levels', [])
        status = result.get('status')
        worker_id = result.get('worker_id', '')
        n_years = result.get('n_years', 0)

        # Create a row for each return period
        for period, level in zip(return_periods, return_levels):
            rows.append({
                'spi_name': spi_name,
                'region_name': region_name,
                'region_id': region_id,
                'return_period': period,
                'gev_value': level,
                'status': status,
                'worker_id': worker_id,
                'n_years': n_years,
                'session_id': session_id,
                'created_at': created_at,
                'script_version': script_version,
                'source_file': os.path.basename(json_file_path)
            })

    return pd.DataFrame(rows)


def process_all_json_files(json_folder_path):
    """Process all JSON files in the folder and combine into single DataFrame"""
    json_folder = Path(json_folder_path)
    all_dataframes = []

    for json_file in json_folder.glob('*.json'):
        logger.info(f"Processing {json_file.name}")
        df = json_to_long_table(json_file)
        all_dataframes.append(df)

    if all_dataframes:
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        logger.info(
            f"Combined {len(all_dataframes)} files into {len(combined_df)} rows"
        )
        return combined_df
    else:
        logger.warning("No JSON files found")
        return pd.DataFrame()


def upload_to_gcs(parquet_file_path,
                  bucket_name,
                  gcs_path,
                  service_account_path=None):
    """Upload parquet file to Google Cloud Storage"""
    try:
        if service_account_path and os.path.exists(service_account_path):
            logger.info(f"Using service account: {service_account_path}")
            client = storage.Client.from_service_account_json(
                service_account_path)
        else:
            logger.info("Using default credentials")
            client = storage.Client()

        bucket = client.bucket(bucket_name)
        blob = bucket.blob(gcs_path)

        logger.info(
            f"Uploading {parquet_file_path} to gs://{bucket_name}/{gcs_path}")
        blob.upload_from_filename(parquet_file_path)
        logger.info(f"Upload complete: gs://{bucket_name}/{gcs_path}")
    except Exception as e:
        logger.error(f"GCS upload failed: {e}")
        if not service_account_path:
            logger.info(
                "Consider using --service-account-json parameter or set GOOGLE_APPLICATION_CREDENTIALS"
            )


def main():
    parser = argparse.ArgumentParser(
        description='Convert GEV JSON files to parquet and upload to GCS')
    parser.add_argument('--json-folder',
                        default='gdo-chirps-spi-gev',
                        help='Folder containing JSON files')
    parser.add_argument('--output-file',
                        default='gev_combined_data.parquet',
                        help='Output parquet file name')
    parser.add_argument('--bucket-name',
                        default='gev_e4drr',
                        help='GCS bucket name')
    parser.add_argument('--gcs-path',
                        default='gdo_chirps_spi',
                        help='GCS path prefix')
    parser.add_argument('--upload',
                        action='store_true',
                        help='Upload to GCS after creating parquet file')
    parser.add_argument('--csv',
                        action='store_true',
                        help='Also save as CSV file for inspection')
    parser.add_argument(
        '--service-account-json',
        default='coiled-data-e4drr_202505.json',
        help='Path to service account JSON file for GCS authentication')

    args = parser.parse_args()

    # Process JSON files
    logger.info(f"Processing JSON files from {args.json_folder}")
    combined_df = process_all_json_files(args.json_folder)

    if combined_df.empty:
        logger.error("No data to process")
        return

    # Save to parquet
    logger.info(f"Saving to {args.output_file}")
    combined_df.to_parquet(args.output_file, index=False)

    # Save to CSV if requested
    if args.csv:
        csv_file = args.output_file.replace('.parquet', '.csv')
        logger.info(f"Saving to {csv_file}")
        combined_df.to_csv(csv_file, index=False)

    # Display summary
    logger.info(f"Summary:")
    logger.info(f"  Total rows: {len(combined_df)}")
    logger.info(
        f"  Unique SPI types: {sorted(combined_df['spi_name'].unique())}")
    logger.info(f"  Unique regions: {combined_df['region_id'].nunique()}")
    logger.info(
        f"  Return periods: {sorted(combined_df['return_period'].unique())}")

    # Upload to GCS if requested
    if args.upload:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        gcs_file_path = f"{args.gcs_path}/gev_combined_data_{timestamp}.parquet"
        upload_to_gcs(args.output_file, args.bucket_name, gcs_file_path,
                      args.service_account_json)


if __name__ == '__main__':
    main()
