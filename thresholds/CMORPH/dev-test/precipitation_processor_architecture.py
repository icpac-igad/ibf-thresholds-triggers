#!/usr/bin/env python3
"""
NOAA CDR Precipitation Dataset Processor Architecture
Adapted from virtualizarr_gdo_spi_processor_json.py for CMORPH and PERSIANN datasets
"""

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import boto3
from botocore import UNSIGNED
from botocore.config import Config

import xarray as xr
from virtualizarr import open_virtual_dataset
from virtualizarr.parsers import HDFParser
from virtualizarr.registry import ObjectStoreRegistry

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Dataset configuration
PRECIPITATION_DATASETS = {
    'cmorph': {
        'name': 'CMORPH',
        'bucket': 'noaa-cdr-precip-cmorph-pds',
        'base_path': 'data/30min/8km/',
        'description': 'Climate Prediction Center MORPHing technique - 30min, 8km',
        'temporal_resolution': '30min',
        'spatial_resolution': '8km'
    },
    'persiann': {
        'name': 'PERSIANN',
        'bucket': 'noaa-cdr-precip-persiann-pds',
        'base_path': 'data/',
        'description': 'Precipitation Estimation from Remotely Sensed Information using Artificial Neural Networks',
        'temporal_resolution': 'daily',
        'spatial_resolution': '25km'
    }
}


class PrecipitationProcessor:
    """Base class for processing precipitation datasets with VirtualiZarr"""
    
    def __init__(self, dataset_name: str, target_year: int, output_dir: str = ".", 
                 timeout: int = 30):
        """
        Initialize the precipitation processor
        
        Args:
            dataset_name: Name of dataset ('cmorph' or 'persiann')
            target_year: Year to process (e.g., 1983)
            output_dir: Directory to save JSON reference files
            timeout: HTTP request timeout in seconds
        """
        if dataset_name not in PRECIPITATION_DATASETS:
            raise ValueError(f"Dataset {dataset_name} not supported. Choose from: {list(PRECIPITATION_DATASETS.keys())}")
        
        self.dataset_name = dataset_name
        self.dataset_config = PRECIPITATION_DATASETS[dataset_name]
        self.target_year = target_year
        self.timeout = timeout
        
        # Setup output directory
        self.output_dir = Path(output_dir)
        self.dataset_output_dir = self.output_dir / f"{dataset_name}_{target_year}"
        self.dataset_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup S3 client for direct access
        self.s3_client = boto3.client(
            's3',
            config=Config(signature_version=UNSIGNED)  # For public buckets
        )
            
        logger.info(f"Initialized {self.dataset_config['name']} processor for year {target_year}")
        logger.info(f"Output directory: {self.dataset_output_dir}")
    
    def get_nc_files_s3_direct(self) -> List[str]:
        """
        Get NetCDF files directly from S3 bucket using boto3
        
        Returns:
            List of S3 URLs to .nc files
        """
        logger.info(f"Listing NetCDF files from S3 bucket: {self.dataset_config['bucket']}")
        
        try:
            bucket_name = self.dataset_config['bucket']
            prefix = f"{self.dataset_config['base_path']}{self.target_year}/"
            
            nc_files = []
            paginator = self.s3_client.get_paginator('list_objects_v2')
            
            for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        key = obj['Key']
                        if key.endswith('.nc'):
                            # Construct S3 URL
                            s3_url = f"https://{bucket_name}.s3.amazonaws.com/{key}"
                            nc_files.append(s3_url)
            
            nc_files = sorted(nc_files)
            logger.info(f"Found {len(nc_files)} NetCDF files for {self.target_year}")
            
            return nc_files
            
        except Exception as e:
            logger.error(f"Error listing S3 objects: {e}")
            return []
    
    
    def get_nc_files(self) -> List[str]:
        """
        Get list of NetCDF files using S3 direct access
        
        Returns:
            List of URLs to .nc files
        """
        if not self.s3_client:
            logger.error("S3 client not initialized")
            return []
        
        nc_files = self.get_nc_files_s3_direct()
        return nc_files
    
    def create_virtual_dataset(self, nc_url: str) -> Optional[xr.Dataset]:
        """
        Create a virtual dataset from a single NetCDF file URL
        
        Args:
            nc_url: URL to NetCDF file
            
        Returns:
            Virtual dataset or None if failed
        """
        try:
            logger.debug(f"Creating virtual dataset for: {Path(nc_url).name}")
            
            # Create registry and parser for VirtualiZarr
            registry = ObjectStoreRegistry()
            parser = HDFParser()
            
            # Use VirtualiZarr with minimal data loading for efficiency
            vds = open_virtual_dataset(
                nc_url,
                registry=registry,
                parser=parser,
                loadable_variables=[],  # Don't load any variables initially
                decode_times=False      # Avoid temporal parsing issues
            )
            
            return vds
            
        except Exception as e:
            logger.error(f"Error creating virtual dataset from {nc_url}: {e}")
            return None
    
    def export_to_json(self, dataset: xr.Dataset, filename: str) -> bool:
        """
        Export virtual dataset to Kerchunk reference JSON file
        
        Args:
            dataset: Virtual dataset to export
            filename: Output filename (without path)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            output_path = self.dataset_output_dir / filename
            logger.debug(f"Exporting to JSON: {filename}")
            
            # Convert to Kerchunk reference format
            refs_dict = dataset.virtualize.to_kerchunk(format='dict')
            
            # Save to JSON file
            with open(output_path, 'w') as f:
                json.dump(refs_dict, f, indent=2)
            
            file_size_mb = output_path.stat().st_size / 1024 / 1024
            logger.info(f"✓ Saved {filename} ({file_size_mb:.2f} MB)")
            
            return True
            
        except Exception as e:
            logger.error(f"Error exporting {filename} to JSON: {e}")
            return False
    
    def process_individual_files(self, nc_files: List[str], max_files: Optional[int] = None) -> Dict[str, Any]:
        """
        Process each NetCDF file individually and save as separate JSON files
        
        Args:
            nc_files: List of NetCDF file URLs
            max_files: Maximum number of files to process (for testing)
            
        Returns:
            Processing summary dictionary
        """
        if max_files:
            process_files = nc_files[:max_files]
            logger.info(f"TESTING MODE: Processing {max_files} files out of {len(nc_files)}")
        else:
            process_files = nc_files
            logger.info(f"FULL PROCESSING: Processing all {len(nc_files)} files")
        
        successful_files = []
        failed_files = []
        
        for i, nc_url in enumerate(process_files, 1):
            filename = Path(nc_url).name
            
            try:
                logger.info(f"Processing file {i}/{len(process_files)}: {filename}")
                
                # Create virtual dataset
                vds = self.create_virtual_dataset(nc_url)
                if vds is None:
                    failed_files.append(filename)
                    continue
                
                # Generate output filename
                file_number = str(i).zfill(3)
                output_filename = f"{self.dataset_name}_{self.target_year}_file{file_number}.json"
                
                # Export to JSON
                success = self.export_to_json(vds, output_filename)
                
                if success:
                    successful_files.append({
                        'original_filename': filename,
                        'json_filename': output_filename,
                        'url': nc_url
                    })
                else:
                    failed_files.append(filename)
                
            except Exception as e:
                logger.error(f"✗ Failed to process {filename}: {e}")
                failed_files.append(filename)
                continue
        
        # Create processing summary
        summary = {
            'dataset_info': {
                'name': self.dataset_config['name'],
                'dataset_key': self.dataset_name,
                'year': self.target_year,
                'temporal_resolution': self.dataset_config['temporal_resolution'],
                'spatial_resolution': self.dataset_config['spatial_resolution']
            },
            'processing_summary': {
                'total_files_available': len(nc_files),
                'total_files_processed': len(process_files),
                'successful_files': len(successful_files),
                'failed_files': len(failed_files),
                'success_rate': f"{len(successful_files)/len(process_files)*100:.1f}%" if process_files else "0%",
                'processing_timestamp': datetime.now().isoformat()
            },
            'successful_files': successful_files,
            'failed_files': failed_files if failed_files else None
        }
        
        # Save summary file
        summary_filename = f"{self.dataset_name}_{self.target_year}_processing_summary.json"
        summary_path = self.dataset_output_dir / summary_filename
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Created processing summary: {summary_path}")
        logger.info(f"Successfully processed {len(successful_files)}/{len(process_files)} files for {self.dataset_config['name']} {self.target_year}")
        
        return summary
    
    def validate_json_files(self) -> Dict[str, Any]:
        """
        Validate generated JSON files by attempting to load them
        
        Returns:
            Validation results dictionary
        """
        logger.info(f"Validating JSON files for {self.dataset_name} {self.target_year}")
        
        json_files = list(self.dataset_output_dir.glob("*.json"))
        # Exclude summary files from validation
        json_files = [f for f in json_files if not f.name.endswith('_summary.json')]
        
        valid_files = []
        invalid_files = []
        
        for json_file in json_files:
            try:
                # Test JSON loading
                with open(json_file, 'r') as f:
                    refs_dict = json.load(f)
                
                # Basic structure validation
                if isinstance(refs_dict, dict) and 'refs' in refs_dict:
                    valid_files.append(json_file.name)
                    logger.debug(f"✓ Valid: {json_file.name}")
                else:
                    invalid_files.append({
                        'filename': json_file.name,
                        'error': 'Missing refs structure'
                    })
                    logger.warning(f"✗ Invalid structure: {json_file.name}")
                
            except Exception as e:
                invalid_files.append({
                    'filename': json_file.name,
                    'error': str(e)
                })
                logger.error(f"✗ Failed validation: {json_file.name} - {e}")
        
        validation_results = {
            'validation_info': {
                'dataset': self.dataset_name,
                'year': self.target_year,
                'validation_timestamp': datetime.now().isoformat()
            },
            'validation_summary': {
                'total_json_files': len(json_files),
                'valid_files': len(valid_files),
                'invalid_files': len(invalid_files),
                'validation_success_rate': f"{len(valid_files)/len(json_files)*100:.1f}%" if json_files else "0%"
            },
            'valid_files': valid_files,
            'invalid_files': invalid_files if invalid_files else None
        }
        
        # Save validation results
        validation_filename = f"{self.dataset_name}_{self.target_year}_validation_results.json"
        validation_path = self.dataset_output_dir / validation_filename
        
        with open(validation_path, 'w') as f:
            json.dump(validation_results, f, indent=2)
        
        logger.info(f"Validation complete: {len(valid_files)}/{len(json_files)} files valid")
        logger.info(f"Validation results saved: {validation_path}")
        
        return validation_results


class CMORPHProcessor(PrecipitationProcessor):
    """Specialized processor for CMORPH 30-minute precipitation data"""
    
    def __init__(self, target_year: int, **kwargs):
        super().__init__('cmorph', target_year, **kwargs)


class PERSIANNProcessor(PrecipitationProcessor):
    """Specialized processor for PERSIANN daily precipitation data"""
    
    def __init__(self, target_year: int, **kwargs):
        super().__init__('persiann', target_year, **kwargs)


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="Process NOAA CDR precipitation datasets using VirtualiZarr",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process CMORPH data for 1983
  python precipitation_processor.py --dataset cmorph --year 1983
  
  # Process PERSIANN data for 1983 with validation
  python precipitation_processor.py --dataset persiann --year 1983 --validate
  
  # Process both datasets for 1983
  python precipitation_processor.py --dataset cmorph persiann --year 1983
  
  # Testing mode (process only first 5 files)
  python precipitation_processor.py --dataset cmorph --year 1983 --max-files 5
        """
    )
    
    parser.add_argument(
        '--dataset',
        nargs='+',
        choices=['cmorph', 'persiann'],
        required=True,
        help='Precipitation datasets to process'
    )
    
    parser.add_argument(
        '--year',
        type=int,
        required=True,
        help='Target year to process (e.g., 1983)'
    )
    
    parser.add_argument(
        '--output-dir',
        default='.',
        help='Output directory for JSON files (default: current directory)'
    )
    
    parser.add_argument(
        '--max-files',
        type=int,
        help='Maximum number of files to process (for testing)'
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Run validation on generated JSON files'
    )
    
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=30,
        help='HTTP request timeout in seconds (default: 30)'
    )
    
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info(f"Starting precipitation dataset processing for year {args.year}")
    logger.info(f"Datasets: {', '.join(args.dataset)}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Process each dataset
    processors = []
    successful_datasets = []
    failed_datasets = []
    
    for dataset_name in args.dataset:
        try:
            logger.info(f"\\n{'='*60}")
            logger.info(f"Processing {dataset_name.upper()} dataset")
            logger.info(f"{'='*60}")
            
            # Initialize processor
            if dataset_name == 'cmorph':
                processor = CMORPHProcessor(
                    target_year=args.year,
                    output_dir=args.output_dir,
                    timeout=args.timeout
                )
            elif dataset_name == 'persiann':
                processor = PERSIANNProcessor(
                    target_year=args.year,
                    output_dir=args.output_dir,
                    timeout=args.timeout
                )
            
            processors.append(processor)
            
            # Get NetCDF files
            nc_files = processor.get_nc_files()
            if not nc_files:
                logger.error(f"No NetCDF files found for {dataset_name} {args.year}")
                failed_datasets.append(dataset_name)
                continue
            
            # Process files
            summary = processor.process_individual_files(nc_files, args.max_files)
            
            if summary['processing_summary']['successful_files'] > 0:
                successful_datasets.append(dataset_name)
                logger.info(f"✓ Successfully processed {dataset_name}")
            else:
                failed_datasets.append(dataset_name)
                logger.error(f"✗ Failed to process any files for {dataset_name}")
            
            # Run validation if requested
            if args.validate:
                logger.info(f"\\n--- Validating {dataset_name} JSON files ---")
                validation_results = processor.validate_json_files()
                
        except Exception as e:
            logger.error(f"Error processing {dataset_name}: {e}")
            failed_datasets.append(dataset_name)
            continue
    
    # Final summary
    logger.info(f"\\n{'='*60}")
    logger.info(f"PROCESSING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Year: {args.year}")
    logger.info(f"Successful datasets: {len(successful_datasets)} - {', '.join(successful_datasets) if successful_datasets else 'None'}")
    
    if failed_datasets:
        logger.warning(f"Failed datasets: {len(failed_datasets)} - {', '.join(failed_datasets)}")
    
    # Exit with appropriate code
    if failed_datasets and not successful_datasets:
        logger.error("All datasets failed to process")
        exit(1)
    elif failed_datasets:
        logger.warning("Some datasets failed to process")
        exit(2)
    else:
        logger.info("All requested datasets processed successfully")
        exit(0)


if __name__ == "__main__":
    main()