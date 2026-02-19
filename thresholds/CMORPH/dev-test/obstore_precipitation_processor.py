#!/usr/bin/env python3
"""
NOAA CDR Precipitation Dataset Processor using obstore library
Uses obstore.store.S3Store for file listing and VirtualiZarr for NetCDF processing
"""

import logging
import warnings
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import json

from obstore.store import S3Store
import xarray as xr
from virtualizarr import open_virtual_dataset
from virtualizarr.parsers import HDFParser
from virtualizarr.registry import ObjectStoreRegistry

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Dataset configuration
DATASETS = {
    'cmorph': {
        'bucket_name': 'noaa-cdr-precip-cmorph-pds',
        'base_path': 'data/30min/8km/',
        'description': 'CMORPH 30-minute 8km precipitation data'
    },
    'persiann': {
        'bucket_name': 'noaa-cdr-precip-persiann-pds', 
        'base_path': 'data/',
        'description': 'PERSIANN daily precipitation data'
    }
}


class ObstorePrecipitationProcessor:
    """Precipitation processor using obstore for S3 access"""
    
    def __init__(self, dataset_name: str, target_year: int, output_dir: str = "."):
        """
        Initialize processor
        
        Args:
            dataset_name: 'cmorph' or 'persiann'
            target_year: Year to process
            output_dir: Output directory for JSON files
        """
        if dataset_name not in DATASETS:
            raise ValueError(f"Dataset {dataset_name} not supported. Choose from: {list(DATASETS.keys())}")
        
        self.dataset_name = dataset_name
        self.dataset_config = DATASETS[dataset_name]
        self.target_year = target_year
        
        # Setup output directory
        self.output_dir = Path(output_dir)
        self.dataset_output_dir = self.output_dir / f"{dataset_name}_{target_year}"
        self.dataset_output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized {dataset_name.upper()} processor for year {target_year}")
        logger.info(f"Output directory: {self.dataset_output_dir}")
    
    def list_nc_files_for_year(self) -> List[str]:
        """
        List all NetCDF files for the target year using obstore
        
        Returns:
            List of file paths (relative to bucket root)
        """
        logger.info(f"Listing NetCDF files for {self.dataset_name} {self.target_year}")
        
        # Create S3Store for the year directory
        year_prefix = f"{self.dataset_config['base_path']}{self.target_year}/"
        
        store = S3Store(
            bucket_name=self.dataset_config['bucket_name'],
            prefix=year_prefix,
            region="us-east-1",
            skip_signature=True
        )
        
        try:
            # List all objects in the year directory
            list_result = list(store.list())
            
            if not list_result:
                logger.warning(f"No objects found in {year_prefix}")
                return []
            
            # The result is a list containing one element which is a list of file info dicts
            all_objects = list_result[0]
            
            # Filter for .nc files and extract paths
            nc_files = []
            for obj_info in all_objects:
                if isinstance(obj_info, dict) and 'path' in obj_info:
                    if obj_info['path'].endswith('.nc'):
                        nc_files.append(obj_info['path'])
            
            nc_files.sort()
            logger.info(f"Found {len(nc_files)} NetCDF files for {self.target_year}")
            
            return nc_files
            
        except Exception as e:
            logger.error(f"Error listing files: {e}")
            return []
    
    def convert_to_s3_url(self, file_path: str) -> str:
        """
        Convert file path to S3 URL
        
        Args:
            file_path: File name only (from obstore list result)
            
        Returns:
            Full S3 URL
        """
        bucket_name = self.dataset_config['bucket_name']
        # Add the full path prefix for the dataset and year
        full_path = f"{self.dataset_config['base_path']}{self.target_year}/{file_path}"
        return f"https://{bucket_name}.s3.amazonaws.com/{full_path}"
    
    def create_virtual_dataset_simple(self, file_path: str) -> Optional[xr.Dataset]:
        """
        Create virtual dataset using simple approach
        
        Args:
            file_path: File path relative to bucket root
            
        Returns:
            Virtual dataset or None if failed
        """
        try:
            # Convert to S3 URL for VirtualiZarr
            s3_url = self.convert_to_s3_url(file_path)
            
            logger.debug(f"Creating virtual dataset for: {Path(file_path).name}")
            
            # Create registry and parser for VirtualiZarr (updated API)
            registry = ObjectStoreRegistry()
            parser = HDFParser()
            
            # Create virtual dataset
            vds = open_virtual_dataset(
                s3_url,
                registry=registry,
                parser=parser,
                loadable_variables=[],
                decode_times=False
            )
            
            return vds
            
        except Exception as e:
            logger.error(f"Error creating virtual dataset from {file_path}: {e}")
            return None
    
    def export_to_json(self, dataset: xr.Dataset, filename: str) -> bool:
        """
        Export virtual dataset to JSON
        
        Args:
            dataset: Virtual dataset
            filename: Output filename
            
        Returns:
            Success status
        """
        try:
            output_path = self.dataset_output_dir / filename
            
            # Convert to Kerchunk reference
            refs_dict = dataset.virtualize.to_kerchunk(format='dict')
            
            # Save to JSON
            with open(output_path, 'w') as f:
                json.dump(refs_dict, f, indent=2)
            
            file_size_mb = output_path.stat().st_size / 1024 / 1024
            logger.info(f"✓ Saved {filename} ({file_size_mb:.2f} MB)")
            
            return True
            
        except Exception as e:
            logger.error(f"Error exporting {filename}: {e}")
            return False
    
    def process_files(self, max_files: Optional[int] = None) -> Dict[str, Any]:
        """
        Process NetCDF files and create JSON references
        
        Args:
            max_files: Maximum number of files to process (for testing)
            
        Returns:
            Processing summary
        """
        # Get list of NetCDF files
        nc_files = self.list_nc_files_for_year()
        
        if not nc_files:
            logger.error(f"No NetCDF files found for {self.dataset_name} {self.target_year}")
            return {'error': 'No files found'}
        
        # Limit files if specified
        if max_files:
            process_files = nc_files[:max_files]
            logger.info(f"Processing first {max_files} files out of {len(nc_files)} available")
        else:
            process_files = nc_files
            logger.info(f"Processing all {len(nc_files)} files")
        
        successful_files = []
        failed_files = []
        
        for i, file_path in enumerate(process_files, 1):
            filename = Path(file_path).name
            
            try:
                logger.info(f"Processing file {i}/{len(process_files)}: {filename}")
                
                # Create virtual dataset
                vds = self.create_virtual_dataset_simple(file_path)
                
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
                        'original_path': file_path
                    })
                else:
                    failed_files.append(filename)
                
            except Exception as e:
                logger.error(f"✗ Failed to process {filename}: {e}")
                failed_files.append(filename)
                continue
        
        # Create summary
        summary = {
            'dataset_info': {
                'name': self.dataset_name.upper(),
                'year': self.target_year,
                'description': self.dataset_config['description']
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
        
        # Save summary
        summary_filename = f"{self.dataset_name}_{self.target_year}_processing_summary.json"
        summary_path = self.dataset_output_dir / summary_filename
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Processing complete: {len(successful_files)}/{len(process_files)} files successful")
        logger.info(f"Summary saved: {summary_path}")
        
        return summary


def test_file_listing():
    """Test file listing functionality"""
    logger.info("Testing file listing with obstore...")
    
    # Test CMORPH listing for a specific day (smaller dataset)
    logger.info("Testing CMORPH file listing for specific day...")
    
    store = S3Store(
        bucket_name='noaa-cdr-precip-cmorph-pds',
        prefix="data/30min/8km/1998/01/01/",  # Single day
        region="us-east-1", 
        skip_signature=True
    )
    
    try:
        list_result = list(store.list())
        
        if not list_result:
            logger.error("No objects found")
            return False
            
        # Extract file paths from the nested structure
        all_objects = list_result[0]
        nc_files = []
        for obj_info in all_objects:
            if isinstance(obj_info, dict) and 'path' in obj_info:
                if obj_info['path'].endswith('.nc'):
                    nc_files.append(obj_info['path'])
        
        logger.info(f"Found {len(nc_files)} NetCDF files for 1998-01-01")
        
        if nc_files:
            logger.info("First 5 files:")
            for i, file_path in enumerate(nc_files[:5], 1):
                logger.info(f"  {i}. {file_path}")
        
        return len(nc_files) > 0
        
    except Exception as e:
        logger.error(f"File listing test failed: {e}")
        return False


def test_virtual_dataset():
    """Test virtual dataset creation"""
    logger.info("Testing virtual dataset creation...")
    
    # Test with PERSIANN (simpler, daily files)
    store = S3Store(
        bucket_name='noaa-cdr-precip-persiann-pds',
        prefix="data/1983/",
        region="us-east-1",
        skip_signature=True
    )
    
    try:
        list_result = list(store.list())
        
        if not list_result:
            logger.error("No objects found")
            return False
            
        # Extract file paths from the nested structure
        all_objects = list_result[0]
        nc_files = []
        for obj_info in all_objects:
            if isinstance(obj_info, dict) and 'path' in obj_info:
                if obj_info['path'].endswith('.nc'):
                    nc_files.append(obj_info['path'])
        
        if not nc_files:
            logger.error("No NetCDF files found for testing")
            return False
        
        # Test first file (nc_files contains just the filename, need to add full path)
        test_file = nc_files[0]
        full_path = f"data/1983/{test_file}"  # Add the prefix path
        s3_url = f"https://noaa-cdr-precip-persiann-pds.s3.amazonaws.com/{full_path}"
        
        logger.info(f"Testing virtual dataset with: {Path(test_file).name}")
        
        # Try VirtualiZarr approach with updated API
        registry = ObjectStoreRegistry()
        parser = HDFParser()
        
        vds = open_virtual_dataset(
            s3_url,
            registry=registry,
            parser=parser,
            loadable_variables=[],
            decode_times=False
        )
        
        if vds is not None:
            logger.info(f"✓ Virtual dataset created successfully")
            logger.info(f"  Dimensions: {dict(vds.dims)}")
            logger.info(f"  Data variables: {list(vds.data_vars.keys())}")
            logger.info(f"  Coordinates: {list(vds.coords.keys())}")
            return True
        else:
            logger.error("Failed to create virtual dataset")
            return False
            
    except Exception as e:
        logger.error(f"Virtual dataset test failed: {e}")
        return False


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Process precipitation datasets using obstore and VirtualiZarr"
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run tests only'
    )
    
    parser.add_argument(
        '--dataset',
        choices=['cmorph', 'persiann'],
        help='Dataset to process'
    )
    
    parser.add_argument(
        '--year',
        type=int,
        help='Year to process'
    )
    
    parser.add_argument(
        '--max-files',
        type=int,
        default=5,
        help='Maximum number of files to process (default: 5 for testing)'
    )
    
    parser.add_argument(
        '--output-dir',
        default='.',
        help='Output directory'
    )
    
    args = parser.parse_args()
    
    if args.test:
        logger.info("Running tests...")
        logger.info("="*50)
        
        # Test file listing
        listing_success = test_file_listing()
        logger.info("="*50)
        
        # Test virtual dataset
        vds_success = test_virtual_dataset()
        logger.info("="*50)
        
        # Results
        logger.info("TEST RESULTS:")
        logger.info(f"File listing: {'✓ PASS' if listing_success else '✗ FAIL'}")
        logger.info(f"Virtual dataset: {'✓ PASS' if vds_success else '✗ FAIL'}")
        
        if listing_success and vds_success:
            logger.info("🎉 All tests passed!")
            return 0
        else:
            logger.error("❌ Some tests failed!")
            return 1
    
    elif args.dataset and args.year:
        logger.info(f"Processing {args.dataset.upper()} data for year {args.year}")
        
        processor = ObstorePrecipitationProcessor(
            dataset_name=args.dataset,
            target_year=args.year,
            output_dir=args.output_dir
        )
        
        summary = processor.process_files(max_files=args.max_files)
        
        if 'error' in summary:
            logger.error(f"Processing failed: {summary['error']}")
            return 1
        
        success_rate = float(summary['processing_summary']['success_rate'].rstrip('%'))
        
        if success_rate >= 80:
            logger.info("✓ Processing completed successfully")
            return 0
        else:
            logger.warning(f"⚠ Processing completed with issues (success rate: {success_rate}%)")
            return 1
    
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())