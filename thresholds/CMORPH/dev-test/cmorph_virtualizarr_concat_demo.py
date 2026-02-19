#!/usr/bin/env python3
"""
CMORPH VirtualiZarr Concatenation Demo

This script demonstrates:
1. Using obstore to list CMORPH NetCDF files from S3
2. Opening multiple files with VirtualiZarr using open_virtual_dataset
3. Concatenating them into a single virtual dataset with xarray.concat
4. Saving to Icechunk store

Based on the working patterns from working_virtualizarr_processor.py
"""

import logging
import warnings
import json
import os
from pathlib import Path
from typing import List, Optional
from datetime import datetime

import xarray as xr
import icechunk
from obstore.store import S3Store, from_url
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

# CMORPH dataset configuration (from working processor)
CMORPH_CONFIG = {
    'bucket': 's3://noaa-cdr-precip-cmorph-pds/',
    'base_path': 'data/30min/8km/',
    'description': 'CMORPH 30-minute 8km precipitation data'
}


class CMORPHVirtualiZarrConcatenator:
    """CMORPH processor using VirtualiZarr with concatenation"""
    
    def __init__(self, target_year: int, output_dir: str = "."):
        """
        Initialize CMORPH processor
        
        Args:
            target_year: Year to process
            output_dir: Output directory
        """
        self.target_year = target_year
        self.bucket = CMORPH_CONFIG['bucket']
        self.base_path = CMORPH_CONFIG['base_path']
        
        # Setup output directory
        self.output_dir = Path(output_dir)
        self.dataset_output_dir = self.output_dir / f"cmorph_{target_year}"
        self.dataset_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup obstore following working pattern
        self.store = from_url(self.bucket, region="us-east-1", skip_signature=True)
        self.registry = ObjectStoreRegistry({self.bucket: self.store})
        self.parser = HDFParser()
        
        logger.info(f"Initialized CMORPH processor for year {target_year}")
        logger.info(f"Output directory: {self.dataset_output_dir}")
        logger.info(f"S3 bucket: {self.bucket}")
    
    def list_cmorph_files(self, max_files: Optional[int] = None) -> List[str]:
        """
        List CMORPH NetCDF files using obstore (following working pattern)
        
        Args:
            max_files: Maximum number of files to list
            
        Returns:
            List of file paths relative to bucket
        """
        logger.info(f"Listing CMORPH files for {self.target_year}")
        
        # Create S3Store for listing (following working pattern)
        bucket_name = self.bucket.replace('s3://', '').rstrip('/')
        year_prefix = f"{self.base_path}{self.target_year}/"
        
        list_store = S3Store(
            bucket_name=bucket_name,
            prefix=year_prefix,
            region="us-east-1",
            skip_signature=True
        )
        
        try:
            # Collect all files by iterating through all results
            nc_files = []
            
            # Get the list iterator
            list_iterator = list_store.list()
            
            # Iterate through all pages of results
            for page in list_iterator:
                if isinstance(page, (list, tuple)):
                    for obj_info in page:
                        if isinstance(obj_info, dict) and 'path' in obj_info:
                            if obj_info['path'].endswith('.nc'):
                                # Return full path relative to bucket
                                full_path = f"{year_prefix}{obj_info['path']}"
                                nc_files.append(full_path)
                elif isinstance(page, dict) and 'path' in page:
                    if page['path'].endswith('.nc'):
                        # Return full path relative to bucket
                        full_path = f"{year_prefix}{page['path']}"
                        nc_files.append(full_path)
            
            nc_files.sort()
            logger.info(f"Found {len(nc_files)} total NetCDF files")
            
            # Limit files if specified
            if max_files and len(nc_files) > max_files:
                nc_files = nc_files[:max_files]
                logger.info(f"Limited to first {max_files} files")
            
            return nc_files
            
        except Exception as e:
            logger.error(f"Error listing files: {e}")
            # Fallback to original method if the iterator approach fails
            try:
                logger.info("Trying fallback listing method...")
                list_result = list(list_store.list())
                
                if not list_result:
                    logger.warning(f"No objects found in {year_prefix}")
                    return []
                
                # Extract file paths from nested structure
                all_objects = list_result[0] if isinstance(list_result[0], list) else list_result
                nc_files = []
                
                for obj_info in all_objects:
                    if isinstance(obj_info, dict) and 'path' in obj_info:
                        if obj_info['path'].endswith('.nc'):
                            full_path = f"{year_prefix}{obj_info['path']}"
                            nc_files.append(full_path)
                
                nc_files.sort()
                logger.info(f"Fallback found {len(nc_files)} NetCDF files")
                
                # Limit files if specified
                if max_files and len(nc_files) > max_files:
                    nc_files = nc_files[:max_files]
                    logger.info(f"Limited to first {max_files} files")
                
                return nc_files
                
            except Exception as e2:
                logger.error(f"Fallback listing also failed: {e2}")
                return []
    
    def create_virtual_datasets(self, file_paths: List[str]) -> List[xr.Dataset]:
        """
        Create virtual datasets from file paths using VirtualiZarr
        
        Args:
            file_paths: List of file paths relative to bucket
            
        Returns:
            List of virtual datasets
        """
        logger.info(f"Creating {len(file_paths)} virtual datasets...")
        
        virtual_datasets = []
        
        for i, file_path in enumerate(file_paths, 1):
            try:
                # Construct full URL (following working pattern)
                url = f"{self.bucket}{file_path}"
                
                logger.debug(f"Processing {i}/{len(file_paths)}: {Path(file_path).name}")
                
                # Create virtual dataset (following working pattern)
                vds = open_virtual_dataset(
                    url=url,
                    parser=self.parser,
                    registry=self.registry
                )
                
                virtual_datasets.append(vds)
                logger.debug(f"✓ Created virtual dataset {i}")
                
            except Exception as e:
                logger.error(f"✗ Failed to create virtual dataset from {file_path}: {e}")
                continue
        
        logger.info(f"Successfully created {len(virtual_datasets)} virtual datasets")
        return virtual_datasets
    
    def concatenate_virtual_datasets(self, virtual_datasets: List[xr.Dataset]) -> xr.Dataset:
        """
        Concatenate virtual datasets along time dimension
        
        Args:
            virtual_datasets: List of virtual datasets
            
        Returns:
            Concatenated virtual dataset
        """
        logger.info(f"Concatenating {len(virtual_datasets)} virtual datasets...")
        
        try:
            # Concatenate using xarray (following requested pattern)
            virtual_ds = xr.concat(
                virtual_datasets,
                dim='time',
                coords='minimal',
                compat='override',
                combine_attrs='override'
            )
            
            logger.info(f"✓ Concatenated dataset created")
            logger.info(f"  Shape: {dict(virtual_ds.dims)}")
            logger.info(f"  Data variables: {list(virtual_ds.data_vars.keys())}")
            logger.info(f"  Time range: {len(virtual_datasets)} time steps")
            
            return virtual_ds
            
        except Exception as e:
            logger.error(f"Error concatenating datasets: {e}")
            raise
    
    def save_to_icechunk_local(self, virtual_ds: xr.Dataset, repo_name: str = "cmorph_concat") -> bool:
        """
        Save concatenated virtual dataset to local Icechunk store
        
        Args:
            virtual_ds: Concatenated virtual dataset
            repo_name: Repository name
            
        Returns:
            Success status
        """
        logger.info("Saving to local Icechunk store...")
        
        try:
            # Setup local Icechunk storage
            icechunk_path = self.dataset_output_dir / repo_name
            storage = icechunk.local_filesystem_storage(
                path=str(icechunk_path),
            )
            
            # Repository configuration
            config = icechunk.RepositoryConfig.default()
            
            # Configure virtual chunk container pointing to CMORPH S3 bucket
            config.set_virtual_chunk_container(
                icechunk.VirtualChunkContainer(
                    self.bucket,
                    icechunk.s3_store(region="us-east-1")
                )
            )
            
            # Set up credentials for S3 access
            credentials = icechunk.containers_credentials({
                self.bucket: icechunk.s3_credentials(anonymous=True)
            })
            
            # Create repository
            try:
                repo = icechunk.Repository.create(storage, config, credentials)
                logger.info("✓ Created new Icechunk repository")
            except Exception as e:
                logger.info(f"Repository exists, trying to open: {e}")
                repo = icechunk.Repository.open(storage, config, credentials)
                logger.info("✓ Opened existing Icechunk repository")
            
            # Create writable session
            session = repo.writable_session("main")
            
            # Write virtual dataset
            logger.info("Writing virtual dataset to Icechunk...")
            virtual_ds.to_zarr(
                store=session,
                mode='w',
                consolidated=True
            )
            
            # Commit
            snapshot_id = session.commit(f"CMORPH {self.target_year} concatenated data - {datetime.now().isoformat()}")
            logger.info(f"✓ Committed with snapshot ID: {snapshot_id}")
            
            # Verify by reading back
            logger.info("Verifying written data...")
            readonly_session = repo.readonly_session("main")
            reopened_ds = xr.open_zarr(readonly_session)
            
            logger.info(f"✓ Verification successful")
            logger.info(f"  Reopened dataset shape: {dict(reopened_ds.dims)}")
            logger.info(f"  Variables: {list(reopened_ds.data_vars)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving to Icechunk: {e}")
            return False
    
    def save_to_icechunk_gcs(self, virtual_ds: xr.Dataset, gcs_bucket: str, gcs_path: str, 
                             service_account_json: str, repo_name: str = "cmorph_concat") -> bool:
        """
        Save concatenated virtual dataset to GCS Icechunk store
        
        Args:
            virtual_ds: Concatenated virtual dataset
            gcs_bucket: GCS bucket name
            gcs_path: Path within bucket
            service_account_json: Path to service account JSON
            repo_name: Repository name
            
        Returns:
            Success status
        """
        logger.info("Saving to GCS Icechunk store...")
        
        try:
            # Setup GCS storage
            storage = icechunk.gcs_storage(
                bucket=gcs_bucket,
                prefix=gcs_path,
                service_account_json_path=service_account_json
            )
            
            # Repository configuration
            config = icechunk.RepositoryConfig.default()
            
            # Configure virtual chunk container pointing to CMORPH S3 bucket
            config.set_virtual_chunk_container(
                icechunk.VirtualChunkContainer(
                    self.bucket,
                    icechunk.s3_store(region="us-east-1")
                )
            )
            
            # Set up credentials for S3 access
            credentials = icechunk.containers_credentials({
                self.bucket: icechunk.s3_credentials(anonymous=True)
            })
            
            # Create repository
            try:
                repo = icechunk.Repository.create(storage, config, credentials)
                logger.info("✓ Created new GCS Icechunk repository")
            except Exception as e:
                logger.info(f"Repository exists, trying to open: {e}")
                repo = icechunk.Repository.open(storage, config, credentials)
                logger.info("✓ Opened existing GCS Icechunk repository")
            
            # Create writable session
            session = repo.writable_session("main")
            
            # Write virtual dataset
            logger.info("Writing virtual dataset to GCS Icechunk...")
            virtual_ds.to_zarr(
                store=session,
                mode='w',
                consolidated=True
            )
            
            # Commit
            snapshot_id = session.commit(f"CMORPH {self.target_year} concatenated data - {datetime.now().isoformat()}")
            logger.info(f"✓ Committed to GCS with snapshot ID: {snapshot_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving to GCS Icechunk: {e}")
            return False
    
    def process_and_concatenate(self, max_files: Optional[int] = None, 
                               save_to_icechunk: bool = True,
                               gcs_config: Optional[dict] = None) -> dict:
        """
        Main processing pipeline: list files, create virtual datasets, concatenate, and save
        
        Args:
            max_files: Maximum number of files to process
            save_to_icechunk: Whether to save to Icechunk
            gcs_config: GCS configuration dict with 'bucket', 'path', 'service_account_json'
            
        Returns:
            Processing summary
        """
        start_time = datetime.now()
        
        try:
            # Step 1: List files
            file_paths = self.list_cmorph_files(max_files=max_files)
            
            if not file_paths:
                return {'error': 'No files found'}
            
            # Step 2: Create virtual datasets  
            virtual_datasets = self.create_virtual_datasets(file_paths)
            
            if not virtual_datasets:
                return {'error': 'No virtual datasets created'}
            
            # Step 3: Concatenate
            concatenated_ds = self.concatenate_virtual_datasets(virtual_datasets)
            
            # Step 4: Save to Icechunk (optional)
            icechunk_success = False
            if save_to_icechunk:
                if gcs_config:
                    icechunk_success = self.save_to_icechunk_gcs(
                        concatenated_ds, 
                        gcs_config['bucket'], 
                        gcs_config['path'], 
                        gcs_config['service_account_json']
                    )
                else:
                    icechunk_success = self.save_to_icechunk_local(concatenated_ds)
            
            # Create summary
            processing_time = (datetime.now() - start_time).total_seconds()
            
            summary = {
                'dataset_info': {
                    'name': 'CMORPH',
                    'year': self.target_year,
                    'description': CMORPH_CONFIG['description']
                },
                'processing_summary': {
                    'files_found': len(file_paths),
                    'virtual_datasets_created': len(virtual_datasets),
                    'concatenation_success': True,
                    'icechunk_save_success': icechunk_success,
                    'processing_time_seconds': processing_time,
                    'timestamp': datetime.now().isoformat()
                },
                'dataset_info_result': {
                    'dimensions': dict(concatenated_ds.dims),
                    'data_variables': list(concatenated_ds.data_vars.keys()),
                    'coordinates': list(concatenated_ds.coords.keys())
                },
                'file_paths': file_paths
            }
            
            # Save summary
            summary_path = self.dataset_output_dir / f"cmorph_{self.target_year}_concat_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Processing completed in {processing_time:.2f} seconds")
            logger.info(f"Summary saved: {summary_path}")
            
            return summary
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return {'error': str(e)}


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="CMORPH VirtualiZarr concatenation demo"
    )
    
    parser.add_argument(
        '--year',
        type=int,
        default=1998,
        help='Year to process (default: 1998)'
    )
    
    parser.add_argument(
        '--max-files',
        type=int,
        default=5,
        help='Maximum number of files to process (default: 5)'
    )
    
    parser.add_argument(
        '--output-dir',
        default='.',
        help='Output directory (default: current)'
    )
    
    parser.add_argument(
        '--no-icechunk',
        action='store_true',
        help='Skip Icechunk save step'
    )
    
    parser.add_argument(
        '--gcs-bucket',
        help='GCS bucket for Icechunk storage'
    )
    
    parser.add_argument(
        '--gcs-path',
        default='icechunk-stores/cmorph-concat',
        help='GCS path for Icechunk storage'
    )
    
    parser.add_argument(
        '--service-account-json',
        help='Path to GCS service account JSON file'
    )
    
    args = parser.parse_args()
    
    logger.info(f"CMORPH VirtualiZarr Concatenation Demo")
    logger.info(f"Year: {args.year}, Max files: {args.max_files}")
    
    # Initialize processor
    processor = CMORPHVirtualiZarrConcatenator(
        target_year=args.year,
        output_dir=args.output_dir
    )
    
    # Setup GCS config if provided
    gcs_config = None
    if args.gcs_bucket and args.service_account_json:
        gcs_config = {
            'bucket': args.gcs_bucket,
            'path': args.gcs_path,
            'service_account_json': args.service_account_json
        }
        logger.info(f"Will save to GCS: gs://{args.gcs_bucket}/{args.gcs_path}")
    
    # Process and concatenate
    summary = processor.process_and_concatenate(
        max_files=args.max_files,
        save_to_icechunk=not args.no_icechunk,
        gcs_config=gcs_config
    )
    
    # Check results
    if 'error' in summary:
        logger.error(f"Processing failed: {summary['error']}")
        return 1
    
    processing_summary = summary.get('processing_summary', {})
    files_found = processing_summary.get('files_found', 0)
    datasets_created = processing_summary.get('virtual_datasets_created', 0)
    icechunk_success = processing_summary.get('icechunk_save_success', False)
    
    logger.info(f"Results:")
    logger.info(f"  Files found: {files_found}")
    logger.info(f"  Virtual datasets created: {datasets_created}")
    logger.info(f"  Concatenation: ✓")
    logger.info(f"  Icechunk save: {'✓' if icechunk_success else '✗' if not args.no_icechunk else 'Skipped'}")
    
    if datasets_created > 0:
        logger.info("🎉 Demo completed successfully!")
        return 0
    else:
        logger.error("❌ Demo failed!")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())