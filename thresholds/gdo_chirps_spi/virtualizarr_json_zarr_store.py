#!/usr/bin/env python3
"""
Modular SPI processor: subset to East Africa and store as Zarr in GCS
Supports SPI1, SPI3, SPI6, SPI9, SPI12, SPI24, SPI48
Direct Zarr storage without Icechunk
"""

import os
import xarray as xr
import zarr
import gcsfs
import warnings
import time
import glob
import gc
import logging
import argparse
from pathlib import Path
from typing import Dict, List

warnings.filterwarnings('ignore')

# Configuration class for better organization
class SPIConfig:
    def __init__(self, spi_type: str, prefix: str, bucket: str = "cdi_arco", 
                 service_account: str = "coiled-data-e4drr_202505.json"):
        self.spi_type = spi_type.lower()
        self.prefix = prefix
        self.bucket = bucket
        self.service_account = service_account
        self.region_bounds = {"lat_min": -12, "lat_max": 23, "lon_min": 21, "lon_max": 53}
        
        # Validate SPI type
        valid_types = ['spi1', 'spi3', 'spi6', 'spi9', 'spi12', 'spi24', 'spi48']
        if self.spi_type not in valid_types:
            raise ValueError(f"Invalid SPI type: {spi_type}. Must be one of {valid_types}")
    
    @property
    def data_directory(self) -> str:
        """Get the data directory based on SPI type"""
        return f"{self.spi_type}/"
    
    @property
    def file_pattern(self) -> str:
        """Get the file pattern based on SPI type"""
        return f"{self.spi_type}/{self.spi_type}_file*.json"
    
    @property
    def zarr_path(self) -> str:
        """Get the full GCS Zarr path"""
        return f"gs://{self.bucket}/{self.prefix}.zarr"
    
    @property
    def zarr_group(self) -> str:
        """Get the Zarr group name based on SPI type"""
        return f"{self.spi_type}_data"

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SPIProcessor:
    """Main SPI processing class"""
    
    def __init__(self, config: SPIConfig):
        self.config = config
        self.fs = None
        
    def setup_gcs_filesystem(self):
        """Setup GCS filesystem with authentication"""
        try:
            self.fs = gcsfs.GCSFileSystem(token=self.config.service_account)
            logger.info("✅ GCS filesystem initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize GCS filesystem: {e}")
            raise

    def zarr_store_exists(self) -> bool:
        """Check if Zarr store already exists in GCS"""
        try:
            zarr_path = self.config.zarr_path
            return self.fs.exists(zarr_path)
        except Exception:
            return False

    def backup_zarr_metadata(self, zarr_path: str) -> str:
        """Create backup of Zarr metadata before append operation"""
        try:
            backup_path = f"{zarr_path}_backup_{int(time.time())}"
            
            # Copy .zmetadata file if it exists
            metadata_path = f"{zarr_path}/.zmetadata"
            backup_metadata = f"{backup_path}/.zmetadata"
            
            if self.fs.exists(metadata_path):
                self.fs.copy(metadata_path, backup_metadata)
                logger.info(f"Backed up metadata to {backup_metadata}")
                return backup_path
            
        except Exception as e:
            logger.warning(f"Failed to backup metadata: {e}")
        
        return None

    def restore_zarr_metadata(self, zarr_path: str, backup_path: str):
        """Restore Zarr metadata from backup"""
        try:
            if backup_path and self.fs.exists(f"{backup_path}/.zmetadata"):
                self.fs.copy(f"{backup_path}/.zmetadata", f"{zarr_path}/.zmetadata")
                logger.info(f"Restored metadata from backup")
        except Exception as e:
            logger.error(f"Failed to restore metadata: {e}")

    def cleanup_backup(self, backup_path: str):
        """Clean up backup files"""
        try:
            if backup_path and self.fs.exists(backup_path):
                self.fs.rm(backup_path, recursive=True)
                logger.info(f"Cleaned up backup {backup_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup backup: {e}")

    def process_and_append_file(self, file_path: str, is_first: bool = False) -> Dict[str, any]:
        """Process single SPI file and append to Zarr store with safety measures"""
        
        backup_path = None
        
        try:
            logger.info(f"Processing {file_path}")
            
            # Load dataset
            ds = xr.open_dataset(file_path)
            logger.info(f"Loaded {file_path}, shape: {dict(ds.sizes)}")
            
            # Handle band dimension if it exists (squeeze it out if single band)
            if 'band' in ds.dims and ds.sizes['band'] == 1:
                ds = ds.squeeze('band', drop=True)
                logger.info(f"Squeezed band dimension, new shape: {dict(ds.sizes)}")
            
            # Subset to East Africa region
            ds_subset = ds.where(
                (ds.lat >= self.config.region_bounds["lat_min"]) & 
                (ds.lat <= self.config.region_bounds["lat_max"]) &
                (ds.lon >= self.config.region_bounds["lon_min"]) & 
                (ds.lon <= self.config.region_bounds["lon_max"]),
                drop=True
            ).compute()
            
            logger.info(f"Subset computed for {file_path}, shape: {dict(ds_subset.sizes)}")
            
            # Clear original dataset to free memory
            del ds
            gc.collect()
            
            # Setup GCS Zarr store
            zarr_path = self.config.zarr_path
            
            # Try different approaches for GCS Zarr storage
            try:
                # Method 1: Direct zarr path with gcsfs
                import fsspec
                
                # Create filesystem options
                fs_options = {'token': self.config.service_account}
                
                # Write or append data
                if is_first:
                    logger.info(f"Writing first file {file_path} to {zarr_path}")
                    ds_subset.to_zarr(
                        store=zarr_path,
                        group=self.config.zarr_group,
                        mode='w',
                        consolidated=True,
                        storage_options=fs_options
                    )
                else:
                    logger.info(f"Appending {file_path} to {zarr_path}")
                    ds_subset.to_zarr(
                        store=zarr_path,
                        group=self.config.zarr_group,
                        append_dim='time',
                        consolidated=True,
                        storage_options=fs_options
                    )
                    
            except Exception as direct_error:
                logger.warning(f"Direct method failed: {direct_error}")
                logger.info("Trying FSMap method...")
                
                # Method 2: FSMap approach (fallback)
                mapper = self.fs.get_mapper(zarr_path)
                
                if is_first:
                    logger.info(f"Writing first file {file_path} to {zarr_path} (FSMap)")
                    ds_subset.to_zarr(
                        store=mapper,
                        group=self.config.zarr_group,
                        mode='w',
                        consolidated=True
                    )
                else:
                    logger.info(f"Appending {file_path} to {zarr_path} (FSMap)")
                    ds_subset.to_zarr(
                        store=mapper,
                        group=self.config.zarr_group,
                        append_dim='time',
                        consolidated=True
                    )
            
            logger.info(f"Successfully stored {file_path} in Zarr")
            
            # Clean up successful backup
            if backup_path:
                self.cleanup_backup(backup_path)
            
            # Cleanup
            del ds_subset
            gc.collect()
            
            return {'status': 'success', 'file_name': Path(file_path).name}
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            
            # If we have a backup and this wasn't the first file, try to restore
            if backup_path and not is_first:
                logger.info("Attempting to restore from backup after error...")
                self.restore_zarr_metadata(self.config.zarr_path, backup_path)
                self.cleanup_backup(backup_path)
            
            # Cleanup on error
            for var_name in ['ds', 'ds_subset']:
                try:
                    if var_name in locals():
                        del locals()[var_name]
                except:
                    pass
            gc.collect()
            
            return {'status': 'failed', 'file_name': Path(file_path).name, 'error': str(e)}

    def find_files(self) -> List[str]:
        """Find all files matching the SPI pattern"""
        files = sorted(glob.glob(self.config.file_pattern))
        return files

    def validate_environment(self) -> bool:
        """Validate that required files and directories exist"""
        # Check service account file
        if not os.path.exists(self.config.service_account):
            print(f"❌ Service account file not found: {self.config.service_account}")
            return False
        
        # Check data directory
        if not os.path.exists(self.config.data_directory):
            print(f"❌ Data directory not found: {self.config.data_directory}")
            return False
        
        # Check for data files
        files = self.find_files()
        if not files:
            print(f"❌ No {self.config.spi_type.upper()} files found in '{self.config.data_directory}' directory")
            print(f"Expected files matching pattern: {self.config.file_pattern}")
            return False
        
        return True

    def process_all_files(self) -> Dict[str, int]:
        """Process all files for the given SPI type"""
        print("=" * 80)
        print(f"{self.config.spi_type.upper()} East Africa Processing - Direct Zarr Version")
        print(f"Zarr Path: {self.config.zarr_path}")
        print(f"Group: {self.config.zarr_group}")
        print("=" * 80)
        
        # Validate environment
        if not self.validate_environment():
            return {'successful': 0, 'total': 0, 'failed': 0}
        
        # Setup GCS filesystem
        try:
            self.setup_gcs_filesystem()
        except Exception as e:
            print(f"❌ Failed to setup GCS: {e}")
            return {'successful': 0, 'total': 0, 'failed': 0}
        
        print("Setting up Zarr store...")
        
        # Check if store exists
        store_exists = self.zarr_store_exists()
        if store_exists:
            print("✅ Found existing Zarr store")
        else:
            print("📦 Creating new Zarr store")
        
        # Find all files
        files = self.find_files()
        print(f"Found {len(files)} {self.config.spi_type.upper()} files to process")
        
        # Process files sequentially
        successful = 0
        total_files = len(files)
        
        try:
            # Process first file separately to initialize the store
            first_file = files[0]
            print(f"\nInitializing store with first file: {Path(first_file).name}")
            
            first_result = self.process_and_append_file(first_file, is_first=not store_exists)
            
            if first_result['status'] == 'success':
                print(f"✅ {first_result['file_name']}: stored successfully")
                successful += 1
            else:
                print(f"❌ {first_result['file_name']}: {first_result.get('error', 'Unknown error')}")
                print("Cannot continue without storing the first file.")
                return {'successful': 0, 'total': total_files, 'failed': 1}
            
            # Process remaining files
            if total_files > 1:
                print(f"\nProcessing remaining {total_files - 1} files...")
                
                for i, file_path in enumerate(files[1:], 1):
                    print(f"\nProgress: {i}/{total_files - 1} - Processing {Path(file_path).name}")
                    
                    result = self.process_and_append_file(file_path, is_first=False)
                    
                    if result['status'] == 'success':
                        print(f"  ✅ {result['file_name']}")
                        successful += 1
                    else:
                        print(f"  ❌ {result['file_name']}: {result.get('error', 'Unknown error')}")
                    
                    # Progress update every 10 files
                    if i % 10 == 0:
                        print(f"\n📊 Progress Update: {successful}/{i + 1} files successful so far")
                        
                    # Small delay to prevent overwhelming the system
                    time.sleep(1)
            
            print("\n" + "=" * 80)
            print(f"🎉 {self.config.spi_type.upper()} Processing Complete!")
            print(f"📦 Zarr Store: {self.config.zarr_path}")
            print(f"✅ Successfully processed: {successful}/{total_files} files")
            print(f"❌ Failed: {total_files - successful}/{total_files} files")
            
            if successful == total_files:
                print("🌟 All files processed successfully!")
            elif successful > 0:
                print(f"⚠️  Partial success - {successful} files were processed")
            else:
                print("💥 No files were processed successfully")
                
            print("=" * 80)
            
            return {'successful': successful, 'total': total_files, 'failed': total_files - successful}
            
        except KeyboardInterrupt:
            print(f"\n⚠️ Processing interrupted by user")
            print(f"📊 Progress at interruption: {successful}/{total_files} files successful")
            return {'successful': successful, 'total': total_files, 'failed': total_files - successful}
            
        except Exception as e:
            print(f"\n❌ Unexpected error in main process: {e}")
            import traceback
            traceback.print_exc()
            print(f"📊 Progress at error: {successful}/{total_files} files successful")
            return {'successful': successful, 'total': total_files, 'failed': total_files - successful}

def create_parser():
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Process SPI data files and store as Zarr in GCS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python spi_processor.py spi1 my_spi1_store
  python spi_processor.py spi1,spi3,spi6 analysis_data
  python spi_processor.py spi12 drought_monitoring
  python spi_processor.py spi1,spi24,spi48 long_term_analysis --bucket my_bucket
        """
    )
    
    parser.add_argument('spi_types', 
                       help='SPI type(s) to process. Single: spi1 or Multiple: spi1,spi3,spi6')
    
    parser.add_argument('prefix', 
                       help='Base prefix for the Zarr stores')
    
    parser.add_argument('--bucket', 
                       default='cdi_arco',
                       help='GCS bucket name (default: cdi_arco)')
    
    parser.add_argument('--service-account', 
                       default='coiled-data-e4drr_202505.json',
                       help='Service account JSON file (default: coiled-data-e4drr_202505.json)')
    
    return parser

def parse_spi_types(spi_types_str: str) -> List[str]:
    """Parse comma-separated SPI types and validate them"""
    valid_types = ['spi1', 'spi3', 'spi6', 'spi9', 'spi12', 'spi24', 'spi48']
    
    # Split by comma and clean up
    spi_types = [spi_type.strip().lower() for spi_type in spi_types_str.split(',')]
    
    # Validate each type
    invalid_types = [spi_type for spi_type in spi_types if spi_type not in valid_types]
    if invalid_types:
        raise ValueError(f"Invalid SPI type(s): {invalid_types}. Must be from {valid_types}")
    
    # Remove duplicates while preserving order
    unique_types = []
    for spi_type in spi_types:
        if spi_type not in unique_types:
            unique_types.append(spi_type)
    
    return unique_types

def main():
    """
    Main function with command line interface
    To open the uploaded zarr, the group has to be mentioned 
    credentials_path = 'coiled-data-e4drr_202505.json'
    scopes = ["https://www.googleapis.com/auth/devstorage.read_write"]
    credentials = service_account.Credentials.from_service_account_file(
           credentials_path, scopes=scopes
    )
    storage_options = {'token': credentials}
    zarr_path = 'gs://cdi_arco/spi3_spi3.zarr'

    # Fix: Add the group parameter
    ds = xr.open_dataset(
         zarr_path,
         engine='zarr',
         group='spi3_data',  # ← This is the key fix!
         consolidated=False,
         storage_options=storage_options)
    ds 
    
    """
    parser = create_parser()
    args = parser.parse_args()
    
    try:
        # Parse SPI types
        spi_types = parse_spi_types(args.spi_types)
        
        print(f"🎯 Processing {len(spi_types)} SPI type(s): {', '.join([s.upper() for s in spi_types])}")
        print(f"📦 Base prefix: {args.prefix}")
        print(f"☁️  Bucket: {args.bucket}")
        print("=" * 80)
        
        overall_results = {'total_successful': 0, 'total_files': 0, 'total_failed': 0}
        
        # Process each SPI type
        for i, spi_type in enumerate(spi_types, 1):
            print(f"\n🔄 [{i}/{len(spi_types)}] Starting {spi_type.upper()} processing...")
            
            # Create configuration with type-specific prefix
            spi_prefix = f"{args.prefix}_{spi_type}"
            config = SPIConfig(
                spi_type=spi_type,
                prefix=spi_prefix,
                bucket=args.bucket,
                service_account=args.service_account
            )
            
            # Create processor and run
            processor = SPIProcessor(config)
            results = processor.process_all_files()
            
            # Update overall results
            overall_results['total_successful'] += results['successful']
            overall_results['total_files'] += results['total']
            overall_results['total_failed'] += results['failed']
            
            print(f"✅ {spi_type.upper()} completed: {results['successful']}/{results['total']} files successful")
            print(f"📦 Stored at: gs://{args.bucket}/{spi_prefix}.zarr")
        
        # Final summary
        print("\n" + "🎉" * 80)
        print("OVERALL PROCESSING SUMMARY")
        print("🎉" * 80)
        print(f"📊 Total SPI types processed: {len(spi_types)}")
        print(f"📁 Total files processed: {overall_results['total_files']}")
        print(f"✅ Total successful: {overall_results['total_successful']}")
        print(f"❌ Total failed: {overall_results['total_failed']}")
        
        success_rate = (overall_results['total_successful'] / overall_results['total_files'] * 100) if overall_results['total_files'] > 0 else 0
        print(f"📈 Success rate: {success_rate:.1f}%")
        
        print(f"\n📦 Created Zarr stores:")
        for spi_type in spi_types:
            print(f"  - gs://{args.bucket}/{args.prefix}_{spi_type}.zarr")
        
        if overall_results['total_failed'] == 0 and overall_results['total_files'] > 0:
            print("\n🌟 All files across all SPI types processed successfully!")
            exit(0)
        elif overall_results['total_successful'] > 0:
            print("\n⚠️  Some files failed - check logs above for details")
            exit(1)
        else:
            print("\n💥 No files were processed successfully")
            exit(2)
            
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        exit(3)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        exit(4)

if __name__ == "__main__":
    main()
