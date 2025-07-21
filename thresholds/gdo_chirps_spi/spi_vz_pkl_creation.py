#!/usr/bin/env python3
"""
SPI VirtualiZarr PKL Creation Script
- Uses xr.open_mfdataset to concatenate individual JSON files 
- Saves virtual datasets as PKL files
- Uses only first 34 JSON files (excluding file035.json)
- Supports all SPI types: spi1, spi3, spi6, spi9, spi12, spi24, spi48
"""

import json
import xarray as xr
import numpy as np
import logging
import time
import pickle
import os
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_spi_concat_json(spi_type: str = "spi1") -> bool:
    """Create the concatenation JSON file with file list for xr.open_mfdataset"""
    
    spi_folder = Path(spi_type)
    if not spi_folder.exists():
        logger.error(f"Folder {spi_folder} does not exist")
        return False
    
    # Get all individual JSON files (not summary or concatenated files)
    json_files = sorted([
        str(f) for f in spi_folder.glob("*.json") 
        if "file" in f.name and "concatenated" not in f.name and "summary" not in f.name and "virtual_concat" not in f.name
    ])
    
    if not json_files:
        logger.error(f"No JSON files found in {spi_folder}")
        return False
    
    logger.info(f"Found {len(json_files)} JSON files for {spi_type}")
    
    # Create the concatenation info structure
    concat_info = {
        f"{spi_type}_virtual_concat": {
            "file_list": json_files,
            "total_files": len(json_files),
            "spi_type": spi_type,
            "description": f"Virtual concatenation file list for {spi_type.upper()} - use with xr.open_mfdataset()"
        }
    }
    
    # Save to the SPI folder
    output_file = spi_folder / f"{spi_type}_virtual_concat.json"
    
    with open(output_file, 'w') as f:
        json.dump(concat_info, f, indent=2)
    
    logger.info(f"✓ Created {output_file}")
    logger.info(f"  Contains {len(json_files)} files for xr.open_mfdataset()")
    
    return True

def create_virtual_spi_dataset(spi_type: str = "spi1", max_files: int = 34) -> bool:
    """
    Create virtual SPI dataset using first 34 files and save as PKL
    """
    
    logger.info(f"Creating virtual {spi_type.upper()} dataset using first {max_files} files...")
    start_time = time.time()
    
    # Check if concatenation JSON exists, create it if not
    json_path = f'{spi_type}/{spi_type}_virtual_concat.json'
    
    if not os.path.exists(json_path):
        logger.info(f"Virtual dataset JSON file not found: {json_path}")
        logger.info(f"Creating virtual concatenation JSON file for {spi_type}...")
        
        if not create_spi_concat_json(spi_type):
            logger.error(f"Failed to create concatenation JSON for {spi_type}")
            return False
        
        logger.info(f"✓ Successfully created concatenation JSON file")
    
    with open(json_path, 'r') as f:
        virtual_info = json.load(f)
    
    # Get file list and limit to first 34 files
    file_list = virtual_info[f'{spi_type}_virtual_concat']['file_list']
    
    # Use only first max_files files
    file_list = file_list[:max_files]
    logger.info(f"Using first {len(file_list)} files out of {len(virtual_info[f'{spi_type}_virtual_concat']['file_list'])} total")
    
    # Print first few and last few files for verification
    logger.info(f"First file: {file_list[0]}")
    logger.info(f"Last file: {file_list[-1]}")
    
    try:
        # Create virtual dataset using xr.open_mfdataset
        logger.info(f"Opening {len(file_list)} files with xr.open_mfdataset...")
        
        ds = xr.open_mfdataset(
            file_list,
            engine='kerchunk',
            concat_dim='time',
            combine='nested',
            chunks={},  # Keep original file chunks initially
            parallel=False,  # More stable than parallel
            decode_times=True,
            data_vars='minimal',  # Only load essential variables
            coords='minimal'      # Only load essential coordinates
        )
        
        logger.info(f"✓ Virtual dataset created successfully")
        logger.info(f"  Shape: {dict(ds.sizes)}")
        logger.info(f"  Variables: {list(ds.data_vars)}")
        logger.info(f"  Time range: {ds.time.min().values} to {ds.time.max().values}")
        
        # Save as PKL file
        output_file = f"{spi_type}_{max_files}files_virtual_dataset.pkl"
        logger.info(f"Saving dataset to {output_file}...")
        
        with open(output_file, 'wb') as f:
            pickle.dump(ds, f)
        
        # Save info file
        info_file = f"{spi_type}_{max_files}files_info.json"
        info_data = {
            'creation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'spi_type': spi_type,
            'total_files': len(file_list),
            'max_files_used': max_files,
            'total_time_steps': ds.sizes.get('time', 0),
            'shape': dict(ds.sizes),
            'time_range': [str(ds.time.min().values), str(ds.time.max().values)],
            'processing_time_minutes': (time.time() - start_time) / 60,
            'variables': list(ds.data_vars),
            'file_list_sample': {
                'first': file_list[0],
                'last': file_list[-1],
                'total': len(file_list)
            }
        }
        
        with open(info_file, 'w') as f:
            json.dump(info_data, f, indent=2)
        
        logger.info(f"✓ Dataset saved successfully!")
        logger.info(f"  PKL file: {output_file}")
        logger.info(f"  Info file: {info_file}")
        logger.info(f"  Total time steps: {ds.sizes.get('time', 0)}")
        logger.info(f"  Processing time: {(time.time() - start_time)/60:.2f} minutes")
        
        # Test that the saved file can be loaded
        logger.info("Testing saved PKL file...")
        with open(output_file, 'rb') as f:
            test_ds = pickle.load(f)
        logger.info(f"✓ PKL file loads correctly, shape: {dict(test_ds.sizes)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to create virtual dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    
    # Get SPI type from command line argument
    spi_type = sys.argv[1] if len(sys.argv) > 1 else "spi1"
    
    # Validate SPI type
    valid_spi_types = ["spi1", "spi3", "spi6", "spi9", "spi12", "spi24", "spi48"]
    if spi_type not in valid_spi_types:
        print(f"Error: Invalid SPI type '{spi_type}'. Valid options: {valid_spi_types}")
        return 1
    
    logger.info("="*70)
    logger.info(f"SPI VIRTUALIZARR PKL CREATION - {spi_type.upper()}")
    logger.info("="*70)
    
    # Create virtual dataset and save as PKL (using first 34 files)
    success = create_virtual_spi_dataset(spi_type, max_files=34)
    
    if success:
        logger.info("="*70)
        logger.info("✓ PKL CREATION COMPLETED SUCCESSFULLY")
        logger.info("="*70)
        return 0
    else:
        logger.error("✗ PKL CREATION FAILED")
        return 1

if __name__ == "__main__":
    exit(main())