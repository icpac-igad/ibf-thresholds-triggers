#!/usr/bin/env python3
"""
Script to examine the structure of NetCDF SPI files and determine optimal chunking
for conversion to Zarr format.
"""

import os
import glob
import xarray as xr
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def examine_nc_files(directory, pattern="SPI*.nc"):
    """
    Examines NetCDF files in the given directory to understand their structure.
    
    Args:
        directory: Directory containing NetCDF files
        pattern: Glob pattern to match files
        
    Returns:
        Dictionary with information about the files
    """
    logger.info(f"Examining NetCDF files in {directory} with pattern {pattern}")
    
    # Get a list of all NetCDF files matching the pattern
    file_paths = glob.glob(os.path.join(directory, pattern))
    logger.info(f"Found {len(file_paths)} files")
    
    if not file_paths:
        logger.error(f"No files found in {directory} matching pattern {pattern}")
        return None
    
    # Group files by SPI type (SPI1, SPI3, etc.)
    spi_groups = {}
    for file_path in file_paths:
        # Extract SPI type from filename
        filename = os.path.basename(file_path)
        spi_type = filename.split('_')[0]  # Assumes naming like "SPI48_gamma_..."
        
        if spi_type not in spi_groups:
            spi_groups[spi_type] = []
        
        spi_groups[spi_type].append(file_path)
    
    # Print info about each SPI group
    for spi_type, files in spi_groups.items():
        logger.info(f"{spi_type}: {len(files)} files")
    
    # Examine the first file from each group
    structure_info = {}
    
    for spi_type, files in spi_groups.items():
        sample_file = files[0]
        logger.info(f"Opening sample file for {spi_type}: {os.path.basename(sample_file)}")
        
        try:
            ds = xr.open_dataset(sample_file)
            
            # Get basic dataset info
            info = {
                'dimensions': {dim: len(ds[dim]) for dim in ds.dims},
                'variables': list(ds.data_vars),
                'coords': list(ds.coords),
                'attrs': ds.attrs,
                'encoding': {var: ds[var].encoding for var in ds.variables},
                'dtypes': {var: str(ds[var].dtype) for var in ds.variables},
                'chunks': getattr(ds, 'chunks', None),
                'file_size_mb': os.path.getsize(sample_file) / (1024 * 1024),
                'file_count': len(files),
                'total_size_mb': len(files) * os.path.getsize(sample_file) / (1024 * 1024)
            }
            
            structure_info[spi_type] = info
            
            # Print key information
            logger.info(f"  Dimensions: {info['dimensions']}")
            logger.info(f"  Variables: {info['variables']}")
            logger.info(f"  File size: {info['file_size_mb']:.2f} MB")
            logger.info(f"  Total size for {spi_type}: {info['total_size_mb']:.2f} MB")
            
            # Close the dataset
            ds.close()
            
        except Exception as e:
            logger.error(f"Error examining {sample_file}: {str(e)}")
    
    return structure_info

def suggest_chunks(structure_info):
    """
    Suggest optimal chunking for Zarr conversion based on the structure of the NetCDF files.
    
    Args:
        structure_info: Dictionary with information about the NetCDF files
        
    Returns:
        Dictionary with suggested chunking for each SPI type
    """
    logger.info("Suggesting optimal chunking strategy for Zarr conversion")
    
    chunk_suggestions = {}
    
    for spi_type, info in structure_info.items():
        dimensions = info['dimensions']
        
        # Default chunking strategy
        chunks = {}
        
        # Time dimension chunking
        if 'time' in dimensions:
            # For time dimension, chunk by year (12 months) if possible
            time_len = dimensions['time']
            if time_len >= 12:
                chunks['time'] = min(12, time_len)
            else:
                chunks['time'] = time_len
        
        # Spatial dimension chunking
        # For lat/lon dimensions, we want to balance between:
        # - Chunks that are too small (inefficient I/O)
        # - Chunks that are too large (memory issues)
        
        # Latitude chunking
        if 'latitude' in dimensions:
            lat_len = dimensions['latitude']
            # Aim for chunks of around 5-10 degrees if possible
            chunks['latitude'] = min(20, lat_len)
        
        # Longitude chunking
        if 'longitude' in dimensions:
            lon_len = dimensions['longitude']
            # Aim for chunks of around 5-10 degrees if possible
            chunks['longitude'] = min(20, lon_len)
        
        # For other dimensions, don't chunk (use full dimension)
        for dim in dimensions:
            if dim not in chunks:
                chunks[dim] = dimensions[dim]
        
        chunk_suggestions[spi_type] = chunks
        logger.info(f"Suggested chunks for {spi_type}: {chunks}")
    
    return chunk_suggestions

if __name__ == "__main__":
    directory = "ecmwf_spi"
    structure_info = examine_nc_files(directory)
    
    if structure_info:
        chunk_suggestions = suggest_chunks(structure_info)
        
        # Print summary
        logger.info("\nSummary of SPI Files:")
        for spi_type, info in structure_info.items():
            logger.info(f"{spi_type}: {info['file_count']} files, {info['total_size_mb']:.2f} MB total")
            logger.info(f"  Dimensions: {info['dimensions']}")
            logger.info(f"  Suggested chunks: {chunk_suggestions[spi_type]}")
    else:
        logger.error("No structure information available to suggest chunks.")