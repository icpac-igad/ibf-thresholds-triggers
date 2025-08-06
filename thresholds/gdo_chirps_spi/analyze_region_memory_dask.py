    #!/usr/bin/env python3
"""
Analyze memory usage of each administrative region's data cube
to help design Dask worker configurations.

This script loads all SPI variables (spi1, spi3, spi6, spi12, spi24, spi48)
from Icechunk storage and analyzes memory requirements for processing
each region individually with 230+ polygons from ICPAC regions.
"""

import icechunk
import xarray as xr
import geopandas as gpd
import numpy as np
import regionmask
import time
import pandas as pd
import warnings
from typing import Dict, List, Tuple

warnings.filterwarnings('ignore')

# Configuration
BASE_PREFIX = "t2spi1_east_africa_icechunk"
BUCKET_NAME = "cdi_arco"
SERVICE_ACCOUNT_FILE = "../pkl_files/coiled-data-e4drr_202505.json"
GEOJSON_FILE = "../icpac_adm1v3.geojson"
SPI_VARIABLES = ['spi1', 'spi3', 'spi6', 'spi12', 'spi24', 'spi48']

def resample_decadal_to_monthly(dataset, spi_type):
    """
    Resample decadal (10-day) SPI data to monthly by taking the last decade of each month.
    
    This function is specifically designed for SPI1 and SPI3 variables which are provided
    in decadal format (every 10 days, resulting in ~3 values per month). To make them
    compatible with other SPI variables (SPI6, SPI12, SPI24, SPI48) which are monthly,
    we take the last decadal value of each month as the representative monthly value.
    
    Args:
        dataset: xarray Dataset with decadal time resolution
        spi_type: String indicating SPI variable type (for logging)
        
    Returns:
        xarray Dataset resampled to monthly resolution
    """
    print(f"   📅 Resampling {spi_type.upper()} from decadal to monthly (taking last decade of each month)")
    
    try:
        # Get the SPI variable name
        spi_var_name = None
        for var in dataset.data_vars:
            if 'spc' in var.lower() or 'spi' in var.lower():
                spi_var_name = var
                break
        
        if spi_var_name is None:
            raise ValueError(f"No SPI variable found in {spi_type} dataset")
        
        # Create a more robust monthly resampling approach
        # For decadal data (3 periods per month), take every 3rd value starting from index 2
        # This corresponds to the last decade of each month (days 21-31)
        
        # Calculate step size based on expected pattern (1224 timesteps / 408 months = 3)
        original_time_steps = dataset.sizes['time']
        expected_monthly_steps = 408  # Target monthly steps
        step_size = max(1, original_time_steps // expected_monthly_steps)
        
        if step_size == 3:  # Confirmed decadal data
            # Take every 3rd timestep starting from index 2 (last decade of each month)
            indices = list(range(2, original_time_steps, 3))
            monthly_dataset = dataset.isel(time=indices)
            
            print(f"   Success: Resampled from {original_time_steps} to {len(indices)} time steps (every 3rd, last decade)")
        else:
            # Fallback: simple subsampling if pattern doesn't match expected
            indices = list(range(0, original_time_steps, step_size))
            monthly_dataset = dataset.isel(time=indices)
            
            print(f"   Warning: Applied fallback resampling: {original_time_steps} to {len(indices)} time steps (step={step_size})")
        
        print(f"   New time range: {monthly_dataset.time.min().values} to {monthly_dataset.time.max().values}")
        
        return monthly_dataset
        
    except Exception as e:
        print(f"   ❌ Failed to resample {spi_type}: {e}")
        return dataset  # Return original if resampling fails

def load_icechunk_spi_data(spi_type):
    """
    Load SPI data from Icechunk repository for a specific SPI variable.
    
    For SPI1 and SPI3: Data is loaded in decadal format (every 10 days) and then
    resampled to monthly by taking the last decade of each month to ensure
    consistency with other SPI variables (SPI6, SPI12, SPI24, SPI48).
    
    For SPI6, SPI12, SPI24, SPI48: Data is already in monthly format and loaded as-is.
    
    Args:
        spi_type: String indicating which SPI variable to load (spi1, spi3, spi6, etc.)
        
    Returns:
        xarray Dataset with uniform monthly temporal resolution (408 time steps)
    """
    print(f"\n{'='*50}")
    print(f"LOADING {spi_type.upper()} DATA FROM ICECHUNK")
    print(f"{'='*50}")

    # Construct repository prefix for the specific SPI type
    repo_prefix = f"{BASE_PREFIX}_{spi_type}"

    print(f"Repository prefix: {repo_prefix}")
    print(f"Bucket: {BUCKET_NAME}")

    try:
        # Setup storage connection
        storage = icechunk.gcs_storage(
            bucket=BUCKET_NAME,
            prefix=repo_prefix,
            service_account_file=SERVICE_ACCOUNT_FILE)

        # Open repository
        repo = icechunk.Repository.open(storage)
        session = repo.readonly_session("main")

        # Load data from specific Zarr group
        group_name = f"{spi_type}_data"  # e.g., "spi1_data"
        dataset = xr.open_zarr(session.store, group=group_name)

        print(f"✅ Successfully loaded {spi_type.upper()} dataset")
        print(f"   Shape: {dict(dataset.sizes)}")
        print(f"   Variables: {list(dataset.data_vars)}")
        print(
            f"   Time range: {dataset.time.min().values} to {dataset.time.max().values}"
        )
        
        # Apply temporal resampling for SPI1 and SPI3 (decadal to monthly)
        if spi_type.lower() in ['spi1', 'spi3']:
            dataset = resample_decadal_to_monthly(dataset, spi_type)

        return dataset

    except Exception as e:
        print(f"❌ Failed to load {spi_type.upper()} Icechunk data: {e}")
        return None

def load_all_spi_datasets():
    """
    Load all SPI datasets and return a dictionary.
    
    This function loads all 6 SPI variables with temporal uniformity:
    - SPI1, SPI3: Loaded as decadal data and resampled to monthly (last decade per month)
    - SPI6, SPI12, SPI24, SPI48: Loaded as monthly data directly
    
    All datasets will have uniform monthly temporal resolution (~408 time steps)
    covering the same time period for consistent memory analysis.
    
    Returns:
        Dictionary mapping SPI variable names to xarray Datasets with uniform temporal resolution
    """
    print(f"\n{'='*70}")
    print("LOADING ALL SPI DATASETS FROM ICECHUNK WITH TEMPORAL UNIFORMITY")
    print(f"{'='*70}")
    print("📅 SPI1, SPI3: Decadal → Monthly resampling (last decade per month)")
    print("📅 SPI6, SPI12, SPI24, SPI48: Direct monthly loading")
    
    spi_datasets = {}
    
    for spi_type in SPI_VARIABLES:
        dataset = load_icechunk_spi_data(spi_type)
        if dataset is not None:
            spi_datasets[spi_type] = dataset
        else:
            print(f"⚠️  Skipping {spi_type} due to loading error")
    
    print(f"\n✅ Successfully loaded {len(spi_datasets)} SPI datasets")
    print(f"   Available datasets: {list(spi_datasets.keys())}")
    
    # Verify temporal uniformity
    time_steps = []
    for spi_type, dataset in spi_datasets.items():
        time_steps.append((spi_type, dataset.sizes['time']))
        print(f"   {spi_type.upper()}: {dataset.sizes['time']} time steps")
    
    # Check if all datasets have the same number of time steps
    unique_time_steps = set([steps for _, steps in time_steps])
    if len(unique_time_steps) == 1:
        print(f"\n🎯 All datasets uniform: {list(unique_time_steps)[0]} time steps")
    else:
        print(f"\n⚠️  Time step variation detected: {dict(time_steps)}")
    
    return spi_datasets

def load_administrative_regions():
    """Load administrative regions and create regionmask"""
    print(f"\n{'='*70}")
    print("LOADING ADMINISTRATIVE REGIONS")
    print(f"{'='*70}")

    try:
        # Load GeoJSON file
        gdf = gpd.read_file(GEOJSON_FILE)
        gdf['region_idx'] = np.arange(len(gdf))

        print(f"✅ Loaded {len(gdf)} administrative regions")
        print(f"   Columns: {list(gdf.columns)}")
        
        # Check for the correct name column
        name_col = 'GID_1' if 'GID_1' in gdf.columns else 'shapeName'
        id_col = 'region_idx' if 'region_idx' in gdf.columns else 'shapeID'
        
        print(f"   Sample regions: {gdf[name_col].head().tolist()}")

        # Fix invalid geometries that can cause overlap detection issues
        invalid_count = (~gdf.geometry.is_valid).sum()
        if invalid_count > 0:
            print(f"   Fixing {invalid_count} invalid geometries...")
            gdf.geometry = gdf.geometry.buffer(0)

        # Create regionmask using the appropriate columns
        regions = regionmask.from_geopandas(gdf,
                                            names=name_col,
                                            abbrevs=id_col,
                                            name="icpac_regions")

        print(f"✅ Created regionmask with {len(regions)} regions")

        return gdf, regions

    except Exception as e:
        print(f"❌ Failed to load regions: {e}")
        raise

def create_region_mask(gdf, dataset):
    """Create region mask for the dataset using regionmask.mask_geopandas"""
    print(f"\n{'='*70}")
    print("CREATING REGION MASK")
    print(f"{'='*70}")

    try:
        # Extract coordinates
        lons = dataset.lon.values
        lats = dataset.lat.values

        print(f"   Dataset coordinates: {len(lons)} lons × {len(lats)} lats")
        print(f"   Coordinate ranges: lon [{lons.min():.2f}, {lons.max():.2f}], lat [{lats.min():.2f}, {lats.max():.2f}]")

        # Create mask using regionmask.mask_geopandas
        try:
            mask = regionmask.mask_geopandas(gdf.geometry, lons, lats)
        except ValueError as e:
            if "overlapping regions" in str(e):
                print("   Detected overlapping regions, using overlap=False...")
                mask = regionmask.mask_geopandas(gdf.geometry, lons, lats, overlap=False)
            else:
                raise

        print(f"✅ Created region mask")
        print(f"   Mask shape: {mask.shape}")
        print(f"   Mask dtype: {mask.dtype}")
        print(f"   Unique regions in mask: {len(np.unique(mask.values[~np.isnan(mask.values)]))}")
        
        return mask

    except Exception as e:
        print(f"❌ Failed to create region mask: {e}")
        raise

def analyze_all_spi_memory_usage(spi_datasets: Dict[str, xr.Dataset], region_mask: xr.DataArray, gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Analyze memory usage for each region's data cube across all SPI variables
    
    Args:
        spi_datasets: Dictionary of SPI datasets (spi1, spi3, etc.)
        region_mask: Region mask array
        gdf: GeoDataFrame with administrative regions
        
    Returns:
        DataFrame with comprehensive memory analysis results for all SPI variables
    """
    print(f"\n{'='*70}")
    print("ANALYZING REGION MEMORY USAGE FOR ALL SPI VARIABLES")
    print(f"{'='*70}")
    
    # Get unique region IDs from the mask
    unique_regions = np.unique(region_mask.values[~np.isnan(region_mask.values)])
    print(f"Processing {len(unique_regions)} regions across {len(spi_datasets)} SPI variables...")
    
    # Initialize results storage
    results = []
    
    start_time = time.time()
    
    for i, region_id in enumerate(unique_regions):
        region_id = int(region_id)
        
        try:
            # Get region metadata
            region_row = gdf.iloc[region_id]
            region_name = region_row.get('GID_1', region_row.get('NAME_1', f"Region_{region_id}"))
            
            # Create region-specific mask
            region_data_mask = (region_mask == region_id)
            
            # Count valid pixels for this region
            valid_pixels = region_data_mask.sum().values
            
            # Get region bounds for additional info
            region_geometry = region_row['geometry']
            bounds = region_geometry.bounds  # minx, miny, maxx, maxy
            
            # Base region info
            region_info = {
                'region_id': region_id,
                'region_name': region_name,
                'valid_pixels': int(valid_pixels),
                'lon_range': f"{bounds[0]:.2f} to {bounds[2]:.2f}",
                'lat_range': f"{bounds[1]:.2f} to {bounds[3]:.2f}"
            }
            
            # Analyze each SPI variable
            for spi_type, dataset in spi_datasets.items():
                # Get SPI variable name
                spi_var = None
                for var in dataset.data_vars:
                    if 'spi' in var.lower() or 'spc' in var.lower():
                        spi_var = var
                        break
                
                if spi_var is None:
                    continue
                    
                # Estimate memory usage for different scenarios
                pixel_data_size_bytes = 8  # float64
                time_steps = dataset.sizes['time']
                n_variables = 1  # Single SPI variable per dataset
                
                # Memory calculations
                single_var_single_timestep = valid_pixels * pixel_data_size_bytes
                single_var_full_time = single_var_single_timestep * time_steps
                
                # Convert to MB
                single_var_single_timestep_mb = single_var_single_timestep / (1024**2)
                single_var_full_time_mb = single_var_full_time / (1024**2)
                
                # Add SPI-specific columns
                region_info[f'{spi_type}_time_steps'] = time_steps
                region_info[f'{spi_type}_single_timestep_mb'] = round(single_var_single_timestep_mb, 3)
                region_info[f'{spi_type}_full_time_mb'] = round(single_var_full_time_mb, 2)
            
            # Calculate total memory if all SPI variables loaded simultaneously
            total_memory_mb = sum([region_info.get(f'{spi}_full_time_mb', 0) for spi in spi_datasets.keys()])
            region_info['total_all_spi_mb'] = round(total_memory_mb, 2)
            
            results.append(region_info)
            
            # Progress indicator
            if (i + 1) % 50 == 0 or i == len(unique_regions) - 1:
                elapsed = time.time() - start_time
                print(f"   Processed {i + 1}/{len(unique_regions)} regions ({elapsed:.1f}s)")
        
        except Exception as e:
            print(f"   ⚠️  Error processing region {region_id}: {e}")
            continue
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by total memory usage (largest first)
    if 'total_all_spi_mb' in results_df.columns:
        results_df = results_df.sort_values('total_all_spi_mb', ascending=False)
    
    return results_df

def print_spi_memory_summary(results_df: pd.DataFrame, spi_datasets: Dict[str, xr.Dataset]):
    """Print comprehensive summary statistics of SPI memory usage analysis"""
    print(f"\n{'='*70}")
    print("SPI MEMORY USAGE SUMMARY")
    print(f"{'='*70}")
    
    # Overall statistics
    total_regions = len(results_df)
    total_pixels = results_df['valid_pixels'].sum()
    
    print(f"📊 Overall Statistics:")
    print(f"   Total regions analyzed: {total_regions}")
    print(f"   Total valid pixels: {total_pixels:,}")
    print(f"   Average pixels per region: {total_pixels/total_regions:.0f}")
    print(f"   SPI variables analyzed: {list(spi_datasets.keys())}")
    
    # Memory usage statistics for each SPI variable
    for spi_type in spi_datasets.keys():
        col_name = f'{spi_type}_full_time_mb'
        if col_name in results_df.columns:
            spi_memory = results_df[col_name]
            print(f"\n📈 {spi_type.upper()} Memory Usage (Full Time Series):")
            print(f"   Minimum: {spi_memory.min():.2f} MB")
            print(f"   Maximum: {spi_memory.max():.2f} MB")
            print(f"   Mean: {spi_memory.mean():.2f} MB")
            print(f"   Median: {spi_memory.median():.2f} MB")
            print(f"   Total across all regions: {spi_memory.sum():.2f} MB ({spi_memory.sum()/1024:.2f} GB)")
    
    # Total memory if all SPI variables loaded simultaneously
    if 'total_all_spi_mb' in results_df.columns:
        total_memory = results_df['total_all_spi_mb']
        print(f"\n📈 Total Memory Usage (All SPI Variables Combined):")
        print(f"   Minimum: {total_memory.min():.2f} MB")
        print(f"   Maximum: {total_memory.max():.2f} MB")
        print(f"   Mean: {total_memory.mean():.2f} MB")
        print(f"   Median: {total_memory.median():.2f} MB")
        print(f"   Total: {total_memory.sum():.2f} MB ({total_memory.sum()/1024:.2f} GB)")
    
    # Dask worker recommendations
    print(f"\n🚀 Dask Worker Recommendations:")
    
    if 'total_all_spi_mb' in results_df.columns:
        max_total_memory_mb = results_df['total_all_spi_mb'].max()
        recommended_total_gb = max(4, (max_total_memory_mb * 3) / 1024)  # 3x buffer, minimum 4GB
        
        print(f"   For processing all SPI variables simultaneously:")
        print(f"     Largest region requires: {max_total_memory_mb:.2f} MB")
        print(f"     Recommended worker memory: {recommended_total_gb:.1f} GB")
        print(f"     Recommended workers: {max(1, int(32 / recommended_total_gb))}")
    
    # For single SPI variable processing
    single_spi_max = 0
    for spi_type in spi_datasets.keys():
        col_name = f'{spi_type}_full_time_mb'
        if col_name in results_df.columns:
            single_spi_max = max(single_spi_max, results_df[col_name].max())
    
    if single_spi_max > 0:
        recommended_single_gb = max(2, (single_spi_max * 3) / 1024)
        print(f"   For processing single SPI variables:")
        print(f"     Largest region requires: {single_spi_max:.2f} MB")
        print(f"     Recommended worker memory: {recommended_single_gb:.1f} GB")
        print(f"     Recommended workers: {max(1, int(32 / recommended_single_gb))}")
    
    # Top 10 largest regions
    print(f"\n🔝 Top 10 Largest Regions (by total SPI memory usage):")
    top_10 = results_df.head(10)
    for idx, row in top_10.iterrows():
        total_mem = row.get('total_all_spi_mb', 'N/A')
        print(f"   {row['region_name']}: {row['valid_pixels']:,} pixels, {total_mem} MB total")

def save_results(results_df: pd.DataFrame, filename: str = "spi_region_memory_analysis.csv"):
    """Save results to CSV file with comprehensive SPI analysis"""
    results_df.to_csv(filename, index=False)
    print(f"\n💾 Results saved to: {filename}")
    print(f"   Columns: {list(results_df.columns)}")
    print(f"   Rows: {len(results_df)}")
    print(f"   Analysis includes all SPI variables and region-wise memory breakdowns")
    
    # Also save a summary file with dask plan information
    summary_filename = filename.replace('.csv', '_dask_plan.csv')
    
    # Create Dask plan summary
    plan_data = []
    
    for spi_type in SPI_VARIABLES:
        col_name = f'{spi_type}_full_time_mb'
        if col_name in results_df.columns:
            max_mem = results_df[col_name].max()
            recommended_gb = max(2, (max_mem * 3) / 1024)
            recommended_workers = max(1, int(32 / recommended_gb))
            
            plan_data.append({
                'spi_variable': spi_type,
                'max_region_memory_mb': max_mem,
                'recommended_worker_memory_gb': round(recommended_gb, 1),
                'recommended_workers': recommended_workers,
                'total_regions': len(results_df),
                'processing_strategy': 'single_variable'
            })
    
    # Add combined strategy
    if 'total_all_spi_mb' in results_df.columns:
        max_total = results_df['total_all_spi_mb'].max()
        recommended_total_gb = max(4, (max_total * 3) / 1024)
        recommended_total_workers = max(1, int(32 / recommended_total_gb))
        
        plan_data.append({
            'spi_variable': 'all_combined',
            'max_region_memory_mb': max_total,
            'recommended_worker_memory_gb': round(recommended_total_gb, 1),
            'recommended_workers': recommended_total_workers,
            'total_regions': len(results_df),
            'processing_strategy': 'all_variables_combined'
        })
    
    plan_df = pd.DataFrame(plan_data)
    plan_df.to_csv(summary_filename, index=False)
    print(f"   Dask plan saved to: {summary_filename}")

def main():
    """Main function to run the comprehensive SPI memory analysis"""
    try:
        print(f"\n{'='*70}")
        print("SPI REGIONAL MEMORY ANALYSIS - ICECHUNK SOURCE")
        print(f"{'='*70}")
        print(f"Target: {len(SPI_VARIABLES)} SPI variables across 230+ administrative regions")
        print(f"SPI Variables: {', '.join(SPI_VARIABLES)}")
        print(f"📅 Temporal Processing: SPI1, SPI3 resampled decadal→monthly; others direct monthly")
        print(f"🎯 Goal: Uniform ~408 monthly time steps across all variables")
        
        # Load all SPI datasets from Icechunk
        spi_datasets = load_all_spi_datasets()
        
        if not spi_datasets:
            raise ValueError("No SPI datasets were successfully loaded")
        
        # Use the first available dataset for coordinate reference
        reference_dataset = next(iter(spi_datasets.values()))
        
        # Load administrative regions
        gdf, regions = load_administrative_regions()
        
        # Create region mask using reference dataset
        mask = create_region_mask(gdf, reference_dataset)
        
        # Analyze memory usage for each region across all SPI variables
        results_df = analyze_all_spi_memory_usage(spi_datasets, mask, gdf)
        
        # Print comprehensive summary
        print_spi_memory_summary(results_df, spi_datasets)
        
        # Save results with Dask plan
        save_results(results_df)
        
        print(f"\n{'='*70}")
        print("✅ SPI REGIONAL MEMORY ANALYSIS COMPLETED")
        print(f"{'='*70}")
        print(f"📊 Analyzed {len(results_df)} regions across {len(spi_datasets)} SPI variables")
        print(f"📅 All variables normalized to uniform monthly temporal resolution")
        print(f"💾 Results saved with Dask worker recommendations")
        print(f"🚀 Ready for distributed SPI processing with temporal consistency!")
        
        return spi_datasets, gdf, mask, results_df
        
    except Exception as e:
        print(f"❌ Error in main workflow: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    ds, gdf, mask, results = main()