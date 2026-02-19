#!/usr/bin/env python3
"""
CMORPH 2020-01-01 VirtualiZarr to Local Icechunk Processor

Process all NetCDF files from s3://noaa-cdr-precip-cmorph-pds/data/30min/8km/2020/01/01/
using VirtualiZarr and store in a local Icechunk repository with plotting capabilities.

Features:
1. List all NetCDF files from CMORPH 2020-01-01
2. Create virtual datasets using VirtualiZarr (no data download)
3. Concatenate along time dimension
4. Store in local Icechunk repository
5. Reopen and validate the Icechunk store
6. Plot precipitation data for analysis

Usage:
    python cmorph_20200101_local_icechunk_processor.py --process
    python cmorph_20200101_local_icechunk_processor.py --process --plot
    python cmorph_20200101_local_icechunk_processor.py --reopen-only --plot
"""

import logging
import warnings
import argparse
from pathlib import Path
from typing import List, Optional
from datetime import datetime

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from obstore.store import S3Store, from_url
from virtualizarr import open_virtual_dataset
from virtualizarr.parsers import HDFParser
from virtualizarr.registry import ObjectStoreRegistry
import icechunk

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# CMORPH configuration
CMORPH_CONFIG = {
    'bucket': 's3://noaa-cdr-precip-cmorph-pds/',
    'base_path': 'data/30min/8km/',
    'region': 'us-east-1',
    'description': 'CMORPH 30-minute 8km precipitation data'
}

# Target date
TARGET_DATE = '2020/01/01'
ICECHUNK_STORE_PATH = './cmorph_20200101_icechunk_store'


class CMORPH20200101Processor:
    """Process CMORPH data for 2020-01-01 to local Icechunk store"""

    def __init__(self, icechunk_path: str = ICECHUNK_STORE_PATH):
        """
        Initialize processor

        Args:
            icechunk_path: Path for local Icechunk store
        """
        self.bucket = CMORPH_CONFIG['bucket']
        self.base_path = CMORPH_CONFIG['base_path']
        self.target_date = TARGET_DATE
        self.icechunk_path = Path(icechunk_path)

        # Setup VirtualiZarr components
        self.store = from_url(self.bucket, region=CMORPH_CONFIG['region'], skip_signature=True)
        self.registry = ObjectStoreRegistry({self.bucket: self.store})
        self.parser = HDFParser()

        logger.info(f"Initialized CMORPH processor for {self.target_date}")
        logger.info(f"Icechunk store path: {self.icechunk_path}")

    def list_nc_files(self) -> List[str]:
        """
        List all NetCDF files for 2020/01/01

        Returns:
            List of file paths relative to bucket
        """
        logger.info(f"Listing NetCDF files for {self.target_date}")

        # Create S3Store for listing
        bucket_name = self.bucket.replace('s3://', '').rstrip('/')
        target_prefix = f"{self.base_path}{self.target_date}/"

        list_store = S3Store(
            bucket_name=bucket_name,
            prefix=target_prefix,
            region=CMORPH_CONFIG['region'],
            skip_signature=True
        )

        try:
            # List all objects
            list_result = list(list_store.list())

            if not list_result:
                logger.warning(f"No objects found in {target_prefix}")
                return []

            # Extract file paths from nested structure
            all_objects = list_result[0]
            nc_files = []

            for obj_info in all_objects:
                if isinstance(obj_info, dict) and 'path' in obj_info:
                    if obj_info['path'].endswith('.nc'):
                        # Return full path including prefix
                        full_path = f"{target_prefix}{obj_info['path']}"
                        nc_files.append(full_path)

            nc_files.sort()
            logger.info(f"Found {len(nc_files)} NetCDF files for {self.target_date}")

            # Log first and last files
            if nc_files:
                logger.info(f"  First file: {Path(nc_files[0]).name}")
                logger.info(f"  Last file: {Path(nc_files[-1]).name}")

            return nc_files

        except Exception as e:
            logger.error(f"Error listing files: {e}")
            return []

    def create_virtual_datasets(self, file_paths: List[str], max_files: Optional[int] = None) -> List[xr.Dataset]:
        """
        Create virtual datasets for multiple files

        Args:
            file_paths: List of file paths relative to bucket
            max_files: Optional limit on number of files to process

        Returns:
            List of virtual datasets
        """
        # Limit files if specified
        if max_files:
            file_paths = file_paths[:max_files]
            logger.info(f"Processing first {max_files} files out of {len(file_paths)} available")

        logger.info(f"Creating virtual datasets for {len(file_paths)} files...")

        virtual_datasets = []
        failed_files = []

        for i, file_path in enumerate(file_paths, 1):
            filename = Path(file_path).name

            try:
                # Construct full URL
                url = f"{self.bucket}{file_path}"

                logger.debug(f"Processing {i}/{len(file_paths)}: {filename}")

                # Create virtual dataset
                vds = open_virtual_dataset(
                    url=url,
                    parser=self.parser,
                    registry=self.registry
                )

                virtual_datasets.append(vds)

                if i % 10 == 0:
                    logger.info(f"  Processed {i}/{len(file_paths)} files...")

            except Exception as e:
                logger.error(f"Failed to process {filename}: {e}")
                failed_files.append(filename)
                continue

        logger.info(f"✓ Created {len(virtual_datasets)} virtual datasets")

        if failed_files:
            logger.warning(f"✗ Failed files ({len(failed_files)}): {failed_files[:5]}...")

        return virtual_datasets

    def concatenate_datasets(self, virtual_datasets: List[xr.Dataset]) -> xr.Dataset:
        """
        Concatenate virtual datasets along time dimension

        Args:
            virtual_datasets: List of virtual datasets

        Returns:
            Concatenated virtual dataset
        """
        logger.info(f"Concatenating {len(virtual_datasets)} virtual datasets along time dimension...")

        try:
            concatenated_ds = xr.concat(
                virtual_datasets,
                dim='time',
                coords='minimal',
                compat='override',
                combine_attrs='override'
            )

            logger.info(f"✓ Concatenated dataset created")
            logger.info(f"  Dimensions: {dict(concatenated_ds.dims)}")
            logger.info(f"  Data variables: {list(concatenated_ds.data_vars.keys())}")
            logger.info(f"  Coordinates: {list(concatenated_ds.coords.keys())}")

            # Time range info
            if 'time' in concatenated_ds.coords:
                time_coord = concatenated_ds['time']
                logger.info(f"  Time range: {time_coord.min().values} to {time_coord.max().values}")
                logger.info(f"  Time steps: {len(time_coord)}")

            return concatenated_ds

        except Exception as e:
            logger.error(f"Error concatenating datasets: {e}")
            raise

    def create_icechunk_store(self, virtual_ds: xr.Dataset, overwrite: bool = False) -> icechunk.Repository:
        """
        Create local Icechunk repository and write virtual dataset

        Args:
            virtual_ds: Concatenated virtual dataset
            overwrite: Whether to overwrite existing store

        Returns:
            Icechunk repository
        """
        logger.info(f"Creating Icechunk store at {self.icechunk_path}...")

        # Check if store already exists
        if self.icechunk_path.exists() and not overwrite:
            logger.warning(f"Icechunk store already exists at {self.icechunk_path}")
            logger.info("Use --overwrite flag to replace, or use --reopen-only to read existing store")
            raise FileExistsError(f"Store exists: {self.icechunk_path}")

        # Setup local filesystem storage
        storage = icechunk.local_filesystem_storage(
            path=str(self.icechunk_path)
        )

        # Setup repository configuration with virtual chunk container
        config = icechunk.RepositoryConfig.default()

        # Configure virtual chunk container pointing to CMORPH S3 bucket
        virtual_chunk_container_url = self.bucket
        config.set_virtual_chunk_container(
            icechunk.VirtualChunkContainer(
                virtual_chunk_container_url,
                icechunk.s3_store(region=CMORPH_CONFIG['region'])
            )
        )

        # Setup credentials for anonymous S3 access
        credentials = icechunk.containers_credentials({
            virtual_chunk_container_url: icechunk.s3_credentials(anonymous=True)
        })

        # Create repository
        logger.info("Creating Icechunk repository...")
        repo = icechunk.Repository.create(storage, config, credentials)
        logger.info("✓ Repository created")

        # Write virtual dataset to Icechunk using virtualizarr's to_icechunk method
        logger.info("Writing virtual dataset to Icechunk store...")
        session = repo.writable_session("main")

        # Use VirtualiZarr's built-in to_icechunk method with the session's store
        virtual_ds.virtualize.to_icechunk(session.store)
        logger.info("✓ Wrote virtual dataset to Icechunk")

        # Commit the session
        snapshot_id = session.commit(f"CMORPH data for {self.target_date} - {len(virtual_ds.time)} timesteps")
        logger.info(f"✓ Committed data with snapshot ID: {snapshot_id}")

        return repo

    def reopen_icechunk_store(self) -> xr.Dataset:
        """
        Reopen existing Icechunk store and return dataset

        Returns:
            Dataset from Icechunk store
        """
        logger.info(f"Reopening Icechunk store from {self.icechunk_path}...")

        if not self.icechunk_path.exists():
            raise FileNotFoundError(f"Icechunk store not found: {self.icechunk_path}")

        # Setup storage
        storage = icechunk.local_filesystem_storage(
            path=str(self.icechunk_path)
        )

        # Setup configuration
        config = icechunk.RepositoryConfig.default()

        # Configure virtual chunk container
        virtual_chunk_container_url = self.bucket
        config.set_virtual_chunk_container(
            icechunk.VirtualChunkContainer(
                virtual_chunk_container_url,
                icechunk.s3_store(region=CMORPH_CONFIG['region'])
            )
        )

        # Setup credentials
        credentials = icechunk.containers_credentials({
            virtual_chunk_container_url: icechunk.s3_credentials(anonymous=True)
        })

        # Open repository
        repo = icechunk.Repository.open(storage, config, credentials)
        logger.info("✓ Repository opened")

        # Get readonly session
        readonly_session = repo.readonly_session("main")

        # Open dataset
        ds = xr.open_zarr(readonly_session, consolidated=True)
        logger.info("✓ Dataset opened from Icechunk store")
        logger.info(f"  Dimensions: {dict(ds.dims)}")
        logger.info(f"  Data variables: {list(ds.data_vars.keys())}")
        logger.info(f"  Coordinates: {list(ds.coords.keys())}")

        if 'time' in ds.coords:
            logger.info(f"  Time range: {ds.time.min().values} to {ds.time.max().values}")
            logger.info(f"  Time steps: {len(ds.time)}")

        return ds

    def plot_precipitation_analysis(self, ds: xr.Dataset, output_dir: str = './plots'):
        """
        Create comprehensive precipitation analysis plots

        Args:
            ds: Dataset from Icechunk store
            output_dir: Directory for saving plots
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        logger.info("Creating precipitation analysis plots...")

        # Determine precipitation variable name
        precip_var = None
        for var in ['cmorph', 'precipitation', 'precip', 'pr']:
            if var in ds.data_vars:
                precip_var = var
                break

        if precip_var is None:
            logger.warning(f"No precipitation variable found. Available: {list(ds.data_vars.keys())}")
            return

        logger.info(f"Using precipitation variable: '{precip_var}'")

        # 1. Time series plot - Daily mean precipitation
        logger.info("Creating time series plot...")
        fig, ax = plt.subplots(figsize=(14, 6))

        # Compute spatial mean for each timestep
        precip_timeseries = ds[precip_var].mean(dim=['lat', 'lon'])

        ax.plot(ds.time.values, precip_timeseries.values, 'b-', linewidth=1.5)
        ax.set_xlabel('Time (UTC)', fontsize=12)
        ax.set_ylabel('Mean Precipitation (mm/hr)', fontsize=12)
        ax.set_title(f'CMORPH Mean Precipitation - {self.target_date}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Format x-axis to show hours
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        plt.xticks(rotation=45)

        plt.tight_layout()
        timeseries_path = output_path / 'cmorph_20200101_timeseries.png'
        plt.savefig(timeseries_path, dpi=150, bbox_inches='tight')
        logger.info(f"  ✓ Saved: {timeseries_path}")
        plt.close()

        # 2. Spatial plot - Daily mean precipitation
        logger.info("Creating daily mean spatial plot...")
        daily_mean = ds[precip_var].mean(dim='time')

        fig, ax = plt.subplots(figsize=(14, 8), subplot_kw={'projection': None})

        im = daily_mean.plot(
            ax=ax,
            cmap='Blues',
            vmin=0,
            vmax=daily_mean.quantile(0.95).values,
            cbar_kwargs={'label': 'Mean Precipitation (mm/hr)', 'shrink': 0.8}
        )

        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.set_title(f'CMORPH Daily Mean Precipitation - {self.target_date}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        daily_mean_path = output_path / 'cmorph_20200101_daily_mean.png'
        plt.savefig(daily_mean_path, dpi=150, bbox_inches='tight')
        logger.info(f"  ✓ Saved: {daily_mean_path}")
        plt.close()

        # 3. Multi-panel plot - Selected timesteps
        logger.info("Creating multi-panel timestep plot...")

        # Select 6 evenly spaced timesteps
        n_times = len(ds.time)
        selected_indices = np.linspace(0, n_times-1, 6, dtype=int)

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for idx, time_idx in enumerate(selected_indices):
            ax = axes[idx]
            time_val = ds.time.isel(time=time_idx).values
            precip_snapshot = ds[precip_var].isel(time=time_idx)

            im = precip_snapshot.plot(
                ax=ax,
                cmap='Blues',
                vmin=0,
                vmax=daily_mean.quantile(0.95).values,
                add_colorbar=False
            )

            # Format time label
            time_str = np.datetime_as_string(time_val, unit='m')
            ax.set_title(f'{time_str} UTC', fontsize=11, fontweight='bold')
            ax.set_xlabel('Longitude', fontsize=10)
            ax.set_ylabel('Latitude', fontsize=10)
            ax.grid(True, alpha=0.3)

        # Add colorbar
        fig.colorbar(im, ax=axes, orientation='horizontal', pad=0.05,
                     label='Precipitation (mm/hr)', shrink=0.8)

        fig.suptitle(f'CMORPH Precipitation Snapshots - {self.target_date}',
                     fontsize=16, fontweight='bold', y=0.98)

        plt.tight_layout()
        snapshots_path = output_path / 'cmorph_20200101_snapshots.png'
        plt.savefig(snapshots_path, dpi=150, bbox_inches='tight')
        logger.info(f"  ✓ Saved: {snapshots_path}")
        plt.close()

        # 4. Summary statistics plot
        logger.info("Creating summary statistics plot...")

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Max precipitation
        precip_max = ds[precip_var].max(dim='time')
        im1 = precip_max.plot(
            ax=axes[0],
            cmap='Reds',
            cbar_kwargs={'label': 'Max Precipitation (mm/hr)', 'shrink': 0.8}
        )
        axes[0].set_title('Maximum Precipitation', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # Standard deviation
        precip_std = ds[precip_var].std(dim='time')
        im2 = precip_std.plot(
            ax=axes[1],
            cmap='YlOrRd',
            cbar_kwargs={'label': 'Std Dev (mm/hr)', 'shrink': 0.8}
        )
        axes[1].set_title('Precipitation Variability', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        # Total accumulated
        precip_total = ds[precip_var].sum(dim='time') * 0.5  # Convert to total mm (30-min intervals)
        im3 = precip_total.plot(
            ax=axes[2],
            cmap='Blues',
            cbar_kwargs={'label': 'Total Precipitation (mm)', 'shrink': 0.8}
        )
        axes[2].set_title('Total Accumulated Precipitation', fontsize=12, fontweight='bold')
        axes[2].grid(True, alpha=0.3)

        fig.suptitle(f'CMORPH Summary Statistics - {self.target_date}',
                     fontsize=14, fontweight='bold', y=1.02)

        plt.tight_layout()
        stats_path = output_path / 'cmorph_20200101_statistics.png'
        plt.savefig(stats_path, dpi=150, bbox_inches='tight')
        logger.info(f"  ✓ Saved: {stats_path}")
        plt.close()

        logger.info(f"✓ All plots saved to {output_path}/")

    def process_and_store(self, max_files: Optional[int] = None, overwrite: bool = False, plot: bool = False):
        """
        Complete processing pipeline: list files, create virtual datasets,
        concatenate, store in Icechunk, and optionally plot

        Args:
            max_files: Optional limit on number of files to process
            overwrite: Whether to overwrite existing Icechunk store
            plot: Whether to create plots
        """
        logger.info("=" * 70)
        logger.info("CMORPH 2020-01-01 Processing Pipeline")
        logger.info("=" * 70)

        # Step 1: List files
        nc_files = self.list_nc_files()

        if not nc_files:
            logger.error("No files found. Aborting.")
            return

        # Step 2: Create virtual datasets
        virtual_datasets = self.create_virtual_datasets(nc_files, max_files=max_files)

        if not virtual_datasets:
            logger.error("No virtual datasets created. Aborting.")
            return

        # Step 3: Concatenate
        concatenated_ds = self.concatenate_datasets(virtual_datasets)

        # Step 4: Create Icechunk store
        repo = self.create_icechunk_store(concatenated_ds, overwrite=overwrite)

        # Step 5: Reopen and validate
        logger.info("Validating Icechunk store...")
        reopened_ds = self.reopen_icechunk_store()

        logger.info("✓ Validation successful!")

        # Step 6: Optional plotting
        if plot:
            self.plot_precipitation_analysis(reopened_ds)

        logger.info("=" * 70)
        logger.info("Processing complete!")
        logger.info("=" * 70)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="CMORPH 2020-01-01 VirtualiZarr to Local Icechunk Processor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all files and create Icechunk store
  python cmorph_20200101_local_icechunk_processor.py --process

  # Process with plotting
  python cmorph_20200101_local_icechunk_processor.py --process --plot

  # Process only first 10 files (testing)
  python cmorph_20200101_local_icechunk_processor.py --process --max-files 10

  # Reopen existing store and plot
  python cmorph_20200101_local_icechunk_processor.py --reopen-only --plot
        """
    )

    parser.add_argument(
        '--process',
        action='store_true',
        help='Process files and create Icechunk store'
    )

    parser.add_argument(
        '--reopen-only',
        action='store_true',
        help='Reopen existing Icechunk store (no processing)'
    )

    parser.add_argument(
        '--max-files',
        type=int,
        default=None,
        help='Maximum number of files to process (default: all files)'
    )

    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing Icechunk store'
    )

    parser.add_argument(
        '--plot',
        action='store_true',
        help='Create precipitation analysis plots'
    )

    parser.add_argument(
        '--icechunk-path',
        default=ICECHUNK_STORE_PATH,
        help=f'Path for Icechunk store (default: {ICECHUNK_STORE_PATH})'
    )

    args = parser.parse_args()

    # Initialize processor
    processor = CMORPH20200101Processor(icechunk_path=args.icechunk_path)

    if args.process:
        # Process and store
        processor.process_and_store(
            max_files=args.max_files,
            overwrite=args.overwrite,
            plot=args.plot
        )

    elif args.reopen_only:
        # Just reopen and optionally plot
        logger.info("Reopening existing Icechunk store...")
        ds = processor.reopen_icechunk_store()

        if args.plot:
            processor.plot_precipitation_analysis(ds)
        else:
            logger.info("Use --plot flag to create analysis plots")

    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
