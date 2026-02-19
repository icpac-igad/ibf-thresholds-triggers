#!/usr/bin/env python3
"""
CMORPH 2020-01-01 Simple Processor with Plotting

Process all NetCDF files from s3://noaa-cdr-precip-cmorph-pds/data/30min/8km/2020/01/01/
using VirtualiZarr, concatenate them, and create plots.

This simplified version:
1. Lists all NC files from the S3 bucket
2. Creates virtual datasets (no data download)
3. Concatenates along time dimension
4. Computes data into memory for plotting
5. Creates comprehensive precipitation plots

Usage:
    python cmorph_20200101_simple_processor.py --process --plot
    python cmorph_20200101_simple_processor.py --process --max-files 10 --plot
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

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

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


class CMORPH20200101SimpleProcessor:
    """Simple CMORPH processor for 2020-01-01 with plotting"""

    def __init__(self):
        """Initialize processor"""
        self.bucket = CMORPH_CONFIG['bucket']
        self.base_path = CMORPH_CONFIG['base_path']
        self.target_date = TARGET_DATE

        # Setup VirtualiZarr components
        self.store = from_url(self.bucket, region=CMORPH_CONFIG['region'], skip_signature=True)
        self.registry = ObjectStoreRegistry({self.bucket: self.store})
        self.parser = HDFParser()

        logger.info(f"Initialized CMORPH processor for {self.target_date}")

    def list_nc_files(self) -> List[str]:
        """List all NetCDF files for 2020/01/01"""
        logger.info(f"Listing NetCDF files for {self.target_date}")

        bucket_name = self.bucket.replace('s3://', '').rstrip('/')
        target_prefix = f"{self.base_path}{self.target_date}/"

        list_store = S3Store(
            bucket_name=bucket_name,
            prefix=target_prefix,
            region=CMORPH_CONFIG['region'],
            skip_signature=True
        )

        try:
            list_result = list(list_store.list())
            if not list_result:
                logger.warning(f"No objects found in {target_prefix}")
                return []

            all_objects = list_result[0]
            nc_files = []

            for obj_info in all_objects:
                if isinstance(obj_info, dict) and 'path' in obj_info:
                    if obj_info['path'].endswith('.nc'):
                        full_path = f"{target_prefix}{obj_info['path']}"
                        nc_files.append(full_path)

            nc_files.sort()
            logger.info(f"Found {len(nc_files)} NetCDF files")
            if nc_files:
                logger.info(f"  First: {Path(nc_files[0]).name}")
                logger.info(f"  Last: {Path(nc_files[-1]).name}")

            return nc_files

        except Exception as e:
            logger.error(f"Error listing files: {e}")
            return []

    def create_concatenated_dataset(self, file_paths: List[str], max_files: Optional[int] = None) -> xr.Dataset:
        """Create concatenated virtual dataset"""

        if max_files:
            file_paths = file_paths[:max_files]
            logger.info(f"Processing first {max_files} files")

        logger.info(f"Creating virtual datasets for {len(file_paths)} files...")

        virtual_datasets = []
        for i, file_path in enumerate(file_paths, 1):
            try:
                url = f"{self.bucket}{file_path}"
                logger.debug(f"{i}/{len(file_paths)}: {Path(file_path).name}")

                vds = open_virtual_dataset(
                    url=url,
                    parser=self.parser,
                    registry=self.registry
                )
                virtual_datasets.append(vds)

                if i % 10 == 0:
                    logger.info(f"  Processed {i}/{len(file_paths)}...")

            except Exception as e:
                logger.error(f"Failed: {Path(file_path).name}: {e}")
                continue

        logger.info(f"✓ Created {len(virtual_datasets)} virtual datasets")

        # Concatenate
        logger.info("Concatenating datasets...")
        concatenated_ds = xr.concat(
            virtual_datasets,
            dim='time',
            coords='minimal',
            compat='override',
            combine_attrs='override'
        )

        logger.info(f"✓ Concatenated dataset:")
        logger.info(f"  Dimensions: {dict(concatenated_ds.sizes)}")
        logger.info(f"  Variables: {list(concatenated_ds.data_vars.keys())}")
        if 'time' in concatenated_ds:
            logger.info(f"  Time: {concatenated_ds.time.min().values} to {concatenated_ds.time.max().values}")
            logger.info(f"  Timesteps: {len(concatenated_ds.time)}")

        return concatenated_ds

    def plot_precipitation(self, ds: xr.Dataset, output_dir: str = './plots'):
        """Create precipitation plots"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        logger.info("Creating precipitation plots...")

        # Find precipitation variable
        precip_var = None
        for var in ['cmorph', 'precipitation', 'precip', 'pr']:
            if var in ds.data_vars:
                precip_var = var
                break

        if precip_var is None:
            logger.warning(f"No precipitation variable. Available: {list(ds.data_vars.keys())}")
            return

        logger.info(f"Using variable: '{precip_var}'")

        # Load data (this will download it)
        logger.info("Loading precipitation data from S3...")
        precip_data = ds[precip_var].load()
        logger.info("✓ Data loaded")

        # 1. Time series - Spatial mean
        logger.info("Plot 1/4: Time series...")
        fig, ax = plt.subplots(figsize=(14, 6))

        mean_precip = precip_data.mean(dim=['lat', 'lon'])
        ax.plot(ds.time.values, mean_precip.values, 'b-', linewidth=1.5)
        ax.set_xlabel('Time (UTC)', fontsize=12)
        ax.set_ylabel('Mean Precipitation (mm/hr)', fontsize=12)
        ax.set_title(f'CMORPH Mean Precipitation - {TARGET_DATE}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path / 'cmorph_20200101_timeseries.png', dpi=150)
        logger.info(f"  ✓ Saved timeseries")
        plt.close()

        # 2. Daily mean spatial plot
        logger.info("Plot 2/4: Daily mean...")
        daily_mean = precip_data.mean(dim='time')

        fig, ax = plt.subplots(figsize=(14, 8))
        im = daily_mean.plot(
            ax=ax,
            cmap='Blues',
            vmin=0,
            vmax=float(daily_mean.quantile(0.95)),
            cbar_kwargs={'label': 'Mean Precipitation (mm/hr)', 'shrink': 0.8}
        )
        ax.set_title(f'CMORPH Daily Mean - {TARGET_DATE}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / 'cmorph_20200101_daily_mean.png', dpi=150)
        logger.info(f"  ✓ Saved daily mean")
        plt.close()

        # 3. Snapshots
        logger.info("Plot 3/4: Snapshots...")
        n_times = len(ds.time)
        selected_indices = np.linspace(0, n_times-1, min(6, n_times), dtype=int)

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        vmax = float(daily_mean.quantile(0.95))

        for idx, time_idx in enumerate(selected_indices):
            ax = axes[idx]
            snapshot = precip_data.isel(time=time_idx)
            time_str = str(ds.time.isel(time=time_idx).values)[:16]

            snapshot.plot(
                ax=ax,
                cmap='Blues',
                vmin=0,
                vmax=vmax,
                add_colorbar=False
            )
            ax.set_title(f'{time_str} UTC', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)

        fig.colorbar(im, ax=axes, orientation='horizontal', pad=0.05,
                     label='Precipitation (mm/hr)', shrink=0.8)
        fig.suptitle(f'CMORPH Snapshots - {TARGET_DATE}', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(output_path / 'cmorph_20200101_snapshots.png', dpi=150)
        logger.info(f"  ✓ Saved snapshots")
        plt.close()

        # 4. Statistics
        logger.info("Plot 4/4: Statistics...")
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Max
        precip_max = precip_data.max(dim='time')
        precip_max.plot(ax=axes[0], cmap='Reds',
                       cbar_kwargs={'label': 'Max (mm/hr)', 'shrink': 0.8})
        axes[0].set_title('Maximum', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # Std dev
        precip_std = precip_data.std(dim='time')
        precip_std.plot(ax=axes[1], cmap='YlOrRd',
                       cbar_kwargs={'label': 'Std Dev (mm/hr)', 'shrink': 0.8})
        axes[1].set_title('Variability', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        # Total
        precip_total = precip_data.sum(dim='time') * 0.5
        precip_total.plot(ax=axes[2], cmap='Blues',
                         cbar_kwargs={'label': 'Total (mm)', 'shrink': 0.8})
        axes[2].set_title('Total Accumulated', fontsize=12, fontweight='bold')
        axes[2].grid(True, alpha=0.3)

        fig.suptitle(f'CMORPH Statistics - {TARGET_DATE}', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(output_path / 'cmorph_20200101_statistics.png', dpi=150)
        logger.info(f"  ✓ Saved statistics")
        plt.close()

        logger.info(f"✓ All plots saved to {output_path}/")

    def process(self, max_files: Optional[int] = None, plot: bool = False):
        """Main processing pipeline"""
        logger.info("=" * 70)
        logger.info("CMORPH 2020-01-01 Processing")
        logger.info("=" * 70)

        # List files
        nc_files = self.list_nc_files()
        if not nc_files:
            logger.error("No files found")
            return None

        # Create concatenated dataset
        ds = self.create_concatenated_dataset(nc_files, max_files=max_files)

        # Plot if requested
        if plot:
            self.plot_precipitation(ds)

        logger.info("=" * 70)
        logger.info("✓ Processing complete!")
        logger.info("=" * 70)

        return ds


def main():
    parser = argparse.ArgumentParser(
        description="CMORPH 2020-01-01 Simple Processor with Plotting"
    )

    parser.add_argument('--process', action='store_true', help='Process files')
    parser.add_argument('--max-files', type=int, default=None, help='Max files (default: all)')
    parser.add_argument('--plot', action='store_true', help='Create plots')

    args = parser.parse_args()

    if not args.process:
        parser.print_help()
        return 1

    processor = CMORPH20200101SimpleProcessor()
    ds = processor.process(max_files=args.max_files, plot=args.plot)

    return 0 if ds is not None else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
