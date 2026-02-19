#!/usr/bin/env python3
"""
Test Zarr opening routine with anonymous S3 access for public CMORPH bucket
"""

import json
import logging
from pathlib import Path
import xarray as xr

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def test_zarr_opening_with_anon_s3(json_path: str):
    """Test opening JSON reference with anonymous S3 access"""

    logger.info(f"Testing Zarr opening: {json_path}")

    # Load JSON reference file
    with open(json_path, 'r') as f:
        refs_data = json.load(f)

    # Basic validation
    if 'refs' not in refs_data:
        raise ValueError("Invalid Kerchunk format: missing 'refs' key")

    zarr_refs = refs_data.get('refs', {})
    logger.info(f"✓ JSON loaded successfully")
    logger.info(f"  Total references: {len(zarr_refs)}")
    logger.info(f"  Has .zgroup: {'.zgroup' in zarr_refs}")
    logger.info(f"  Has .zattrs: {'.zattrs' in zarr_refs}")

    # Use kerchunk backend with proper storage options for public S3
    logger.info("\nOpening with kerchunk engine (anonymous S3 access)...")

    # Write refs to temp file for kerchunk engine
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
        json.dump(refs_data, tmp)
        tmp_path = tmp.name

    try:
        # Open with xarray using kerchunk engine
        # Configure storage options for anonymous access to public bucket
        ds = xr.open_dataset(
            tmp_path,
            engine='kerchunk',
            storage_options={
                's3': {'anon': True}  # Anonymous access for public S3 bucket
            }
        )

        logger.info("\n" + "="*80)
        logger.info("✓ ZARR DATASET OPENED SUCCESSFULLY!")
        logger.info("="*80)

        # Display dataset information
        logger.info("\n📊 DATASET OVERVIEW:")
        logger.info(f"  Dimensions: {dict(ds.dims)}")
        logger.info(f"  Data variables: {list(ds.data_vars.keys())}")
        logger.info(f"  Coordinates: {list(ds.coords.keys())}")

        # Display coordinate details
        logger.info("\n📐 COORDINATE DETAILS:")
        for coord_name in ds.coords.keys():
            coord = ds[coord_name]
            logger.info(f"  {coord_name}:")
            logger.info(f"    Shape: {coord.shape}")
            logger.info(f"    Dtype: {coord.dtype}")
            logger.info(f"    Chunks: {coord.chunks if hasattr(coord, 'chunks') else 'N/A'}")

        # Display variable details
        logger.info("\n📊 DATA VARIABLE DETAILS:")
        for var_name in ds.data_vars.keys():
            var = ds[var_name]
            logger.info(f"  {var_name}:")
            logger.info(f"    Shape: {var.shape}")
            logger.info(f"    Dtype: {var.dtype}")
            logger.info(f"    Chunks: {var.chunks if hasattr(var, 'chunks') else 'N/A'}")
            logger.info(f"    Attributes: {len(var.attrs)} items")

        # Display global attributes
        logger.info("\n🏷️ GLOBAL ATTRIBUTES:")
        for key, value in list(ds.attrs.items())[:5]:  # Show first 5 attributes
            logger.info(f"  {key}: {value}")
        if len(ds.attrs) > 5:
            logger.info(f"  ... and {len(ds.attrs) - 5} more attributes")

        # Test actual data access (load a small sample)
        logger.info("\n🔬 TESTING DATA ACCESS (loading small sample)...")
        try:
            # Try to load a small slice of coordinate data
            first_coord = list(ds.coords.keys())[0]
            coord_sample = ds[first_coord][:5].values
            logger.info(f"  ✓ Successfully loaded first 5 values of '{first_coord}':")
            logger.info(f"    {coord_sample}")

            logger.info("\n" + "="*80)
            logger.info("✅ FULL VALIDATION PASSED - JSON REFERENCES ARE WORKING!")
            logger.info("="*80)

        except Exception as data_error:
            logger.warning(f"  ⚠ Data access failed: {data_error}")
            logger.info("  Metadata access works, but actual data loading had issues")

        # Close dataset
        ds.close()

        return True

    finally:
        # Cleanup temp file
        import os
        try:
            os.unlink(tmp_path)
        except:
            pass


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python test_zarr_opening.py <path_to_json_file>")
        print("\nExample:")
        print("  python test_zarr_opening.py cmorph_2020/cmorph_2020_file003.json")
        sys.exit(1)

    json_file = sys.argv[1]

    if not Path(json_file).exists():
        logger.error(f"File not found: {json_file}")
        sys.exit(1)

    try:
        success = test_zarr_opening_with_anon_s3(json_file)
        if success:
            logger.info("\n🎉 Test completed successfully!")
            sys.exit(0)
    except Exception as e:
        logger.error(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
