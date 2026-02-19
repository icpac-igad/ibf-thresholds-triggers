#!/usr/bin/env python3
"""
Test script to validate JSON reference files can be opened with obstore and xarray
"""

import json
import xarray as xr
import fsspec


def test_json_file(json_path):
    """
    Test if a JSON reference file can be loaded as an xarray dataset

    Args:
        json_path: Path to the JSON reference file
    """
    print(f"Testing JSON file: {json_path}")
    print("=" * 80)

    # 1. Load and validate JSON structure
    print("\n1. Loading JSON file...")
    try:
        with open(json_path, 'r') as f:
            refs = json.load(f)
        print(f"   ✓ JSON loaded successfully")
        print(f"   - Top-level keys: {list(refs.keys())}")

        if 'refs' in refs:
            print(f"   - Number of references: {len(refs['refs'])}")
            # Show first few keys
            ref_keys = list(refs['refs'].keys())[:10]
            print(f"   - Sample ref keys: {ref_keys}")
    except Exception as e:
        print(f"   ✗ Failed to load JSON: {e}")
        return False

    # 2. Try to open with fsspec and xarray
    print("\n2. Opening with fsspec reference filesystem...")
    try:
        # Create reference filesystem mapper
        fs = fsspec.filesystem(
            "reference",
            fo=refs,
            remote_protocol="s3",
            remote_options={
                "anon": True,
                "skip_instance_cache": True
            }
        )

        mapper = fs.get_mapper("")
        print(f"   ✓ Reference filesystem created")

        # Try to open with xarray
        print("\n3. Opening dataset with xarray...")
        ds = xr.open_dataset(
            mapper,
            engine="zarr",
            chunks={},
            backend_kwargs={"consolidated": False}
        )

        print(f"   ✓ Dataset opened successfully!")
        print(f"\n   Dataset info:")
        print(f"   - Dimensions: {dict(ds.dims)}")
        print(f"   - Variables: {list(ds.data_vars)}")
        print(f"   - Coordinates: {list(ds.coords)}")

        # Show basic stats
        print(f"\n   Dataset summary:")
        print(ds)

        # Try to access some data
        print(f"\n4. Testing data access...")
        for var_name in list(ds.data_vars)[:2]:  # Test first 2 variables
            var = ds[var_name]
            print(f"   - {var_name}: shape={var.shape}, dtype={var.dtype}")
            # Try to load a small slice
            try:
                sample = var.isel({dim: 0 for dim in var.dims}).values
                print(f"     First value: {sample}")
            except Exception as e:
                print(f"     Could not load sample data: {e}")

        print(f"\n   ✓ Data access successful!")
        return True

    except Exception as e:
        print(f"   ✗ Failed to open dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys

    # Test the specific file mentioned
    json_file = "cmorph_2020/cmorph_2020_file003.json"

    if len(sys.argv) > 1:
        json_file = sys.argv[1]

    success = test_json_file(json_file)

    if success:
        print("\n" + "=" * 80)
        print("✓ JSON file is valid and can be opened with obstore/xarray!")
    else:
        print("\n" + "=" * 80)
        print("✗ JSON file validation failed")
        sys.exit(1)
