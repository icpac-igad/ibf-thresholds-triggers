#!/usr/bin/env python3
"""
Check how VirtualiZarr opens JSON files and displays xarray dataset output
"""

from virtualizarr import open_virtual_dataset

# Open the JSON file
json_file = "cmorph_2020/cmorph_2020_file003.json"

print(f"Opening: {json_file}")
print("=" * 80)

try:
    # Open dataset using xarray with kerchunk engine (as in validation_framework.py)
    import xarray as xr
    vds = xr.open_dataset(json_file, engine='kerchunk')

    print("\n📊 XARRAY DATASET OUTPUT:")
    print("=" * 80)
    print(vds)

    print("\n\n📋 DATASET INFO:")
    print("=" * 80)
    print(f"Dimensions: {dict(vds.dims)}")
    print(f"Coordinates: {list(vds.coords)}")
    print(f"Data variables: {list(vds.data_vars)}")

    print("\n\n🔍 ATTRIBUTES:")
    print("=" * 80)
    for key, value in vds.attrs.items():
        print(f"  {key}: {value}")

    print("\n\n📐 VARIABLE DETAILS:")
    print("=" * 80)
    for var_name in vds.data_vars:
        var = vds[var_name]
        print(f"\n{var_name}:")
        print(f"  Shape: {var.shape}")
        print(f"  Dims: {var.dims}")
        print(f"  Dtype: {var.dtype}")
        print(f"  Chunks: {var.chunks}")

    print("\n\n✅ SUCCESS: File opened successfully with VirtualiZarr")

except Exception as e:
    print(f"\n❌ ERROR: {type(e).__name__}")
    print(f"Message: {str(e)}")
    import traceback
    print("\nFull traceback:")
    traceback.print_exc()
