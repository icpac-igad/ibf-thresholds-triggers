#!/usr/bin/env python3
"""
Debug script to understand obstore.list() return format
"""

from obstore.store import S3Store

# Create store for a specific day to test
store = S3Store(
    bucket_name='noaa-cdr-precip-cmorph-pds',
    prefix="data/30min/8km/1998/01/01/",
    region="us-east-1", 
    skip_signature=True
)

print("Testing store.list() output format...")

try:
    result = store.list()
    print(f"Type of result: {type(result)}")
    
    # Convert to list
    items = list(result)
    print(f"Number of items: {len(items)}")
    
    if items:
        print(f"Type of first item: {type(items[0])}")
        print(f"First item: {items[0]}")
        
        # Show first 5 items
        print("\nFirst 5 items:")
        for i, item in enumerate(items[:5]):
            print(f"  {i+1}. {item} (type: {type(item)})")
            
        # Check if items have attributes
        first_item = items[0]
        print(f"\nFirst item attributes: {dir(first_item)}")
        
        # Try to access path or key attribute
        if hasattr(first_item, 'path'):
            print(f"First item path: {first_item.path}")
        if hasattr(first_item, 'key'):
            print(f"First item key: {first_item.key}")
        if hasattr(first_item, 'location'):
            print(f"First item location: {first_item.location}")
        
        # Try string conversion
        print(f"String representation: {str(first_item)}")
            
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()