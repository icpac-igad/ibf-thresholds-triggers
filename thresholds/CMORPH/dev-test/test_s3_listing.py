#!/usr/bin/env python3
"""
Test script to verify S3 file listing functionality
Tests the modified precipitation processor without BeautifulSoup4
"""

import sys
import logging
from precipitation_processor_architecture import CMORPHProcessor, PERSIANNProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def test_cmorph_listing():
    """Test CMORPH S3 file listing"""
    logger.info("Testing CMORPH S3 file listing for year 1998...")
    
    try:
        processor = CMORPHProcessor(target_year=1998, output_dir="./test_output")
        nc_files = processor.get_nc_files()
        
        logger.info(f"Found {len(nc_files)} CMORPH files for 1998")
        
        # Show first few files as examples
        if nc_files:
            logger.info("First 5 files:")
            for i, file_url in enumerate(nc_files[:5], 1):
                logger.info(f"  {i}. {file_url}")
        
        return len(nc_files) > 0
        
    except Exception as e:
        logger.error(f"CMORPH test failed: {e}")
        return False

def test_persiann_listing():
    """Test PERSIANN S3 file listing"""
    logger.info("Testing PERSIANN S3 file listing for year 1983...")
    
    try:
        processor = PERSIANNProcessor(target_year=1983, output_dir="./test_output")
        nc_files = processor.get_nc_files()
        
        logger.info(f"Found {len(nc_files)} PERSIANN files for 1983")
        
        # Show first few files as examples
        if nc_files:
            logger.info("First 5 files:")
            for i, file_url in enumerate(nc_files[:5], 1):
                logger.info(f"  {i}. {file_url}")
        
        return len(nc_files) > 0
        
    except Exception as e:
        logger.error(f"PERSIANN test failed: {e}")
        return False

def test_virtual_dataset_creation():
    """Test creating a virtual dataset from one file"""
    logger.info("Testing virtual dataset creation...")
    
    try:
        # Test with PERSIANN first (likely smaller files)
        processor = PERSIANNProcessor(target_year=1983, output_dir="./test_output")
        nc_files = processor.get_nc_files()
        
        if not nc_files:
            logger.warning("No NetCDF files found for testing virtual dataset creation")
            return False
        
        # Try to create virtual dataset from first file
        test_file = nc_files[0]
        logger.info(f"Testing virtual dataset creation with: {test_file}")
        
        vds = processor.create_virtual_dataset(test_file)
        
        if vds is not None:
            logger.info(f"✓ Virtual dataset created successfully")
            logger.info(f"  Dimensions: {dict(vds.dims)}")
            logger.info(f"  Data variables: {list(vds.data_vars.keys())}")
            logger.info(f"  Coordinates: {list(vds.coords.keys())}")
            return True
        else:
            logger.error("Failed to create virtual dataset")
            return False
            
    except Exception as e:
        logger.error(f"Virtual dataset creation test failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("Starting S3 listing and VirtualiZarr tests...")
    logger.info("="*60)
    
    test_results = {}
    
    # Test CMORPH listing
    test_results['cmorph_listing'] = test_cmorph_listing()
    logger.info("="*60)
    
    # Test PERSIANN listing  
    test_results['persiann_listing'] = test_persiann_listing()
    logger.info("="*60)
    
    # Test virtual dataset creation
    test_results['virtual_dataset'] = test_virtual_dataset_creation()
    logger.info("="*60)
    
    # Summary
    logger.info("TEST RESULTS SUMMARY:")
    logger.info("-" * 30)
    for test_name, result in test_results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{test_name}: {status}")
    
    # Overall result
    all_passed = all(test_results.values())
    
    if all_passed:
        logger.info("\n🎉 All tests passed! The modified processor is working correctly.")
        sys.exit(0)
    else:
        logger.error("\n❌ Some tests failed. Check the logs above for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()