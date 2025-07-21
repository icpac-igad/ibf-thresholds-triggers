# SPI Processor Documentation

## Overview

The SPI Processor is a modular Python script designed to process Standardized Precipitation Index (SPI) data files and store them efficiently in Icechunk repositories on Google Cloud Storage. It supports multiple SPI timescales (1, 3, 6, 9, 12, 24, and 48 months) and provides a complete workflow for data ingestion, processing, and storage.

## What is Icechunk?

Icechunk is a next-generation storage format for scientific data that provides:
- **Version control** for datasets (like Git for data)
- **Efficient cloud storage** with chunked, compressed data
- **Time-travel capabilities** to access historical versions
- **Transactional commits** ensuring data integrity
- **Zarr compatibility** for seamless integration with scientific Python ecosystem

### Understanding Icechunk Repositories vs Prefixes

**Important Distinction**: In Icechunk, a "prefix" is NOT the same as a "repository":

- **Repository**: A complete Icechunk repository with version control, stored at a specific GCS path
- **Prefix**: The GCS path/folder where the repository is stored

```
GCS Structure:
bucket_name/
└── prefix_path/                    # ← This is the "prefix" 
    ├── .icechunk/                  # ← Repository metadata
    │   ├── config.json
    │   ├── refs/
    │   └── snapshots/
    └── data/                       # ← Zarr data chunks
        └── group_name/
            ├── lat/
            ├── lon/
            └── spi/
```

### Repository Creation vs Opening

The script handles repository lifecycle automatically:

1. **First Run**: Attempts to open existing repository
   ```
   Repository doesn't exist, creating new one...
   ✅ Created new repository
   ```

2. **Subsequent Runs**: Opens existing repository
   ```
   ✅ Opened existing repository
   ```

This is normal behavior - the error message about "repository doesn't exist" is expected when running for the first time with a new prefix.

## Data Architecture

### Repository Structure

Each SPI type creates a separate Icechunk repository with the following structure:

```
GCS Bucket: cdi_arco/
├── t2spi1_east_africa_icechunk_spi3/    # SPI3 repository (complete path = prefix)
│   ├── .icechunk/                       # Version control metadata
│   │   ├── config.json                  # Repository configuration
│   │   ├── refs/                        # Branch references
│   │   │   └── main                     # Main branch pointer
│   │   └── snapshots/                   # Commit snapshots
│   │       ├── abc123.json              # Individual commits
│   │       └── def456.json
│   └── data/                            # Zarr data storage
│       └── spi3_data/                   # Zarr group containing actual data
│           ├── .zarray                  # Zarr metadata
│           ├── .zattrs                  # Attributes
│           ├── lat/                     # Latitude coordinate chunks
│           ├── lon/                     # Longitude coordinate chunks  
│           ├── time/                    # Time coordinate chunks
│           └── spi/                     # SPI values array chunks
├── t2spi1_east_africa_icechunk_spi6/    # SPI6 repository
│   ├── .icechunk/
│   └── data/
│       └── spi6_data/                   # Separate Zarr group for SPI6
└── t2spi1_east_africa_icechunk_spi12/   # SPI12 repository
    ├── .icechunk/
    └── data/
        └── spi12_data/                  # Separate Zarr group for SPI12
```

**Key Points:**
- Each prefix creates a **complete, independent repository**
- The prefix (`t2spi1_east_africa_icechunk_spi3`) is the **full GCS path** to the repository
- Each repository contains **one Zarr group** with SPI data for that timescale
- Repositories are **isolated** - no shared data between SPI types

### Zarr Group Organization

Each SPI type is stored in its own Zarr group with consistent structure:

- **Group name**: `{spi_type}_data` (e.g., `spi1_data`, `spi3_data`)
- **Coordinates**: 
  - `lat`: Latitude values (subset to East Africa: -12° to 23°)
  - `lon`: Longitude values (subset to East Africa: 21° to 53°)
  - `time`: Time dimension (varies by input files)
- **Data variables**: SPI values and associated metadata

## Expected Behavior During Processing

### Normal Repository Lifecycle Messages

When you run the script, you'll see these messages which are **completely normal**:

#### First Time Running (New Prefix)
```bash
python spi_processor.py spi3,spi6 my_new_study

🎯 Processing 2 SPI type(s): SPI3, SPI6
📦 Base prefix: my_new_study
================================================================================
🔄 [1/2] Starting SPI3 processing...
================================================================================
SPI3 East Africa Processing - Local Version
Prefix: my_new_study_spi3
================================================================================
Setting up Icechunk repository...
Repository doesn't exist, creating new one... (  × the repository doesn't exist)
✅ Created new repository
```

**This is expected!** The error message is normal when a repository doesn't exist yet.

#### Subsequent Runs (Existing Prefix)
```bash
python spi_processor.py spi3 my_new_study

Setting up Icechunk repository...
✅ Opened existing repository
```

### Understanding the Messages

- **"Repository doesn't exist"**: Normal for first-time runs with new prefixes
- **"Creating new one"**: Script automatically creates the repository
- **"✅ Created new repository"**: Success! Repository is ready for data
- **"✅ Opened existing repository"**: Repository already exists, ready to append data

### Repository vs Prefix Clarification

```python
# In your example:
base_prefix = "t2spi1_east_africa_icechunk"    # Your input
actual_prefix = "t2spi1_east_africa_icechunk_spi3"  # What gets created

# The actual_prefix becomes the full GCS path:
# gs://cdi_arco/t2spi1_east_africa_icechunk_spi3/
```

**Each SPI type gets its own complete repository** - they don't share storage or version history.

### 1. Repository Initialization

```python
# First file creates the repository and Zarr group
ds_subset.to_zarr(session.store, group="spi1_data", mode='w', consolidated=False)
```

### 2. Data Appending

```python
# Subsequent files append along time dimension
ds_subset.to_zarr(session.store, group="spi1_data", append_dim='time', consolidated=False)
```

### 3. Version Control

Each file processing creates a commit:

```python
commit_message = f"Added {Path(file_path).name} to {spi_type.upper()}"
session.commit(commit_message)
```

## Usage Examples

### Single SPI Type

```bash
# Process SPI1 data
python spi_processor.py spi1 drought_monitoring_2025

# Process SPI12 data with custom bucket
python spi_processor.py spi12 annual_analysis --bucket my_climate_bucket
```

### Multiple SPI Types

```bash
# Process multiple types simultaneously
python spi_processor.py spi1,spi3,spi6 comprehensive_drought_study

# Process all short-term indices
python spi_processor.py spi1,spi3,spi6,spi9 short_term_analysis

# Process long-term indices
python spi_processor.py spi12,spi24,spi48 long_term_climate_trends
```

## Accessing Stored Data

### Opening Different SPI Repositories

```python
import icechunk
import xarray as xr

# Your base prefix from the command
base_prefix = "t2spi1_east_africa_icechunk"

# Each SPI type has its own repository
spi_types = ['spi3', 'spi6', 'spi9', 'spi12', 'spi24', 'spi48']

datasets = {}
for spi_type in spi_types:
    # Construct the actual repository prefix
    repo_prefix = f"{base_prefix}_{spi_type}"
    
    # Open the specific repository
    storage = icechunk.gcs_storage(
        bucket="cdi_arco",
        prefix=repo_prefix,  # e.g., "t2spi1_east_africa_icechunk_spi3"
        service_account_file="your_credentials.json"
    )
    
    repo = icechunk.Repository.open(storage)
    session = repo.readonly_session("main")
    
    # Load the data from the specific Zarr group
    group_name = f"{spi_type}_data"  # e.g., "spi3_data"
    datasets[spi_type] = xr.open_zarr(session.store, group=group_name)
    
    print(f"✅ Loaded {spi_type.upper()}: {datasets[spi_type].sizes}")

# Now you have separate datasets for each SPI type
print("\nAvailable datasets:")
for spi_type, ds in datasets.items():
    print(f"  {spi_type}: {ds.spi.shape} (time, lat, lon)")
```

### Expected Dataset Structure

```python
<xarray.Dataset>
Dimensions:  (lat: 140, lon: 128, time: 360)  # Example dimensions
Coordinates:
  * lat      (lat) float64 -12.0 -11.75 -11.5 ... 22.5 22.75 23.0
  * lon      (lon) float64 21.0 21.25 21.5 ... 52.5 52.75 53.0  
  * time     (time) datetime64[ns] 1990-01-01 ... 2019-12-01
Data variables:
    spi      (time, lat, lon) float32 ...
Attributes:
    title:       Standardized Precipitation Index
    description: SPI calculated for East Africa region
    region:      East Africa (lat: -12 to 23, lon: 21 to 53)
```

## Repository Discovery and Management

### Listing Repositories in a Bucket/Prefix

Unfortunately, **Icechunk doesn't provide a direct method to list all repositories** in a bucket or under a prefix. However, you can use GCS tools to discover repositories:

#### Method 1: Using Google Cloud SDK

```bash
# List all objects under your base prefix
gsutil ls -d gs://cdi_arco/t2spi1_east_africa_icechunk*

# Expected output:
# gs://cdi_arco/t2spi1_east_africa_icechunk_spi3/
# gs://cdi_arco/t2spi1_east_africa_icechunk_spi6/
# gs://cdi_arco/t2spi1_east_africa_icechunk_spi9/
# gs://cdi_arco/t2spi1_east_africa_icechunk_spi12/
# gs://cdi_arco/t2spi1_east_africa_icechunk_spi24/
# gs://cdi_arco/t2spi1_east_africa_icechunk_spi48/
```

#### Method 2: Using Python GCS Client

```python
from google.cloud import storage
import re

def list_spi_repositories(bucket_name, base_prefix, service_account_file):
    """List all SPI repositories under a base prefix"""
    
    # Initialize GCS client
    client = storage.Client.from_service_account_json(service_account_file)
    bucket = client.bucket(bucket_name)
    
    # List all prefixes that start with base_prefix
    prefixes = set()
    blobs = bucket.list_blobs(prefix=base_prefix)
    
    for blob in blobs:
        # Extract the repository prefix (first two path segments)
        path_parts = blob.name.split('/')
        if len(path_parts) >= 2:
            repo_prefix = '/'.join(path_parts[:1])  # Take first segment as repo prefix
            prefixes.add(repo_prefix)
    
    # Filter for SPI repositories
    spi_repos = []
    spi_pattern = re.compile(r'.*_spi\d+

## Version Control Operations

### Listing Commits in a Repository

```python
import icechunk

# Open specific repository  
storage = icechunk.gcs_storage(
    bucket="cdi_arco",
    prefix="t2spi1_east_africa_icechunk_spi3",
    service_account_file="coiled-data-e4drr_202505.json"
)
repo = icechunk.Repository.open(storage)

# Get commit history
commits = list(repo.ancestry("main"))
print(f"Repository has {len(commits)} commits:")

for i, commit in enumerate(commits[:10]):  # Show last 10 commits
    print(f"\n{i+1}. Commit: {commit.id[:8]}...")
    print(f"   Message: {commit.message}")
    print(f"   Date: {commit.timestamp}")
    
# Example output:
# 1. Commit: abc12345...
#    Message: Added spi3_file045.json to SPI3
#    Date: 2025-01-15T10:30:00Z
#
# 2. Commit: def67890...
#    Message: Added spi3_file044.json to SPI3  
#    Date: 2025-01-15T10:29:15Z
```

### Accessing Historical Versions

```python
# Open specific commit
commit_id = "abc123..."  # From commit list
session = repo.readonly_session(commit_id)
historical_ds = xr.open_zarr(session.store, group="spi1_data")
```

### Branching and Merging

```python
# Create a new branch
session = repo.writable_session("main")
session.branch("experimental_processing")

# Work on branch
# ... make changes ...
session.commit("Experimental data processing")

# Merge back to main (if needed)
main_session = repo.writable_session("main")
main_session.merge("experimental_processing")
```

## Repository Management

### Checking Repository Status

```python
# Check repository info
print(f"Repository path: {repo.path}")
print(f"Current branch: main")

# Check storage usage
# (Note: Specific storage info methods depend on Icechunk version)
```

### Multiple Repository Access

```python
# Access different SPI types from same processing run
prefixes = ["drought_study_spi1", "drought_study_spi3", "drought_study_spi6"]

datasets = {}
for prefix in prefixes:
    spi_type = prefix.split('_')[-1]  # Extract spi1, spi3, etc.
    
    storage = icechunk.gcs_storage(
        bucket="cdi_arco",
        prefix=prefix,
        service_account_file="credentials.json"
    )
    
    repo = icechunk.Repository.open(storage)
    session = repo.readonly_session("main")
    
    datasets[spi_type] = xr.open_zarr(session.store, group=f"{spi_type}_data")

# Now you have datasets['spi1'], datasets['spi3'], datasets['spi6']
```

## Data Analysis Examples

### Basic Data Exploration

```python
# Load SPI1 data
ds = datasets['spi1']

# Check data range
print(f"Time range: {ds.time.min().values} to {ds.time.max().values}")
print(f"Spatial extent: lat {ds.lat.min().values} to {ds.lat.max().values}")
print(f"SPI range: {ds.spi.min().values} to {ds.spi.max().values}")

# Calculate statistics
mean_spi = ds.spi.mean(dim='time')
std_spi = ds.spi.std(dim='time')
```

### Combining Multiple SPI Types

```python
# Combine different SPI timescales for analysis
spi_combined = xr.Dataset({
    'spi1': datasets['spi1'].spi,
    'spi3': datasets['spi3'].spi,
    'spi6': datasets['spi6'].spi
})

# Calculate correlations between different timescales
correlation = xr.corr(spi_combined.spi1, spi_combined.spi3, dim='time')
```

## Error Handling and Recovery

### Common Issues and Solutions

1. **Repository already exists**:
   ```python
   # Script automatically tries to open existing repository first
   # If creation fails, it attempts to open existing one
   ```

2. **Partial processing failures**:
   ```python
   # Script continues processing remaining files
   # Failed files are logged but don't stop the process
   ```

3. **Memory issues with large files**:
   ```python
   # Script uses explicit garbage collection
   # Processes files sequentially to manage memory
   ```

### Recovery from Interrupted Processing

```python
# Check last commit to see what was processed
commits = list(repo.ancestry("main"))
last_commit = commits[0]
print(f"Last processed: {last_commit.message}")

# Resume processing from specific file if needed
# (Manual identification of last processed file from commit messages)
```

## Performance Considerations

### Storage Efficiency

- **Chunking**: Data is automatically chunked for efficient access
- **Compression**: Zarr provides built-in compression
- **Regional subsetting**: Only East Africa data is stored, reducing size by ~80%

### Processing Speed

- **Sequential processing**: Prevents memory overflow
- **Batch processing**: Can process multiple SPI types in one run
- **Local computation**: Subsetting computed locally before upload

### Network Optimization

- **Minimal transfers**: Only processed, subset data uploaded
- **Efficient protocols**: Uses optimized GCS storage protocols
- **Credential caching**: Service account credentials cached per session

## Best Practices

### File Organization

```
project/
├── spi_processor.py           # Main script
├── credentials.json           # Service account file
├── spi1/                     # SPI1 input files
│   ├── spi1_file001.json
│   └── spi1_file002.json
├── spi3/                     # SPI3 input files
└── logs/                     # Processing logs (optional)
```

### Naming Conventions

- **Repository prefixes**: Use descriptive names like `drought_monitoring_2025`
- **Commit messages**: Automatically include filename and SPI type
- **Branch names**: Use descriptive names for experimental work

### Data Validation

```python
# Validate data after processing
ds = xr.open_zarr(session.store, group="spi1_data")

# Check for expected variables
assert 'spi' in ds.data_vars
assert 'lat' in ds.coords
assert 'lon' in ds.coords
assert 'time' in ds.coords

# Validate spatial bounds
assert ds.lat.min() >= -12
assert ds.lat.max() <= 23
assert ds.lon.min() >= 21
assert ds.lon.max() <= 53
```

## Troubleshooting

### Common Error Messages

1. **"Service account file not found"**:
   - Ensure `coiled-data-e4drr_202505.json` is in working directory
   - Or specify custom path with `--service-account`

2. **"No SPI files found"**:
   - Check directory structure matches expected pattern
   - Ensure files are named `spi1_file*.json`, etc.

3. **"Repository creation failed"**:
   - Check GCS permissions
   - Verify bucket exists and is accessible
   - Confirm service account has write permissions

### Debug Mode

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with verbose output
python spi_processor.py spi1 test_prefix --service-account debug_credentials.json
```

This documentation provides a comprehensive guide to understanding, using, and managing SPI data with the Icechunk-based processing system.)
    
    for prefix in prefixes:
        if spi_pattern.match(prefix):
            spi_repos.append(prefix)
    
    return sorted(spi_repos)

# Usage
repos = list_spi_repositories(
    bucket_name="cdi_arco",
    base_prefix="t2spi1_east_africa_icechunk",
    service_account_file="coiled-data-e4drr_202505.json"
)

print("Found SPI repositories:")
for repo in repos:
    spi_type = repo.split('_')[-1]  # Extract spi3, spi6, etc.
    print(f"  - {repo} ({spi_type.upper()})")
```

#### Method 3: Convention-Based Discovery

Since our script follows a predictable naming pattern, you can programmatically construct repository names:

```python
def discover_spi_repositories(base_prefix, bucket="cdi_arco", service_account_file="coiled-data-e4drr_202505.json"):
    """Discover SPI repositories by attempting to open each expected type"""
    import icechunk
    
    spi_types = ['spi1', 'spi3', 'spi6', 'spi9', 'spi12', 'spi24', 'spi48']
    available_repos = {}
    
    for spi_type in spi_types:
        repo_prefix = f"{base_prefix}_{spi_type}"
        
        try:
            # Try to open the repository
            storage = icechunk.gcs_storage(
                bucket=bucket,
                prefix=repo_prefix,
                service_account_file=service_account_file
            )
            repo = icechunk.Repository.open(storage)
            
            # Get basic info
            session = repo.readonly_session("main")
            commits = list(repo.ancestry("main"))
            
            available_repos[spi_type] = {
                'prefix': repo_prefix,
                'commit_count': len(commits),
                'latest_commit': commits[0].message if commits else 'No commits',
                'repository': repo
            }
            
        except Exception as e:
            # Repository doesn't exist or can't be accessed
            continue
    
    return available_repos

# Usage
repos = discover_spi_repositories("t2spi1_east_africa_icechunk")

print(f"Found {len(repos)} SPI repositories:")
for spi_type, info in repos.items():
    print(f"\n📊 {spi_type.upper()}:")
    print(f"   Prefix: {info['prefix']}")
    print(f"   Commits: {info['commit_count']}")
    print(f"   Latest: {info['latest_commit']}")
```

### Listing Groups Within a Repository

Once you have a repository open, you can list the Zarr groups:

```python
import icechunk
import zarr

# Open a specific repository
storage = icechunk.gcs_storage(
    bucket="cdi_arco",
    prefix="t2spi1_east_africa_icechunk_spi3",
    service_account_file="coiled-data-e4drr_202505.json"
)

repo = icechunk.Repository.open(storage)
session = repo.readonly_session("main")

# Open as Zarr store to inspect structure
zarr_store = zarr.open(session.store, mode='r')

print("Available groups in repository:")
def list_groups(group, prefix=""):
    for key in group.keys():
        full_path = f"{prefix}/{key}" if prefix else key
        item = group[key]
        
        if hasattr(item, 'keys'):  # It's a group
            print(f"📁 Group: {full_path}")
            list_groups(item, full_path)  # Recursive for nested groups
        else:  # It's an array
            print(f"📊 Array: {full_path} {item.shape} {item.dtype}")

list_groups(zarr_store)

# Expected output for SPI repository:
# 📁 Group: spi3_data
# 📊 Array: spi3_data/lat (140,) float64
# 📊 Array: spi3_data/lon (128,) float64  
# 📊 Array: spi3_data/time (360,) datetime64[ns]
# 📊 Array: spi3_data/spi (360, 140, 128) float32
```

### Complete Repository Explorer Function

```python
def explore_spi_repositories(base_prefix, bucket="cdi_arco", service_account_file="coiled-data-e4drr_202505.json"):
    """Complete exploration of all SPI repositories under a base prefix"""
    import icechunk
    import xarray as xr
    import zarr
    
    print(f"🔍 Exploring repositories with base prefix: {base_prefix}")
    print("=" * 80)
    
    spi_types = ['spi1', 'spi3', 'spi6', 'spi9', 'spi12', 'spi24', 'spi48']
    
    for spi_type in spi_types:
        repo_prefix = f"{base_prefix}_{spi_type}"
        
        try:
            # Open repository
            storage = icechunk.gcs_storage(
                bucket=bucket,
                prefix=repo_prefix,
                service_account_file=service_account_file
            )
            repo = icechunk.Repository.open(storage)
            session = repo.readonly_session("main")
            
            print(f"\n📦 Repository: {repo_prefix}")
            
            # Get commit info
            commits = list(repo.ancestry("main"))
            print(f"   📝 Commits: {len(commits)}")
            if commits:
                print(f"   📅 Latest: {commits[0].message}")
                print(f"   🕐 Date: {commits[0].timestamp}")
            
            # Get Zarr structure
            zarr_store = zarr.open(session.store, mode='r')
            print(f"   📁 Groups: {list(zarr_store.keys())}")
            
            # Get data info
            group_name = f"{spi_type}_data"
            if group_name in zarr_store:
                group = zarr_store[group_name]
                print(f"   📊 Arrays in {group_name}:")
                for key in group.keys():
                    array = group[key]
                    print(f"      - {key}: {array.shape} {array.dtype}")
                
                # Quick data summary
                if 'spi' in group:
                    spi_array = group['spi']
                    print(f"   📈 SPI data shape: {spi_array.shape}")
                    print(f"   📊 SPI range: {spi_array[:].min():.2f} to {spi_array[:].max():.2f}")
            
        except Exception as e:
            print(f"\n❌ {repo_prefix}: Not found or inaccessible")
            continue
    
    print("\n" + "=" * 80)

# Usage
explore_spi_repositories("t2spi1_east_africa_icechunk")
```

### Repository Management Summary

| Task | Method | Notes |
|------|--------|-------|
| **List all repositories** | GCS client or gsutil | No direct Icechunk method |
| **Check if repo exists** | Try to open with try/catch | Most reliable approach |
| **List groups in repo** | Zarr.open(session.store) | Access Zarr structure |
| **Get repository info** | repo.ancestry("main") | Commit history and metadata |
| **Bulk discovery** | Convention-based iteration | Use known SPI naming pattern |

**Key Limitation**: Icechunk doesn't provide repository discovery APIs, so you need to use GCS-level tools or follow naming conventions to find repositories.

### Listing Commits

```python
# Get commit history
commits = list(repo.ancestry("main"))
for commit in commits[:10]:  # Show last 10 commits
    print(f"Commit: {commit.id}")
    print(f"Message: {commit.message}")
    print(f"Timestamp: {commit.timestamp}")
    print("---")
```

### Accessing Historical Versions

```python
# Open specific commit
commit_id = "abc123..."  # From commit list
session = repo.readonly_session(commit_id)
historical_ds = xr.open_zarr(session.store, group="spi1_data")
```

### Branching and Merging

```python
# Create a new branch
session = repo.writable_session("main")
session.branch("experimental_processing")

# Work on branch
# ... make changes ...
session.commit("Experimental data processing")

# Merge back to main (if needed)
main_session = repo.writable_session("main")
main_session.merge("experimental_processing")
```

## Repository Management

### Checking Repository Status

```python
# Check repository info
print(f"Repository path: {repo.path}")
print(f"Current branch: main")

# Check storage usage
# (Note: Specific storage info methods depend on Icechunk version)
```

### Multiple Repository Access

```python
# Access different SPI types from same processing run
prefixes = ["drought_study_spi1", "drought_study_spi3", "drought_study_spi6"]

datasets = {}
for prefix in prefixes:
    spi_type = prefix.split('_')[-1]  # Extract spi1, spi3, etc.
    
    storage = icechunk.gcs_storage(
        bucket="cdi_arco",
        prefix=prefix,
        service_account_file="credentials.json"
    )
    
    repo = icechunk.Repository.open(storage)
    session = repo.readonly_session("main")
    
    datasets[spi_type] = xr.open_zarr(session.store, group=f"{spi_type}_data")

# Now you have datasets['spi1'], datasets['spi3'], datasets['spi6']
```

## Data Analysis Examples

### Basic Data Exploration

```python
# Load SPI1 data
ds = datasets['spi1']

# Check data range
print(f"Time range: {ds.time.min().values} to {ds.time.max().values}")
print(f"Spatial extent: lat {ds.lat.min().values} to {ds.lat.max().values}")
print(f"SPI range: {ds.spi.min().values} to {ds.spi.max().values}")

# Calculate statistics
mean_spi = ds.spi.mean(dim='time')
std_spi = ds.spi.std(dim='time')
```

### Combining Multiple SPI Types

```python
# Combine different SPI timescales for analysis
spi_combined = xr.Dataset({
    'spi1': datasets['spi1'].spi,
    'spi3': datasets['spi3'].spi,
    'spi6': datasets['spi6'].spi
})

# Calculate correlations between different timescales
correlation = xr.corr(spi_combined.spi1, spi_combined.spi3, dim='time')
```

## Error Handling and Recovery

### Common Issues and Solutions

1. **Repository already exists**:
   ```python
   # Script automatically tries to open existing repository first
   # If creation fails, it attempts to open existing one
   ```

2. **Partial processing failures**:
   ```python
   # Script continues processing remaining files
   # Failed files are logged but don't stop the process
   ```

3. **Memory issues with large files**:
   ```python
   # Script uses explicit garbage collection
   # Processes files sequentially to manage memory
   ```

### Recovery from Interrupted Processing

```python
# Check last commit to see what was processed
commits = list(repo.ancestry("main"))
last_commit = commits[0]
print(f"Last processed: {last_commit.message}")

# Resume processing from specific file if needed
# (Manual identification of last processed file from commit messages)
```

## Performance Considerations

### Storage Efficiency

- **Chunking**: Data is automatically chunked for efficient access
- **Compression**: Zarr provides built-in compression
- **Regional subsetting**: Only East Africa data is stored, reducing size by ~80%

### Processing Speed

- **Sequential processing**: Prevents memory overflow
- **Batch processing**: Can process multiple SPI types in one run
- **Local computation**: Subsetting computed locally before upload

### Network Optimization

- **Minimal transfers**: Only processed, subset data uploaded
- **Efficient protocols**: Uses optimized GCS storage protocols
- **Credential caching**: Service account credentials cached per session

## Best Practices

### File Organization

```
project/
├── spi_processor.py           # Main script
├── credentials.json           # Service account file
├── spi1/                     # SPI1 input files
│   ├── spi1_file001.json
│   └── spi1_file002.json
├── spi3/                     # SPI3 input files
└── logs/                     # Processing logs (optional)
```

### Naming Conventions

- **Repository prefixes**: Use descriptive names like `drought_monitoring_2025`
- **Commit messages**: Automatically include filename and SPI type
- **Branch names**: Use descriptive names for experimental work

### Data Validation

```python
# Validate data after processing
ds = xr.open_zarr(session.store, group="spi1_data")

# Check for expected variables
assert 'spi' in ds.data_vars
assert 'lat' in ds.coords
assert 'lon' in ds.coords
assert 'time' in ds.coords

# Validate spatial bounds
assert ds.lat.min() >= -12
assert ds.lat.max() <= 23
assert ds.lon.min() >= 21
assert ds.lon.max() <= 53
```

## Troubleshooting

### Common Error Messages

1. **"Service account file not found"**:
   - Ensure `coiled-data-e4drr_202505.json` is in working directory
   - Or specify custom path with `--service-account`

2. **"No SPI files found"**:
   - Check directory structure matches expected pattern
   - Ensure files are named `spi1_file*.json`, etc.

3. **"Repository creation failed"**:
   - Check GCS permissions
   - Verify bucket exists and is accessible
   - Confirm service account has write permissions

### Debug Mode

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with verbose output
python spi_processor.py spi1 test_prefix --service-account debug_credentials.json
```

This documentation provides a comprehensive guide to understanding, using, and managing SPI data with the Icechunk-based processing system.
