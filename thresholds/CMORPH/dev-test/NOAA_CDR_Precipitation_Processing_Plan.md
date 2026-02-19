# NOAA CDR Precipitation Datasets Processing Plan
## VirtualiZarr JSON Metadata Generation for CMORPH and PERSIANN

### Project Overview
This project focuses on processing two NOAA Climate Data Record (CDR) precipitation datasets using the VirtualiZarr Python library to generate JSON metadata files for data streaming capabilities. The goal is to create Kerchunk reference files that enable efficient cloud-native data access without downloading the entire NetCDF files.

### Datasets

#### Dataset 1: CMORPH (Climate Prediction Center MORPHing technique)
- **Location**: `https://noaa-cdr-precip-cmorph-pds.s3.amazonaws.com/index.html#data/30min/8km/`
- **Description**: High-resolution satellite-based precipitation estimates
- **Temporal Resolution**: 30-minute intervals
- **Spatial Resolution**: 8km
- **Target Year**: 1983 (initial focus)
- **Data Format**: NetCDF files
- **Expected Structure**: 
  - Directory: `/data/30min/8km/YYYY/`
  - Files: Daily or sub-daily NetCDF files with precipitation data

#### Dataset 2: PERSIANN (Precipitation Estimation from Remotely Sensed Information using Artificial Neural Networks)
- **Location**: `https://noaa-cdr-precip-persiann-pds.s3.amazonaws.com/index.html#data/`
- **Description**: Neural network-based precipitation estimates from satellite data
- **Target Year**: 1983 (initial focus)
- **Data Format**: NetCDF files
- **Expected Structure**: 
  - Directory: `/data/YYYY/`
  - Files: Regular temporal NetCDF files with precipitation estimates

### Processing Architecture

Based on the sample script `virtualizarr_gdo_spi_processor_json.py`, we will adapt the approach for precipitation datasets:

#### Core Components

1. **PrecipitationProcessor Class**
   - Extends the SPIProcessor pattern for precipitation data
   - Handles AWS S3 bucket navigation
   - Manages NetCDF file discovery and processing

2. **Dataset-Specific Processors**
   - `CMORPHProcessor`: Handles 30-minute, 8km resolution data
   - `PERSIANNProcessor`: Handles PERSIANN-specific data structure

3. **VirtualiZarr Integration**
   - Uses `open_virtual_dataset()` for memory-efficient processing
   - Generates Kerchunk references without loading full datasets
   - Handles chunking and concatenation issues

### Processing Phases

#### Phase 1: Initial Setup and Testing (1983 Data)

**CMORPH Processing (1983)**
1. **Directory Discovery**
   - Navigate to `/data/30min/8km/1983/`
   - Identify all NetCDF files for 1983
   - Extract file naming patterns and temporal coverage

2. **Individual File Processing**
   - Process each NetCDF file individually (following sample script pattern)
   - Generate individual JSON reference files
   - Avoid concatenation to prevent chunking conflicts

3. **JSON Generation**
   - Create pure Kerchunk reference files for each NetCDF
   - Store in organized directory structure: `/output/cmorph_1983/`
   - Naming convention: `cmorph_1983_file001.json`, `cmorph_1983_file002.json`, etc.

**PERSIANN Processing (1983)**
1. **Directory Discovery**
   - Navigate to `/data/1983/`
   - Identify all NetCDF files for 1983
   - Extract file naming patterns and temporal coverage

2. **Individual File Processing**
   - Process each NetCDF file using VirtualiZarr
   - Generate individual JSON reference files
   - Handle PERSIANN-specific metadata and structure

3. **JSON Generation**
   - Create Kerchunk reference files
   - Store in organized directory: `/output/persiann_1983/`
   - Naming convention: `persiann_1983_file001.json`, `persiann_1983_file002.json`, etc.

#### Phase 2: JSON Validation and Sanity Checks

**Validation Steps**
1. **File Integrity Checks**
   - Verify JSON structure and format
   - Check Kerchunk reference validity
   - Ensure all required metadata is present

2. **Data Access Testing**
   - Load JSON references back into xarray
   - Verify data can be accessed without downloading original NetCDF
   - Test chunking and data slicing operations

3. **Metadata Verification**
   - Compare original NetCDF metadata with JSON references
   - Verify coordinate systems and variable attributes
   - Check temporal and spatial bounds

4. **Performance Benchmarks**
   - Measure JSON file sizes vs original NetCDF files
   - Test data access speed using JSON references
   - Validate memory efficiency gains

#### Phase 3: Scaling Strategy for Multi-Year Processing

**Scaling Approach**
1. **Parallel Processing Framework**
   - Implement multiprocessing for concurrent file handling
   - Use batch processing for large numbers of NetCDF files
   - Configure memory-efficient processing pipelines

2. **Year-by-Year Expansion**
   - After 1983 validation, expand to 1984-1990
   - Then scale to full dataset coverage (1998-present for CMORPH, 1983-present for PERSIANN)
   - Monitor and optimize processing time and resource usage

3. **Storage Organization**
   - Hierarchical JSON storage by year and month
   - Summary files for dataset catalogs
   - Efficient indexing for rapid data discovery

### Implementation Details

#### Key Processing Parameters
```python
# VirtualiZarr configuration (based on sample script)
vds = open_virtual_dataset(
    url,
    indexes={},
    loadable_variables=[],  # Minimal loading for speed
    decode_times=False      # Avoid temporal parsing issues
)
```

#### Error Handling Strategy
1. **Robust File Processing**
   - Continue processing on individual file failures
   - Log detailed error messages and failed files
   - Generate summary reports for troubleshooting

2. **Network Resilience**
   - Implement retry logic for S3 access
   - Handle timeout and connection issues
   - Resume processing from checkpoint on failure

#### Output Structure
```
output/
├── cmorph_1983/
│   ├── cmorph_1983_file001.json
│   ├── cmorph_1983_file002.json
│   ├── ...
│   └── cmorph_1983_summary.json
├── persiann_1983/
│   ├── persiann_1983_file001.json
│   ├── persiann_1983_file002.json
│   ├── ...
│   └── persiann_1983_summary.json
└── validation_results/
    ├── cmorph_validation_report.json
    └── persiann_validation_report.json
```

### Success Criteria

#### Phase 1 Success Metrics
- Successfully process ≥90% of 1983 NetCDF files for both datasets
- Generate valid Kerchunk JSON references for all processed files
- JSON file size reduction of ≥95% compared to original NetCDF files

#### Phase 2 Success Metrics
- 100% of generated JSON files pass validation tests
- Data access through JSON references matches original NetCDF data
- Performance benchmarks show ≥10x improvement in data discovery time

#### Phase 3 Success Metrics
- Scalable processing pipeline capable of handling full dataset archives
- Processing time scales linearly with number of files
- Automated quality control and error reporting system

### Technical Requirements

#### Dependencies
```python
- xarray
- virtualizarr
- kerchunk
- boto3 (for S3 access)
- requests
- beautifulsoup4 (for directory listing)
- json
- pathlib
```

#### Resource Requirements
- Memory: 8-16 GB RAM for efficient processing
- Storage: Sufficient space for JSON outputs (estimated 1-5% of original data size)
- Network: Stable high-bandwidth connection for S3 access

### Risk Mitigation

#### Potential Challenges
1. **Large Dataset Size**: CMORPH 30-minute data generates many files per day
2. **Network Latency**: S3 access speed may vary
3. **Metadata Complexity**: Precipitation datasets may have complex coordinate systems
4. **Chunking Issues**: VirtualiZarr concatenation challenges (addressed by individual file approach)

#### Mitigation Strategies
1. **Batch Processing**: Process files in manageable chunks
2. **Caching**: Implement local caching for frequently accessed metadata
3. **Monitoring**: Real-time processing status and error tracking
4. **Checkpointing**: Save progress to enable resume on failure

### Next Steps
1. Implement dataset-specific processors based on the sample script pattern
2. Test initial processing on small subset of 1983 data
3. Validate JSON outputs and refine processing parameters
4. Scale to full 1983 dataset processing
5. Implement validation and sanity check frameworks
6. Design and implement multi-year scaling strategy

This plan provides a comprehensive roadmap for generating JSON metadata files from NOAA CDR precipitation datasets, enabling efficient cloud-native data streaming while maintaining data integrity and accessibility.