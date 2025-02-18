import xarray as xr
from google.oauth2 import service_account
import fsspec
import numpy as np
# Load dataset (assuming it's already loaded as 'db0')
# or load as in the previous steps
# db0 = xr.open_dataset("seas51-af-20241012.grib", engine="cfgrib")
# GCS path
ds=xr.open_dataset('/data/chirps-v2.0.monthly.nc')

def calculate_chunk_size_in_bytes(ds, chunk_sizes):
    # Calculate total dataset size in bytes and bytes per element
    total_size = sum(v.nbytes for v in ds.values())  # Total size in bytes
    total_elements = sum(v.size for v in ds.values())  # Total number of elements
    bytes_per_element = total_size / total_elements

    # Calculate number of elements in each chunk
    chunk_elements = np.prod([min(ds.sizes[dim], chunk) for dim, chunk in chunk_sizes.items()])

    # Calculate chunk size in bytes
    chunk_size_bytes = chunk_elements * bytes_per_element

    return chunk_size_bytes

# Example usage
chunk_size = {
    'time': 52,  # About 1/10th of the time dimension
    'latitude': 200,  # 1/10th of the latitude dimension
    'longitude': 720  # 1/10th of the longitude dimension
}

chunk_sizes = {
    "time": 52,
    "latitude": 400,
    "longitude": 600,
}

chunk_sizes = {
    "latitude": 100,
    "longitude": 100,
}



# Assuming `ds` is your xarray dataset
estimated_chunk_size_bytes = calculate_chunk_size_in_bytes(ds, chunk_sizes)
print(f"Estimated chunk size: {estimated_chunk_size_bytes / (1024 * 1024):.2f} MB")


gcs_zarr_path = "gs://seas51/chirps_v2_monthly_20241012.zarr"  # Replace with your actual bucket name
# Define chunk sizes with latitude and longitude every 2 points
# Define path to your credentials JSON file
credentials_path = "../coiled-data-key.json"
# Path to your GCS credentials JSON file
#credentials_path = "coiled-data-key.json"
# Specify the correct GCS scope
scopes = ["https://www.googleapis.com/auth/devstorage.read_write"]
# Create credentials object with the required scope
credentials = service_account.Credentials.from_service_account_file(
    credentials_path, scopes=scopes
)

gcs_zarr_path = "gs://seas51/chirps_v2_monthly_20241012.zarr"
ds.chunk(chunk_sizes).to_zarr(store=gcs_zarr_path, mode="w", storage_options={"token": credentials})
