import xarray as xr
from google.oauth2 import service_account
import fsspec
# Load dataset (assuming it's already loaded as 'db0')
# or load as in the previous steps
# db0 = xr.open_dataset("seas51-af-20241012.grib", engine="cfgrib")
# GCS path
ds=xr.open_dataset('/data/chirps-v2.0.monthly.nc')

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

# Create an fsspec filesystem with GCS using the service account
#fs = fsspec.filesystem("gcs", token=credentials)
# Save to Zarr in GCS
chunk_sizes = {
    "time": 52,
    "latitude": 400,
    "longitude": 600,
}
gcs_zarr_path = "gs://seas51/chirps_v2_monthly_20241012.zarr"
ds.chunk(chunk_sizes).to_zarr(store=gcs_zarr_path, mode="w", storage_options={"token": credentials})
