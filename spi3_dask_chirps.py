import dask
from dask.distributed import Client
import xarray as xr
import fsspec
from google.oauth2 import service_account
from xclim.indices import standardized_precipitation_index
import coiled

def calculate_spi3_chirps():
    # Connect to Coiled cluster
    cluster = coiled.Cluster(
        n_workers=10,
        worker_memory="16GiB",
        worker_cpu=4,
        name="spi3-chirps-calculation"
    )
    client = Client(cluster)
    
    # GCS credentials
    credentials = service_account.Credentials.from_service_account_file(
        "../../coiled-data-key.json",
        scopes=["https://www.googleapis.com/auth/devstorage.read_write"]
    )
    
    # Open CHIRPS dataset from GCS
    gcs_zarr_path = "gs://seas51/chirps_v2_monthly_20241012.zarr"
    mapper = fsspec.get_mapper(gcs_zarr_path, token=credentials)
    
    # Open with Dask chunks
    ds = xr.open_zarr(
        mapper,
        chunks={"time": 120, "latitude": 100, "longitude": 100},  # Adjust based on cluster size
        consolidated=False
    )
    
    # Calculate SPI-3 using Dask
    print("Calculating SPI-3...")
    spi3 = standardized_precipitation_index(
        ds["precip"],
        freq="MS",
        window=3,
        dist="gamma",
        method="APP",
        cal_start="1991-01-01",
        cal_end="2018-01-01",
        fitkwargs={"floc": 0}
    )
    
    # Write results back to GCS
    output_path = "gs://seas51/chirps_spi3_5km.zarr"
    print(f"Writing results to {output_path}")
    spi3.to_dataset(name="spi3").to_zarr(
        fsspec.get_mapper(output_path, token=credentials),
        mode="w",
        consolidated=True
    )
    
    print("SPI-3 calculation complete!")
    client.close()
    cluster.close()

if __name__ == "__main__":
    calculate_spi3_chirps()
