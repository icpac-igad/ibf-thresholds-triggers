import xarray as xr
import dask.array as da
from dask.distributed import Client
from dateutil.relativedelta import relativedelta
from calendar import monthrange
import pandas as pd
from climate_indices.indices import spi, Distribution
from climate_indices.compute import Periodicity
import xarray as xr
from typing import Dict
import pandas as pd

# Start a Dask client
client = Client()

p1_input_data = "/home/ea_seas_v51_1981_2023.grib"
db = xr.open_dataset(
    p1_input_data,
    engine="cfgrib",
    chunks={"time": 1},  # Adjust chunk size as needed
    backend_kwargs=dict(time_dims=("forecastMonth", "time")),
)


# Define a function that transforms the data for a single time index
def transform_data(data_at_time):
    valid_time = [
        pd.to_datetime(data_at_time.time.values) + relativedelta(months=fcmonth - 1)
        for fcmonth in data_at_time.forecastMonth
    ]
    data_at_time = data_at_time.assign_coords(valid_time=("forecastMonth", valid_time))
    numdays = [monthrange(dtat.year, dtat.month)[1] for dtat in valid_time]
    data_at_time = data_at_time.assign_coords(numdays=("forecastMonth", numdays))
    data_at_time_tp = data_at_time * data_at_time.numdays * 24 * 60 * 60 * 1000
    data_at_time_tp.attrs["units"] = "mm"
    data_at_time_tp.attrs["long_name"] = "Total precipitation"
    return data_at_time_tp


# Apply the function in parallel using map_blocks or apply_ufunc
# Note that the exact function call will depend on the shape of your data and the operation you want to perform
cont_db = xr.concat(
    [transform_data(db.sel(time=time_index)) for time_index in db.time], dim="time"
).persist()
cont_db = cont_db.compute()


def spi_wrapper(
    obj: xr.DataArray,
    precip_var: str,
    scale: int,
    distribution: Distribution,
    data_start_year: int,
    calibration_year_initial: int,
    calibration_year_final: int,
    periodicity: Periodicity,
    fitting_params: Dict = None,
) -> xr.DataArray:
    # compute SPI for this timeseries
    """
    The function from https://gist.github.com/monocongo/978348233b4bde80e9bcc52fe8e4150c
    using climate_indices lib https://github.com/monocongo/climate_indices
    """
    spi_data = spi(
        values=obj[
            precip_var
        ].to_numpy(),  # TODO find why we need to use the variable name rather than already using the variables's DataArray (i.e. why is obj a Dataset?)
        scale=scale,
        distribution=distribution,
        data_start_year=data_start_year,
        calibration_year_initial=calibration_year_initial,
        calibration_year_final=calibration_year_final,
        periodicity=periodicity,
        fitting_params=fitting_params,
    )
    # TODO for some reason this is necessary for the nClimGrid low-resolution example NetCDFs
    # TODO find out why
    spi_data = spi_data.flatten()
    # print(spi_data.shape)
    # TODO for some reason this is necessary for the NCO-modified nClimGrid normal-resolution example NetCDFs
    # TODO find out why
    spi_data = spi_data.reshape(spi_data.size, 1)
    # create the return DataArray (copy of input object's geospatial dims/coords plus SPI data)
    da_spi = xr.DataArray(
        dims=obj[precip_var].dims,
        coords=obj[precip_var].coords,
        attrs={
            "description": "SPI computed by the climate_indices Python package",
            "references": "https://github.com/monocongo/climate_indices",
            "valid_min": -3.09,  # this should mirror climate_indices.indices._FITTED_INDEX_VALID_MIN
            "valid_max": 3.09,  # this should mirror climate_indices.indices._FITTED_INDEX_VALID_MAX
        },
        data=spi_data,
    )
    return da_spi


def spi3_prod_name_creator(ds_ens):
    """
    Convenience function to generate a list of SPI product
    names, such as MAM, so that can be used to filter the
    SPI product from dataframe

    Parameters
    ----------
    ds_ens : xarray dataframe
        The data farme with SPI output organized for
        the period 1981-2023.

    Returns
    -------
    spi_prod_list : String list
        List of names with iteration of SPI3 product names such as
        ['JFM','FMA','MAM',......]

    """
    db = pd.DataFrame()
    db["dt"] = ds_ens["valid_time"].values
    db["month"] = db["dt"].dt.strftime("%b").astype(str).str[0]
    db["year"] = db["dt"].dt.strftime("%Y")
    db["spi_prod"] = (
        db.groupby("year")["month"].shift(2)
        + db.groupby("year")["month"].shift(1)
        + db.groupby("year")["month"].shift(0)
    )
    spi_prod_list = db["spi_prod"].tolist()
    return spi_prod_list


def apply_spi(sc):
    spi_ds = (
        sc.groupby("grid_cells")
        .apply(
            spi_wrapper,
            precip_var="tprate",
            scale=3,
            distribution=Distribution.gamma,
            data_start_year=1981,
            calibration_year_initial=1981,
            calibration_year_final=2018,
            periodicity=Periodicity.monthly,
        )
        .unstack("grid_cells")
    )
    spi_ds1 = spi_ds.to_dataset(name="tprate")
    spi_prod_list = spi3_prod_name_creator(spi_ds1)
    spi_ds1 = spi_ds1.assign_coords(spi_prod=("time", spi_prod_list))
    return spi_ds1


lt1_db = cont_db.sel(forecastMonth=3)


########trail 1
lt1_db = cont_db.sel(forecastMonth=6)

# cld1 = lt1_db.chunk(chunks={'number': 20,'time':32,'latitude':8,'longitude':8})
# cld1 = lt1_db.chunk(chunks={'number': 1})

sc = lt1_db.sel(number=2).stack(grid_cells=("latitude", "longitude"))
client = Client()
cont_cld1 = xr.concat(
    [
        apply_spi(lt1_db.sel(number=n_idx).stack(grid_cells=("latitude", "longitude")))
        for n_idx in lt1_db.number
    ],
    dim="number",
).persist()
cont_cld1 = apply_spi(sc).persist()
sp_lt1 = cont_cld1.compute()

#########trail 2
lt1_db = cont_db.sel(forecastMonth=6)

# cld1 = lt1_db.chunk(chunks={'number': 20,'time':32,'latitude':8,'longitude':8})
# cld1 = lt1_db.chunk(chunks={'number': 1})
client = Client()

lt1_dbm1 = lt1_db.sel(number=1)
sc = lt1_dbm1.stack(grid_cells=("latitude", "longitude"))


sc = sc.chunk({"time": -1, "grid_cells": -1})

spi_ds = (
    sc.groupby("grid_cells")
    .apply(
        spi_wrapper,
        precip_var="tprate",
        scale=3,
        distribution=Distribution.gamma,
        data_start_year=1981,
        calibration_year_initial=1981,
        calibration_year_final=2018,
        periodicity=Periodicity.monthly,
    )
    .unstack("grid_cells")
)

spi1_ds = spi_ds.compute()

###########trail 3, dask map method
