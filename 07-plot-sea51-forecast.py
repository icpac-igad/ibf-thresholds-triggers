#!/usr/bin/env python3
"""
nspired by the utils_plots.run_map_plot function but focused on a single time period.

Usage:
    python single_month_forecast_plot.py --region_id kmj --season MAM --lead_time 3 --year 2023 --month 4
"""
import os
import argparse
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from datetime import datetime, timedelta
import logging
import sys
import geopandas as gp 
from matplotlib.colors import ListedColormap, BoundaryNorm

from climpred import HindcastEnsemble

# Adding necessary paths to import modules from the project
sys.path.append('.')

from vthree_utils import BinCreateParams, get_threshold, get_region_bounds, get_credentials
from vthree_utils import make_obs_fct_dataset, seas51_patch_empirical_probability
from utils_plots import helper_stamp_plot
from vthree_utils import spi3_prod_name_creator
from vthree_utils import generate_trigger_dict

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def save_forecast_to_netcdf(dm_fct_mod, dm_fct_sev, dm_fct_ext, params, year, month, output_dir="./"):
    """
    Save forecast probability data to a NetCDF file
    
    Args:
        dm_fct_mod (xarray.DataArray/Dataset): Moderate drought forecast probability
        dm_fct_sev (xarray.DataArray/Dataset): Severe drought forecast probability
        dm_fct_ext (xarray.DataArray/Dataset): Extreme drought forecast probability
        params (BinCreateParams): Parameters object
        year (int): Year of the forecast initialization
        month (int): Month of the forecast initialization
        output_dir (str): Directory to save the output file
        
    Returns:
        str: Path to the saved NetCDF file
    """
    # Create a dataset to hold all three variables
    ds = xr.Dataset()
    
    # Extract DataArrays if we have Datasets
    # For moderate drought probability
    if isinstance(dm_fct_mod, xr.Dataset) and params.spi_prod_name in dm_fct_mod:
        mod_array = dm_fct_mod[params.spi_prod_name]
    else:
        mod_array = dm_fct_mod
        
    # For severe drought probability
    if isinstance(dm_fct_sev, xr.Dataset) and params.spi_prod_name in dm_fct_sev:
        sev_array = dm_fct_sev[params.spi_prod_name]
    else:
        sev_array = dm_fct_sev
        
    # For extreme drought probability
    if isinstance(dm_fct_ext, xr.Dataset) and params.spi_prod_name in dm_fct_ext:
        ext_array = dm_fct_ext[params.spi_prod_name]
    else:
        ext_array = dm_fct_ext
    
    # Add variables to the dataset with appropriate names
    ds['mod_prob'] = mod_array
    ds['sev_prob'] = sev_array  
    ds['ext_prob'] = ext_array
    
    # Add useful metadata
    ds.attrs['description'] = f'SEAS51 SPI3 empirical probabilities for {params.sc_season_str.upper()}'
    ds.attrs['creation_date'] = datetime.now().strftime('%Y-%m-%d')
    ds.attrs['year'] = year
    ds.attrs['month'] = month
    ds.attrs['lead_time'] = params.lead_int
    ds.attrs['region_id'] = params.region_id
    
    # Construct the filename
    filename = f"seas51_spi3_{params.sc_season_str}_eprob_{year}_{month:02d}.nc"
    output_path = os.path.join(output_dir, filename)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the dataset to a NetCDF file
    ds.to_netcdf(output_path)
    
    logger.info(f"Saved forecast probabilities to {output_path}")
    return output_path


def create_classified_colormap(vmin, vmax, cmap_name='Blues'):
    """
    Create a classified colormap with 5 classes between vmin and vmax
    
    Args:
        vmin (float): Minimum value for the colormap
        vmax (float): Maximum value for the colormap
        cmap_name (str): Base colormap name
        
    Returns:
        tuple: (cmap, norm, bounds) - the colormap, normalization objects, and boundary values
    """
    # Create 5 equally spaced class boundaries
    bounds = np.linspace(vmin, vmax, 6)
    
    # Get the base colormap using the recommended approach
    base_cmap = plt.colormaps[cmap_name]
    
    # Sample 5 colors from the base colormap
    colors = [base_cmap(i) for i in np.linspace(0, 1, 5)]
    
    # Create a new colormap with these 5 colors
    cmap = ListedColormap(colors)
    
    # Create a normalization to map values to colormap indices
    norm = BoundaryNorm(bounds, cmap.N)
    
    return cmap, norm, bounds


def create_binary_trigger_map(forecast_prob, trigger_value):
    """
    Create a binary map based on whether forecast probability exceeds trigger value
    
    Args:
        forecast_prob: Forecast probability xarray DataArray
        trigger_value: Threshold value to compare against
        
    Returns:
        xarray DataArray with binary values (1 where forecast exceeds trigger, 0 otherwise)
    """
    # Create a copy of the input
    binary_map = forecast_prob.copy()
    
    # Convert to binary values - carefully handling the comparison to maintain NaN values
    # First create a mask of valid (non-NaN) values
    valid_mask = ~np.isnan(binary_map)
    
    # Now set all valid values to 0 initially
    binary_map = binary_map.where(~valid_mask, 0)
    
    # Then set values >= trigger_value to 1
    exceeds_trigger = (forecast_prob >= trigger_value) & valid_mask
    binary_map = binary_map.where(~exceeds_trigger, 1)
    
    # Preserve attributes
    if hasattr(forecast_prob, 'attrs'):
        binary_map.attrs = forecast_prob.attrs.copy()
    binary_map.attrs['description'] = f'Binary trigger map (threshold: {trigger_value})'
    
    return binary_map


def get_forecast_data_only(params):
    """
    Retrieves and processes forecast data without observation alignment constraints.
    This function is designed for operational forecasting where you need to use the 
    latest SEAS51 data, regardless of whether observations exist for validation.
    
    Parameters:
    - params (BinCreateParams): Parameter object containing region_id, season_str, lead_int, etc.
    
    Returns:
    - xarray.Dataset: Processed forecast dataset ready for visualization
    """
    try:
        logger.info(f"Retrieving forecast-only data for region: {params.region_id}")
        
        # Load forecast dataset 
        if len(params.season_str) == 3:
            # SPI3 dataset
            kn_fct = xr.open_dataset(os.path.join(params.data_path, params.fct_netcdf_file))
            logger.info("Loaded SPI3 forecast dataset")
        else:
            # SPI4 dataset
            kn_fct = xr.open_dataset(os.path.join(params.data_path, params.fct_netcdf_file))
            logger.info("Loaded SPI4 forecast dataset")
            
        
        # Subset to region
        a_fc = kn_fct
        #logger.info("Subsetted forecast to given region")
        
        # Add climpred to get valid_time in forecast
        try:
            hindcast = HindcastEnsemble(a_fc)
            a_fc1 = hindcast.get_initialized()
            logger.debug("Added climpred HindcastEnsemble to add valid_time in forecast")
        except ImportError:
            logger.warning("climpred not available, calculating valid_time manually")
            # Manual calculation of valid_time if climpred not available
            init_times = a_fc.init.values
            lead_times = a_fc.lead.values
            valid_times = np.array([
                pd.Timestamp(init) + pd.DateOffset(months=int(lead))
                for init in init_times
                for lead in lead_times
            ]).reshape(len(init_times), len(lead_times))
            
            a_fc1 = a_fc.assign_coords(valid_time=(("init", "lead"), valid_times))
        
        # Subset to specified lead time
        a_fc2 = a_fc1.isel(lead=params.lead_int)
        
        # Add SPI product names for filtering
        if len(params.season_str) == 3:
            spi_prod_list = spi3_prod_name_creator(a_fc2, "valid_time")
        else:
            spi_prod_list = spi4_prod_name_creator(a_fc2, "valid_time")
            
        logger.info(f"Added SPI product in forecast dataset, filtering to {params.season_str}")
        a_fc2 = a_fc2.assign_coords(spi_prod=("init", spi_prod_list))
        a_fc3 = a_fc2.where(a_fc2.spi_prod == params.season_str, drop=True)
        
        # Include all forecast dates, including future dates without observations
        logger.info(f"Final forecast dataset contains {len(a_fc3.init)} initialization dates")
        
        return a_fc3
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except ValueError as e:
        logger.error(f"Value error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_forecast_data_only: {e}")
        raise


def forecast_plot_datatree(ens_data, fct_mod, fct_sev, fct_ext,td_mod, td_sev, td_ext):
    """

    DEPRECATED to replaced it with xarray datatree
    Helper function to prepare and combine data for stamp plot.

    Parameters:
    ens_data (xarray.Dataset): Ensemble data.
    obs_data (xarray.Dataset): Observation data.
    fct_mod (xarray.Dataset): Moderate forecast data.
    fct_sev (xarray.Dataset): Severe forecast data.
    fct_ext (xarray.Dataset): Extreme forecast data.

    Returns:
    xarray.Dataset: Combined dataset for stamp plot.
    """
    try:
        logger.info("Starting datatree with helper_stamp_plot function")
        seas51tree = xr.DataTree()
        for member in ens_data.member:
            member_data = ens_data.sel(member=member)
            seas51tree[f"ensemble/member_{int(member)}"] = xr.DataTree(
                name=f"member_{int(member)}", dataset=member_data
            )

        seas51tree["fct_mod"] = xr.DataTree(name="fct_mod", dataset=fct_mod)
        seas51tree["fct_sev"] = xr.DataTree(name="fct_sev", dataset=fct_sev)
        seas51tree["fct_ext"] = xr.DataTree(name="fct_ext", dataset=fct_ext)
        seas51tree["td_mod"] = xr.DataTree(name="td_mod", dataset=fct_mod)
        seas51tree["td_sev"] = xr.DataTree(name="td_sev", dataset=fct_sev)
        seas51tree["td_ext"] = xr.DataTree(name="td_ext", dataset=fct_ext)
        logger.info(f"made the combined_data as xarray datatree {seas51tree}")
        logger.info("helper_stamp_plot function completed successfully")
        return seas51tree

    except KeyError as e:
        logger.error(f"KeyError in helper_stamp_plot: {str(e)}")
        raise
    except ValueError as e:
        logger.error(f"ValueError in helper_stamp_plot: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in helper_stamp_plot: {str(e)}")
        raise


def get_2d_data(data_array):
    """Extract a 2D slice from a potentially multi-dimensional array"""
    # Handle None or empty case
    if data_array is None:
        return None
    
    # Get values
    try:
        values = data_array.values
    except AttributeError:
        # If it's already a numpy array
        values = data_array
    
    # If already 2D, return as is
    if len(values.shape) == 2:
        return values
    
    # Remove dimensions of size 1
    try:
        values = np.squeeze(values)
    except:
        pass
    
    # If still more than 2D, take the first index of extra dimensions
    while len(values.shape) > 2:
        values = values[0]
    
    return values


def mdplot_single_row(dstree, params, shapefile_df, output_dir):
    """
    Create a single row plot with ensemble members, empirical probabilities, and binary trigger maps
    
    Args:
        dstree (xarray.DataTree): Data tree containing ensemble members, forecasts and binary maps
        params (BinCreateParams): Parameters for the forecast
        shapefile_df (GeoDataFrame): Shapefile DataFrame for region boundaries
        output_dir (str): Directory to save output maps
    """
    region_geom = shapefile_df["geometry"].values[0]
    
    # Configure the plot layout
    # We need to display 51 ensemble members + 3 forecast categories + 3 binary maps = 57 plots
    # Removing observation plot means we need room for 56 plots
    fig = plt.figure(figsize=(25, 20))
    
    # Calculate grid dimensions - 8x8 grid = 64 cells (enough for all plots)
    n_rows = 8
    n_cols = 8
    
    # Setup the layout
    gs = plt.GridSpec(n_rows, n_cols, figure=fig)
    
    # Get coordinates for plotting
    members = list(dstree["ensemble"].children.keys())
    lats = dstree["ensemble/member_0"].ds.lat.values
    lons = dstree["ensemble/member_0"].ds.lon.values
    # Log information about ensemble members
    print(f"Number of ensemble members: {len(members)}")
    print(f"Member names: {members[:5]}... + {len(members)-5} more")
    
    # Get the most recent init date for title formatting
    init_dates = dstree["fct_mod"].ds.init.values
    latest_init = init_dates[-1]
    valid_times = dstree["fct_mod"].ds.valid_time.values
    latest_valid = valid_times[-1]
    
    # 1. Plot the ensemble members
    member_count = 0
    for i in range(n_rows):
        for j in range(n_cols):
            if member_count < len(members):
                # Create subplot with Cartopy projection
                ax = fig.add_subplot(gs[i, j], projection=ccrs.PlateCarree())
                
                # Get the member data
                member_key = members[member_count]
                
                try:
                    # Get the member data
                    member_data = dstree[f"ensemble/{member_key}"].ds[params.spi_prod_name]
                    
                    # Extract a specific time slice (most recent init date)
                    data = member_data.values[-1, :, :]
                    
                    # Plot the data
                    pcm = ax.pcolormesh(
                        lons, lats, data,
                        cmap="RdBu", transform=ccrs.PlateCarree(),
                        vmin=-4, vmax=4
                    )
                    
                    # Add title with member info and year
                    ax.set_title(f'{latest_init.year} m{member_key.split("_")[1]}', fontsize=8)
                except Exception as e:
                    print(f"Error plotting member {member_key}: {e}")
                    ax.set_title(f'm{member_key.split("_")[1]} (error)', fontsize=8, color='red')
                
                # Remove tick labels
                ax.set_xticks([])
                ax.set_yticks([])
                
                # Add region boundary for first member only
                if member_count == 0:
                    ax.add_geometries([region_geom], crs=ccrs.PlateCarree(), edgecolor="black", facecolor="none")
                
                member_count += 1
    
    # Define special plots (probabilistic forecasts, binary maps) - removed observation
    special_plots = [
        ("fct_mod", f"Mod EP {latest_valid.strftime('%b-%Y')}", "Blues", None),
        ("fct_sev", f"Sev EP {latest_valid.strftime('%b-%Y')}", "Blues", None),
        ("fct_ext", f"Ext EP {latest_valid.strftime('%b-%Y')}", "Blues", None),
        ("td_mod", f"Mod Trigger {latest_valid.strftime('%b-%Y')}", "RdYlGn_r", (0, 1)),
        ("td_sev", f"Sev Trigger {latest_valid.strftime('%b-%Y')}", "RdYlGn_r", (0, 1)),
        ("td_ext", f"Ext Trigger {latest_valid.strftime('%b-%Y')}", "RdYlGn_r", (0, 1))
    ]
    
    # Calculate where to place the special plots (in remaining grid cells)
    special_cells = []
    for i in range(n_rows):
        for j in range(n_cols):
            if i * n_cols + j >= member_count:
                special_cells.append((i, j))
    
    # Ensure we have enough cells for special plots
    if len(special_cells) < len(special_plots):
        print(f"Warning: Not enough grid cells for all special plots. Need {len(special_plots)}, have {len(special_cells)}")
    
    # Determine the min and max values for empirical probability plots
    # First, collect all forecast data to find the overall min/max
    forecast_data = []
    for key in ["fct_mod", "fct_sev", "fct_ext"]:
        try:
            data = dstree[key].ds[params.spi_prod_name].values[-1, :, :]
            forecast_data.extend(data.flatten())
        except:
            pass
    
    # Calculate min/max values, defaulting to 0-1 if there's an issue
    try:
        forecast_min = max(0, min(filter(lambda x: not np.isnan(x), forecast_data)))
        forecast_max = min(1, max(filter(lambda x: not np.isnan(x), forecast_data)))
        # Round to nearest 0.05 for cleaner bounds
        forecast_min = max(0, round(forecast_min * 20) / 20)
        forecast_max = min(1, round(forecast_max * 20) / 20)
        # Ensure min and max are different for proper colorbar
        if forecast_min == forecast_max:
            forecast_min = 0
            forecast_max = 1
    except:
        forecast_min = 0
        forecast_max = 1
    
    print(f"Using forecast range: ({forecast_min}, {forecast_max})")
    
    # Plot each special item
    for idx, ((key, title, cmap, vrange), (i, j)) in enumerate(zip(special_plots, special_cells)):
        if idx >= len(special_cells):
            break
            
        ax = fig.add_subplot(gs[i, j], projection=ccrs.PlateCarree())
        
        try:
            # Get the data based on the key
            if key.startswith("fct_"):
                # Set dynamic range for forecast probability data
                plot_data = dstree[key].ds[params.spi_prod_name].values[-1, :, :]
                actual_vrange = (forecast_min, forecast_max)

                 # Create classified colormap with 5 classes
                classified_cmap, classified_norm, class_bounds = create_classified_colormap(
                    actual_vrange[0], actual_vrange[1], cmap_name=cmap
                )
                # Use continuous colormap
                pcm = ax.pcolormesh(
                    lons, lats, plot_data,
                    cmap=classified_cmap, 
                    norm=classified_norm,
                    transform=ccrs.PlateCarree(),
                    )
            # For binary trigger maps with imshow
            elif key.startswith("td_"):
                plot_data = dstree[key].ds[params.spi_prod_name].values[-1, :, :]
                
                # Create discrete colormap for binary data
                from matplotlib.colors import ListedColormap
                binary_cmap = ListedColormap(['red', 'green'])
                
                # Set background color for NaN values to white with transparency
                binary_cmap.set_bad('white', alpha=0)
                
                # Use imshow with explicit extent
                lat_min, lat_max = lats.min(), lats.max()
                lon_min, lon_max = lons.min(), lons.max()
                
                pcm = ax.imshow(
                    plot_data,
                    cmap=binary_cmap,
                    vmin=0, vmax=1,
                    extent=[lon_min, lon_max, lat_min, lat_max],
                    transform=ccrs.PlateCarree(),
                    origin='upper'
                )
            else:
                # Other data types use specified range or default
                plot_data = dstree[key].ds[params.spi_prod_name].values[-1, :, :]
                actual_vrange = vrange if vrange else (0, 1)
                pcm = ax.pcolormesh(
                    lons, lats, plot_data,
                    cmap=cmap, transform=ccrs.PlateCarree(),
                    vmin=actual_vrange[0], vmax=actual_vrange[1]
                )
            
            # Add title
            ax.set_title(title, fontsize=8)
            
            # Remove tick labels
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add region boundary to first special plot
            if idx == 0:
                ax.add_geometries([region_geom], crs=ccrs.PlateCarree(), edgecolor="black", facecolor="none")
        
        except Exception as e:
            print(f"Error plotting {key}: {e}")
            ax.set_title(f"Error: {key}", fontsize=8, color='red')
        
    # Add colorbars
    # Ensemble colorbar
    cbar_ax1 = fig.add_axes([0.92, 0.7, 0.02, 0.2])
    cbar1 = plt.colorbar(
        plt.cm.ScalarMappable(norm=plt.Normalize(vmin=-4, vmax=4), cmap="RdBu"),
        cax=cbar_ax1
    )
    cbar1.set_label(f"{params.spi_prod_name} (Ensemble)")
    
    # Forecast probability colorbar with dynamic range
    cbar_ax2 = fig.add_axes([0.92, 0.4, 0.02, 0.2])
    forecast_cmap, forecast_norm, forecast_bounds = create_classified_colormap( forecast_min, forecast_max, cmap_name="Blues")
    cbar2 = plt.colorbar(
        plt.cm.ScalarMappable(norm=forecast_norm, cmap=forecast_cmap),
        cax=cbar_ax2,
        ticks=forecast_bounds
    )
    cbar2.set_label("Drought probability")    
    # Binary trigger colorbar with discrete values
    cbar_ax3 = fig.add_axes([0.92, 0.1, 0.02, 0.2])
    binary_cmap = ListedColormap(['red', 'green'])
    binary_norm = BoundaryNorm([0, 0.5, 1.0001], binary_cmap.N)
    cbar3 = plt.colorbar(
        plt.cm.ScalarMappable(norm=binary_norm, cmap=binary_cmap),
        cax=cbar_ax3
    )
    cbar3.set_label("Trigger exceeded (0=No, 1=Yes)")
    cbar3.set_ticks([0.25, 0.75])  # Center the ticks in each section
    cbar3.set_ticklabels(["Yes (1)", "No (0)"])  # Add descriptive labels
    
    # Add main title with latest init and valid dates
    region_name = params.region_name_dict[params.region_id]
    season_str = params.sc_season_str.upper()
    
    fig.suptitle(
        f"{region_name} SEAS51 SPI Forecast ({season_str})\n"
        f"Init: {latest_init.strftime('%Y-%m-%d')}, Valid: {latest_valid.strftime('%Y-%m-%d')}, Lead: {params.lead_int} months",
        fontsize=16,
        weight="bold",
        y=0.98
    )
    
    # Save the figure
    output_file = f"{output_dir}/{params.region_id}_{params.sc_season_str}_lt{params.lead_int}.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"Plot saved to {output_file}")
    return output_file 

def main():
    """Main function to parse arguments and run the script
    
    python 07-plot-sea51-forecast.py --region_id kmj --season JJA --lead_time 3 --year 2025 --month 4 --use_shpfile --shapefile_path '../../data/kmj_polygon.shp' --output_dir './'

    """
    parser = argparse.ArgumentParser(description="Generate forecast plots for a specific month")
    parser.add_argument("--region_id", default="kmj", help="Region ID (e.g., kmj)")
    parser.add_argument("--season", default="MAM", help="Season (e.g., MAM, JJA)")
    parser.add_argument("--lead_time", type=int, default=3, help="Lead time in months")
    parser.add_argument("--year", type=int, default=2023, help="Target year")
    parser.add_argument("--month", type=int, default=4, help="Target month (1-12)")
    parser.add_argument("--use_shpfile", action="store_true", help="Use local shapefiles")
    parser.add_argument("--shapefile_path", help="Path to local shapefile")
    parser.add_argument("--output_dir", help="Output directory for plots")
    
    args = parser.parse_args()
    
    # Create parameters object
    params = BinCreateParams(
        region_id=0,
        season_str=args.season,
        lead_int=args.lead_time,
        level="mod",
        region_name_dict={0: "Karamoja", 1: "Marsabit", 2: "Wajir"},
        spi_prod_name="spi3",
        data_path="./",
        output_path="./output/",
        spi4_data_path="",
        obs_netcdf_file=f"kmj_obs_spi3_masked.nc",
        fct_netcdf_file=f"kmj_rgr_seas51_spi3_masked.nc",
        service_account_json="",
        gcs_file_url="",
        region_filter=args.region_id
    )
    
    # Set season string
    params.sc_season_str = args.season.lower()
    
    # Run the plot generation
    ens_data = get_forecast_data_only(params)
    # Get thresholds (you might need to adjust this if threshold values are normally derived from observations)
    threshold_dict = get_threshold(params.region_id, params.sc_season_str)
    # Calculate empirical probabilities
    fct_mod, fct_sev, fct_ext = seas51_patch_empirical_probability(ens_data, threshold_dict)

    month=args.month
    year=args.year
    dm_ens_data=ens_data.sel(init=(ens_data.init.dt.year == year) & (ens_data.init.dt.month == month))
    dm_fct_mod = fct_mod.sel(init=(fct_mod.init.dt.year == year) & (fct_mod.init.dt.month == month))
    dm_fct_sev = fct_sev.sel(init=(fct_sev.init.dt.year == year) & (fct_sev.init.dt.month == month))
    dm_fct_ext = fct_ext.sel(init=(fct_ext.init.dt.year == year) & (fct_ext.init.dt.month == month))

    # Save forecast data to NetCDF
    netcdf_path = save_forecast_to_netcdf(
        dm_fct_mod, dm_fct_sev, dm_fct_ext, 
        params, year, month, 
        output_dir=args.output_dir or params.output_path
    )
    print(f"Saved forecast data to {netcdf_path}")

    params.output_path='output/'

    td,df=generate_trigger_dict(params)
    tdfm=create_binary_trigger_map(dm_fct_mod, td['mod']/100)
    tdfs=create_binary_trigger_map(dm_fct_sev, td['sev']/100)
    tdfe=create_binary_trigger_map(dm_fct_ext, td['ext']/100)
    fct_dt=forecast_plot_datatree(ens_data, fct_mod, fct_sev, fct_ext,tdfm, tdfs, tdfe)
    
    if args.use_shpfile:
        if args.shapefile_path:
            shapefile_df = gp.read_file(args.shapefile_path)
        else:
            shapefile_df = gp.read_file('../../data/kmj_polygon.shp')  # Default path
    else:
        pass 

    output_dir=params.output_path
    mdplot_single_row(fct_dt, params, shapefile_df, output_dir)



if __name__ == "__main__":
    main()
