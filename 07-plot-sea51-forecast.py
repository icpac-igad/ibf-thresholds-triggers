#!/usr/bin/env python3
"""
SEAS51 Forecast Plot Generator with Single Threshold/Trigger

Generates ensemble forecast stamp plots with empirical probability and binary
trigger maps for drought anticipatory action decision-making.

Run with --help for detailed usage information and examples.
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
from io import StringIO


from climpred import HindcastEnsemble

# Adding necessary paths to import modules from the project
sys.path.append('.')

from vthree_utils import BinCreateParams, spi3_prod_name_creator

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def save_forecast_to_netcdf(dm_fct_prob, params, year, month, threshold, trigger, output_dir="./"):
    """
    Save forecast probability data to a NetCDF file

    Args:
        dm_fct_prob (xarray.DataArray/Dataset): Drought forecast probability
        params (BinCreateParams): Parameters object
        year (int): Year of the forecast initialization
        month (int): Month of the forecast initialization
        threshold (float): SPI threshold used
        trigger (float): Trigger probability used
        output_dir (str): Directory to save the output file

    Returns:
        str: Path to the saved NetCDF file
    """
    # Create a dataset to hold the probability variable
    ds = xr.Dataset()

    # Extract DataArray if we have Dataset
    if isinstance(dm_fct_prob, xr.Dataset) and params.spi_prod_name in dm_fct_prob:
        prob_array = dm_fct_prob[params.spi_prod_name]
    else:
        prob_array = dm_fct_prob

    # Add variable to the dataset
    ds['drought_prob'] = prob_array

    # Add useful metadata
    ds.attrs['description'] = f'SEAS51 SPI3 empirical probability for {params.sc_season_str.upper()}'
    ds.attrs['creation_date'] = datetime.now().strftime('%Y-%m-%d')
    ds.attrs['year'] = year
    ds.attrs['month'] = month
    ds.attrs['lead_time'] = params.lead_int
    ds.attrs['region_id'] = params.region_id
    ds.attrs['threshold'] = threshold
    ds.attrs['trigger'] = trigger

    # Construct the filename with threshold info
    threshold_str = f"{abs(threshold):.2f}".replace('.', 'p')
    trigger_str = f"{trigger*100:.1f}".replace('.', 'p')
    filename = f"kmj_seas51_spi3_{params.sc_season_str}_eprob_{year}_{month:02d}_th{threshold_str}_tr{trigger_str}.nc"
    output_path = os.path.join(output_dir, filename)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the dataset to a NetCDF file
    ds.to_netcdf(output_path)

    logger.info(f"Saved forecast probabilities to {output_path}")
    return output_path, ds


def create_classified_colormap(vmin, vmax, cmap_name='Blues'):
    """
    Create a classified colormap with 5 classes between vmin and vmax
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
    """
    # Create a copy of the input
    binary_map = forecast_prob.copy()

    # Create a mask for zero values - treat them as NaN
    zero_mask = forecast_prob == 0

    # Create a mask for existing NaN values
    nan_mask = np.isnan(forecast_prob)

    # Create a combined mask for all values to be treated as NaN
    combined_mask = zero_mask | nan_mask

    # Create a mask for values that exceed the trigger (must be both > 0 and >= trigger)
    exceeds_trigger = (forecast_prob > 0) & (forecast_prob >= trigger_value)

    # Initialize all values as NaN
    binary_map = xr.full_like(forecast_prob, np.nan)

    # Set non-NaN and non-zero values that don't exceed trigger to 0
    binary_map = xr.where((~combined_mask) & (~exceeds_trigger), 0, binary_map)

    # Then set values that exceed the trigger to 1
    binary_map = xr.where(exceeds_trigger, 1, binary_map)

    return binary_map


def get_forecast_data_only(params):
    """
    Retrieves and processes forecast data without observation alignment constraints.
    """
    try:
        logger.info(f"Retrieving forecast-only data for region: {params.region_id}")

        # Load forecast dataset
        if len(params.season_str) == 3:
            kn_fct = xr.open_dataset(os.path.join(params.data_path, params.fct_netcdf_file))
            logger.info("Loaded SPI3 forecast dataset")
        else:
            kn_fct = xr.open_dataset(os.path.join(params.data_path, params.fct_netcdf_file))
            logger.info("Loaded SPI4 forecast dataset")

        # Subset to region
        a_fc = kn_fct

        # Add climpred to get valid_time in forecast
        try:
            hindcast = HindcastEnsemble(a_fc)
            a_fc1 = hindcast.get_initialized()
            logger.debug("Added climpred HindcastEnsemble to add valid_time in forecast")
        except ImportError:
            logger.warning("climpred not available, calculating valid_time manually")
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


def calculate_empirical_probability_single(ens_data, threshold):
    """
    Calculate empirical probability for a single threshold value.

    Args:
        ens_data: Ensemble forecast data with 'member' dimension
        threshold: SPI threshold value (e.g., -0.68)

    Returns:
        xarray.DataArray: Empirical probability of exceeding threshold
    """
    # Count members below threshold
    below_threshold = (ens_data < threshold).sum(dim='member')

    # Calculate probability (proportion of members below threshold)
    total_members = ens_data.sizes['member']
    probability = below_threshold / total_members

    return probability


def forecast_plot_datatree_single(ens_data, fct_prob, td_prob, params):
    """
    Helper function to prepare and combine data for stamp plot with single threshold.
    """
    try:
        logger.info("Starting datatree with single threshold")
        seas51tree = xr.DataTree()

        for member in ens_data.member:
            member_data = ens_data.sel(member=member)
            seas51tree[f"ensemble/member_{int(member)}"] = xr.DataTree(
                name=f"member_{int(member)}", dataset=member_data
            )

        seas51tree["fct_prob"] = xr.DataTree(name="fct_prob", dataset=fct_prob)
        seas51tree["td_prob"] = xr.DataTree(name="td_prob", dataset=td_prob.to_dataset(name="trigger_exceeded"))

        logger.info(f"Made the combined_data as xarray datatree {seas51tree}")
        return seas51tree

    except KeyError as e:
        logger.error(f"KeyError in forecast_plot_datatree_single: {str(e)}")
        raise
    except ValueError as e:
        logger.error(f"ValueError in forecast_plot_datatree_single: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in forecast_plot_datatree_single: {str(e)}")
        raise


def mdplot_single_threshold(dstree, params, threshold, trigger, shapefile_df, output_dir):
    """
    Create a plot with ensemble members, empirical probability, and binary trigger map
    for a single threshold/trigger combination.
    """
    region_geom = shapefile_df["geometry"].values[0]

    # Configure the plot layout
    # We need to display 51 ensemble members + 1 forecast probability + 1 binary map = 53 plots
    fig = plt.figure(figsize=(25, 18))

    # Calculate grid dimensions - 8x7 grid = 56 cells (enough for all plots)
    n_rows = 8
    n_cols = 7

    # Setup the layout
    gs = plt.GridSpec(n_rows, n_cols, figure=fig)

    # Get coordinates for plotting
    members = list(dstree["ensemble"].children.keys())
    lats = dstree["ensemble/member_0"].ds.lat.values
    lons = dstree["ensemble/member_0"].ds.lon.values

    print(f"Number of ensemble members: {len(members)}")

    # Get the most recent init date for title formatting
    init_dates = dstree["fct_prob"].ds.init.values
    latest_init = init_dates[-1]
    valid_times = dstree["fct_prob"].ds.valid_time.values
    latest_valid = valid_times[-1]

    # 1. Plot the ensemble members
    member_count = 0
    for i in range(n_rows):
        for j in range(n_cols):
            if member_count < len(members):
                ax = fig.add_subplot(gs[i, j], projection=ccrs.PlateCarree())

                member_key = members[member_count]

                try:
                    member_data = dstree[f"ensemble/{member_key}"].ds[params.spi_prod_name]
                    data = member_data.values[-1, :, :]

                    pcm = ax.pcolormesh(
                        lons, lats, data,
                        cmap="RdBu", transform=ccrs.PlateCarree(),
                        vmin=-4, vmax=4
                    )

                    ax.set_title(f'{latest_init.year} m{member_key.split("_")[1]}', fontsize=8)
                except Exception as e:
                    print(f"Error plotting member {member_key}: {e}")
                    ax.set_title(f'm{member_key.split("_")[1]} (error)', fontsize=8, color='red')

                ax.set_xticks([])
                ax.set_yticks([])

                if member_count == 0:
                    ax.add_geometries([region_geom], crs=ccrs.PlateCarree(), edgecolor="black", facecolor="none")

                member_count += 1

    # Define special plots (1 probability forecast, 1 binary map)
    special_plots = [
        ("fct_prob", f"Emp. Prob (th={threshold})", "Blues", None),
        ("td_prob", f"Trigger (>{trigger*100:.1f}%)", "RdYlGn_r", (0, 1)),
    ]

    # Calculate where to place the special plots
    special_cells = []
    for i in range(n_rows):
        for j in range(n_cols):
            if i * n_cols + j >= member_count:
                special_cells.append((i, j))

    # Determine the min and max values for empirical probability plot
    try:
        fct_data = dstree["fct_prob"].ds[params.spi_prod_name].values[-1, :, :]
        forecast_data = fct_data.flatten()
        forecast_min = max(0, np.nanmin(forecast_data))
        forecast_max = min(1, np.nanmax(forecast_data))
        forecast_min = max(0, round(forecast_min * 20) / 20)
        forecast_max = min(1, round(forecast_max * 20) / 20)
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
            if key == "fct_prob":
                plot_data = dstree[key].ds[params.spi_prod_name].values[-1, :, :]
                actual_vrange = (forecast_min, forecast_max)

                classified_cmap, classified_norm, class_bounds = create_classified_colormap(
                    actual_vrange[0], actual_vrange[1], cmap_name=cmap
                )
                pcm = ax.pcolormesh(
                    lons, lats, plot_data,
                    cmap=classified_cmap,
                    norm=classified_norm,
                    transform=ccrs.PlateCarree(),
                )

            elif key == "td_prob":
                td_dataset = dstree[key].ds
                var_name = list(td_dataset.data_vars)[0]
                raw_data = td_dataset[var_name].values
                plot_data = np.squeeze(raw_data)

                print(f"Binary map shape for {key}: {raw_data.shape} → {plot_data.shape}")

                binary_cmap = ListedColormap(['green', 'red'])
                binary_cmap.set_bad('white', alpha=0.6)
                binary_norm = BoundaryNorm([0, 0.5, 1.0001], binary_cmap.N)

                pcm = ax.pcolormesh(
                    lons, lats, plot_data,
                    cmap=binary_cmap,
                    norm=binary_norm,
                    transform=ccrs.PlateCarree(),
                    shading='auto'
                )

            ax.set_title(title, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

            if idx == 0:
                ax.add_geometries([region_geom], crs=ccrs.PlateCarree(), edgecolor="black", facecolor="none")

        except Exception as e:
            print(f"Error plotting {key}: {e}")
            ax.set_title(f"Error: {key}", fontsize=8, color='red')

    # Add colorbars
    # Ensemble colorbar
    cbar_ax1 = fig.add_axes([0.92, 0.65, 0.02, 0.25])
    cbar1 = plt.colorbar(
        plt.cm.ScalarMappable(norm=plt.Normalize(vmin=-4, vmax=4), cmap="RdBu"),
        cax=cbar_ax1
    )
    cbar1.set_label(f"{params.spi_prod_name} (Ensemble)")

    # Forecast probability colorbar
    cbar_ax2 = fig.add_axes([0.92, 0.35, 0.02, 0.25])
    forecast_cmap, forecast_norm, forecast_bounds = create_classified_colormap(forecast_min, forecast_max, cmap_name="Blues")
    cbar2 = plt.colorbar(
        plt.cm.ScalarMappable(norm=forecast_norm, cmap=forecast_cmap),
        cax=cbar_ax2,
        ticks=forecast_bounds
    )
    cbar2.set_label("Drought probability")

    # Binary trigger colorbar
    cbar_ax3 = fig.add_axes([0.92, 0.05, 0.02, 0.25])
    binary_cmap = ListedColormap(['green', 'red'])
    binary_norm = BoundaryNorm([0, 0.5, 1.0001], binary_cmap.N)
    cbar3 = plt.colorbar(
        plt.cm.ScalarMappable(norm=binary_norm, cmap=binary_cmap),
        cax=cbar_ax3
    )
    cbar3.set_label("Trigger exceeded (0=No, 1=Yes)")
    cbar3.set_ticks([0.25, 0.75])
    cbar3.set_ticklabels(["No (0)", "Yes (1)"])

    # Add main title
    region_name = params.region_name_dict[params.region_id]
    season_str = params.sc_season_str.upper()

    fig.suptitle(
        f"{region_name} SEAS51 SPI Forecast ({season_str})\n"
        f"Init: {latest_init.strftime('%Y-%m-%d')}, Valid: {latest_valid.strftime('%Y-%m')}, Lead: {params.lead_int} months\n"
        f"Threshold: {threshold}, Trigger: {trigger*100:.1f}%",
        fontsize=16,
        weight="bold",
        y=0.98
    )

    # Save the figure with threshold/trigger info in filename
    threshold_str = f"{abs(threshold):.2f}".replace('.', 'p')
    trigger_str = f"{trigger*100:.1f}".replace('.', 'p')
    output_file = f"{output_dir}/{params.region_id}_{params.sc_season_str}_lt{params.lead_int}_th{threshold_str}_tr{trigger_str}.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Plot saved to {output_file}")
    return output_file


def main():
    """Main function to parse arguments and run the script."""

    description = """
SEAS51 Forecast Plot Generator with Single Threshold/Trigger

Generates ensemble forecast stamp plots with empirical probability and binary
trigger maps for drought anticipatory action decision-making.

WORKFLOW STEPS:
  1. Load SEAS51 SPI3 forecast ensemble data for the specified region/season
  2. Filter forecast to target year/month initialization date
  3. Calculate empirical probability of drought (SPI < threshold) across ensemble
  4. Create binary trigger map (probability >= trigger value)
  5. Save forecast probability to NetCDF file
  6. Generate stamp plot showing all 51 ensemble members + probability + trigger map
"""

    epilog = """
EXAMPLES:

  Basic usage with required arguments:
  ------------------------------------
  python 07-plot-sea51-forecast-v4-newrun.py \\
      --threshold -0.68 --trigger 0.152 \\
      --use_shpfile --shapefile_path ./kmj_polygon.shp

  Full specification for JJA 2025 forecast initialized in April:
  --------------------------------------------------------------
  python 07-plot-sea51-forecast-v4-newrun.py \\
      --region_id kmj \\
      --season JJA \\
      --lead_time 3 \\
      --year 2025 \\
      --month 4 \\
      --threshold -0.68 \\
      --trigger 0.152 \\
      --use_shpfile \\
      --shapefile_path ./kmj_polygon.shp \\
      --fct_file ./kmj_rgr_seas51_spi3_masked.nc \\
      --output_dir ./output

  MAM 2026 forecast with different threshold/trigger:
  ---------------------------------------------------
  python 07-plot-sea51-forecast-v4-newrun.py \\
      --season MAM \\
      --year 2025 \\
      --month 12 \\
      --threshold -0.5 \\
      --trigger 0.20 \\
      --use_shpfile \\
      --shapefile_path ./kmj_polygon.shp

OUTPUT FILES:
  - {region}_seas51_spi3_{season}_eprob_{year}_{month}_th{threshold}_tr{trigger}.nc
  - {region}_{season}_lt{lead}_th{threshold}_tr{trigger}.png

NOTES:
  - Threshold is the SPI value below which drought occurs (typically negative)
  - Trigger is the probability threshold for AA activation (0-1 scale, e.g., 0.152 = 15.2%)
  - Lead time is months from initialization to target season (e.g., 3 for April→JJA)
"""

    parser = argparse.ArgumentParser(
        description=description,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Region and season arguments
    parser.add_argument("--region_id", default="kmj",
                        help="Region identifier (default: kmj)")
    parser.add_argument("--season", default="MAM",
                        help="Target season: MAM, JJA, OND, etc. (default: MAM)")
    parser.add_argument("--lead_time", type=int, default=3,
                        help="Lead time in months from init to season (default: 3)")

    # Forecast initialization date
    parser.add_argument("--year", type=int, default=2023,
                        help="Forecast initialization year (default: 2023)")
    parser.add_argument("--month", type=int, default=4,
                        help="Forecast initialization month 1-12 (default: 4)")

    # Threshold and trigger (required)
    parser.add_argument("--threshold", type=float, required=True,
                        help="SPI drought threshold, e.g., -0.68 for moderate drought")
    parser.add_argument("--trigger", type=float, required=True,
                        help="Trigger probability (0-1 scale), e.g., 0.152 for 15.2%%")

    # Data input files
    parser.add_argument("--fct_file", type=str,
                        default="/srv/202512-itt/v2-kmj-dec2025/kmj_rgr_seas51_spi3_masked.nc",
                        help="Path to SEAS51 SPI3 forecast NetCDF file")

    # Shapefile options
    parser.add_argument("--use_shpfile", action="store_true",
                        help="Use local shapefile for region boundary (required)")
    parser.add_argument("--shapefile_path", type=str,
                        help="Path to shapefile (.shp or .geojson) for region boundary")

    # Output options
    parser.add_argument("--output_dir", default="./",
                        help="Output directory for plots and NetCDF (default: ./)")

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
        obs_netcdf_file="",  # Not used in this script
        fct_netcdf_file=args.fct_file,
        service_account_json="",
        gcs_file_url="",
        region_filter=args.region_id
    )

    # Set season string
    params.sc_season_str = args.season.lower()

    # Run the plot generation
    ens_data = get_forecast_data_only(params)

    # Calculate empirical probability with single threshold
    logger.info(f"Calculating empirical probability with threshold: {args.threshold}")
    fct_prob = calculate_empirical_probability_single(ens_data[params.spi_prod_name], args.threshold)

    # Convert to dataset for consistency
    fct_prob_ds = fct_prob.to_dataset(name=params.spi_prod_name)

    # Select specific month and year
    month = args.month
    year = args.year
    dm_ens_data = ens_data.sel(init=(ens_data.init.dt.year == year) & (ens_data.init.dt.month == month))
    dm_fct_prob = fct_prob_ds.sel(init=(fct_prob_ds.init.dt.year == year) & (fct_prob_ds.init.dt.month == month))

    # Save forecast data to NetCDF
    netcdf_path, epds = save_forecast_to_netcdf(
        dm_fct_prob, params, year, month, args.threshold, args.trigger,
        output_dir=args.output_dir
    )
    print(f"Saved forecast data to {netcdf_path}")

    # Create binary trigger map
    td_prob = create_binary_trigger_map(epds['drought_prob'], args.trigger)

    # Create datatree for plotting
    fct_dt = forecast_plot_datatree_single(ens_data, fct_prob_ds, td_prob, params)

    # Load shapefile
    if args.use_shpfile:
        if args.shapefile_path:
            shapefile_df = gp.read_file(args.shapefile_path)
        else:
            shapefile_df = gp.read_file('../../data/kmj_polygon.shp')
    else:
        raise ValueError("Shapefile is required for plotting. Use --use_shpfile flag.")

    print(f'Threshold: {args.threshold}, Trigger: {args.trigger*100:.1f}%')
    print(f"Binary trigger values: {td_prob.values}")

    # Generate plot
    output_file = mdplot_single_threshold(
        fct_dt, params, args.threshold, args.trigger,
        shapefile_df, args.output_dir
    )

    return output_file


if __name__ == "__main__":
    main()
