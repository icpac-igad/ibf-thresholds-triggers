import os
from dotenv import load_dotenv
import pytest
import logging

import pandas as pd
import geopandas as gp
import xarray as xr
import numpy as np
import cftime

from utils import ken_mask_creator
from utils import make_obs_fct_dataset
from utils import get_threshold
from utils import empirical_probability
from utils import seas51_patch_empirical_probability
from utils import prepare_data_for_concat
from utils import helper_stamp_plot

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Test_Ken_mask_creator:
    # Mock the geopandas read_file function
    @pytest.fixture
    def mock_geopandas_read_file(monkeypatch):
        def mock_read_file(filename):
            if 'Karamoja_boundary_dissolved.shp' in filename:
                return gp.GeoDataFrame({'geometry': [None]})
            elif 'wajir_mbt_extent.shp' in filename:
                return gp.GeoDataFrame({'geometry': [None, None]})
        monkeypatch.setattr(gp, 'read_file', mock_read_file)

    # Test the ken_mask_creator function
    def test_ken_mask_creator(mock_geopandas_read_file):
        #data_path = "/path/to/data/"
        """
        The inpute shape file geomerty is for KIMWA region is corrupted
        the region mask is not working, tested here only for the bounds and extent
        """
        load_dotenv()
        data_path=os.getenv("ea_input_path")
        logger.info(f'{data_path}')
        the_mask, rl_dict, mds2 = ken_mask_creator(data_path)

        # Test the output types
        #assert isinstance(the_mask, regionmask.Regions)
        assert isinstance(the_mask, list)
        assert isinstance(rl_dict, dict)
        assert isinstance(mds2, gp.GeoDataFrame)

        # Test the contents of rl_dict
        expected_rl_dict = {0: "Karamoja", 1: "Marsabit", 2: "Wajir"}
        assert rl_dict == expected_rl_dict

        # Test the structure of mds2
        assert list(mds2.columns) == ["geometry", "region", "region_name"]
        assert len(mds2) == 3

        # Test the values in mds2
        assert list(mds2['region']) == [0, 1, 2]
        assert list(mds2['region_name']) == ["Karamoja", "Marsabit", "Wajir"]



class TestMakeObsFctDataset:
    @pytest.fixture
    def data_path(self):
        # Adjust this to the actual path where your test data is stored
        load_dotenv()
        data_path=os.getenv("ea_input_path")
        return data_path

    def test_spi3_dataset_loading(self, data_path, caplog):
        # Test SPI3 dataset loading
        season_str = "MAM"
        region_id = 0
        lead_int = 0

        obs_data, ens_data = make_obs_fct_dataset(data_path, region_id, season_str, lead_int)

        assert "Loaded SPI3 datasets" in caplog.text
        assert isinstance(obs_data, (xr.DataArray, xr.Dataset))
        assert isinstance(ens_data, (xr.DataArray, xr.Dataset))
        
        if isinstance(obs_data, xr.Dataset):
            assert "spi3" in obs_data.data_vars
        else:
            assert "spi3" in obs_data.name

        if isinstance(ens_data, xr.Dataset):
            assert "spi3" in ens_data.data_vars
        else:
            assert "spi3" in ens_data.name 

class TestEmpiricalProbablity:
    @pytest.fixture
    def dummy_ens_data(self):
        member = np.arange(51)
        init = pd.date_range('1982-02-01', '2023-02-01', freq='YS')  # Use pd.date_range for 'init'

        # Calculate 'valid_time' by adding 3 months to 'init'
        valid_time = init + pd.DateOffset(months=3)
        valid_time = [cftime.DatetimeProlepticGregorian(date.year, date.month, date.day) for date in valid_time]

        lat = np.linspace(-5, 5, 11)
        lon = np.linspace(30, 35, 6)

        data = np.random.uniform(low=-4, high=4, size=(51, len(init), 11, 6))

        ens_data = xr.DataArray(
            data,
            dims=['member', 'init', 'lat', 'lon'],
            coords={
                'member': member,
                'init': init,
                'valid_time': ('init', valid_time),  # Assign 'valid_time' as a coordinate
                'lat': lat,
                'lon': lon
            }
        )
        ens_data1 = ens_data.to_dataset(name='spi3')
        return ens_data1
    def test_empirical_probability_valid_input(self, dummy_ens_data):
        threshold_dict = {'mod': -1, 'sev': -2, 'ext': -3}

        fct_mod, fct_sev, fct_ext = empirical_probability(dummy_ens_data, threshold_dict)

        # Assertions
        logger.info(fct_mod)
        logger.info(fct_sev)
        logger.info(fct_ext)
        logger.info(fct_ext['init'].values)
        logger.info(fct_ext.dims)

        assert isinstance(fct_mod, xr.Dataset)
        assert isinstance(fct_sev, xr.Dataset)
        assert isinstance(fct_ext, xr.Dataset)
        assert len(fct_mod.init.values) == len(fct_sev.init.values) == len(fct_ext.init.values) 
        assert 'init' in fct_mod.dims and 'lat' in fct_mod.dims and 'lon' in fct_mod.dims

    def test_empirical_probability_invalid_input_type(self):
        with pytest.raises(ValueError) as exc_info:
            empirical_probability(np.array([1, 2, 3]), {})
        assert str(exc_info.value) == "ens_data must be an xarray.Dataset"

    @pytest.mark.skip(reason="test is low use")
    def test_empirical_probability_missing_member_dimension(self):
        data_no_member = xr.DataArray(np.random.rand(40, 10), coords={'init': np.arange(40)})
        data_no_member = data_no_member.to_dataset(name='spi3')

        with pytest.raises(ValueError) as exc_info:
            empirical_probability(data_no_member, {})
        assert str(exc_info.value) == "ens_data must have a 'member' dimension"
    
    @pytest.mark.skip(reason="test is low use")
    def test_empirical_probability_missing_threshold_key(self, dummy_ens_data):
        threshold_dict_missing_key = {'mod': -1, 'ext': -3}  # Missing 'sev' key

        with pytest.raises(KeyError) as exc_info:
            empirical_probability(dummy_ens_data, threshold_dict_missing_key)
        assert str(exc_info.value) == "threshold_dict is missing required key: sev"

    def test_empirical_probability_calculation(self, dummy_ens_data):
        threshold_dict = {'mod': -1, 'sev': -2, 'ext': -3}

        fct_mod, fct_sev, fct_ext = empirical_probability(dummy_ens_data, threshold_dict)

        # Check if calculated probabilities are within expected range [0, 1]
        assert (fct_mod >= 0).all() and (fct_mod <= 1).all()
        assert (fct_sev >= 0).all() and (fct_sev <= 1).all()
        assert (fct_ext >= 0).all() and (fct_ext <= 1).all()

        # You can add more specific assertions here to check the actual 
        # calculation logic based on your 'empirical_probability' function
    def test_seas51_patch_empirical_probability(self, dummy_ens_data, caplog):
        ens_data = dummy_ens_data 
        # Test SPI3 dataset loading
        season_str='MAM'
        sc_season_str=season_str.lower()
        region_id=0
        threshold_dict=get_threshold(region_id, sc_season_str)
        logger.info(threshold_dict) 
        fct_mod, fct_sev, fct_ext = seas51_patch_empirical_probability(ens_data,threshold_dict)
        logger.info(fct_mod)
        logger.info(fct_sev)
        logger.info(fct_ext)
        logger.info(fct_ext['init'].values)
        assert "Empirical probabilities calculated successfully" in caplog.text
        assert isinstance(fct_mod, xr.Dataset)
        assert isinstance(fct_sev, xr.Dataset)
        assert isinstance(fct_ext, xr.Dataset)
        assert np.array_equal(fct_mod['init'].values, ens_data['init'].values)
       

class TestHelperStamp:
    @pytest.fixture
    def d_ens_dt(self):
        member = np.arange(51)
        init = np.arange(42)
        lat = np.linspace(-5, 5, 11)
        lon = np.linspace(30, 35, 6)
        
        data = np.random.uniform(low=-4, high=4, size=(51, 42, 11, 6))
        
        ens_data = xr.DataArray(
            data,
            dims=['member', 'init', 'lat', 'lon'],
            coords={
                'member': member,
                'init': init,
                'lat': lat,
                'lon': lon
            }
        )
        ens_data1=ens_data.to_dataset(name='spi3')
        return ens_data1
    @pytest.fixture
    def d_obs_dt(self):
        time = np.arange(42)
        lat = np.linspace(-5, 5, 11)
        lon = np.linspace(30, 35, 6)
        
        data = np.random.uniform(low=-4, high=4, size=(42, 11, 6))
        
        obs_data = xr.DataArray(
            data,
            dims=['time', 'lat', 'lon'],
            coords={
                'time': time,
                'lat': lat,
                'lon': lon
            }
        )
        obs_data1=obs_data.to_dataset(name='spi3')
        return obs_data1
    @pytest.fixture
    def d_emp_prob_dt(self, d_ens_dt):
        ens_data=d_ens_dt
        season_str='MAM'
        sc_season_str=season_str.lower()
        region_id=0
        threshold_dict=get_threshold(region_id, sc_season_str)
        fct_mod, fct_sev, fct_ext = seas51_patch_empirical_probability(ens_data,threshold_dict)
        return fct_mod, fct_sev, fct_ext


    def test_seas51_patch_empirical_probability(self,d_ens_dt,d_obs_dt,d_emp_prob_dt, caplog):
        ens_data = d_ens_dt
        obs_data = d_obs_dt
        fct_mod, fct_sev, fct_ext = d_emp_prob_dt
        # Test SPI3 dataset loading

        ds=helper_stamp_plot(ens_data,obs_data,fct_mod,fct_sev,fct_ext)        
        assert "helper_stamp_plot function completed successfully" in caplog.text
        assert isinstance(ds, xr.Dataset)
        assert isinstance(fct_sev, xr.Dataset)
        assert isinstance(fct_ext, xr.Dataset)
       











#    def test_spi4_dataset_loading(self, data_path, caplog):
#        # Test SPI4 dataset loading
#        season_str = "JJAS"
#        region_id = 0
#        lead_int = 0
#
#        obs_data, ens_data = make_obs_fct_dataset(data_path, region_id, season_str, lead_int)
#
#        assert "Loaded SPI4 datasets" in caplog.text
#        assert isinstance(obs_data, xr.DataArray)
#        assert isinstance(ens_data, xr.DataArray)
#        assert "spi4" in obs_data.name
#        assert "spi4" in ens_data.name
#
#    def test_dataset_selection(self, data_path):
#        # Test dataset selection based on lat/lon bounds
#        season_str = "MAM"
#        region_id = 0
#        lead_int = 0
#
#        obs_data, ens_data = make_obs_fct_dataset(data_path, region_id, season_str, lead_int)
#
#        # Check if the selection was applied correctly
#        # You may need to adjust these assertions based on your actual data
#        assert obs_data.lon.min() >= llon
#        assert obs_data.lon.max() <= ulon
#        assert obs_data.lat.min() >= llat
#        assert obs_data.lat.max() <= ulat
#        assert ens_data.lon.min() >= llon
#        assert ens_data.lon.max() <= ulon
#        assert ens_data.lat.min() >= llat
#        assert ens_data.lat.max() <= ulat
#


class TestArrangeObsFctStampplot:
    @pytest.fixture
    def mock_forecast_data(self):
        # Create a mock forecast dataset
        dates = pd.date_range(start='1979-01-01', end='2022-12-31', freq='YS')
        members = np.arange(1, 52)
        lats = np.linspace(-5, 5, 5)
        lons = np.linspace(30, 40, 5)
        
        data = np.random.randn(len(dates), len(members), len(lats), len(lons))
        
        return xr.Dataset(
            data_vars={'spi3': (['init', 'member', 'lat', 'lon'], data)},
            coords={
                'init': dates,
                'member': members,
                'lat': lats,
                'lon': lons
            }
        )

    @pytest.fixture
    def mock_obs_data(self):
        # Create a mock observation dataset
        dates = pd.date_range(start='1979-01-01', end='2022-12-31', freq='YS')
        lats = np.linspace(-5, 5, 5)
        lons = np.linspace(30, 40, 5)
        
        data = np.random.randn(len(dates), len(lats), len(lons))
        
        return xr.Dataset(
            data_vars={'spi3': (['time', 'lat', 'lon'], data)},
            coords={
                'time': dates,
                'lat': lats,
                'lon': lons
            }
        )

    def test_arrange_obs_fct_stampplot_dimensions(self, mock_forecast_data, mock_obs_data):
        # Test if the resulting dataset has the correct dimensions
        result = arrange_obs_fct_stampplot(mock_forecast_data, mock_obs_data)
        
        assert 'init' in result.dims
        assert 'member' in result.dims
        assert 'lat' in result.dims
        assert 'lon' in result.dims
        assert result.dims['member'] == 52  # 51 forecast members + 1 observation member

    def test_arrange_obs_fct_stampplot_data_integrity(self, mock_forecast_data, mock_obs_data):
        # Test if the data from both forecast and observations are correctly merged
        result = arrange_obs_fct_stampplot(mock_forecast_data, mock_obs_data)
        
        # Check if the first 51 members are from the forecast data
        np.testing.assert_array_equal(result.spi3.isel(member=slice(0, 51)), mock_forecast_data.spi3)
        
        # Check if the 52nd member is from the observation data
        np.testing.assert_array_equal(result.spi3.isel(member=51).dropna(dim='init'), mock_obs_data.spi3)

    def test_arrange_obs_fct_stampplot_time_alignment(self, mock_forecast_data, mock_obs_data):
        # Test if the time dimension is correctly aligned
        result = arrange_obs_fct_stampplot(mock_forecast_data, mock_obs_data)
        
        assert (result.init == mock_forecast_data.init).all()
        assert (result.init == mock_obs_data.time).all()

    def test_arrange_obs_fct_stampplot_missing_data(self, mock_forecast_data, mock_obs_data):
        # Test handling of missing data
        # Remove some data from observations
        mock_obs_data = mock_obs_data.isel(time=slice(0, -5))
        
        result = arrange_obs_fct_stampplot(mock_forecast_data, mock_obs_data)
        
        # Check if the last 5 time steps for the observation member are NaN
        assert np.isnan(result.spi3.isel(member=51, init=slice(-5, None))).all()

    def test_arrange_obs_fct_stampplot_attributes(self, mock_forecast_data, mock_obs_data):
        # Test if important attributes are preserved
        mock_forecast_data.attrs['forecast_attribute'] = 'test_forecast'
        mock_obs_data.attrs['obs_attribute'] = 'test_obs'
        
        result = arrange_obs_fct_stampplot(mock_forecast_data, mock_obs_data)
        
        assert 'forecast_attribute' in result.attrs
        assert result.attrs['forecast_attribute'] == 'test_forecast'
        assert 'obs_attribute' in result.attrs
        assert result.attrs['obs_attribute'] == 'test_obs'

    def test_arrange_obs_fct_stampplot_coordinate_values(self, mock_forecast_data, mock_obs_data):
        # Test if coordinate values are preserved
        result = arrange_obs_fct_stampplot(mock_forecast_data, mock_obs_data)
        
        np.testing.assert_array_equal(result.lat, mock_forecast_data.lat)
        np.testing.assert_array_equal(result.lon, mock_forecast_data.lon)
        np.testing.assert_array_equal(result.init, mock_forecast_data.init)

    def test_arrange_obs_fct_stampplot_error_handling(self, mock_forecast_data, mock_obs_data):
        # Test error handling for mismatched coordinates
        mock_obs_data_mismatch = mock_obs_data.assign_coords(lat=mock_obs_data.lat + 1)
        
        with pytest.raises(ValueError):
            arrange_obs_fct_stampplot(mock_forecast_data, mock_obs_data_mismatch)
