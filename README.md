## AA triggers selection using forecast verification 

The following Jupyter notebooks and Python scripts need to be executed
sequentially:

1. **Notebook** `src/01-input-spi-seas51`: Contains methods for processing
   SEAS51 data and calculating the Standardized Precipitation Index (SPI).
   
2. **Notebook** `src/02-input-spi-chrips`: Contains methods for processing
   CHIRPS data and calculating SPI.

3. **Python Script** `src/run_kmj.py`: Performs 2D and 1D forecast
   verification, calculating metrics such as AUROC using bootstrapping. It
   subsets and saves trigger data and metrics as CSV files.

4. **Python Script** `src/run_map.py`: Generates stamp plot maps of SPI for
   forecast ensemble members, CHIRPS observations, and threshold exceedance
   empirical probabilities from NetCDF files.

5. **Python Script** `src/run_bar.py`: After manually selecting trigger values
   from the CSV file in step 3, this script creates bar plots of selected
   triggers, comparing 1D analyses of observations and forecasts.

6. **Python Script** `src/run_heatmap.py`: Generates a heatmap of decisive
   triggers, displaying False Alarm Rate (FAR) and Hit Rate (HR).

7. **Python Script** `src/run_latex_table.py`: Creates a LaTeX longtable for
   inclusion in the analysis report, listing trigger values with AUROC scores
   greater than 0.5 and highlighting manually selected triggers.

To compile the report, use the LaTeX template provided in the `doc/` folder.

