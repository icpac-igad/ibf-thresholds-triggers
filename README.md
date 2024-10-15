## AA triggers selection using forecast verification 

The notesbooks and script needed to be run one after another.

1. Notebook `src/01-input-spi-seas51`, methods for processing SEAS51 and
   calculate SPI
2. Notebook `src/02-input-spi-chrips`, methods for processing CHRIPS and
   calcualte SPI
3. The python script `src/run_kmj.py`, to do the 2d and 1d forecast
   verification, metrices such as AUROC with bootstrap, subset the triggers and
   save the triggers and metrices as csv files
4. The python script `src/run_map.py`, to do the stamp plot map of SPI in
   forecast ensemble members, CHIRPS observations and threshold exceedance
   emprical probablity from netcdf files
5. The python script `src/run_bar.py`. This has to be run after manually
   selecting the trigger values from the csv file in step 3. This script makes
   bar plots of selected triggers with comparision of 1d analysis comparing
   observation and forecast.
6. The python script `src/run_heatmap.py`. This makes decised triggers
   reflecting it with FAR and HR in heatmap format. 
7. The python script `src/run_latex_table.py`. Which makes latex longtable to
   be used with the report on analysis. Which shows the list of trigger values
   having AUROC score >0.5 and manually selected triggers marked in the table.  

Using the latex tempalte in doc/ the report can be compiled. 

