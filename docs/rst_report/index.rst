Thresholds and triggers for Anticipatory Action in Karamoja Region
==================================================================

Introduction
-----------

This document details the analysis for verifying seasonal drought forecasts for anticipatory action in the Karamoja region, including thresholds and triggers.

A trigger is a forecast issued at a certain lead time that exceeds both the danger level and the probability threshold, leading to the initiation of predefined actions. Thresholds, on the other hand, are past drought extreme observations that can be set forward to monitor future forecasts based on predefined triggers. This helps determine whether a given forecast exceeds the threshold, aiding in decision-making for anticipatory action.

In Karamoja, the danger levels for drought, defined as thresholds (Mild, Moderate, Severe), are predefined through an analysis of the frequency of long-term SPI values observed in the region (Figure 1) and subsequent stakeholder consultations during 2023, comparing these values with the drought disaster impact experienced in the region (Table 1). The current analysis explores the long-term monthly SPI variability in Karamoja, verifying forecast quality and determining the trigger for each threshold in terms of probability for defined danger levels for SPI products, namely SPI3-MAM and SPI3-JJA (Figure 2).

The final product is a trigger value (empirical probability of ensemble forecasts exceeding the given threshold ranges) reflected in terms of False Alarm Ratio and Hit Rate with respect to different lead times and threshold levels.

.. table:: Table 1: SPI threshold for Karamoja. Which is higher side of the range set in the consultation workshop 2023
   :widths: 20 20 20 20
   
   ====  =======  =======  =======
         **Mild** **Moderate** **Severe**
   ====  =======  =======  =======
   MAM   -0.43    -0.67    -0.84
   JJA   -0.43    -0.67    -0.84
   ====  =======  =======  =======

The method closely follows the paper by Gabriela et.al 2024 and 2023:

- Guimarães Nobre, G.; Towner, J.; Nhantumbo, B.; João da Conceição Marcos Matuele, C.; Raiva, I.; Pasqui, M.; Quaresima, S.; Bonifácio, R. Ready, Set, Go! An Anticipatory Action System against Droughts. EGUsphere 2024, 2024, 1–30. https://doi.org/10.5194/egusphere-2024-538.

- Nobre, Gabriela Guimarães, et al. "Forecasting, thresholds, and triggers: Towards developing a Forecast-based Financing system for droughts in Mozambique." Climate Services 30 (2023): 100344.

Analysis Steps
-------------

1. **Data Processing of SEAS5 and CHIRPS** (scripts for this step: https://github.com/icpac-igad/ibf-thresholds-triggers/blob/xarray-method/01-input-spi-seas51.ipynb and https://github.com/icpac-igad/ibf-thresholds-triggers/blob/xarray-method/02-input-spi-chrips.ipynb)

   - CHIRPS monthly mean observation data, upscaled from 5km to 25km.
   - SEAS5 forecast data, lead time 1-6 months, downscaled from 100km to 25km.
   - SPI calculation on CHIRPS data.
   - Processing SEAS51 data.
   - Replacing time steps with lead time.
   - Converting precipitation from m/s to mm/month.
   - SPI calculation on SEAS51 data, considering SPI product and month lead time as shown in Figure 1.

2. **Forecast verification stats and plotting of outputs**

   - Aligning SPI3 values of observation and forecast for each month shown in Figure 1 (Months 11, 12, 1, 2 for SPI MAM). Calculate AUROC score with bootstrapping and other metrics in 2D and 1D forms (https://github.com/icpac-igad/ibf-thresholds-triggers/blob/xarray-method/run_kmj.py).
   - Display the decided triggers in heatmap table form (https://github.com/icpac-igad/ibf-thresholds-triggers/blob/xarray-method/run_heatmap.py).
   - Create a bar chart plot for observation and forecast with a table showing forecast performance for past years (https://github.com/icpac-igad/ibf-thresholds-triggers/blob/xarray-method/run_heatmap.py).
   - Generate a table of other potential triggers that can be chosen based on user needs (https://github.com/icpac-igad/ibf-thresholds-triggers/blob/xarray-method/run_latex_table.py).
   - Map the observation and forecast in 2D map format with region shapefile overlay, including members, observation, and forecast empirical probability for each season and lead time (https://github.com/icpac-igad/ibf-thresholds-triggers/blob/xarray-method/run_map.py).

JJA
---

The following figures and tables show the analysis related to the June, July, August season SPI3. The outcome of the analysis on selected triggers is shown in Figure 8. These trigger values, representing the empirical probability of exceeding given thresholds, can be used along with monthly SEAS51 SPI3 forecasts for JJA for the given lead time. The bar plot in Figure 9 further justifies the selected triggers by showing observed drought categories count (odc) for the analysis period and the overall performance of SPI for JJA by SEAS51 and the chosen trigger. Tables 5 to 7 present the availability of alternative trigger values. The maps in Figures 10 to 12 show the time series plot of the dataset used for forecast verification analysis.

It can be noted that the forecast quality is poorer compared to MAM forecasts, and trigger values are not available for the "Severe" threshold category. This can potentially be improved by bias correction or post-processing methods.

.. figure:: _static/dt_0_jja.png
   :width: 80%
   :align: center
   
   Figure 8: Selected JJA triggers for Karamoja

.. figure:: _static/0_jja.png
   :width: 110%
   :align: center
   
   Figure 9: Overview of JJA Hit rate(%) and False alarm ratio(%) per SPI indicator, lead time of the forecasting information in months and region wise. The chosen trigger value is displayed within each tile

.. _table5:

.. include:: ../../output/0_jja_lt2.rst

.. _table6:

.. include:: ../../output/0_jja_lt3.rst

.. _table7:

.. include:: ../../output/0_jja_lt4.rst

.. figure:: _static/map_0_jja_lt2.png
   :width: 100%
   :align: center
   
   Figure 10: Time series map plot of JJA for Karamoja region for the lead time 2

.. figure:: _static/map_0_jja_lt3.png
   :width: 100%
   :align: center
   
   Figure 11: Time series map plot of JJA for Karamoja region for the lead time 3

.. figure:: _static/map_0_jja_lt4.png
   :width: 100%
   :align: center
   
   Figure 12: Time series map plot of JJA for Karamoja region for the lead time 4
