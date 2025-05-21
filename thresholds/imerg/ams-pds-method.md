# Extreme Precipitation Analysis: AMS vs PDS Approaches

## Overview of Extreme Value Analysis Methods

This document summarizes the key differences between Annual Maximum Series (AMS) and Partial Duration Series (PDS) approaches for extreme precipitation analysis, with implementation examples using the xclim library.

## Annual Maximum Series (AMS)

### Definition and Characteristics
- Extracts exactly one maximum value per year (or per block)
- Yields a sample size equal to the number of years in the record
- Typically fitted using the Generalized Extreme Value (GEV) distribution
- Standard approach in many hydrological applications

### Implementation in xclim
```python
# Extract annual maxima using select_resample_op
annual_maxima = select_resample_op(precipitation_data, 
                                  op="max", 
                                  freq="YS",  # Year Start
                                  month=[5, 6, 7, 8, 9, 10])  # Optional seasonal filter

# Fit GEV distribution
from lmoments3.distr import gev
params = fit(annual_maxima, dist=gev, method="PWM")

# Calculate return levels
return_level = parametric_quantile(params, q=1-1/100)  # 100-year return level
```

### Advantages
- Simple and widely used
- Direct correspondence between return period T and annual exceedance probability (1/T)
- Follows extreme value theory for block maxima

### Limitations
- Wastes information by using only one value per year
- May include relatively small "maxima" from quiet years
- Less accurate for shorter return periods (< 10 years)
- Requires longer record lengths for robust estimates

## Partial Duration Series (PDS)

### Definition and Characteristics
- Also known as Peaks Over Threshold (POT)
- Includes all events above a specified threshold, regardless of when they occur
- Sample size depends on threshold selection and record length
- Requires ensuring independence between events (e.g., minimum time separation)
- Theoretically fitted using the Generalized Pareto Distribution (GPD)
- Requires accounting for the average exceedance rate (λ) when relating to return periods

### Manual Implementation with xclim
```python
# Extract peaks above threshold
threshold = float(precipitation_data.where(precipitation_data > 0).quantile(0.98))
peaks_above_threshold = precipitation_data.where(precipitation_data > threshold)

# Ensure independence (minimum separation between events)
min_separation_days = 3
# ... [code to ensure independence] ...
independent_peaks = precipitation_data.sel(time=independent_peak_indices)

# Calculate exceedances (difference above threshold)
exceedances = independent_peaks - threshold

# Calculate average exceedance rate per year
years_in_record = (independent_peaks.time.max() - independent_peaks.time.min()) / np.timedelta64(1, 'Y')
lambda_value = len(independent_peaks) / years_in_record

# Fit GPD to exceedances above threshold
from lmoments3.distr import gpa
params = fit(exceedances, dist=gpa, method="PWM")

# For T-year return period, calculate non-exceedance probability
T = 100  # 100-year return period
non_exceedance_prob = 1 - (-np.log(1 - 1/T)) / lambda_value

# Calculate return level (add threshold back to exceedance)
return_level = parametric_quantile(params, q=non_exceedance_prob) + threshold
```

### Advantages
- Uses more data points, potentially yielding more robust estimates
- Better represents the actual frequency of extreme events
- More accurate for shorter return periods
- Can work with shorter record lengths

### Limitations
- More complex to implement
- Requires careful threshold selection and declustering
- Return period calculation requires accounting for the exceedance rate
- Threshold selection can significantly impact results

## xclim and NOAA Atlas 14 Methodological Differences

It is important to note that xclim's built-in frequency analysis functions are primarily based on the AMS approach, as evidenced by the `select_resample_op` function with `freq="YS"` parameter. In contrast, many official precipitation frequency estimates, such as those provided by NOAA's Precipitation Frequency Data Server (PFDS) available at https://hdsc.nws.noaa.gov/pfds/pfds_map_ak.html, are developed using the PDS approach. NOAA's Atlas 14 and similar precipitation frequency analyses typically employ PDS methods because they provide more robust estimates, particularly for shorter return periods that are critical for many design applications. The PDS approach used by NOAA involves selecting all peaks above a threshold, ensuring their independence, fitting appropriate statistical distributions, and applying specific procedures for frequency conversion. This methodological difference is significant when comparing results from xclim's default implementation versus official precipitation frequency estimates, and explains why manual implementation of PDS analysis (as shown in this document) may be necessary to produce results comparable to official standards.

## Key Implementation Differences

### 1. Event Selection
- **AMS**: Uses `select_resample_op` with `freq="YS"` to select annual maxima
- **PDS**: Manually identifies peaks above a threshold and ensures independence

### 2. Distribution Selection
- **AMS**: Typically uses GEV distribution
- **PDS**: Theoretically should use GPD for exceedances above threshold

### 3. Parameter Estimation
- Both approaches can use Method of Moments (MM), Maximum Likelihood (ML), or Probability Weighted Moments (PWM)
- PWM (implemented through L-moments) is often preferred for extreme value analysis due to its robustness with small samples

### 4. Return Period Calculation
- **AMS**: Direct relationship between return period T and non-exceedance probability (1-1/T)
- **PDS**: Requires conversion using the average exceedance rate: F(x) = 1 - (-ln(1-1/T))/λ

## Confidence Intervals

For both approaches, confidence intervals can be calculated using bootstrap resampling:

```python
# Bootstrap for confidence intervals
n_bootstrap = 1000
bootstrap_return_levels = []

for i in range(n_bootstrap):
    # Sample with replacement
    bootstrap_indices = np.random.choice(len(original_data), size=len(original_data), replace=True)
    bootstrap_sample = original_data[bootstrap_indices]
    
    # Fit distribution
    bootstrap_params = fit(bootstrap_sample, dist=distribution, method="PWM")
    
    # Calculate return level
    bootstrap_return_level = parametric_quantile(bootstrap_params, q=non_exceedance_prob)
    bootstrap_return_levels.append(float(bootstrap_return_level))

# Calculate 90% confidence interval
lower_ci = np.percentile(bootstrap_return_levels, 5)
upper_ci = np.percentile(bootstrap_return_levels, 95)
```

## Common Time Aggregations for Precipitation Analysis

When analyzing extreme precipitation, multiple time aggregations are typically considered:

- 30 minutes: Short-duration, high-intensity events (urban flooding)
- 1-3 hours: Medium-duration events (urban and small catchment flooding)
- 6-24 hours: Daily-scale events (river flooding, landslides)
- 48-72 hours: Multi-day events (regional flooding)
- Weekly: Extended wet periods (groundwater flooding, reservoir management)

## Guidance on Choosing Between AMS and PDS

1. **Record Length**:
   - Short records (< 20 years): PDS generally preferred
   - Long records (> 50 years): Differences diminish, either may be appropriate

2. **Return Period Interest**:
   - Short return periods (2-10 years): PDS typically provides better estimates
   - Long return periods (> 100 years): Both have high uncertainty

3. **Sample Size Requirements**:
   - For reliable extreme value analysis, PDS can provide more data points
   - As a rule of thumb, at least 20-30 samples are recommended for fitting extreme value distributions

4. **Threshold Selection for PDS**:
   - Common methods include:
     - Fixed percentile (e.g., 98th or 99th percentile)
     - Average events per year approach (e.g., top 3 events per year)
     - Physical significance threshold

## Conclusion

Both AMS and PDS approaches have their place in extreme precipitation analysis. The choice between them should be guided by the specific application, data availability, and the return periods of interest. For comprehensive analysis, implementing both methods and comparing results can provide valuable insights into the uncertainty and robustness of extreme precipitation estimates.

The xclim library directly supports AMS through its high-level functions, while PDS requires manual implementation of threshold exceedance extraction and special handling for return period calculations.
