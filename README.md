# gdg-hackathon
Downscaling of Satellite based air quality map using AI/ML
Description: Develop an AI/ML (Artificial Intelligence/Machine Learning) model to generate fine spatial resolution air quality map from coarse resolution satellite data. It should utilise existing python-based ML libraries. Developed model need to be validated with unseen independent data. Challenge: To utilise large satellite data having gaps under cloudy conditions To select suitable ML algorithm and ensure optimal fitting of ML model for desired accuracy To validate model output with unseen independent data Usage: To enhance air quality knowledge, Sharpen focus at local level Users: Researchers and government bodies monitoring/working on air quality assessment Available Solutions (if Yes, reasons for not using them): Individual components are available, comprehensive and proven solution does not exist. Desired Outcome: Fine resolution air quality map of NO2

Data source: 
1. Satellite derived daily Tropospheric NO2 from either of the following links: (a) Daily Tropospheric NO2 from TROPOMI/Sentinel-5p – Swath data 
https://search.earthdata.nasa.gov/search/granules?p=C2089270961-
GES_DISC&pg[0][v]=f&pg[0][gsk]=-start_date&q=tropomi%20no2&tl=1726635700.002!3!!  
(b)	Daily Tropospheric NO2 from TROPOMI/Sentinel-5p (using google earth engine) – gridded geotif format  
https://developers.google.com/earth-
engine/datasets/catalog/COPERNICUS_S5P_OFFL_L3_NO2#description  
(c)	Daily Tropospheric NO2 from OMI/Aura – gridded data 
https://search.earthdata.nasa.gov/search/granules?p=C1266136111GES_DISC&pg[0][v]=f&pg[0][gsk]=start_date&q=omi%20tropospheric%20no2&tl=1726635700.002!3!! 
(d)	Daily Tropospheric NO2 from OMI/Aura – gridded data https://measures.gesdisc.eosdis.nasa.gov/data/MINDS/OMI_MINDS_NO2d.1.1/2024/ 
 
2. Either of the above data (daily tropospheric NO2) to be used in conjunction with ground-based NO2 concentration monitored by CPCB: https://app.cpcbccr.com/ccr/#/caaqm-dashboard-all/caaqmlanding (go to Advance Search to download data for different stations) 
 
Machine learning algorithm 
Generally, Random Forest, XGBoost and Neural Network (ANN/CNN) are good for downscaling. However, students may explore different AI/ML algorithms and can decide themselves which algorithm to be used.  
 
