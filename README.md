# Energy-Price-Forecasting

**Objective**:

- The aim of this project is to compare different deep neural network architectures on the task of predicting the next hours' electricity price by using the past values of the electricity price as well as those of another features related to energy generation and weather. conditions.


**Background information provided by Kaggle**:  

Two different datasets were provided containing hourly information about the electricity generation and weather in Spain for the period 2015-2019 (4 years). In particular:

- Weather data: Contains hourly information about the weather conditions (e.g. temperature, wind speed, humidity, rainfall, qualitative desctiption) of 5 major cities in Spain (Madrid, Barcelona, Valencia, Seville and Bilbao).

- Energy data: Contains hourly information about the generation of energy in Spain. In particular, there is info (in MW) about the amount of electricty generated by the various energy sources (fossil gas, fossil hard coal and wind energy dominate the energy grid), as well as about the total load (energy demand) of the national grid and the price of energy (€/MWh). Note: Since the generation of each energy type is in MW and the time-series contains hourly info, the value of each cell represents MWh (Megawatt hours).

**Constraints**:
- Contrary to Cross-Sectional Data, Longitudinal Data has a time-component that implicitly orders each row. Therefore, it is not a good idea to use K-Fold cross validation to test model performance. A viable approach is to use Forward Chaining to preserve the data's temporal structure.