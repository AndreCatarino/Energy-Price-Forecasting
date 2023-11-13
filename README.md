# Energy-Price-Forecasting

Objective:

- The primary objective of this project is to employ various deep neural network architectures to predict the electricity price for the next hour. The prediction utilizes historical data on electricity prices, energy generation from different sources, and weather conditions in five major cities in Spain (Madrid, Barcelona, Valencia, Seville, and Bilbao) over the period 2015-2019.

Background information provided by Kaggle:

- Two key datasets were provided, encompassing hourly information on weather conditions and energy generation in Spain. The weather data includes details such as temperature, wind speed, humidity, rainfall, and qualitative descriptions for the specified cities. The energy data comprises information on electricity generation by various sources, total grid load (energy demand), and energy prices (€/MWh).

Constraints:

- Longitudinal data, inherent in time-series datasets, requires a careful approach to model evaluation. K-Fold cross-validation is unsuitable due to the temporal structure; hence, Forward Chaining is implemented to preserve the data's temporal order.

Approach:

- The project explores multiple deep learning architectures, including LSTM, TCN, an attention mechanism and hybrid models like CNN-LSTM, to predict the next hour's electricity price (sequence-to-vector). Forward chaining is employed to assess the best performing model. During the model evaluation on the test set, the performance of the selected model is then compared against a persistence model baseline, surpassing the baseline in all considered regression metrics (MAE, MSE, RMSE, and R2).