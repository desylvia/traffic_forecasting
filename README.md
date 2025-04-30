# traffic_forecasting

This repository is about internet traffic forecasting using the biggest internet traffic in Indonesia. The model used is constrained linear regression with modification to include seasonality. We choosed this traditional model to keep error rate (we use SMAPE) under 5% (stakeholder request) because other more sophisticated models seems to fail to keep error rate under 5% or at least under 10%.

model.py is model code we use and utils.py is utilization functions.

All code is in python, input from database and output also store to database. Database is using maria db older version.

This code is only one of many layers in many internet layers that i forecasted.
