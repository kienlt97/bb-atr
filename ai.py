import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import lime
import lime.lime_tabular
import shap
from eodhd import APIClient

api_key = 'lwaoJYVsMOYVNIBXma32k3PoNzhB5kJ7A6TcRv6cQEqPUTEBMBZHPWiFKZ7bIRqM'     # passkey (saved in bashrc for linux)
api_secret = 'aDpaIwHf9GVJBiI36aUye5Y2zd1LKCPAUjKIMD9N5ZhzJBqNOJN6Jy09Waw7HBjO' # secret (saved in bashrc for linux)
api = APIClient(api_key)

# Step 1: Extract AAPL historical data

symbol = 'AAPL'
start_date = '2020-01-01'
end_date = '2022-01-01'

stock_data = client.get_historical_data(symbol, 'd', start_date, end_date)
stock_data = stock_data.drop(['symbol', 'interval'], axis = 1)

stock_data.tail()