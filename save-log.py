import numpy as np
import pandas as pd
from binance.client import Client 
from dateutil import parser

api_key = 'lwaoJYVsMOYVNIBXma32k3PoNzhB5kJ7A6TcRv6cQEqPUTEBMBZHPWiFKZ7bIRqM'     # passkey (saved in bashrc for linux)
api_secret = 'aDpaIwHf9GVJBiI36aUye5Y2zd1LKCPAUjKIMD9N5ZhzJBqNOJN6Jy09Waw7HBjO' # secret (saved in bashrc for linux)


def get_historical_data(start_str, end_str):
    bars = client.get_historical_klines(
        symbol='ETHUSDT',
        interval='1m',
        start_str=start_str,
        end_str=end_str)
    
    for line in bars:        # Keep only first 6 columns, 'date' 'open' 'high' 'low' 'close','volume'
        del line[7:]

    df = pd.DataFrame(bars, columns=['open_time', 'open', 'high', 'low', 'close','volume','close_time']) #  2 dimensional tabular data

    df['open'] = pd.to_numeric(df['open'], errors='coerce').fillna(0).astype(float)
    df['high'] =  pd.to_numeric(df['high'], errors='coerce').fillna(0).astype(float)
    df['low'] =  pd.to_numeric(df['low'], errors='coerce').fillna(0).astype(float)
    df['close'] =  pd.to_numeric(df['close'], errors='coerce').fillna(0).astype(float)
    df['volume'] =  pd.to_numeric(df['volume'], errors='coerce').fillna(0).astype(float)
    df['close_time'] = pd.to_datetime(df['close_time'] , unit='ms') # index set to first column = date_and_time
    return df

def log_out(df):
    output = df[['open_time', 'open', 'high', 'low', 'close','volume','close_time']]
    output.set_index('open_time', inplace=True)
    output.index = pd.to_datetime(output.index, unit='ms') # index set to first column = date_and_time

    with open('output-btc.txt', 'w') as f:
        f.write(output.to_string())

if __name__ == '__main__':
 
    client = Client(api_key, api_secret, tld ='us')
    start_str = str(parser.parse('2024-04-24 17:00:00'))
    end_str = str(parser.parse('2024-04-24 23:59:00'))
    
    df = get_historical_data(start_str, end_str)
    log_out(df)