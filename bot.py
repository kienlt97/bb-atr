import os
import websocket as wb
from pprint import pprint

# import talib
# import numpy as np
from binance.client import Client
from binance.enums import *
# from dotenv import load_dotenv
# from database_insert import create_table
# from base_sql import Session
# from price_data_sql import CryptoPrice
# import redis
import json
import datetime 
# from fastapi.encoders import jsonable_encoder
import pandas as pd
import matplotlib.dates as mpl_dates
from bb_atr import backTest
# load_dotenv()

# this functions creates the table if it does not exist
# create_table()

# create a session
# session = Session()

BINANCE_SOCKET = "wss://stream.binance.com:9443/stream?streams=nearusdt@kline_1m"
# BINANCE_SOCKET = "wss://stream.binance.com:9443/stream?streams=ethusdt@kline_3m/btcusdt@kline_3m"

closed_prices = []
API_KEY = os.environ.get("API_KEY")
API_SECRET = os.environ.get("API_SECRET")
client = Client(API_KEY, API_SECRET)
symbol = 'NEARUSDT'
starttime = '1 day ago UTC'  # to start for 1 day ago
interval = '1m'
# rc = redis.Redis(host='192.168.40.6', port=6379, db=1)

def on_open(ws):
    # ws.send("{'event':'addChannel','channel':'ethusdt@kline_1m'}")
    print("connection opened")


def on_close(ws):
    print("closed connection")


def on_error(ws, error):
    print(error)

# def save_redis(crypto):
#     rc.set(crypto.id, json.dumps( jsonable_encoder(crypto), indent=4))
def get_historical_data(symbol):
    bars = client.get_historical_klines(symbol, interval, starttime)

    for line in bars:  # Keep only first 6 columns, 'date' 'open' 'high' 'low' 'close','volume'
        del line[6:]

    df = pd.DataFrame(bars, columns=['date', 'open', 'high', 'low', 'close', 'volume'])  # 2 dimensional tabular data
  
    # df['date'] = df['date'].apply(lambda x: pd.to_datetime(x, unit='ms').strftime('%d/%m/%Y'))
    df['open'] = pd.to_numeric(df['open'], errors='coerce').fillna(0).astype(float)
    df['high'] = pd.to_numeric(df['high'], errors='coerce').fillna(0).astype(float)
    df['low'] = pd.to_numeric(df['low'], errors='coerce').fillna(0).astype(float)
    df['close'] = pd.to_numeric(df['close'], errors='coerce').fillna(0).astype(float)
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0).astype(float)
    df['date'] = df['date'].apply(lambda x: datetime.datetime.fromtimestamp(x / 1000.0).strftime('%Y-%m-%d %H:%M:%S') )
    # print(df)
    return df

# df_final = get_historical_data(symbol)


def on_message(ws, message):
    # global df_final

    message = json.loads(message)
    # pprint(message)
    candle = message["data"]["k"]
    # pprint(candle)
    # if is_candle_closed:
    symbol = candle["s"]
    # pprint(symbol)
    closed = candle["c"]
    open = candle["o"]
    high = candle["h"]
    low = candle["l"]
    volume = candle["v"]
    interval = candle["i"]
    kline_closed = candle["x"]  
    event_time = message["data"]["E"]
    event_time = datetime.datetime.fromtimestamp(event_time / 1000.0).strftime('%Y-%m-%d %H:%M:%S')
    start_time = datetime.datetime.fromtimestamp(candle["t"] / 1000.0).strftime('%Y-%m-%d %H:%M:%S')
    close_time = datetime.datetime.fromtimestamp(candle["T"] / 1000.0).strftime('%Y-%m-%d %H:%M:%S')

    if kline_closed:
        # pprint(f"closed: {closed}")
        # pprint(f"open: {open}")
        # pprint(f"high: {high}")
        # pprint(f"low: {low}")
        # pprint(f"volume: {volume}")
        # pprint(f"interval: {interval}")
        # pprint(f"event_time: {event_time}")
        # pprint(f"start_time: {start_time}")
        # pprint(f"close_time: {close_time}")
        # print(f"kline_closed: {kline_closed}")
        
        data = [event_time, float(open), float(high), float(low), float(closed), float(volume)]
        df = pd.DataFrame([data], columns=['date', 'open', 'high', 'low', 'close', 'volume'])
        # df_final = df_final._append(df, ignore_index=True)
        # print(df_final)
        # create price entries
        backTest(df)
        print("==========================================================================")

    # Create a datetime object     

    # crypto = CryptoPrice(
    #     crypto_name=symbol,
    #     open_price=open,
    #     close_price=closed,
    #     high_price=high,
    #     low_price=low,
    #     volume=volume,
    #     interval=interval,
    #     created_time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") ,
    #     event_time = event_time
    # )
    # session.add(crypto)
    # session.commit()
    # save_redis(crypto)
    # session.close()


ws = wb.WebSocketApp(BINANCE_SOCKET, on_open=on_open, on_close=on_close, on_error=on_error, on_message=on_message)
# ws1 = wb.WebSocketApp(B_S, on_open=on_open, on_close=on_close, on_error=on_error, on_message=on_message)
ws.run_forever()
# ws1.run_forever()