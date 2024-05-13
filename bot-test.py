import os
import websocket as wb
from pprint import pprint

from binance.client import Client
from binance.enums import *
import json
import datetime 
import pandas as pd
from test import backTest 

BINANCE_SOCKET = "wss://stream.binance.com:9443/stream?streams=tfuelusdt@kline_1m"

closed_prices = []
API_KEY = os.environ.get("API_KEY")
API_SECRET = os.environ.get("API_SECRET")
client = Client(API_KEY, API_SECRET)

def on_open(ws):
    # ws.send("{'event':'addChannel','channel':'ethusdt@kline_1m'}")
    print("connection opened")

def on_close(ws):
    print("closed connection")

def on_error(ws, error):
    print(error)

def on_message(ws, message):
    message = json.loads(message)
    # pprint(message)
    candle = message["data"]["k"]
    # pprint(candle)
    # if is_candle_closed:
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
        data = [event_time, float(open), float(high), float(low), float(closed), float(volume)]
        df = pd.DataFrame([data], columns=['date', 'open', 'high', 'low', 'close', 'volume'])
        backTest(df)


ws = wb.WebSocketApp(BINANCE_SOCKET, on_open=on_open, on_close=on_close, on_error=on_error, on_message=on_message)
ws.run_forever()
 