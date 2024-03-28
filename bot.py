import os
import websocket as wb
from pprint import pprint

# import talib
# import numpy as np
from binance.client import Client
from binance.enums import *
from dotenv import load_dotenv
from database_insert import create_table
from base_sql import Session
from price_data_sql import CryptoPrice
import redis
import json
import datetime 
from fastapi.encoders import jsonable_encoder
import pandas as pd
import matplotlib.dates as mpl_dates

load_dotenv()

# this functions creates the table if it does not exist
create_table()

# create a session
session = Session()

BINANCE_SOCKET = "wss://stream.binance.com:9443/stream?streams=ethusdt@kline_3m"
# BINANCE_SOCKET = "wss://stream.binance.com:9443/stream?streams=ethusdt@kline_3m/btcusdt@kline_3m"
TRADE_SYMBOL = "ETHUSD"
closed_prices = []

API_KEY = os.environ.get("API_KEY")
API_SECRET = os.environ.get("API_SECRET")
client = Client(API_KEY, API_SECRET, tld="us")

rc = redis.Redis(host='192.168.40.6', port=6379, db=1)

def order(side, size, order_type=ORDER_TYPE_MARKET, symbol=TRADE_SYMBOL):
    # order_type = "MARKET" if side == "buy" else "LIMIT"
    try:
        order = client.create_order(
            symbol=symbol,
            side=side,
            type=order_type,
            quantity=size,
        )
        print(order)
        return True
    except Exception as e:
        print(e)
        return False

def on_open(ws):
    # ws.send("{'event':'addChannel','channel':'ethusdt@kline_1m'}")
    print("connection opened")


def on_close(ws):
    print("closed connection")


def on_error(ws, error):
    print(error)

def save_redis(crypto):
    rc.set(crypto.id, json.dumps( jsonable_encoder(crypto), indent=4))
    
def on_message(ws, message):
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
    event_time = pd.to_datetime(message["data"]["E"], unit='ms').to_pydatetime()
            
    # df['date'] = df['date'].apply(mpl_dates.date2num)
    # print(candle)
    # pprint(f"closed: {closed}")
    # pprint(f"open: {open}")
    # pprint(f"high: {high}")
    # pprint(f"low: {low}")
    # pprint(f"volume: {volume}")
    # pprint(f"interval: {interval}")
    # pprint(f"event_time: {event_time}")
    # # create price entries
    # print(symbol)
    # print("==========================================================================")
    # Create a datetime object     

    crypto = CryptoPrice(
        crypto_name=symbol,
        open_price=open,
        close_price=closed,
        high_price=high,
        low_price=low,
        volume=volume,
        interval=interval,
        created_time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") ,
        event_time = event_time
    )
    session.add(crypto)
    session.commit()
    save_redis(crypto)
    session.close()


ws = wb.WebSocketApp(BINANCE_SOCKET, on_open=on_open, on_close=on_close, on_error=on_error, on_message=on_message)
# ws1 = wb.WebSocketApp(B_S, on_open=on_open, on_close=on_close, on_error=on_error, on_message=on_message)
ws.run_forever()
# ws1.run_forever()