import os
import websocket as wb
from pprint import pprint
from datetime import datetime
import numpy as np
from binance.client import Client
from binance.enums import *
from dotenv import load_dotenv
from database_insert import create_table
from base_sql import Session
from price_data_sql import CryptoPrice
import redis
import json
from fastapi.encoders import jsonable_encoder
import pandas as pd
import matplotlib.dates as mpl_dates
from types import SimpleNamespace
from math import floor
import time
from termcolor import colored as cl
from termcolor import colored
from itertools import groupby


load_dotenv()
# this functions creates the table if it does not exist
create_table()
# create a session
session = Session()

currentDateAndTime = datetime.now()
start_time = time.time()
BINANCE_SOCKET = "wss://stream.binance.com:9443/stream?streams=ethusdt@kline_1m"
# BINANCE_SOCKET = "wss://stream.binance.com:9443/stream?streams=ethusdt@kline_3m/btcusdt@kline_3m"
TRADE_SYMBOL = "ETHUSD"
symbol = TRADE_SYMBOL
global closed_prices
closed_prices = []
flag = 0

API_KEY = os.environ.get("API_KEY")
API_SECRET = os.environ.get("API_SECRET")
client = Client(API_KEY, API_SECRET, tld="us")
arr_fu = []

rc = redis.Redis(host='192.168.40.11', port=6379, db=1)
 

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
    # arr_fu = rc.get()
    global closed_prices
    closed_prices = getAllData()
    print("size closed_prices: {}".format(len(closed_prices)))
    print("connection opened")


def on_close(ws):
    print("closed connection")

def on_error(ws, error):
    print(error)

def save_redis(crypto):
    rc.set(crypto.id, json.dumps( jsonable_encoder(crypto), indent=4))

def getAllData():
    closed_prices = []
    for key in rc.scan_iter():
        closed_prices.append(CryptoPrice(**json.loads(rc.get(key))))
    return closed_prices

def get_kc(high, low, close, kc_lookback, multiplier, atr_lookback):
    try:
        tr1 = pd.DataFrame(high - low)
        tr2 = pd.DataFrame(abs(high - close.shift()))
        tr3 = pd.DataFrame(abs(low - close.shift()))
        frames = [tr1, tr2, tr3]
        tr = pd.concat(frames, axis=1, join='inner').max(axis=1)
        atr = tr.ewm(alpha=1 / atr_lookback).mean()

        kc_middle = close.ewm(kc_lookback).mean()
        kc_upper = close.ewm(kc_lookback).mean() + multiplier * atr
        kc_lower = close.ewm(kc_lookback).mean() - multiplier * atr

        return kc_middle, kc_upper, kc_lower
    except Exception as e:
        print("exception: ", e)

# KELTNER CHANNEL STRATEGY
def implement_kc_strategy(prices, kc_upper, kc_lower, date_time):
    buy_price = []
    sell_price = []
    kc_signal = []
    date_signal = []
    fomatDT = ''

    if flag == 0:
        fomatDT = '%d/%m/%Y'
    else:
        fomatDT = '%m/%Y'

    # date_time = date_time.apply(lambda x: pd.to_datetime(x, unit='ms').strftime(fomatDT))
 
    signal = 0
    for i in range(0, len(prices)):
        date_signal.append(date_time[i])
        if prices[i] < kc_lower[i] and i < len(prices) - 1 and prices[i + 1] > prices[i]:
            if signal != 1:
                buy_price.append(prices[i])
                sell_price.append(np.nan)
                signal = 1
                kc_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                kc_signal.append(0)
        elif prices[i] > kc_upper[i] and i < len(prices) - 1 and prices[i + 1] < prices[i]:
            if signal != -1:
                buy_price.append(np.nan)
                sell_price.append(prices[i])
                signal = -1
                kc_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                kc_signal.append(0)
        else:
            buy_price.append(np.nan)
            sell_price.append(np.nan)
            kc_signal.append(0)


    return buy_price, sell_price, kc_signal, date_signal

def groupDateTime(arr_result, symbol, investment_value, currency):
    # Group the tuples by key and calculate the sum of values for each group
    grouped = [(key, sum(value for _, value in group))
                for key, group in groupby(arr_result, key=lambda x: x[0])]
    print("==========================  {}  ======================".format(symbol))
    date = []
    profit = []
    losses = []
    for rs in grouped:
        color = ''
        tk_profit = floor((round(rs[1],3) / investment_value) * 100)
        vnd_profit = '{:,.2f}'.format((round(rs[1]*currency,0))).replace(',','*').replace('.', ',').replace('*','.')
        rs_str = 'date:' + str(rs[0]) +'\t - profit: $'+ str(round(rs[1],3)) + '\t ~ VND: ' +  vnd_profit +   '\t -> ' + str(tk_profit) +'%'
        
        date.append(rs[0])
        if (round(rs[1],3) < 0):
            losses.append(rs[1])
            profit.append(0)
            color = 'red'
        else:
            profit.append(rs[1])
            losses.append(0)
            color = 'green'
        print(colored(rs_str, color))

    list = [profit, losses, date]
    data = pd.DataFrame(list) # Each list would be added as a row
    data = data.transpose() # To Transpose and make each rows as columns
    data.columns = ['profit', 'losses', 'date'] # Rename the columns
    data.head()

    return data

def backTest(df):
    try:
        df['kc_middle'], df['kc_upper'], df['kc_lower'] = get_kc(df['high_price'], df['low_price'], df['close_price'], 20, 2, 10)
        buy_price, sell_price, kc_signal, date_signal = implement_kc_strategy(df['close_price'], df['kc_upper'],
                                                                            df['kc_lower'], df['close_time'])
        position = []
        position_date = []
        for i in range(len(kc_signal)):
            position_date.append(date_signal[i])
            if kc_signal[i] > 1:
                position.append(0)
            else:
                position.append(1)
        for i in range(len(df['close_price'])):
            if kc_signal[i] == 1:
                position[i] = 1
            elif kc_signal[i] == -1:
                position[i] = 0
            else:
                position[i] = position[i - 1]


        close_price = df['close_price']
        kc_upper = df['kc_upper']
        kc_lower = df['kc_lower']
        kc_signal = pd.DataFrame(kc_signal).rename(columns={0: 'kc_signal'}).set_index(df.index)
        position = pd.DataFrame(position).rename(columns={0: 'kc_position'}).set_index(df.index)
        position_date = pd.DataFrame(position_date).rename(columns={0: 'position_date'}).set_index(df.index)
 
        frames = [close_price, kc_upper, kc_lower, kc_signal, position, position_date]
        strategy = pd.concat(frames, join='inner', axis=1)

        intc_ret = pd.DataFrame(np.diff(df['close_price'])).rename(columns={0: 'returns'})
        kc_strategy_ret = []

        for i in range(len(intc_ret)):
            returns = intc_ret['returns'][i] * strategy['kc_position'][i]
            kc_strategy_ret.append(returns)

        kc_strategy_ret_df = pd.DataFrame(kc_strategy_ret).rename(columns={0: 'kc_returns'})

        investment_value = 1000
        currency = 25000
        kc_investment_ret = []
        arr_result = []

        for i in range(len(kc_strategy_ret_df['kc_returns'])):
            number_of_stocks = floor(investment_value / df['close_price'][i])
            returns = number_of_stocks * kc_strategy_ret_df['kc_returns'][i]
            kc_investment_ret.append(returns)
            tp = (strategy['position_date'][i], returns)
            arr_result.append(tp)

        groupDateTime(arr_result, symbol, investment_value, currency)

        kc_investment_ret_df = pd.DataFrame(kc_investment_ret).rename(columns={0: 'investment_returns'})
        total_investment_ret = round(sum(kc_investment_ret_df['investment_returns']), 2)
        profit_percentage = floor((total_investment_ret / investment_value) * 100)
    
        vnd_profit_total = '{:,.2f}'.format((round(total_investment_ret * currency,0))).replace(',','*').replace('.', ',').replace('*','.')
        print(cl('Profit gained from the KC strategy by investing ${}, in INTC : ${} ~ {} VND'.format(investment_value,total_investment_ret, vnd_profit_total), attrs=['bold']))
        print(cl('Profit percentage of the KC strategy : {}%'.format(profit_percentage), attrs=['bold']))
        time_end = float(time.time() - start_time)
        print("--- %s seconds ---" % time_end)
    except Exception as e:
        print("exception backTest: ", e)


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
    open_time = pd.to_datetime(candle["t"], unit='ms').to_pydatetime()
    close_time = pd.to_datetime(candle["T"], unit='ms').to_pydatetime()     

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
        created_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S") ,
        event_time = event_time,
        open_time = open_time,
        close_time = close_time
    )
    closed_prices.append(crypto)

    df = pd.DataFrame([vars(price) for price in closed_prices])

    df['high_price'] = pd.to_numeric(df['high_price'], errors='coerce').fillna(0).astype(float)
    df['low_price'] = pd.to_numeric(df['low_price'], errors='coerce').fillna(0).astype(float)
    df['close_price'] = pd.to_numeric(df['close_price'], errors='coerce').fillna(0).astype(float)

    backTest(df)


    # session.add(crypto)
    # session.commit()
    # save_redis(crypto)
    # session.close()
 
    
ws = wb.WebSocketApp(BINANCE_SOCKET, on_open=on_open, on_close=on_close, on_error=on_error, on_message=on_message)
ws.run_forever()

