# IMPORTING PACKAGES

import pandas as pd
from binance.client import Client
from termcolor import colored
import json
import time as t
from datetime import datetime, timedelta, time, date
from time import sleep
from binance.exceptions import BinanceAPIException, BinanceOrderException
from binance.enums import *


interval = '1m'
api_key = 'lwaoJYVsMOYVNIBXma32k3PoNzhB5kJ7A6TcRv6cQEqPUTEBMBZHPWiFKZ7bIRqM'  # passkey (saved in bashrc for linux)
api_secret = 'aDpaIwHf9GVJBiI36aUye5Y2zd1LKCPAUjKIMD9N5ZhzJBqNOJN6Jy09Waw7HBjO'  # secret (saved in bashrc for linux)
symbol =  "TFUELUSDT"
my_asset = 'TFUEL'
client = Client(api_key, api_secret)
re_num = 0
def get_historical_data(symbol):
    start_time = t.time()
    start_str = datetime.combine(date.today() - timedelta(days=2), time(12, 0, 0))
    end_str =  datetime.combine(date.today(), time(23, 59, 59))
    print("reqest get_historical_data at: {}  with start_time: {} -  end_time: {}".format(datetime.now(), start_str, end_str))

    bars = client.get_historical_klines(symbol, interval, start_str = str(start_str), end_str = str(end_str))

    for line in bars:  # Keep only first 6 columns, 'date' 'open' 'high' 'low' 'close','volume'
        del line[6:]

    df = pd.DataFrame(bars, columns=['date', 'open', 'high', 'low', 'close', 'volume'])  # 2 dimensional tabular data
  
    # df['date'] = df['date'].apply(lambda x: pd.to_datetime(x, unit='ms').strftime('%d/%m/%Y'))
    df['open'] = pd.to_numeric(df['open'], errors='coerce').fillna(0).astype(float)
    df['high'] = pd.to_numeric(df['high'], errors='coerce').fillna(0).astype(float)
    df['low'] = pd.to_numeric(df['low'], errors='coerce').fillna(0).astype(float)
    df['close'] = pd.to_numeric(df['close'], errors='coerce').fillna(0).astype(float)
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0).astype(float)
    df['date'] = df['date'].apply(lambda x: datetime.fromtimestamp(x / 1000.0).strftime('%Y-%m-%d %H:%M:%S') )
    time_end = float(t.time() - start_time)
    print("get_historical_data end  {} about {} seconds ...".format(datetime.now(), time_end))

    return df


# KELTNER CHANNEL CALCULATION
def get_kc(high, low, close, kc_lookback, multiplier, atr_lookback):
    try:
        # print("get_kc start at  {}".format(datetime.now()))
        start_time = t.time()
        tr1 = pd.DataFrame(high - low)
        tr2 = pd.DataFrame(abs(high - close.shift()))
        tr3 = pd.DataFrame(abs(low - close.shift()))
        frames = [tr1, tr2, tr3]
        tr = pd.concat(frames, axis=1, join='inner').max(axis=1)
        atr = tr.ewm(alpha=1 / atr_lookback).mean()

        kc_middle = close.ewm(kc_lookback).mean()
        kc_upper = close.ewm(kc_lookback).mean() + multiplier * atr
        kc_lower = close.ewm(kc_lookback).mean() - multiplier * atr
        time_end = float(t.time() - start_time)
        # print("get_kc end  {} about {} seconds ...".format(datetime.now(), time_end))
        return kc_middle, kc_upper, kc_lower
    except Exception as e:
        print("exption get_kc: {}".format(e))

# KELTNER CHANNEL STRATEGY
data = dict()
cummulative_buy = []
cummulative_sell = []
def implement_kc_strategy(prices, kc_upper, kc_lower, date_time, re_num, prices_real, date_real):
    # print("implement_kc_strategy start at:  {}".format(datetime.now()))
 
     # date_time = date_time.apply(lambda x: pd.to_datetime(x, unit='ms').strftime('%d/%m/%Y'))
    global data
    global cummulative_buy
    global cummulative_sell
    signal = 0
    start_time = t.time()

    for i in range(0, len(prices)):
        buy_value = 0
        sell_value = 0
        buy_key = ''
        sell_key = ''
        if prices[i] < kc_lower[i] and i < len(prices) and prices[i + 1] > prices[i]:
            if signal != 1:
                signal = 1
                buy_key = 'B_'  + str(date_time[i])
                buy_value = prices[i]
        elif prices[i] > kc_upper[i] and i < len(prices) and prices[i + 1] < prices[i]:
            if signal != -1:
                signal = -1
                sell_key = 'S_' + str(date_time[i])
                sell_value = prices[i]

        if sell_value != 0:
            if data.get(sell_key) == None:
                # if re_num != 0:
                #         cummulativeQuoteQty = f_sell()
                #         cummulative_sell.append(cummulativeQuoteQty)
                print(colored("Sell with price {} - at: {} - prices_real: {}, date_real:{}".format(sell_value, date_time[i], prices_real, date_real), 'red'))
                data[sell_key] = sell_value
        elif buy_value != 0:
            if data.get(buy_key) == None:
                # if re_num != 0:
                        # cummulativeQuoteQty = f_buy()
                        # cummulative_buy.append(cummulativeQuoteQty)
                print(colored("Buy with entry price {} - at: {} - prices_real: {}, date_real:{}".format(buy_value, date_time[i], prices_real, date_real, prices_real, date_real), 'green'))
                data[buy_key] = buy_value

    time_end = float(t.time() - start_time)
    # print("implement_kc_strategy end  {} about {} seconds ...".format(datetime.now(), time_end))

def f_sell():
    try:
        quantity_sell = getQuantity(my_asset)
        order = client.order_market_sell(symbol=symbol,quantity = quantity_sell)
        print("Sell with {} quantity".format(quantity_sell))
        sleep(2)
    except BinanceAPIException as e:
        #error handling here
        print('BinanceAPIException f_sell:', e)
    except BinanceOrderException as e:
        #error handling here
        print('BinanceOrderException f_sell: ', e)
    else:
        print(json.dumps(order, indent=2))
        
    return order['cummulativeQuoteQty']


def f_buy():
    try:
        quantity_buy = getQuantity('USDT')
        order = client.order_market_buy(symbol=symbol,quantity = quantity_buy)
        print("Buy with {} quantity".format(quantity_buy))
        sleep(2)
    except BinanceAPIException as e:
        #error handling here
        print('BinanceAPIException f_buy:', e)
    except BinanceOrderException as e:
        #error handling here
        print('BinanceOrderException f_buy: ', e)
    else:
        print(json.dumps(order, indent=2))

    return order['cummulativeQuoteQty']

df = get_historical_data(symbol)

def backTest(df_plus):
    global df
    global re_num
    try:
        df = df._append(df_plus, ignore_index=True)
        # print("bot request at {}".format(datetime.now()))
        # df = get_historical_data(symbol)
        df['kc_middle'], df['kc_upper'], df['kc_lower'] = get_kc(df['high'], df['low'], df['close'], 20, 2, 10)
        implement_kc_strategy(df['close'], df['kc_upper'],df['kc_lower'], df['date'], re_num, df_plus['close'][0], df_plus['date'][0])
        re_num = 1

    except Exception as e:
        print("exception: "+e)

 
def getQuantity(asset):
    balance = client.get_asset_balance(asset = asset)
    trades = client.get_recent_trades(symbol=symbol)
    quantity = (float(balance['free'])) / (float(trades[0]['price']))
 
    response = client.get_symbol_info(symbol=symbol)
    lotSizeFloat = format(float(response["filters"][1]["stepSize"]), '.20f')

    # LotSize
    numberAfterDotLot = str(lotSizeFloat.split(".")[1])
    indexOfOneLot = numberAfterDotLot.find("1")
    if indexOfOneLot == -1:
        quantity = int(quantity)
    else:
        quantity = round(float(quantity), int(indexOfOneLot))
    return quantity