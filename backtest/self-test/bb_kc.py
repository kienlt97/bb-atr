# IMPORTING PACKAGES

import pandas as pd
from binance.client import Client
from termcolor import colored
import json
import time as t
from datetime import datetime, timedelta, time, date
from binance.exceptions import BinanceAPIException, BinanceOrderException
from binance.enums import *
import numpy as np 
from telegram_bot import sendMessage

interval = '1m'
api_key = 'lwaoJYVsMOYVNIBXma32k3PoNzhB5kJ7A6TcRv6cQEqPUTEBMBZHPWiFKZ7bIRqM'  # passkey (saved in bashrc for linux)
api_secret = 'aDpaIwHf9GVJBiI36aUye5Y2zd1LKCPAUjKIMD9N5ZhzJBqNOJN6Jy09Waw7HBjO'  # secret (saved in bashrc for linux)
symbol =  "TFUELUSDT"
my_asset = 'TFUEL'
asset_symbol = 'USDT'
client = Client(api_key, api_secret)

def get_historical_data(symbol):
    try:
        start_time = t.time()
        start_str = datetime.combine(date.today() - timedelta(days=1), time(12, 0, 0))
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
        df['date'] = df['date'].apply(lambda x: datetime.fromtimestamp(x / 1000.0).strftime('%Y-%m-%d %H:%M:%S'))
        time_end = float(t.time() - start_time)
        print("get_historical_data end  {} about {} seconds ...".format(datetime.now(), time_end))
        return df
    except Exception as e:
        print('get_historical_data exception {}'.format(e))

# KELTNER CHANNEL CALCULATION
# BOLLINGER BANDS CALCULATION
def sma(df, lookback):
    try:
        sma = df.rolling(lookback).mean()
        return sma
    except Exception as e:
        print("sma exception: {}".format(e))

def get_bb(df, lookback):
    try:
        std = df.rolling(lookback).std()
        upper_bb = sma(df, lookback) + std * 2
        lower_bb = sma(df, lookback) - std * 2
        middle_bb = sma(df, lookback)
        return upper_bb, middle_bb, lower_bb
    except Exception as e:
        print("get_bb exception: {}".format(e))

# KELTNER CHANNEL CALCULATION
def get_kc(high, low, close, kc_lookback, multiplier, atr_lookback):
    try:
        tr1 = pd.DataFrame(high - low)
        tr2 = pd.DataFrame(abs(high - close.shift()))
        tr3 = pd.DataFrame(abs(low - close.shift()))
        frames = [tr1, tr2, tr3]
        tr = pd.concat(frames, axis = 1, join = 'inner').max(axis = 1)
        atr = tr.ewm(alpha = 1/atr_lookback).mean()

        kc_middle = close.ewm(kc_lookback).mean()
        kc_upper = close.ewm(kc_lookback).mean() + multiplier * atr
        kc_lower = close.ewm(kc_lookback).mean() - multiplier * atr
        return kc_middle, kc_upper, kc_lower
    except Exception as e:
        print("get_kc exception: {}".format(e))
    
# RSI CALCULATION
def get_rsi(close, lookback):
    try:
        ret = close.diff()
        up = []
        down = []
        for i in range(len(ret)):
            if ret[i] < 0:
                up.append(0)
                down.append(ret[i])
            else:
                up.append(ret[i])
                down.append(0)
        up_series = pd.Series(up)
        down_series = pd.Series(down).abs()
        up_ewm = up_series.ewm(com = lookback - 1, adjust = False).mean()
        down_ewm = down_series.ewm(com = lookback - 1,adjust = False).mean()
        rs = up_ewm/down_ewm
        rsi = 100 - (100 / (1 + rs))
        rsi_df = pd.DataFrame(rsi).rename(columns  ={0:'rsi'}).set_index(close.index)
        rsi_df = rsi_df.dropna()
        return rsi_df[3:]
    except Exception as e:
        print("get_rsi exception: {}".format(e))

# TRADING STRATEGY

signalL = 0
profit = 0
def bb_kc_rsi_strategy(prices, upper_bb, lower_bb, kc_upper, kc_lower, rsi, date_trade, re_num):
    # print("bb_kc_rsi_strategy start at:  {}".format(datetime.now()))
    start_time = t.time()
    try:
        global signalL
        global profit 
        lower_bb = lower_bb.to_numpy()
        kc_lower = kc_lower.to_numpy()
        upper_bb = upper_bb.to_numpy()
        kc_upper = kc_upper.to_numpy()
        prices = prices.to_numpy()
        rsi = rsi.to_numpy()
        last_idx = len(prices) - 1
        
        if lower_bb[last_idx] < kc_lower[last_idx] and upper_bb[last_idx] > kc_upper[last_idx] and rsi[last_idx] < 40:
            if signalL != 1:                
                sendMessage("Last date_time: {} |-- price: {} -- | (rsi < 40: Buy, rsi > 60: Sell) -- rsi: {} |-- lower_bb: {} | kc_lower: {} | upper_bb: {} | kc_upper: {} | signal: {}".format(date_trade, prices[last_idx], rsi[last_idx], lower_bb[last_idx], kc_lower[last_idx], upper_bb[last_idx], kc_upper[last_idx], signal))
                print(colored("Last date_time: {} |-- price: {} -- | (rsi < 40: Buy, rsi > 60: Sell) -- rsi: {} |-- lower_bb: {} | kc_lower: {} | upper_bb: {} | kc_upper: {} | signal: {}".format(date_trade, prices[last_idx], rsi[last_idx], lower_bb[last_idx], kc_lower[last_idx], upper_bb[last_idx], kc_upper[last_idx], signalL),'yellow'))
                print(colored("Last Buy with entry price {} - at: {}".format(prices[last_idx], date_trade), 'green'))
                sendMessage("Last Buy with entry price {} - at: {}".format(prices[last_idx], date_trade))
                signalL = 1
                print("===========================================================================================================")

        elif lower_bb[last_idx] < kc_lower[last_idx] and upper_bb[last_idx] > kc_upper[last_idx] and rsi[last_idx] > 60:
            if signalL != -1:
                sendMessage("Last date_time: {} |-- price: {} -- | (rsi < 40: Buy, rsi > 60: Sell) -- rsi: {} |-- lower_bb: {} | kc_lower: {} | upper_bb: {} | kc_upper: {} | signal: {}".format(date_trade, prices[last_idx], rsi[last_idx], lower_bb[last_idx], kc_lower[last_idx], upper_bb[last_idx], kc_upper[last_idx], signal))
                print(colored("Last date_time: {} |-- price: {} -- | (rsi < 40: Buy, rsi > 60: Sell) -- rsi: {} |-- lower_bb: {} | kc_lower: {} | upper_bb: {} | kc_upper: {} | signal: {}".format(date_trade, prices[last_idx], rsi[last_idx], lower_bb[last_idx], kc_lower[last_idx], upper_bb[last_idx], kc_upper[last_idx], signalL),'yellow'))
                print(colored("Last Sell with price {} - at: {}".format(prices[last_idx], date_trade), 'red'))
                sendMessage("Last Sell with price {} - at: {}".format(prices[last_idx], date_trade))
                signalL = -1
                print("===========================================================================================================")

    except Exception as e:
        print("bb_kc_rsi_strategy exception: {}".format(e))

def bb_kc_rsi_strategy1(prices, upper_bb, lower_bb, kc_upper, kc_lower, rsi, date_trade):
    try:
        global profit 
        lower_bb = lower_bb.to_numpy()
        kc_lower = kc_lower.to_numpy()
        upper_bb = upper_bb.to_numpy()
        kc_upper = kc_upper.to_numpy()
        prices = prices.to_numpy()
        rsi = rsi.to_numpy()
        last_idx = len(prices) - 1
        
        signal = 0
        for i in range(len(prices)):
            if i == last_idx: 
                if lower_bb[i] < kc_lower[i] and upper_bb[i] > kc_upper[i] and rsi[i] < 40:
                    if signal != 1:                        
                        print(colored("date_time: {} |-- price: {} -- | (rsi < 40: Buy, rsi > 60: Sell) -- rsi: {} |-- lower_bb: {} | kc_lower: {} | upper_bb: {} | kc_upper: {} | signal: {}".format(date_trade, prices[last_idx], rsi[last_idx], lower_bb[last_idx], kc_lower[last_idx], upper_bb[last_idx], kc_upper[last_idx], signal),'yellow'))
                        sendMessage("For date_time: {} |-- price: {} -- | (rsi < 40: Buy, rsi > 60: Sell) -- rsi: {} |-- lower_bb: {} | kc_lower: {} | upper_bb: {} | kc_upper: {} | signal: {}".format(date_trade, prices[last_idx], rsi[last_idx], lower_bb[last_idx], kc_lower[last_idx], upper_bb[last_idx], kc_upper[last_idx], signal))
                        print(colored("For Buy with entry price {} - at: {}".format(prices[i], date_trade), 'green'))
                        sendMessage("For Buy with entry price {} - at: {}".format(prices[i], date_trade))
                        print("===========================================================================================================")
                        signal = 1
                elif lower_bb[i] < kc_lower[i] and upper_bb[i] > kc_upper[i] and rsi[i] > 60:
                    if signal != -1:
                        print(colored("date_time: {} |-- price: {} -- | (rsi < 40: Buy, rsi > 60: Sell) -- rsi: {} |-- lower_bb: {} | kc_lower: {} | upper_bb: {} | kc_upper: {} | signal: {}".format(date_trade, prices[last_idx], rsi[last_idx], lower_bb[last_idx], kc_lower[last_idx], upper_bb[last_idx], kc_upper[last_idx], signal),'yellow'))
                        sendMessage("For date_time: {} |-- price: {} -- | (rsi < 40: Buy, rsi > 60: Sell) -- rsi: {} |-- lower_bb: {} | kc_lower: {} | upper_bb: {} | kc_upper: {} | signal: {}".format(date_trade, prices[last_idx], rsi[last_idx], lower_bb[last_idx], kc_lower[last_idx], upper_bb[last_idx], kc_upper[last_idx], signal))
                        print(colored("For Sell with price {} - at: {}".format(prices[i], date_trade), 'red'))
                        sendMessage("For Sell with price {} - at: {}".format(prices[i], date_trade))
                        print("===========================================================================================================")
                        signal = -1
            else:
                if lower_bb[i] < kc_lower[i] and upper_bb[i] > kc_upper[i] and rsi[i] < 40:
                    if signal != 1:
                        signal = 1
                elif lower_bb[i] < kc_lower[i] and upper_bb[i] > kc_upper[i] and rsi[i] > 60:
                    if signal != -1:
                        signal = -1        
    except Exception as e:
        print("bb_kc_rsi_strategy exception: {}".format(e))        
         
df = get_historical_data(symbol)
re_num = 0
def backTest(df_plus):
    global df
    global re_num
    try:
        _date = datetime.strptime(str(df_plus['date'][0]), '%Y-%m-%d %H:%M:%S').time() 
        if _date.hour == 0 and _date.minute == 1:
            df = get_historical_data(symbol)

        df = df._append(df_plus, ignore_index=True)
        # print("bot request at {}".format(datetime.now()))
        df['upper_bb'], df['middle_bb'], df['lower_bb'] = get_bb(df['close'], 20)
        df['kc_middle'], df['kc_upper'], df['kc_lower'] = get_kc(df['high'], df['low'], df['close'], 20, 2, 10)

        df['rsi_14'] = get_rsi(df['close'], 14)
        # df = df.dropna()
        bb_kc_rsi_strategy(df['close'], df['upper_bb'], df['lower_bb'], df['kc_upper'], df['kc_lower'], df['rsi_14'], df_plus['date'][0], re_num)
        re_num = 1
        df1 = get_historical_data(symbol)
        df1['upper_bb'], df1['middle_bb'], df1['lower_bb'] = get_bb(df1['close'], 20)
        df1['kc_middle'], df1['kc_upper'], df1['kc_lower'] = get_kc(df1['high'], df1['low'], df1['close'], 20, 2, 10)
        df1['rsi_14'] = get_rsi(df1['close'], 14)

        bb_kc_rsi_strategy1(df1['close'], df1['upper_bb'], df1['lower_bb'], df1['kc_upper'], df1['kc_lower'], df1['rsi_14'], df1['date'][0])
    except Exception as e:
        print("backTest xception: " + e)
 