# IMPORTING PACKAGES

import pandas as pd
from binance.client import Client
from termcolor import colored
import json
import time as t
from datetime import datetime, timedelta, time, date
from binance.exceptions import BinanceAPIException, BinanceOrderException
from binance.enums import *
from telegram_bot import sendMessage
import numpy as np 

interval = '1m'
api_key = 'lwaoJYVsMOYVNIBXma32k3PoNzhB5kJ7A6TcRv6cQEqPUTEBMBZHPWiFKZ7bIRqM'  # passkey (saved in bashrc for linux)
api_secret = 'aDpaIwHf9GVJBiI36aUye5Y2zd1LKCPAUjKIMD9N5ZhzJBqNOJN6Jy09Waw7HBjO'  # secret (saved in bashrc for linux)
symbol =  "TFUELUSDT"
my_asset = 'TFUEL'
asset_symbol = 'USDT'
client = Client(api_key, api_secret)
re_num = 0

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
        sendMessage('📛🆘 get_historical_data exception: {}'.format(e))
        print('get_historical_data exception {}'.format(e))

# KELTNER CHANNEL CALCULATION
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
        sendMessage('📛🆘 get_kc exception: {}'.format(e))
        print("get_kc exception: {}".format(e))

# KELTNER CHANNEL STRATEGY
data = dict()
bougth_value = 0.10783
profit = 0
signal = 1
rule_bougth_value = 0.10998659999999999
count_sent = 0
def implement_kc_strategy(prices, kc_upper, kc_lower, date_time, re_num):
    print("implement_kc_strategy start at:  {}".format(datetime.now()))
    global data
    global bougth_value
    global signal 
    global profit
    global rule_bougth_value
    global count_sent
    start_time = t.time()

    try:
        last_idx = len(prices) - 1
        buy_value = 0
        sell_value = 0    

        print(colored("date_time: {}, --(kc_lower > prices -> buy) kc_lower: {} - prices: {} - kc_upper: {} (prices > kc_upper -> sell) - bougth_value: {}, rule_bougth_value :{} signal:{}".format(date_time[last_idx], kc_lower[last_idx] , prices[last_idx], kc_upper[last_idx] ,bougth_value, rule_bougth_value,signal),'yellow'))

        if prices[last_idx] < kc_lower[last_idx] and prices[last_idx] < prices[last_idx-1]:
            if signal != 1:
                signal = 1
                buy_key = 'B_'  + str(date_time[last_idx])
                buy_value = prices[last_idx]

        elif prices[last_idx] > kc_upper[last_idx] and prices[last_idx] > prices[last_idx - 1]:
            
            if  prices[last_idx] < bougth_value:
                rule_bougth_value = bougth_value*2/100 + bougth_value
                count_sent =+ 1
            elif prices[last_idx] > bougth_value and rule_bougth_value == 0:
                rule_bougth_value = bougth_value

            if signal == 1 and count_sent == 10:
                print(colored("prices:{} - bougth_value: {} - rule_bougth_value: {}".format(prices[last_idx],bougth_value, rule_bougth_value ),'red'))
                sendMessage("❗️❗️❗️ prices:{} - bougth_value: {} - rule_bougth_value: {}".format(prices[last_idx],bougth_value, rule_bougth_value ))

            if prices[last_idx] > rule_bougth_value :
                if signal != -1 and signal != 0:
                    signal = -1
                    sell_key = 'S_' + str(date_time[last_idx])
                    sell_value = prices[last_idx]     
        
        if sell_value != 0:
            if data.get(sell_key) == None:
                if re_num != 0:
                    cummulativeQuoteQty = f_sell()
                    profit = profit + float(cummulativeQuoteQty)
                    sendMessage('👑👑💸💸💰💰💵💵🏦🏦 profit: {}'.format(profit))
                print(colored("Sell with price {} - at: {}".format(sell_value, date_time[last_idx]), 'red'))
                data[sell_key] = sell_value

        elif buy_value != 0:
            if data.get(buy_key) == None:
                if re_num != 0:
                    cummulativeQuoteQty, bougth_price = f_buy()
                    bougth_value = bougth_price
                    profit = profit - float(cummulativeQuoteQty)
                print(colored("Buy with entry price {} - at: {}".format(buy_value, date_time[last_idx]), 'green'))
                data[buy_key] = buy_value
            
        time_end = float(t.time() - start_time)
        print("implement_kc_strategy end {} about {} seconds ...".format(datetime.now(), time_end))
    except Exception as e:
        sendMessage('📛🆘 implement_kc_strategy exception: {}'.format(e))
        print("implement_kc_strategy exception: {}".format(e))
    
def f_sell():
    print("f_sell start at:  {}".format(datetime.now()))
    start_time = t.time()
    cummulativeQuoteQty = 0
    try:
        quantity_sell = getQuantitySell()
        order = client.order_market_sell(symbol=symbol,quantity = quantity_sell)
        print("Sell with {} quantity".format(quantity_sell))
    except BinanceAPIException as e:
        #error handling here
        sendMessage('📛🆘 BinanceAPIException f_sell exception: {}'.format(e))
        print('BinanceAPIException f_sell:', e)
    except BinanceOrderException as e:
        #error handling here
        sendMessage('📛🆘 BinanceOrderException f_sell exception: {}'.format(e))
        print('BinanceOrderException f_sell: ', e)
    else:
        print(json.dumps(order, indent=2))
        cummulativeQuoteQty = order['cummulativeQuoteQty']
        message = ''
        for i in range(0, len(order['fills'])):
            price = order['fills'][i]['price']
            qty = order['fills'][i]['qty']
            commission = order['fills'][i]['commission']
            total_money = float(price) * round(float(qty), 2)
            message = message + "\n📢🧧🧧 Sell with entry price: {} - quantity: {} -> :  at: total money: {} - commission {} ,cummulativeQuoteQty:{} 📢🧧🧧".format(price, round(float(qty), 2), total_money, commission, cummulativeQuoteQty)
        sendMessage(message)
    
    print("f_sell end {} about {} seconds ...".format(datetime.now(), float(t.time() - start_time)))

    return cummulativeQuoteQty

def f_buy():
    print("f_buy start at:  {}".format(datetime.now()))
    start_time = t.time()
    cummulativeQuoteQty = 0
    bougth_price = 0
    try:
        quantity_buy = getQuantityBuy()
        order = client.order_market_buy(symbol=symbol,quantity = quantity_buy)
        print("Buy with {} quantity".format(quantity_buy))
    except BinanceAPIException as e:
        #error handling here
        sendMessage('📛🆘BinanceAPIException f_buy exception: {}'.format(e))
        print('BinanceAPIException f_buy:', e)
    except BinanceOrderException as e:
        #error handling here
        sendMessage('📛🆘 BinanceOrderException f_buy exception: {}'.format(e))
        print('BinanceOrderException f_buy: ', e)
    else:
        cummulativeQuoteQty = order['cummulativeQuoteQty']
        print(json.dumps(order, indent=2))
        message = ''
        for i in range(0, len(order['fills'])):
            price = order['fills'][i]['price']
            qty = order['fills'][i]['qty']
            commission = order['fills'][i]['commission']
            total_money = float(price) * round(float(qty), 2)
            message = message + "\n🔔💹 Buy with entry price: {} - quantity: {} -> :  at: total money: {} - commission {}: ,cummulativeQuoteQty:{}  🔔💹".format(price, round(float(qty), 2), total_money, commission,cummulativeQuoteQty)
            
            if bougth_price <= price:
                bougth_price = price
        sendMessage(message)    

    print("f_buy end {} about {} seconds ...".format(datetime.now(), float(t.time() - start_time)))
    return cummulativeQuoteQty, bougth_price
         
df = get_historical_data(symbol)

def backTest(df_plus):
    global df
    global re_num
    try:
        _date = datetime.strptime(str(df_plus['date'][0]), '%Y-%m-%d %H:%M:%S').time() 
        if _date.hour == 0 and _date.minute == 1:
            df = get_historical_data(symbol)
            sendMessage('✅✅ Change data get_historical_data:')

        df = df._append(df_plus, ignore_index=True)
        print("bot request at {}".format(datetime.now()))
        # df = get_historical_data(symbol)
        df['kc_middle'], df['kc_upper'], df['kc_lower'] = get_kc(df['high'], df['low'], df['close'], 20, 2, 10)
        implement_kc_strategy(df['close'], df['kc_upper'],df['kc_lower'], df['date'], re_num)
        re_num = 1
    except Exception as e:
        sendMessage('📛🆘 backTest exception: {}'.format(e))
        print("backTest xception: " + e)

def getQuantityBuy():
    print("getQuantityBuy start at:  {}".format(datetime.now()))
    start_time = t.time()
    try:
        balance = client.get_asset_balance(asset = asset_symbol)
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

        print("getQuantityBuy end {} about {} seconds ...".format(datetime.now(), float(t.time() - start_time)))
        return quantity
    except Exception as e:
        sendMessage('📛🆘 getQuantityBuy exception: {}'.format(e))
        print("getQuantityBuy xception: " + e)
    

def getQuantitySell():
    print("getQuantitySell start at:  {}".format(datetime.now()))
    start_time = t.time()
    try:
        balance = client.get_asset_balance(asset=my_asset)

        print("getQuantitySell end {} about {} seconds ...".format(datetime.now(), float(t.time() - start_time)))
        return int(float(balance['free']))
    except Exception as e:
        sendMessage('📛🆘 getQuantitySell exception: {}'.format(e))
        print("getQuantitySell xception: " + e)
    
  