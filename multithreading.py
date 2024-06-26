# IMPORTING PACKAGES

import numpy as np
import pandas as pd
from termcolor import colored as cl
from math import floor
from binance.client import Client
from profit_data import Profit
import time as t
from forex_python.converter import CurrencyRates
import concurrent.futures
from dateutil import parser

from datetime import datetime, date, timedelta, time
currentDateAndTime = datetime.now()

start_time = t.time()
c = CurrencyRates()
 
interval = '1m'
api_key = 'lwaoJYVsMOYVNIBXma32k3PoNzhB5kJ7A6TcRv6cQEqPUTEBMBZHPWiFKZ7bIRqM'  # passkey (saved in bashrc for linux)
api_secret = 'aDpaIwHf9GVJBiI36aUye5Y2zd1LKCPAUjKIMD9N5ZhzJBqNOJN6Jy09Waw7HBjO'  # secret (saved in bashrc for linux)

def get_historical_data(symbol):
    start_str = str(datetime.combine(date.today() - timedelta(days=7), time(12, 0, 0)))
    end_str =  str(datetime.combine(date.today(), time(23, 59, 59)))

    bars = client.get_historical_klines(symbol, interval, start_str=start_str,
        end_str=end_str)

    for line in bars:  # Keep only first 6 columns, 'date' 'open' 'high' 'low' 'close','volume'
        del line[6:]

    df = pd.DataFrame(bars, columns=['date', 'open', 'high', 'low', 'close', 'volume'])  # 2 dimensional tabular data
  
    # df['date'] = df['date'].apply(lambda x: pd.to_datetime(x, unit='ms').strftime('%d/%m/%Y'))
    df['open'] = pd.to_numeric(df['open'], errors='coerce').fillna(0).astype(float)
    df['high'] = pd.to_numeric(df['high'], errors='coerce').fillna(0).astype(float)
    df['low'] = pd.to_numeric(df['low'], errors='coerce').fillna(0).astype(float)
    df['close'] = pd.to_numeric(df['close'], errors='coerce').fillna(0).astype(float)
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0).astype(float)

    return df


# KELTNER CHANNEL CALCULATION
def get_kc(high, low, close, kc_lookback, multiplier, atr_lookback):
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


# KELTNER CHANNEL STRATEGY

def implement_kc_strategy(prices, kc_upper, kc_lower, date_time):
    buy_price = []
    sell_price = []
    kc_signal = []
    date_signal = []
    date_time = date_time.apply(lambda x: pd.to_datetime(x, unit='ms').strftime('%d/%m/%Y'))
 
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


def backTest(symbol):
    try:
        symbol = symbol + "USDT"
        df = get_historical_data(symbol)

        df['kc_middle'], df['kc_upper'], df['kc_lower'] = get_kc(df['high'], df['low'], df['close'], 20, 2, 10)
        buy_price, sell_price, kc_signal, date_signal = implement_kc_strategy(df['close'], df['kc_upper'],
                                                                              df['kc_lower'], df['date'])
        position = []
        position_date = []
        for i in range(len(kc_signal)):
            position_date.append(date_signal[i])
            if kc_signal[i] > 1:
                position.append(0)
            else:
                position.append(1)
                x = 0
        for i in range(len(df['close'])):
            if kc_signal[i] == 1:
                position[i] = 1
                # print("Long coin at:        ${}  -  {}".format(buy_price[i], date_signal[i]))
                x = i
            elif kc_signal[i] == -1:
                position[i] = 0
                # print("Take Profit coin at: ${}  - \t{} -> profit: {}".format(sell_price[i], date_signal[i], round(sell_price[i] - buy_price[x], 2)))
            else:
                position[i] = position[i - 1]

        close_price = df['close']
        kc_upper = df['kc_upper']
        kc_lower = df['kc_lower']
        kc_signal = pd.DataFrame(kc_signal).rename(columns={0: 'kc_signal'}).set_index(df.index)
        position = pd.DataFrame(position).rename(columns={0: 'kc_position'}).set_index(df.index)
        position_date = pd.DataFrame(position_date).rename(columns={0: 'position_date'}).set_index(df.index)

        frames = [close_price, kc_upper, kc_lower, kc_signal, position, position_date]
        strategy = pd.concat(frames, join='inner', axis=1)

        intc_ret = pd.DataFrame(np.diff(df['close'])).rename(columns={0: 'returns'})
        kc_strategy_ret = []

        for i in range(len(intc_ret)):
            returns = intc_ret['returns'][i] * strategy['kc_position'][i]
            kc_strategy_ret.append(returns)

        kc_strategy_ret_df = pd.DataFrame(kc_strategy_ret).rename(columns={0: 'kc_returns'})

        investment_value = 300
        kc_investment_ret = []
        arr_result = []

        for i in range(len(kc_strategy_ret_df['kc_returns'])):
            number_of_stocks = floor(investment_value / df['close'][i])
            returns = number_of_stocks * kc_strategy_ret_df['kc_returns'][i]
            kc_investment_ret.append(returns)
            tp = (strategy['position_date'][i], returns)
            arr_result.append(tp)

        kc_investment_ret_df = pd.DataFrame(kc_investment_ret).rename(columns={0: 'investment_returns'})
        total_investment_ret = round(sum(kc_investment_ret_df['investment_returns']), 2)
        profit_percentage = floor((total_investment_ret / investment_value) * 100)
    

        profit_obj = Profit(profit_percentage, investment_value, total_investment_ret, symbol)
        return profit_obj
    except Exception as e:
        print("error: " + e)
 

def f_common (alts_list, fname):
    print("start {}:\t {}".format(fname, currentDateAndTime))
    arr_profit = []
    for sym in alts_list:
        obj = backTest(sym)
        arr_profit.append(obj)

    time_end = float(t.time() - start_time)/60.0
    print("End -- {} - {} minute ---".format(fname,round(time_end,0)))

    return arr_profit

def f1(alts_list, fname):
    return f_common(alts_list, fname)

def f2(alts_list, fname):
    return f_common(alts_list, fname)

def f3(alts_list, fname):
    return f_common(alts_list, fname)

def f4(alts_list, fname):
    return f_common(alts_list, fname)

def f5(alts_list, fname):
    return f_common(alts_list, fname)

def f6(alts_list, fname):
    return f_common(alts_list, fname)

def f7(alts_list, fname):
    return f_common(alts_list, fname)

def f8(alts_list, fname):
    return f_common(alts_list, fname)

def f9(alts_list, fname):
    return f_common(alts_list, fname)

def f10(alts_list, fname):
    return f_common(alts_list, fname)

def f11(alts_list, fname):
    return f_common(alts_list, fname)

def f12(alts_list, fname):
    return f_common(alts_list, fname)

def f13(alts_list, fname):
    return f_common(alts_list, fname)

def f14(alts_list, fname):
    return f_common(alts_list, fname)

def f15(alts_list, fname):
    return f_common(alts_list, fname)

def f16(alts_list, fname):
    return f_common(alts_list, fname)


# if __name__ == '__main__':
with concurrent.futures.ThreadPoolExecutor() as executor:
    api_key = 'lwaoJYVsMOYVNIBXma32k3PoNzhB5kJ7A6TcRv6cQEqPUTEBMBZHPWiFKZ7bIRqM'  # passkey (saved in bashrc for linux)
    api_secret = 'aDpaIwHf9GVJBiI36aUye5Y2zd1LKCPAUjKIMD9N5ZhzJBqNOJN6Jy09Waw7HBjO'  # secret (saved in bashrc for linux)
    client = Client(api_key, api_secret)

    print("Using Binance TestNet Server")
    a = ['1INCH', 'ADA', 'ATOM', 'ANKR', 'ALGO', 'AVAX' ,
                'AAVE', 'AUDIO', 'BAT', 'CHZ', 'COTI', 'FLOW', 
                'APE', 'BNB', 'ETH',   'DOT', 'DOGE', 'EOS', 
                'ETC', 'ENJ', 'EGLD', 'FTM', 'FIL', 'AXS',
                'IOTA', 'ICP', 'KSM', 'LINK', 'LTC', 'GALA',
                'HBAR',  'MATIC', 'MANA', 'NEO', 'NEAR', 'ONE', 
                'RVN', 'SAND', 'XTZ', 'ZEC', 'SOL', 'TFUEL',
                'THETA', 'UNI', 'VET', 'XRP', 'XLM', 'ZIL'
                  ]

    ar1 = a[0:3]
    ar2 = a[3:6]
    ar3 = a[6:9]
    ar4 = a[9:12]
    ar5 = a[12:15]
    ar6 = a[15:18]
    ar7 = a[18:21]
    ar8 = a[21:24]
    ar9 = a[24:27]
    ar10 = a[27:30]
    ar11 = a[30:33]
    ar12 = a[33:36]
    ar13 = a[36:39]
    ar14 = a[39:42]
    ar15 = a[42:45]
    ar16 = a[45:48]

    f1_s = executor.submit(f1, ar1, f1.__name__)
    f2_s = executor.submit(f2, ar2, f2.__name__)
    f3_s = executor.submit(f3, ar3, f3.__name__)
    f4_s = executor.submit(f4, ar4, f4.__name__)
    f5_s = executor.submit(f5, ar5, f5.__name__)
    f6_s = executor.submit(f6, ar6, f6.__name__)
    f7_s = executor.submit(f7, ar7, f7.__name__)
    f8_s = executor.submit(f8, ar8, f8.__name__)
    f9_s = executor.submit(f9, ar9, f9.__name__)
    f10_s = executor.submit(f10, ar10, f10.__name__)
    f11_s = executor.submit(f11, ar11, f11.__name__)
    f12_s = executor.submit(f12, ar12, f12.__name__)
    f13_s = executor.submit(f13, ar13, f13.__name__)
    f14_s = executor.submit(f14, ar14, f14.__name__)
    f15_s = executor.submit(f15, ar15, f15.__name__)
    f16_s = executor.submit(f16, ar16, f16.__name__)

    arr_profit = f1_s.result() + f2_s.result() + f3_s.result() + f4_s.result() + f5_s.result() + f6_s.result() + f7_s.result() + f8_s.result() + f9_s.result() + f10_s.result() + f11_s.result() + f12_s.result() + f13_s.result() + f14_s.result() + f15_s.result() + f16_s.result()
 
    total = 0
    arr_profit.sort(key=lambda x: x.profit_percentage)

    arr_profit = arr_profit[43:48]
    for p in arr_profit:
        if p.profit_percentage > -15:
            total += p.total_investment_ret
            print("================================ {} =======================================".format(p.symbol))
            print(cl('Profit gained from the KC strategy by investing ${}, in INTC : {}'.format(p.investment_value,
                                                                                                p.total_investment_ret),
                    attrs=['bold']))
            print(cl('Profit percentage of the KC strategy : {}%'.format(p.profit_percentage), attrs=['bold']))
 
    
    vnd_profit_total = '{:,.2f}'.format((round(total * 25000,0))).replace(',','*').replace('.', ',').replace('*','.')
    print('${} ~ {}'.format(total, vnd_profit_total))
    
    time_end = float(t.time() - start_time)/60.0
    print("--- %s total minute ---" % round(time_end,0))

