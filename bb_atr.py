# IMPORTING PACKAGES

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from termcolor import colored as cl
from math import floor
from binance.client import Client
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from profit_data import Profit
from itertools import groupby
from termcolor import colored
from decimal import Decimal
from typing import Union, Optional, Dict
import json
from decimal import Decimal as D, ROUND_DOWN, ROUND_UP
import time
from datetime import datetime


plt.rcParams['figure.figsize'] = (20, 10)
plt.style.use('fivethirtyeight')

starttime = '2 day ago UTC'  # to start for 1 day ago
interval = '1m'
# symbol = 'DOGEUSDT'   # Change symbol here e.g. BTCUSDT, BNBBTC, ETHUSDT, NEOBTC
api_key = 'lwaoJYVsMOYVNIBXma32k3PoNzhB5kJ7A6TcRv6cQEqPUTEBMBZHPWiFKZ7bIRqM'  # passkey (saved in bashrc for linux)
api_secret = 'aDpaIwHf9GVJBiI36aUye5Y2zd1LKCPAUjKIMD9N5ZhzJBqNOJN6Jy09Waw7HBjO'  # secret (saved in bashrc for linux)
symbol =  "NEARUSDT"

client = Client(api_key, api_secret)

def get_historical_data(symbol):
    bars = client.get_historical_klines(symbol, interval, starttime)
    print("call get_historical_data")
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
    return df


# KELTNER CHANNEL CALCULATION
def get_kc(high, low, close, kc_lookback, multiplier, atr_lookback):
    print("request get_kc at \t{}".format(datetime.now()))
    start_time = time.time()
    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift()))
    tr3 = pd.DataFrame(abs(low - close.shift()))
    frames = [tr1, tr2, tr3]
    tr = pd.concat(frames, axis=1, join='inner').max(axis=1)
    atr = tr.ewm(alpha=1 / atr_lookback).mean()

    kc_middle = close.ewm(kc_lookback).mean()
    kc_upper = close.ewm(kc_lookback).mean() + multiplier * atr
    kc_lower = close.ewm(kc_lookback).mean() - multiplier * atr
    time_end = float(time.time() - start_time)
    print("--- get_kc end \t{} about {} seconds ---".format(datetime.now(), time_end))
    return kc_middle, kc_upper, kc_lower


# KELTNER CHANNEL STRATEGY
data = dict()

def implement_kc_strategy(prices, kc_upper, kc_lower, date_time):
    print("request implement_kc_strategy at \t{}".format(datetime.now()))
    buy_prices = []
    sell_prices = []
    kc_signal = []
    date_signal = []
    # date_time = date_time.apply(lambda x: pd.to_datetime(x, unit='ms').strftime('%d/%m/%Y'))
    global data
    signal = 0
    start_time = time.time()

    for i in range(0, len(prices)):
        buy_value = 0
        sell_value = 0
        buy_key = ''
        sell_key = ''
        date_signal.append(date_time[i])
        if prices[i] < kc_lower[i] and i < len(prices) - 1 and prices[i + 1] > prices[i]:
            if signal != 1:
                buy_prices.append(prices[i])
                sell_prices.append(np.nan)
                signal = 1
                kc_signal.append(signal)
                buy_key = 'B_'  + str(date_time[i])
                buy_value = prices[i]
            else:
                buy_prices.append(np.nan)
                sell_prices.append(np.nan)
                kc_signal.append(0)
        elif prices[i] > kc_upper[i] and i < len(prices) - 1 and prices[i + 1] < prices[i]:
            if signal != -1:
                buy_prices.append(np.nan)
                sell_prices.append(prices[i])
                signal = -1
                kc_signal.append(signal)
                sell_key = 'S_' + str(date_time[i])
                sell_value = prices[i]
            else:
                buy_prices.append(np.nan)
                sell_prices.append(np.nan)
                kc_signal.append(0)
        else:
            buy_prices.append(np.nan)
            sell_prices.append(np.nan)
            kc_signal.append(0)
        
        if sell_value != 0:
            if data.get(sell_key) == None:
                print(colored("Sell with price {} - at: {}".format(prices[i], date_time[i]), 'red'))
                data[sell_key] = sell_value
            # else:
            #     print("Nothing to do ...")
        elif buy_value != 0:
            if data.get(buy_key) == None:
                print(colored("Buy with entry price {} - at: {}".format(prices[i], date_time[i]), 'green'))
                data[buy_key] = buy_value
            # else:
            #     print("Nothing to do ...")
                
    time_end = float(time.time() - start_time)
    print("--- implement_kc_strategy end \t{} about {} seconds ---".format(datetime.now(), time_end))

    return buy_prices, sell_prices, kc_signal, date_signal


def plot_graph(symbol, df, entry_prices, exit_prices):
    fig = make_subplots(rows=3, cols=1, subplot_titles=['Close + BB-ATR'])

    df.set_index('date', inplace=True)
    df.index = pd.to_datetime(df.index, unit='ms')  # index set to first column = date_and_time

    #  Plot close price
    fig.add_trace(
        go.Line(x=df.index, y=np.array(df['close'], dtype=np.float32), line=dict(color='blue', width=1), name='Close'),
        row=1, col=1)

    #  Plot bollinger bands
    bb_high = df['kc_upper'].astype(float).to_numpy()
    bb_mid = df['kc_middle'].astype(float).to_numpy()
    bb_low = df['kc_lower'].astype(float).to_numpy()
    fig.add_trace(go.Line(x=df.index, y=bb_high, line=dict(color='green', width=1), name='BB High'), row=1, col=1)
    fig.add_trace(go.Line(x=df.index, y=bb_mid, line=dict(color='#ffd866', width=1), name='BB Mid'), row=1, col=1)
    fig.add_trace(go.Line(x=df.index, y=bb_low, line=dict(color='red', width=1), name='BB Low'), row=1, col=1)

    #  Add buy and sell indicators
    fig.add_trace(
        go.Scatter(x=df.index, y=np.array(entry_prices, dtype=np.float32), marker_symbol='arrow-up', marker=dict(
            color='green', size=15
        ), mode='markers', name='Buy'))
    fig.add_trace(
        go.Scatter(x=df.index, y=np.array(exit_prices, dtype=np.float32), marker_symbol='arrow-down', marker=dict(
            color='red', size=15
        ), mode='markers', name='Sell'))

    fig.update_layout(
        title={'text': f'{symbol} with BB-RSI-KC' + '/ interval: ' + interval + '-starttime: ' + starttime, 'x': 0.5},
        autosize=False,
        width=2000, height=3000)
    fig.update_yaxes(range=[0, 1000000000], secondary_y=True)
    fig.update_yaxes(visible=True, secondary_y=True)  # hide range slider

    fig.show()


df = get_historical_data('NEARUSDT')

def backTest(df_plus):
    global df
    try:
        df = df._append(df_plus, ignore_index=True)
        print("request at {}".format(datetime.now()))
        print(df.shape)
        # df = get_historical_data(symbol)
        df['kc_middle'], df['kc_upper'], df['kc_lower'] = get_kc(df['high'], df['low'], df['close'], 20, 2, 10)
        buy_price, sell_price, kc_signal, date_signal = implement_kc_strategy(df['close'], df['kc_upper'],
                                                                              df['kc_lower'], df['date'])

        # plot_graph(symbol, df, buy_price, sell_price)
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

        investment_value = 500
        currency = 25000
        kc_investment_ret = []
        arr_result = []

        for i in range(len(kc_strategy_ret_df['kc_returns'])):
            number_of_stocks = floor(investment_value / df['close'][i])
            returns = number_of_stocks * kc_strategy_ret_df['kc_returns'][i]
            kc_investment_ret.append(returns)
            tp = (strategy['position_date'][i], returns)
            arr_result.append(tp)

 
        # # Group the tuples by key and calculate the sum of values for each group
        # grouped = [(key, sum(value for _, value in group))
        #            for key, group in groupby(arr_result, key=lambda x: x[0])]
        # print("==========================  {}  ======================".format(symbol))

        # for rs in grouped:
        #     color = ''
        #     tk_profit = floor((round(rs[1],3) / investment_value) * 100)
        #     vnd_profit = '{:,.2f}'.format((round(rs[1]*currency,0))).replace(',','*').replace('.', ',').replace('*','.')
        #     rs_str = 'date:' + str(rs[0]) +'\t - profit: $'+ str(round(rs[1],3)) + '\t ~ VND: ' +  vnd_profit +   '\t -> ' + str(tk_profit) +'%'
        #     if (round(rs[1],3) < 0):
        #         color = 'red'
        #     else:
        #         color = 'green'
            
        #     print(colored(rs_str, color))
        

        kc_investment_ret_df = pd.DataFrame(kc_investment_ret).rename(columns={0: 'investment_returns'})
        total_investment_ret = round(sum(kc_investment_ret_df['investment_returns']), 2)
        profit_percentage = floor((total_investment_ret / investment_value) * 100)
    
        vnd_profit_total = '{:,.2f}'.format((round(total_investment_ret * currency,0))).replace(',','*').replace('.', ',').replace('*','.')
        print(cl('Profit gained from the KC strategy by investing ${}, in INTC : ${} ~ {} VND'.format(investment_value,total_investment_ret, vnd_profit_total), attrs=['bold']))
        print(cl('Profit percentage of the KC strategy : {}%'.format(profit_percentage), attrs=['bold']))


        profit_obj = Profit(profit_percentage, investment_value, total_investment_ret, symbol)
        return profit_obj
    except Exception as e:
        print("exception: "+e)

def round_step_size(quantity: Union[float, Decimal], step_size: Union[float, Decimal]) -> float:
    """Rounds a given quantity to a specific step size

    :param quantity: required
    :param step_size: required

    :return: decimal
    """
    quantity = Decimal(str(quantity))
    return float(quantity - quantity % Decimal(str(step_size)))

def getQuantity():
    balance = client.get_asset_balance(asset='USDT')
    trades = client.get_recent_trades(symbol=symbol)
    quantity = (float(balance['free'])) / (float(trades[0]['price'])) * 0.83
    price = (float(trades[0]['price']))
 
    response = client.get_symbol_info(symbol=symbol)
    priceFilterFloat = format(float(response["filters"][0]["tickSize"]), '.20f')
    lotSizeFloat = format(float(response["filters"][1]["stepSize"]), '.20f')
    # PriceFilter
    numberAfterDot = str(priceFilterFloat.split(".")[1])
    indexOfOne = numberAfterDot.find("1")
    if indexOfOne == -1:
        price = int(price)
    else:
        price = round(float(price), int(indexOfOne - 1))
    # LotSize
    numberAfterDotLot = str(lotSizeFloat.split(".")[1])
    indexOfOneLot = numberAfterDotLot.find("1")
    if indexOfOneLot == -1:
        quantity = int(quantity)
    else:
        quantity = round(float(quantity), int(indexOfOneLot))

    return quantity, price

if __name__ == '__main__':
    api_key = 'lwaoJYVsMOYVNIBXma32k3PoNzhB5kJ7A6TcRv6cQEqPUTEBMBZHPWiFKZ7bIRqM'  # passkey (saved in bashrc for linux)
    api_secret = 'aDpaIwHf9GVJBiI36aUye5Y2zd1LKCPAUjKIMD9N5ZhzJBqNOJN6Jy09Waw7HBjO'  # secret (saved in bashrc for linux)
    client = Client(api_key, api_secret)
    # symbol = 'DOGEUSDT'
    df = get_historical_data('NEARUSDT')

    # # order = client.get_order(symbol = symbol, orderId = 4946721858) # 4946721858, 4946503030
    # # print(json.dumps(order, indent=2)) 
    # quantity, price = getQuantity()
    # print(quantity)
    # print(price)
    
    # market_res = client.order_market_sell(symbol=symbol,quantity = 74)
    # market_res = client.order_market_buy(symbol=symbol,quantity = 74)

    # print(json.dumps(market_res, indent=2))


    ################################################
    # print(buy_order)
    # exchange_info = client.get_exchange_info()

    # # for i in range(len(exchange_info['symbols'])) :
    # #     arr_profit.append(backTest(exchange_info['symbols'][i]['symbol'], i))


    print("Using Binance TestNet Server")
    alts_list1 = ['1INCH', 'ADA', 'ATOM', 'ANKR', 'ALGO', 'AVAX', 'AAVE', 'AUDIO',
                  'BAT', 'CHZ', 'COTI', 'FLOW', 'APE', 'BNB', 'ETH',
                  'DOT', 'DOGE', 'EOS', 'ETC', 'ENJ', 'EGLD', 'FTM', 'FIL', 'AXS',
                  'IOTA', 'ICP', 'KSM', 'LINK', 'LTC', 'GALA', 'HBAR',
                  'MATIC', 'MANA', 'NEO', 'NEAR', 'ONE', 'RVN', 'SAND', 'XTZ', 'ZEC',
                  'SOL', 'TFUEL', 'THETA', 'UNI', 'VET', 'XRP', 'XLM', 'ZIL'
                  ]

    arr_profit = []
    alts_list = ['NEAR']
    # alts_list = ['ZEC', 'TFUEL','RVN','ICP','FTM']
    for sym in alts_list:
        arr_profit.append(backTest(sym))


    arr_profit.sort(key=lambda x: x.profit_percentage)
    total = 0
    for p in arr_profit:
        total += p.total_investment_ret
        # if p.profit_percentage > -15:
        #     total += p.total_investment_ret
        #     print("================================ {} =======================================".format(p.symbol))
        #     print(cl('Profit gained from the KC strategy by investing ${}, in INTC : {}'.format(p.investment_value,
        #                                                                                         p.total_investment_ret),
        #              attrs=['bold']))
        #     print(cl('Profit percentage of the KC strategy : {}%'.format(p.profit_percentage), attrs=['bold']))

    vnd_profit_total = '{:,.2f}'.format((round(total * 25000,0))).replace(',','*').replace('.', ',').replace('*','.')
    print('${} ~ {}'.format(total, vnd_profit_total))
