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
import time as t
from forex_python.converter import CurrencyRates
from dateutil import parser
from datetime import datetime, timedelta, time, date

start_time = t.time()
c = CurrencyRates()

plt.rcParams['figure.figsize'] = (20, 10)
plt.style.use('fivethirtyeight')

interval = '1m'
flag = 0

api_key = 'lwaoJYVsMOYVNIBXma32k3PoNzhB5kJ7A6TcRv6cQEqPUTEBMBZHPWiFKZ7bIRqM'  # passkey (saved in bashrc for linux)
api_secret = 'aDpaIwHf9GVJBiI36aUye5Y2zd1LKCPAUjKIMD9N5ZhzJBqNOJN6Jy09Waw7HBjO'  # secret (saved in bashrc for linux)


def get_historical_data(symbol):
    start_str = datetime.combine(date.today() - timedelta(days=7), time(12, 0, 0))
    end_str =  datetime.combine(date.today(), time(23, 59, 59))
    print("reqest get_historical_data with start_time: {} -  end_time: {}".format(start_str, end_str))

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
    fomatDT = ''
    quantity = 1100

    if flag == 0:
        fomatDT = '%Y-%m-%d'
    else:
        fomatDT = '%m/%Y'
    
    date_time = date_time.apply(lambda x: pd.to_datetime(x, unit='ms').strftime(fomatDT)) 
    try:
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
    except Exception as e:
        print("exception implement_kc_strategy: ", e)

    


def plot_graph(symbol, df, entry_prices, exit_prices, dfs):

    fig = make_subplots(rows=1  , cols=1, subplot_titles=['Close + BB','RSI','STC'])
    df['date'] = df['date'].apply(lambda x: datetime.fromtimestamp(x / 1000.0).strftime('%Y-%m-%d %H:%M:%S') )
    df.set_index('date', inplace=True)
 
    #  Plot close price
    fig.add_trace(go.Line(x=df.index, y=np.array(df['close'], dtype=np.float32), line=dict(color='blue', width=1), name='Close'),row=1, col=1)
    
    #  Plot bollinger bands
    bb_high = df['kc_upper'].astype(float).to_numpy()
    bb_mid = df['kc_middle'].astype(float).to_numpy()
    bb_low = df['kc_lower'].astype(float).to_numpy()
    fig.add_trace(go.Line(x=df.index, y=bb_high, line=dict(color='green', width=1), name='BB High'), row=1, col=1)
    fig.add_trace(go.Line(x=df.index, y=bb_mid, line=dict(color='#ffd866', width=1), name='BB Mid'), row=1, col=1)
    fig.add_trace(go.Line(x=df.index, y=bb_low, line=dict(color='red', width=1), name='BB Low'), row=1, col=1)
   
    #  Add buy and sell indicators
    fig.add_trace(go.Scatter(x=df.index, y=np.array(entry_prices, dtype=np.float32), marker_symbol='arrow-up', marker=dict(color='green', size=15), mode='markers', name='Buy'),1,1)
    fig.add_trace(go.Scatter(x=df.index, y=np.array(exit_prices, dtype=np.float32), marker_symbol='arrow-down', marker=dict(color='red', size=15), mode='markers', name='Sell'),1,1)

    fig.update_layout(showlegend=False, title=dict(text="Visualization",font=dict(family="Arial", size=20,color='#283747')))  
    fig.update_yaxes(range=[0, 1000000000], secondary_y=True)
    fig.update_yaxes(visible=True, secondary_y=True)  # hide range slider

    specs = [[{'type':'pie'}, {"type": "bar"}]]
    fig1 = make_subplots(rows=1, cols=2, specs=specs, shared_yaxes = True, subplot_titles=['Pie Chart', 'Grouped Bar Chart'])

    profit = dfs['profit'].to_numpy()
    losses = dfs['losses'].to_numpy()
    date = dfs['date'].to_numpy()
    values = profit + losses

    #My data creation##                    
    fig1.add_trace(go.Pie(
                                labels = date, 
                                values = values,
                                hole = 0.6,
                                marker_colors = ['#353837','#646665', '#8e9492', '#c9d1ce'],
                                textinfo='percent+value',  ## ADD - display both
                                ), 1, 1)  
    
    ## Create individual traces for Male and Female
    fig1.append_trace(go.Bar(x = date, y = profit, name = 'profit', textposition = 'auto', marker=dict(color='green')), 1, 2)
    fig1.append_trace(go.Bar(x = date, y = losses, name = 'losses', textposition = 'auto', marker=dict(color='red')), 1, 2)

    # fig1.show()
    fig.show()


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
 
        for i in range(len(df['close'])):
            if kc_signal[i] == 1:
                position[i] = 1
                # print("Long coin at:        ${}  -  {}".format(buy_price[i], date_signal[i]))
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

        investment_value = 12
        currency = 25000
        kc_investment_ret = []
        arr_result = []

        for i in range(len(kc_strategy_ret_df['kc_returns'])):
            number_of_stocks = floor(investment_value / df['close'][i])
            returns = number_of_stocks * kc_strategy_ret_df['kc_returns'][i]
            kc_investment_ret.append(returns)
            tp = (strategy['position_date'][i], returns)
            arr_result.append(tp)

        dfs = groupDateTime(arr_result, symbol, investment_value, currency)

        kc_investment_ret_df = pd.DataFrame(kc_investment_ret).rename(columns={0: 'investment_returns'})
        total_investment_ret = round(sum(kc_investment_ret_df['investment_returns']), 2)
        profit_percentage = floor((total_investment_ret / investment_value) * 100)
    
        vnd_profit_total = '{:,.2f}'.format((round(total_investment_ret * currency,0))).replace(',','*').replace('.', ',').replace('*','.')
        print(cl('Profit gained from the KC strategy by investing ${}, in INTC : ${} ~ {} VND'.format(investment_value,total_investment_ret, vnd_profit_total), attrs=['bold']))
        print(cl('Profit percentage of the KC strategy : {}%'.format(profit_percentage), attrs=['bold']))
        time_end = float(t.time() - start_time)
        print("--- %s seconds ---" % time_end)
        plot_graph(symbol, df, buy_price, sell_price, dfs)

        profit_obj = Profit(profit_percentage, investment_value, total_investment_ret, symbol)
        return profit_obj
    except Exception as e:
        print("exception backTest: ", e)


def groupDateTime(arr_result, symbol, investment_value, currency):
    # Group the tuples by key and calculate the sum of values for each group
    arr_result.sort(key=lambda x: x[0])
    grouped = [(key, sum(value for _, value in group))
                for key, group in groupby(arr_result, key=lambda x: x[0])]
    print("==========================  {}  ======================".format(symbol))
    date = []
    profit = []
    losses = []
    for rs in grouped:
        if round(rs[1],3) != 0:
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


def getQuantitySell():
    balance = client.get_asset_balance(asset='TFUEL')
    return int(float(balance['free']))

def getQuantityBuy():
    balance = client.get_asset_balance(asset='USDT')
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

import requests

if __name__ == '__main__':

    api_key = 'lwaoJYVsMOYVNIBXma32k3PoNzhB5kJ7A6TcRv6cQEqPUTEBMBZHPWiFKZ7bIRqM'  # passkey (saved in bashrc for linux)
    api_secret = 'aDpaIwHf9GVJBiI36aUye5Y2zd1LKCPAUjKIMD9N5ZhzJBqNOJN6Jy09Waw7HBjO'  # secret (saved in bashrc for linux)
    client = Client(api_key, api_secret)
    print("Using Binance TestNet Server")
 
    token = '7003550557:AAELbBLiBT5Ka686nCZ43kQCIm468gu4Gds'
    method = 'sendMessage'
    chat_id = '1489044599'
        # order = client.get_order(symbol = symbol, orderId = 4946721858) # 4946721858, 4946503030
    # print(json.dumps(order, indent=2)) 
    
    # quantity = getQuantitySell()
    # market_res = client.order_market_sell(symbol=symbol,quantity = quantity)
    # print(json.dumps(market_res, indent=2))

    # quantity = getQuantityBuy()
    # market_res = client.order_market_buy(symbol=symbol,quantity = quantity)
    # print(json.dumps(market_res, indent=2))
   
    # message = ''
    # for i in range(0, len(market_res['fills'])):
    #     price = market_res['fills'][i]['price']
    #     qty = market_res['fills'][i]['qty']
    #     commission = market_res['fills'][i]['commission']
    #     transactTime = market_res['transactTime']
    #     total_money = float(price) * round(float(qty), 2)
    #     message = message + "\n📢🧧🧧 Sell with entry price: {} - quantity: {} -> :  at: total money: {} - commission {}:  asset after cost: {} at time: {} 📢🧧🧧".format(price, round(float(qty), 2), total_money, commission, float(total_money) - float(commission), datetime.fromtimestamp(transactTime / 1000.0).strftime('%Y-%m-%d %H:%M:%S') )
    
    # message = "🔔💹 Buy with entry price: {} - quantity: {} -> :  at: total money: {} - commission {}:  asset after cost: {} at time: {} 🔔💹".format(price, round(float(qty), 2), total_money, commission, float(total_money) - float(commission), datetime.fromtimestamp(transactTime / 1000.0).strftime('%Y-%m-%d %H:%M:%S') )
    # message ="<b>{}</b>&parse_mode=HTML".format(message)
    # url = 'https://api.telegram.org/bot{0}/{1}?chat_id={2}&text={3}'.format(token, method, chat_id, message)
    # response = requests.post(url=url).json()
    # print(response)

    # print("Using Binance TestNet Server")
    # alts_list = ['1INCH', 'ADA', 'ATOM', 'ANKR', 'ALGO', 'AVAX', 'AAVE', 'AUDIO',
    #               'BAT', 'CHZ', 'COTI', 'FLOW', 'APE', 'BNB', 'ETH',
    #               'DOT', 'DOGE', 'EOS', 'ETC', 'ENJ', 'EGLD', 'FTM', 'FIL', 'AXS',
    #               'IOTA', 'ICP', 'KSM', 'LINK', 'LTC', 'GALA', 'HBAR',
    #               'MATIC', 'MANA', 'NEO', 'NEAR', 'ONE', 'RVN', 'SAND', 'XTZ', 'ZEC',
    #               'SOL', 'TFUEL', 'THETA', 'UNI', 'VET', 'XRP', 'XLM', 'ZIL'
    #   
    #             ]

    flag = 0
    arr_profit = []
    alts_list = ['TFUEL']
    # alts_list1 = ['ZEC', 'TFUEL','RVN','ONE','GALA']
    # alts_list = ['TFUEL', 'ZEC','RVN','NEAR','FLOW']
    for sym in alts_list:
        arr_profit.append(backTest(sym))


    # arr_profit.sort(key=lambda x: x.profit_percentage)
    total = 0
    for p in arr_profit:
        total += p.total_investment_ret

    vnd_profit_total = '{:,.2f}'.format((round(total * 25000,0))).replace(',','*').replace('.', ',').replace('*','.')
    print('${} ~ {}'.format(total, vnd_profit_total))