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
from forex_python.converter import CurrencyRates
from dateutil import parser
from datetime import date,datetime

start_time = time.time()
c = CurrencyRates()

plt.rcParams['figure.figsize'] = (20, 10)
plt.style.use('fivethirtyeight')

starttime = ''  # to start for 1 day ago
interval = '1m'
flag = 0
str_date = ''

# symbol = 'DOGEUSDT'   # Change symbol here e.g. BTCUSDT, BNBBTC, ETHUSDT, NEOBTC
api_key = 'lwaoJYVsMOYVNIBXma32k3PoNzhB5kJ7A6TcRv6cQEqPUTEBMBZHPWiFKZ7bIRqM'  # passkey (saved in bashrc for linux)
api_secret = 'aDpaIwHf9GVJBiI36aUye5Y2zd1LKCPAUjKIMD9N5ZhzJBqNOJN6Jy09Waw7HBjO'  # secret (saved in bashrc for linux)


def get_historical_data(symbol):
    starttime = getDayNumber()

    start_str = str(parser.parse('2024-04-22 00:00:00'))
    end_str = str(parser.parse('2024-04-24 23:59:00'))

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
def implement_kc_strategy(prices, kc_upper, kc_lower, date_time, dt):
    buy_price = []
    sell_price = []
    kc_signal = []
    date_signal = []
    fomatDT = ''

    if flag == 0:
        fomatDT = '%d/%m/%Y'
    else:
        fomatDT = '%m/%Y'
    
    if dt == 1:
        date_time = date_time.apply(lambda x: pd.to_datetime(x, unit='ms').strftime(fomatDT))
    else:
        date_time = date_time.apply(lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%f').strftime(fomatDT))

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


def plot_graph(symbol, df, entry_prices, exit_prices, dfs):
    # fig = make_subplots(rows=0, cols=1, subplot_titles=['Close + BB-ATR'])

    # df.set_index('date', inplace=True)
    # df.index = pd.to_datetime(df.index, unit='ms')  # index set to first column = date_and_time

    # #  Plot close price
    # fig.add_trace(
    #     go.Line(x=df.index, y=np.array(df['close'], dtype=np.float32), line=dict(color='blue', width=1), name='Close'),
    #     row=1, col=1)

    # #  Plot bollinger bands
    # bb_high = df['kc_upper'].astype(float).to_numpy()
    # bb_mid = df['kc_middle'].astype(float).to_numpy()
    # bb_low = df['kc_lower'].astype(float).to_numpy()
    # fig.add_trace(go.Line(x=df.index, y=bb_high, line=dict(color='green', width=1), name='BB High'), row=1, col=1)
    # fig.add_trace(go.Line(x=df.index, y=bb_mid, line=dict(color='#ffd866', width=1), name='BB Mid'), row=1, col=1)
    # fig.add_trace(go.Line(x=df.index, y=bb_low, line=dict(color='red', width=1), name='BB Low'), row=1, col=1)

    # #  Add buy and sell indicators
    # fig.add_trace(
    #     go.Scatter(x=df.index, y=np.array(entry_prices, dtype=np.float32), marker_symbol='arrow-up', marker=dict(
    #         color='green', size=15
    #     ), mode='markers', name='Buy'))
    # fig.add_trace(
    #     go.Scatter(x=df.index, y=np.array(exit_prices, dtype=np.float32), marker_symbol='arrow-down', marker=dict(
    #         color='red', size=15
    #     ), mode='markers', name='Sell'))

    # profit = dfs['profit'].to_numpy()
    # losses = dfs['losses'].to_numpy()
    # date = dfs['date'].to_numpy()
    # values = profit + losses


    # fig.add_trace(go.Pie(labels=date,
    #                      values = values,
    #                      hovertemplate = "%{label}: <br>Value: %{value} ",
    #                      showlegend=True,
    #                      textposition='inside',
    #                      rotation=90), 2, 1)
    
    # fig.add_trace(go.Bar(x = date, y = profit, name = 'Profit',marker=dict(color='green')), 2, 2)
    # fig.add_trace(go.Bar(x = date, y = losses,name = 'Losses', marker=dict(color='red')), 2, col=2)

    # fig.update_layout(
    #     title={'text': f'{symbol} with BB-RSI-KC' + '/ interval: ' + interval + '-starttime: ' + starttime, 'x': 0.5},
    #     autosize=False,
    #     width=2000, height=3000)
    # fig.update_yaxes(range=[0, 1000000000], secondary_y=True)
    # fig.update_yaxes(visible=True, secondary_y=True)  # hide range slider

    # fig.show()

    fig = make_subplots(rows=1  , cols=1, subplot_titles=['Close + BB','RSI','STC'])
    df.set_index('date', inplace=True)
    df.index = pd.to_datetime(df.index, unit='ms') # index set to first column = date_and_time
    
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

    fig1.show()
    fig.show()


def backTest(symbol):
    try:
        symbol = symbol + "USDT"
        dt = 1
        
        if dt == 1:
            df = get_historical_data(symbol)
        else:
            df = getdf()


        print("1: ",len(df['close']))
        df['kc_middle'], df['kc_upper'], df['kc_lower'] = get_kc(df['high'], df['low'], df['close'], 20, 2, 10)
        print("2: ",len(df['close']))
        buy_price, sell_price, kc_signal, date_signal = implement_kc_strategy(df['close'], df['kc_upper'],
                                                                              df['kc_lower'], df['date'], dt)
        print("3: ",len(buy_price))

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

        investment_value = 1000
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
        time_end = float(time.time() - start_time)
        print("--- %s seconds ---" % time_end)
        # plot_graph(symbol, df, buy_price, sell_price, dfs)

        profit_obj = Profit(profit_percentage, investment_value, total_investment_ret, symbol)
        return profit_obj
    except Exception as e:
        print("exception: ", e)

import redis
from price_data_sql import CryptoPrice
# rc = redis.Redis(host='192.168.40.11', port=6379, db=1)

def getAllData():
    closed_prices = []
    # for key in rc.scan_iter():
    #     closed_prices.append(CryptoPrice(**json.loads(rc.get(key))))
    return closed_prices

def getdf():
    closed_prices = getAllData()
    df = pd.DataFrame([vars(price) for price in closed_prices])

    df['date'] = df['close_time']
    df['high'] = pd.to_numeric(df['high_price'], errors='coerce').fillna(0).astype(float)
    df['low'] = pd.to_numeric(df['low_price'], errors='coerce').fillna(0).astype(float)
    df['close'] = pd.to_numeric(df['close_price'], errors='coerce').fillna(0).astype(float)

    return df

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

def getDayNumber():
    date1 = parser.parse(str_date)
    date2 = parser.parse(str(date.today()))
    diff =  date2 - date1 

    return str(diff.days) +' day ago UTC' 
    
if __name__ == '__main__':

    api_key = 'lwaoJYVsMOYVNIBXma32k3PoNzhB5kJ7A6TcRv6cQEqPUTEBMBZHPWiFKZ7bIRqM'  # passkey (saved in bashrc for linux)
    api_secret = 'aDpaIwHf9GVJBiI36aUye5Y2zd1LKCPAUjKIMD9N5ZhzJBqNOJN6Jy09Waw7HBjO'  # secret (saved in bashrc for linux)
    client = Client(api_key, api_secret)
    # symbol = 'DOGEUSDT'
    # df = get_historical_data('NEARUSDT')

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
    alts_list = ['1INCH', 'ADA', 'ATOM', 'ANKR', 'ALGO', 'AVAX', 'AAVE', 'AUDIO',
                  'BAT', 'CHZ', 'COTI', 'FLOW', 'APE', 'BNB', 'ETH',
                  'DOT', 'DOGE', 'EOS', 'ETC', 'ENJ', 'EGLD', 'FTM', 'FIL', 'AXS',
                  'IOTA', 'ICP', 'KSM', 'LINK', 'LTC', 'GALA', 'HBAR',
                  'MATIC', 'MANA', 'NEO', 'NEAR', 'ONE', 'RVN', 'SAND', 'XTZ', 'ZEC',
                  'SOL', 'TFUEL', 'THETA', 'UNI', 'VET', 'XRP', 'XLM', 'ZIL'
                  ]

    arr_profit = []
    # alts_list = ['NEAR']
    flag = 0
    str_date = '2024-04-22'
    alts_list1 = ['ZEC', 'TFUEL','RVN','ONE','GALA']
    for sym in alts_list:
        arr_profit.append(backTest(sym))


    # arr_profit.sort(key=lambda x: x.profit_percentage)
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