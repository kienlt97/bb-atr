# IMPORTING PACKAGES

from eodhd import APIClient
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from termcolor import colored as cl
from math import floor
from binance.client import Client
from plotly.subplots import make_subplots
import plotly.graph_objects as go

plt.rcParams['figure.figsize'] = (20,10)
plt.style.use('fivethirtyeight')

starttime = '30 day ago UTC'  # to start for 1 day ago
interval = '3m'
symbol = 'SOLUSDT'   # Change symbol here e.g. BTCUSDT, BNBBTC, ETHUSDT, NEOBTC
api_key = 'lwaoJYVsMOYVNIBXma32k3PoNzhB5kJ7A6TcRv6cQEqPUTEBMBZHPWiFKZ7bIRqM'     # passkey (saved in bashrc for linux)
api_secret = 'aDpaIwHf9GVJBiI36aUye5Y2zd1LKCPAUjKIMD9N5ZhzJBqNOJN6Jy09Waw7HBjO' # secret (saved in bashrc for linux)


def get_historical_data(symbol):
    # valid intervals - 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
    # request historical candle (or klines) data using timestamp from above, interval either every min, hr, day or month
    # starttime = '30 minutes ago UTC' for last 30 mins time
    # e.g. client.get_historical_klines(symbol='ETHUSDTUSDT', '1m', starttime)
    # starttime = '1 Dec, 2017', '1 Jan, 2018'  for last month of 2017
    # e.g. client.get_historical_klines(symbol='BTCUSDT', '1h', '1 Dec, 2017', '1 Jan, 2018')
    #     [
    #     1499040000000,      // Open time
    #     "0.01634790",       // Open
    #     "0.80000000",       // High
    #     "0.01575800",       // Low
    #     "0.01577100",       // Close
    #     "148976.11427815",  // Volume
    #     1499644799999,      // Close time
    #     "2434.19055334",    // Quote asset volume
    #     308,                // Number of trades
    #     "1756.87402397",    // Taker buy base asset volume
    #     "28.46694368",      // Taker buy quote asset volume
    #     "17928899.62484339" // Ignore
    # ]
   
    bars = client.get_historical_klines(symbol, interval, starttime)

    for line in bars:        # Keep only first 6 columns, 'date' 'open' 'high' 'low' 'close','volume'
        del line[6:]

    df = pd.DataFrame(bars, columns=['date', 'open', 'high', 'low', 'close','volume']) #  2 dimensional tabular data

    df['open'] = pd.to_numeric(df['open'], errors='coerce').fillna(0).astype(float)
    df['high'] =  pd.to_numeric(df['high'], errors='coerce').fillna(0).astype(float)
    df['low'] =  pd.to_numeric(df['low'], errors='coerce').fillna(0).astype(float)
    df['close'] =  pd.to_numeric(df['close'], errors='coerce').fillna(0).astype(float)
    df['volume'] =  pd.to_numeric(df['volume'], errors='coerce').fillna(0).astype(float)

    return df

# KELTNER CHANNEL CALCULATION
def get_kc(high, low, close, kc_lookback, multiplier, atr_lookback):
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

# KELTNER CHANNEL STRATEGY

def implement_kc_strategy(prices, kc_upper, kc_lower):
    buy_price = []
    sell_price = []
    kc_signal = []
    signal = 0
    
    for i in range(len(prices)):
        if prices[i] < kc_lower[i] and prices[i+1] > prices[i]:
            if signal != 1:
                buy_price.append(prices[i])
                sell_price.append(np.nan)
                signal = 1
                kc_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                kc_signal.append(0)
        elif prices[i] > kc_upper[i] and prices[i+1] < prices[i]:
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
            
    return buy_price, sell_price, kc_signal

# SPY ETF COMPARISON

def get_benchmark(df, investment_value):
    spy = df['close']
    benchmark = pd.DataFrame(np.diff(spy)).rename(columns = {0:'benchmark_returns'})
    
    investment_value = investment_value
    benchmark_investment_ret = []
    
    for i in range(len(benchmark['benchmark_returns'])):
        number_of_stocks = floor(investment_value/spy[i])
        returns = number_of_stocks*benchmark['benchmark_returns'][i]
        benchmark_investment_ret.append(returns)

    benchmark_investment_ret_df = pd.DataFrame(benchmark_investment_ret).rename(columns = {0:'investment_returns'})
    return benchmark_investment_ret_df

def plot_graph(symbol, df, entry_prices, exit_prices):
    fig = make_subplots(rows=3, cols=1, subplot_titles=['Close + BB-ATR'])
    
    df.set_index('date', inplace=True)
    df.index = pd.to_datetime(df.index, unit='ms') # index set to first column = date_and_time
    
    #  Plot close price
    fig.add_trace(go.Line(x = df.index, y = np.array(df['close'], dtype=np.float32), line=dict(color='blue', width=1), name='Close'), row = 1, col = 1)

    #  Plot bollinger bands
    bb_high = df['kc_upper'].astype(float).to_numpy()
    bb_mid = df['kc_middle'].astype(float).to_numpy()
    bb_low = df['kc_lower'].astype(float).to_numpy()
    fig.add_trace(go.Line(x = df.index, y = bb_high, line=dict(color='green', width=1), name='BB High'), row = 1, col = 1)
    fig.add_trace(go.Line(x = df.index, y = bb_mid, line=dict(color='#ffd866', width=1), name='BB Mid'), row = 1, col = 1)
    fig.add_trace(go.Line(x = df.index, y = bb_low, line=dict(color='red', width=1), name='BB Low'), row = 1, col = 1)
    
    #  Plot RSI
    # fig.add_trace(go.Line(x = df.index, y = np.array(df['rsi_14'], dtype=np.float32) , line=dict(color='blue', width=1), name='RSI'), row = 2, col = 1)

    #  Add buy and sell indicators
    fig.add_trace(go.Scatter(x=df.index, y=np.array(entry_prices, dtype=np.float32), marker_symbol='arrow-up', marker=dict(
        color='green',size=15
    ),mode='markers',name='Buy'))
    fig.add_trace(go.Scatter(x=df.index, y=np.array(exit_prices, dtype=np.float32), marker_symbol='arrow-down', marker=dict(
        color='red',size=15
    ),mode='markers',name='Sell'))
        
    fig.update_layout(
        title={'text':f'{symbol} with BB-RSI-KC' + '/ interval: '+ interval + '-starttime: '+ starttime, 'x':0.5},
        autosize=False,
        width=2000,height=3000)
    fig.update_yaxes(range=[0,1000000000],secondary_y=True)
    fig.update_yaxes(visible=True, secondary_y=True)  #hide range slider

    fig.show()
 

if __name__ == '__main__':

    client = Client(api_key, api_secret, tld ='us')
    print("Using Binance TestNet Server")

    # symbol = 'ETH-USD'

    df = get_historical_data(symbol)
    # df = df.iloc[:,:4]

    df['kc_middle'], df['kc_upper'], df['kc_lower'] = get_kc(df['high'], df['low'], df['close'], 20, 2, 10)

    buy_price, sell_price, kc_signal = implement_kc_strategy(df['close'], df['kc_upper'], df['kc_lower'])
    plot_graph(symbol, df, buy_price, sell_price)

    # print("sum_buy_price", np.nansum(buy_price))   
    # print("sum_sell_price", np.nansum(sell_price))    
    # print("total", np.nansum(sell_price) - np.nansum(buy_price))    

    position = []
    for i in range(len(kc_signal)):
        if kc_signal[i] > 1:
            position.append(0)
        else:
            position.append(1)
    
    for i in range(len(kc_signal)):
        if kc_signal[i] == 1:
            position[i] = 1
        elif kc_signal[i] == -1:
            position[i] = 0
        else:
            position[i] = position[i-1]     
    close_price = df['close']
    kc_upper = df['kc_upper']
    kc_lower = df['kc_lower']
    kc_signal = pd.DataFrame(kc_signal).rename(columns = {0:'kc_signal'}).set_index(df.index)
    position = pd.DataFrame(position).rename(columns = {0:'kc_position'}).set_index(df.index)

    frames = [close_price, kc_upper, kc_lower, kc_signal, position]
    strategy = pd.concat(frames, join = 'inner', axis = 1)

    intc_ret = pd.DataFrame(np.diff(df['close'])).rename(columns = {0:'returns'})
    kc_strategy_ret = []

    for i in range(len(intc_ret)):
        returns = intc_ret['returns'][i]*strategy['kc_position'][i]
        kc_strategy_ret.append(returns)
      
        
    kc_strategy_ret_df = pd.DataFrame(kc_strategy_ret).rename(columns = {0:'kc_returns'})
    investment_value = 1000
    kc_investment_ret = []

    for i in range(len(kc_strategy_ret_df['kc_returns'])):
        number_of_stocks = floor(investment_value/df['close'][i])
        returns = number_of_stocks*kc_strategy_ret_df['kc_returns'][i]
        kc_investment_ret.append(returns)

    kc_investment_ret_df = pd.DataFrame(kc_investment_ret).rename(columns = {0:'investment_returns'})
    total_investment_ret = round(sum(kc_investment_ret_df['investment_returns']), 2)
    profit_percentage = floor((total_investment_ret/investment_value)*100)
    print(cl('Profit gained from the KC strategy by investing ${} in INTC : {}'.format(investment_value, total_investment_ret), attrs = ['bold']))
    print(cl('Profit percentage of the KC strategy : {}%'.format(profit_percentage), attrs = ['bold']))
