# IMPORTING PACKAGES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from binance.client import Client
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from termcolor import colored as cl
from math import floor
from itertools import groupby
from termcolor import colored
from datetime import datetime, timedelta, time, date

plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (20,10)

# EXTRACTING STOCK DATA
# EXTRACTING STOCK DATA
starttime = '60 day ago UTC'  # to start for 1 day ago
interval = '1m'
symbol = 'TFUELUSDT'   # Change symbol here e.g. BTCUSDT, BNBBTC, ETHUSDT, NEOBTC
api_key = 'lwaoJYVsMOYVNIBXma32k3PoNzhB5kJ7A6TcRv6cQEqPUTEBMBZHPWiFKZ7bIRqM'     # passkey (saved in bashrc for linux)
api_secret = 'aDpaIwHf9GVJBiI36aUye5Y2zd1LKCPAUjKIMD9N5ZhzJBqNOJN6Jy09Waw7HBjO' # secret (saved in bashrc for linux)

def get_historical_data(symbol):
 
    start_str = datetime.combine(date.today() - timedelta(days=3), time(12, 0, 0))
    end_str =  datetime.combine(date.today(), time(23, 59, 59))
    print("reqest get_historical_data with start_time: {} -  end_time: {}".format(start_str, end_str))

    bars = client.get_historical_klines(symbol, interval, start_str = str(start_str), end_str = str(end_str))
    for line in bars:        # Keep only first 6 columns, 'date' 'open' 'high' 'low' 'close'
        del line[5:]

    df = pd.DataFrame(bars, columns=['date', 'open', 'high', 'low', 'close']) #  2 dimensional tabular data

    df['open'] = pd.to_numeric(df['open'], errors='coerce').fillna(0).astype(float)
    df['high'] =  pd.to_numeric(df['high'], errors='coerce').fillna(0).astype(float)
    df['low'] =  pd.to_numeric(df['low'], errors='coerce').fillna(0).astype(float)
    df['close'] =  pd.to_numeric(df['close'], errors='coerce').fillna(0).astype(float)

    return df

# BOLLINGER BANDS CALCULATION
def sma(df, lookback):
    sma = df.rolling(lookback).mean()
    return sma

def get_bb(df, lookback):
    std = df.rolling(lookback).std()
    upper_bb = sma(df, lookback) + std * 2
    lower_bb = sma(df, lookback) - std * 2
    middle_bb = sma(df, lookback)
    
    return upper_bb, middle_bb, lower_bb

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

# RSI CALCULATION
def get_rsi(close, lookback):
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

# TRADING STRATEGY
def bb_kc_rsi_strategy(prices, upper_bb, lower_bb, kc_upper, kc_lower, rsi, date_time, rsi_up, rsi_low):
    buy_price = []
    sell_price = []
    bb_kc_rsi_signal = []
    date_signal = []
    signal = 0
    lower_bb = lower_bb.to_numpy()
    kc_lower = kc_lower.to_numpy()
    upper_bb = upper_bb.to_numpy()
    kc_upper = kc_upper.to_numpy()
    prices = prices.to_numpy()
    date_time = date_time.apply(lambda x: pd.to_datetime(x, unit='ms').strftime('%d/%m/%Y')).to_numpy()
    rsi = rsi.to_numpy()
    
    for i in range(len(prices)):
        date_signal.append(date_time[i])
        if lower_bb[i] < kc_lower[i] and upper_bb[i] > kc_upper[i] and rsi[i] < rsi_up:
            if signal != 1:
                buy_price.append(prices[i])
                sell_price.append(np.nan)
                signal = 1
                bb_kc_rsi_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                bb_kc_rsi_signal.append(0)
        elif lower_bb[i] < kc_lower[i] and upper_bb[i] > kc_upper[i] and rsi[i] > rsi_low:
            if signal != -1:
                buy_price.append(np.nan)
                sell_price.append(prices[i])
                signal = -1
                bb_kc_rsi_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                bb_kc_rsi_signal.append(0)
        else:
            buy_price.append(np.nan)
            sell_price.append(np.nan)
            bb_kc_rsi_signal.append(0)
    return buy_price, sell_price, bb_kc_rsi_signal, date_signal


def plot_graph(symbol, df, entry_prices, exit_prices):
    fig = make_subplots(rows=3, cols=1, subplot_titles=['Close + BB-KC'])
    df['date'] = df['date'].apply(lambda x: datetime.fromtimestamp(x / 1000.0).strftime('%Y-%m-%d %H:%M:%S') )
    
    df.set_index('date', inplace=True)
    # df.index = pd.to_datetime(df.index, unit='ms') # index set to first column = date_and_time
    
    #  Plot close price
    fig.add_trace(go.Line(x = df.index, y = np.array(df['close'], dtype=np.float32), line=dict(color='blue', width=1), name='Close'), row = 1, col = 1)

    #  Plot bollinger bands
    bb_high = df['upper_bb'].astype(float).to_numpy()
    bb_mid = df['middle_bb'].astype(float).to_numpy()
    bb_low = df['lower_bb'].astype(float).to_numpy()
    fig.add_trace(go.Line(x = df.index, y = bb_high, line=dict(color='green', width=1), name='BB High'), row = 1, col = 1)
    fig.add_trace(go.Line(x = df.index, y = bb_mid, line=dict(color='#ffd866', width=1), name='BB Mid'), row = 1, col = 1)
    fig.add_trace(go.Line(x = df.index, y = bb_low, line=dict(color='red', width=1), name='BB Low'), row = 1, col = 1)
    
    #  Plot RSI
    fig.add_trace(go.Line(x = df.index, y = np.array(df['rsi_14'], dtype=np.float32) , line=dict(color='blue', width=1), name='RSI'), row = 2, col = 1)

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

def log_out(df):
    output = df[['date', 'open', 'high', 'low', 'close']]
    # output = output1[ (output1['RSI_entry_ind'] == 1)]

    # output = output1[(output1['RSI_entry_ind'] != 0) | (output1['BB_entry_ind'] != 0) | (output1['STC_entry_ind'] != 0) |
    #                   (output1['RSI_exit_ind'] != 0 ) | (output1['BB_exit_ind'] != 0 ) | (output1['STC_exit_ind'] != 0)]

    output.set_index('date', inplace=True)
    output.index = pd.to_datetime(output.index, unit='ms') # index set to first column = date_and_time

    with open('output-btc.txt', 'w') as f:
        f.write(output.to_string())
 
def backTest(df, rsi_up, rsi_low):

    buy_price, sell_price, bb_kc_rsi_signal, date_signal = bb_kc_rsi_strategy(df['close'], df['upper_bb'], df['lower_bb'], df['kc_upper'], df['kc_lower'], df['rsi_14'], df['date'], rsi_up, rsi_low)
    plot_graph(symbol, df, buy_price, sell_price)
 
    # POSITION
    position = []
    position_date = []
    for i in range(len(bb_kc_rsi_signal)):
        position_date.append(date_signal[i])
        if bb_kc_rsi_signal[i] > 1:
            position.append(0)
        else:
            position.append(1)

    for i in range(len(df['close'])):
        if bb_kc_rsi_signal[i] == 1:
            position[i] = 1
        elif bb_kc_rsi_signal[i] == -1:
            position[i] = 0
        else:
            position[i] = position[i-1]


    kc_upper = df['kc_upper']
    kc_lower = df['kc_lower']
    upper_bb = df['upper_bb']
    lower_bb = df['lower_bb']
    rsi = df['rsi_14']
    close_price = df['close']
    bb_kc_rsi_signal = pd.DataFrame(bb_kc_rsi_signal).rename(columns = {0:'bb_kc_rsi_signal'}).set_index(df.index)
    position = pd.DataFrame(position).rename(columns ={0:'bb_kc_rsi_position'}).set_index(df.index)
    position_date = pd.DataFrame(position_date).rename(columns={0: 'position_date'}).set_index(df.index)
    
    frames = [close_price, kc_upper, kc_lower, upper_bb, lower_bb, rsi, bb_kc_rsi_signal, position, position_date]

    strategy = pd.concat(frames, join = 'inner', axis= 1)

    df_ret = pd.DataFrame(np.diff(df['close'])).rename(columns = {0:'returns'})
    bb_kc_rsi_strategy_ret = []

    bb_kc_rsi_position = strategy['bb_kc_rsi_position'].to_numpy()
    df_ret = df_ret['returns'].to_numpy()
    df_close = df['close'].to_numpy()
    
    for i in range(len(df_ret)):
        returns = df_ret[i]*bb_kc_rsi_position[i]
        bb_kc_rsi_strategy_ret.append(returns)

    bb_kc_rsi_strategy_ret_df = pd.DataFrame(bb_kc_rsi_strategy_ret).rename(columns = {0:'bb_kc_rsi_returns'})
    investment_value = 12
    bb_kc_rsi_investment_ret = []
    bb_kc_rsi_returns = bb_kc_rsi_strategy_ret_df['bb_kc_rsi_returns'].to_numpy()
    arr_result = []
    bb_kc_position_date = strategy['position_date'].to_numpy()

    for i in range(len(bb_kc_rsi_strategy_ret_df['bb_kc_rsi_returns'])):
        number_of_stocks = floor(investment_value/df_close[i])
        returns = number_of_stocks*bb_kc_rsi_returns[i]
        bb_kc_rsi_investment_ret.append(returns)
        tp = (bb_kc_position_date[i], returns)
        arr_result.append(tp)

    # Group the tuples by key and calculate the sum of values for each group
    grouped = [(key, sum(value for _, value in group))
                for key, group in groupby(arr_result, key=lambda x: x[0])]

    # grouped.sort(key=lambda a: a[1])
    print("==========================  {} - {}  ======================".format(symbol, str(rsi_up) + '_' + str(rsi_low)))

    for rs in grouped:
        profit = ''
        if (round(rs[1],3) < 0):
            profit = colored(round(rs[1],3), 'red')
        else:
            profit = colored(round(rs[1],3), 'green')
        print('date: {} - profit: ${} - %{}'.format(rs[0],profit, floor((round(rs[1],3)/investment_value)*100)))

    bb_kc_rsi_investment_ret_df = pd.DataFrame(bb_kc_rsi_investment_ret).rename(columns = {0:'investment_returns'})
    total_investment_ret = round(sum(bb_kc_rsi_investment_ret_df['investment_returns']), 2)
    profit_percentage = floor((total_investment_ret/investment_value)*100)
    print('Profit gained from the BB KC RSI strategy by investing $%s in df: %s' % (investment_value,total_investment_ret))
    print(cl('Profit percentage of the KC strategy : {}%'.format(profit_percentage), attrs = ['bold']))

    return total_investment_ret

data = dict()

# BACKTESTING
if __name__ == '__main__':

    client = Client(api_key, api_secret, tld ='us')
    print("Using Binance TestNet Server")

    df = get_historical_data(symbol)
    df['upper_bb'], df['middle_bb'], df['lower_bb'] = get_bb(df['close'], 20)
    df['kc_middle'], df['kc_upper'], df['kc_lower'] = get_kc(df['high'], df['low'], df['close'], 20, 2, 10)

    df['rsi_14'] = get_rsi(df['close'], 14)
    df = df.dropna()

    arr_profit = []
    arr_profit.append(backTest(df, 40,60))

    # for rsi_up in range(0 , 100):
    #     for rsi_low in range (0, 100):
    #         key = str(rsi_up) + '_' + str(rsi_low)
    #         data[key] = backTest(df, rsi_up, rsi_low)

    # sorted_data = sorted(data.items(), key=lambda x:x[1])

    # with open('data-sorted.txt', 'w') as f:
    #     f.write(str(sorted_data) )

    # for d in sorted_data:
    #     print(d)