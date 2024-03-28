# IMPORTING PACKAGES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from binance.client import Client
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from termcolor import colored as cl
from math import floor
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (20,10)

# EXTRACTING STOCK DATA
# EXTRACTING STOCK DATA
starttime = '30 day ago UTC'  # to start for 1 day ago
interval = '3m'
symbol = 'NEARUSDT'   # Change symbol here e.g. BTCUSDT, BNBBTC, ETHUSDT, NEOBTC
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
def bb_kc_rsi_strategy(prices, upper_bb, lower_bb, kc_upper, kc_lower, rsi):
    buy_price = []
    sell_price = []
    bb_kc_rsi_signal = []
    signal = 0
    lower_bb = lower_bb.to_numpy()
    kc_lower = kc_lower.to_numpy()
    upper_bb = upper_bb.to_numpy()
    kc_upper = kc_upper.to_numpy()
    prices = prices.to_numpy()

    rsi = rsi.to_numpy()
    for i in range(len(prices)):
        if lower_bb[i] < kc_lower[i] and upper_bb[i] > kc_upper[i] and rsi[i] < 30:
            if signal != 1:
                buy_price.append(prices[i])
                sell_price.append(np.nan)
                signal = 1
                bb_kc_rsi_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                bb_kc_rsi_signal.append(0)
        elif lower_bb[i] < kc_lower[i] and upper_bb[i] > kc_upper[i] and rsi[i] > 70:
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
    return buy_price, sell_price, bb_kc_rsi_signal


def plot_graph(symbol, df, entry_prices, exit_prices):
    fig = make_subplots(rows=3, cols=1, subplot_titles=['Close + BB-KC'])
    
    df.set_index('date', inplace=True)
    df.index = pd.to_datetime(df.index, unit='ms') # index set to first column = date_and_time
    
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
    output = df[['date', 'open', 'high', 'low', 'close','volume']]
    # output = output1[ (output1['RSI_entry_ind'] == 1)]

    # output = output1[(output1['RSI_entry_ind'] != 0) | (output1['BB_entry_ind'] != 0) | (output1['STC_entry_ind'] != 0) |
    #                   (output1['RSI_exit_ind'] != 0 ) | (output1['BB_exit_ind'] != 0 ) | (output1['STC_exit_ind'] != 0)]

    output.set_index('date', inplace=True)
    output.index = pd.to_datetime(output.index, unit='ms') # index set to first column = date_and_time

    with open('output-btc.txt', 'w') as f:
        f.write(output.to_string())
    
# BACKTESTING
if __name__ == '__main__':

    client = Client(api_key, api_secret, tld ='us')
    print("Using Binance TestNet Server")

    # symbol = 'ETH-USD'

    df = get_historical_data(symbol)

    # log_out(df)

    # print(df)
    df['upper_bb'], df['middle_bb'], df['lower_bb'] = get_bb(df['close'], 20)
    df['kc_middle'], df['kc_upper'], df['kc_lower'] = get_kc(df['high'], df['low'], df['close'], 20, 2, 10)

    df['rsi_14'] = get_rsi(df['close'], 14)
    df = df.dropna()

    
    buy_price, sell_price, bb_kc_rsi_signal = bb_kc_rsi_strategy(df['close'], df['upper_bb'], df['lower_bb'], df['kc_upper'], df['kc_lower'], df['rsi_14'])
    plot_graph(symbol, df, buy_price, sell_price)
 
    # POSITION
    position = []
    for i in range(len(bb_kc_rsi_signal)):
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
    frames = [close_price, kc_upper, kc_lower, upper_bb, lower_bb, rsi, bb_kc_rsi_signal, position]
    strategy = pd.concat(frames, join = 'inner', axis= 1)

    df_ret = pd.DataFrame(np.diff(df['close'])).rename(columns = {0:'returns'})
    bb_kc_rsi_strategy_ret = []

    strategy = strategy['bb_kc_rsi_position'].to_numpy()
    df_ret = df_ret['returns'].to_numpy()
    df_close = df['close'].to_numpy()

    for i in range(len(df_ret)):
        returns = df_ret[i]*strategy[i]
        bb_kc_rsi_strategy_ret.append(returns)
        
    bb_kc_rsi_strategy_ret_df = pd.DataFrame(bb_kc_rsi_strategy_ret).rename(columns = {0:'bb_kc_rsi_returns'})
    investment_value = 300
    bb_kc_rsi_investment_ret = []
    bb_kc_rsi_returns = bb_kc_rsi_strategy_ret_df['bb_kc_rsi_returns'].to_numpy()

    for i in range(len(bb_kc_rsi_strategy_ret_df['bb_kc_rsi_returns'])):
        number_of_stocks = floor(investment_value/df_close[i])
        returns = number_of_stocks*bb_kc_rsi_returns[i]
        bb_kc_rsi_investment_ret.append(returns)

    bb_kc_rsi_investment_ret_df = pd.DataFrame(bb_kc_rsi_investment_ret).rename(columns = {0:'investment_returns'})
    total_investment_ret = round(sum(bb_kc_rsi_investment_ret_df['investment_returns']), 2)
    profit_percentage = floor((total_investment_ret/investment_value)*100)
    print('Profit gained from the BB KC RSI strategy by investing $%s in df: %s' % (investment_value,total_investment_ret))
    print(cl('Profit percentage of the KC strategy : {}%'.format(profit_percentage), attrs = ['bold']))