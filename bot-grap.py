from binance.client import Client
# from binance.websockets import BinanceSocketManager
from binance.streams import BinanceSocketManager
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input
import plotly
import plotly.graph_objs as go
from collections import deque
import pandas as pd
import json

class priceTicker:
    def __init__(self, api_key, secret_key):
        self.ticker_time = deque(maxlen=200)
        self.last_price = deque(maxlen=200)
        client = Client(api_key, secret_key)
        socket = BinanceSocketManager(client)
        socket.options_kline_socket('BTCUSDT', self.on_message, '1m')

        socket.start()
        self.app = dash.Dash()
        self.app.layout = html.Div(
            [
                dcc.Graph(id = 'live-graph', animate = True),
                dcc.Interval(
                    id = 'graph-update',
                    interval = 2000,
                    n_intervals = 0
                )
            ]
        )
        self.app.callback(
            Output('live-graph', 'figure'),
            [Input('graph-update', 'n_intervals')])(self.update_graph)
        #app.run_server(debug=True, host='127.0.0.1', port=16452)
    def on_message(self, message):

        #price = {'time':message['E'], 'price':message['k']['c']}
        self.ticker_time.append(message['E'])
        self.last_price.append(message['k']['c'])
        #print(self.ticker_time, self.last_price)

    def update_graph(self, n):

        data = go.Scatter(
            x = list(self.ticker_time),
            y = list(self.last_price),
            name ='Scatter',
            mode = 'lines+markers'
        )


        return {'data': [data],
                'layout': go.Layout(xaxis=dict(range=[min(self.ticker_time), max(self.ticker_time)]),
                yaxis=dict(range=[min(self.last_price), max(self.last_price)]),)}

def Main():
 
    api_key = 'lwaoJYVsMOYVNIBXma32k3PoNzhB5kJ7A6TcRv6cQEqPUTEBMBZHPWiFKZ7bIRqM'  # passkey (saved in bashrc for linux)
    api_secret = 'aDpaIwHf9GVJBiI36aUye5Y2zd1LKCPAUjKIMD9N5ZhzJBqNOJN6Jy09Waw7HBjO'  # secret (saved in bashrc for linux)
    ticker = priceTicker(api_key, api_secret)

    ticker.app.run_server(debug=True, host='127.0.0.1', port=16452)
if __name__ == '__main__':
    #app.run_server(debug=False, host='127.0.0.1', port=10010)
    Main()
