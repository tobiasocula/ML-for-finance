from pathlib import Path
import keras
import backtrader as bt
import sys, os, json, time
from ...datacode.find_relevant_csv import find_relevant_csv
import pandas as pd
import numpy as np

root = Path.cwd()
model_path = root / 'main' / 'models'
stocks_path = root / 'main' / 'data' / 'stocks_5m'
indices_path = root / 'main' / 'data' / 'indices_5m'

model_name = "multiple_stocks_model_5min_7"
model = keras.models.load_model(os.path.join(model_path, model_name + '.keras'))

with open(os.path.join(model_path, 'model_info.json'), 'r') as f:
    json_data = json.load(f)

model_json = json_data[model_name]
params = model_json['params']

class MyStrategy(bt.Strategy):
    def __init__(self):

        self.trade_log = []
        self.open_trades = {}

        self.opens  = [data.open for data in self.datas]
        self.highs  = [data.high for data in self.datas]
        self.lows   = [data.low for data in self.datas]
        self.closes = [data.close for data in self.datas]
        self.time = self.datas[0].datetime

        self.tickers = stock_symbols + index_symbols

        self.tmapping = {t:i for i, t in enumerate(self.tickers)}
        
        self.sl_ratio = 0.1 # meaning % beneath / above current price
        self.tp_ratio = 1.5 # meaning 1.5x distance to sl_ratio

        self.risk = 0.01

        # print(self.closes[0][i]) -> access close data at time i
        

    def next(self):

        if self.closes[0].get(size=params['epoch_length']):
            # print(self.closes[0].get(size=params['epoch_length'])) # returns 'array' (just like list)
            final_input = np.empty(shape=(params['window_length'], 29))
            final_input[:,0] = self.time.get(size=params['window_length'])
            currprices = []

            close_mins = []
            close_maxs = []

            

            for j, ticker in enumerate(self.tickers):

                opens = self.closes[self.tmapping[ticker]].get(size=params['epoch_length'])
                closes = self.opens[self.tmapping[ticker]].get(size=params['epoch_length'])
                highs = self.highs[self.tmapping[ticker]].get(size=params['epoch_length'])
                lows = self.lows[self.tmapping[ticker]].get(size=params['epoch_length'])

                currprice = closes[-1]
                currprices.append(currprice)

                n_window_closes = closes[len(closes)-params['epoch_length']:]
                n_window_opens = opens[len(closes)-params['epoch_length']:]
                n_window_highs = highs[len(closes)-params['epoch_length']:]
                n_window_lows = lows[len(closes)-params['epoch_length']:]

                p_window_closes = closes[len(closes)-params['window_length']:]
                p_window_opens = opens[len(closes)-params['window_length']:]
                p_window_highs = highs[len(closes)-params['window_length']:]
                p_window_lows = lows[len(closes)-params['window_length']:]

                mClose, MClose = min(n_window_closes), max(n_window_closes)
                mOpens, MOpens = min(n_window_opens), max(n_window_opens)
                mHighs, MHighs = min(n_window_highs), max(n_window_highs)
                mLows, MLows = min(n_window_lows), max(n_window_lows)

                close_mins.append(mClose)
                close_maxs.append(MClose)

                p_window_closes_norm = (np.array(p_window_closes) - mClose) / (MClose - mClose)
                p_window_opens_norm = (np.array(p_window_opens) - mOpens) / (MOpens - mOpens)
                p_window_highs_norm = (np.array(p_window_highs) - mHighs) / (MHighs - mHighs)
                p_window_lows_norm = (np.array(p_window_lows) - mLows) / (MLows - mLows)

                final_input[:,j*4+1] = p_window_closes_norm
                final_input[:,j*4+2] = p_window_opens_norm
                final_input[:,j*4+3] = p_window_highs_norm
                final_input[:,j*4+4] = p_window_lows_norm

            #print(final_input)
            #print(final_input.shape) # shape (100, 29)

            prediction = model.predict(np.expand_dims(final_input, axis=0), verbose=0)[0]
            pred_unnorm = np.array(close_mins) + prediction * (np.array(close_maxs) - np.array(close_maxs))
            
            diffs = pred_unnorm - np.array(currprices)
            for i, data in enumerate(self.datas):

                # use formula: abs(current_price - stoploss) * order_size = risk% * capital (max risk)
                # in long: SL = currprice - risk% * capital / order_size
                # or: order_size = risk% * capital / (cp - sl)
                if not self.getposition(data):
                    if diffs[i] > 0: # enter long
                        sl = currprices[i] * (1 - self.sl_ratio)
                        tp = currprices[i] * (1 + self.sl_ratio) * self.tp_ratio
                        self.buy_bracket(
                            data=data,
                            size=self.risk * self.broker.getvalue() / (currprices[i] - sl),
                            price=None, # market
                            stopprice=sl,
                            limitprice=tp
                        )
                    elif diffs[i] < 0: # enter short
                        sl = currprices[i] * (1 + self.sl_ratio)
                        tp = currprices[i] * (1 - self.sl_ratio) * self.tp_ratio
                        self.sell_bracket(
                            data=data,
                            size=self.risk * self.broker.getvalue() / (sl - currprices[i]),
                            price=None, # market
                            stopprice=sl,
                            limitprice=tp
                        )

    def notify_trade(self, trade):
        # Get symbol from the data feed attached to the trade
        symbol = getattr(trade.data, '_name', 'unknown')

        if trade.justopened:
            # Cache size and open price when trade opens
            self.open_trades[trade.ref] = {
                'size': trade.size,
                'price_open': trade.price,
                'dt_open': bt.num2date(trade.dtopen),
                'bar_open': trade.baropen,
                'symbol': symbol,
            }

        if trade.isclosed:
            # Get cached open info
            open_info = self.open_trades.pop(trade.ref, None)

            size = open_info['size'] if open_info else None
            price_open = open_info['price_open'] if open_info else None
            dt_open = open_info['dt_open'] if open_info else None
            bar_open = open_info['bar_open'] if open_info else None

            dt_close = bt.num2date(trade.dtclose) if trade.dtclose else None
            bar_close = trade.barclose
            price_close = trade.price
            gross_pnl = trade.pnl
            net_pnl = trade.pnlcomm

            print(f"Symbol: {symbol}, opened at {dt_open}, size: {size}, closed at {dt_close}, PnL: {gross_pnl}")

            self.trade_log.append({
                'symbol': symbol,
                'dt_open': dt_open,
                'bar_open': bar_open,
                'size': size,
                'price_open': price_open,
                'dt_close': dt_close,
                'bar_close': bar_close,
                'price_close': price_close,
                'gross_pnl': gross_pnl,
                'net_pnl': net_pnl,
                'balance': self.broker.getvalue()
            })



cerebro = bt.Cerebro()
cerebro.addstrategy(MyStrategy)

stock_symbols = ['AAPL', 'GOOG', 'MSFT', 'TSLA']
index_symbols = ['^DJI', '^GSPC', 'IXIC']

for ss in stock_symbols:
    df = find_relevant_csv(stocks_path, ss)
    df.index = pd.to_datetime(df.index)
    cerebro.adddata(bt.feeds.PandasData(dataname=df, name=ss))
    
for iss in index_symbols:
    df = find_relevant_csv(indices_path, iss)
    df.index = pd.to_datetime(df.index)
    cerebro.adddata(bt.feeds.PandasData(dataname=df, name=iss))

# Set initial capital
cerebro.broker.setcash(100000)

# Run the backtest
cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days)
cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')


results = cerebro.run()
strat = results[0]

# Final return (%)
print(f"Final return: {strat.analyzers.returns.get_analysis()['rnorm100']:.2f}%")

# Sharpe ratio
print(f"Sharpe ratio: {strat.analyzers.sharpe.get_analysis().get('sharperatio', 'N/A')}")

# Max drawdown
dd = strat.analyzers.drawdown.get_analysis()
print(f"Max drawdown: {dd['max']['drawdown']:.2f}%")

# Calmar ratio (Return / Max Drawdown)
calmar = strat.analyzers.returns.get_analysis()['rnorm100'] / dd['max']['drawdown']
print(f"Calmar ratio: {calmar:.2f}")

# Create the DataFrame
trade_df = pd.DataFrame(strat.trade_log)
print(trade_df)

cerebro.plot(style='candlestick', volume=False)