from pathlib import Path
from ...datacode.find_relevant_csv import find_relevant_csv
import keras
import backtrader as bt
import pandas as pd
import numpy as np
import os, json

root = Path.cwd()
model_path = root / 'main' / 'models'
stocks_path = root / 'main' / 'data' / 'stocks_5m'

with open(os.path.join(model_path, 'model_info.json'), 'r') as f:
    json_data = json.load(f)

model_name = "5min_basic_one_candle"

model_json = json_data[model_name]
params = model_json['params']

models = [keras.models.load_model(
    os.path.join(model_path, f"{model_name}_{id}.keras")
) for id in range(9)] # test all 9 models

class MyStrategy(bt.Strategy):
    def __init__(self, model):

        self.trade_log = []
        self.open_trades = {}

        self.model = model

        # data is stored in: self.data (= pd dataframe)

    def next(self):
        if self.close.get(size=params['epoch_length']):
            
            currprice = self.data.close[0]

            p_window_close = self.close.get(size=params['window_length'])
            p_window_high = self.high.get(size=params['window_length'])
            p_window_open = self.open.get(size=params['window_length'])
            p_window_low = self.low.get(size=params['window_length'])

            n_window_close = self.close.get(size=params['epoch_length'])
            n_window_open = self.high.get(size=params['epoch_length'])
            n_window_high = self.open.get(size=params['epoch_length'])
            n_window_low = self.low.get(size=params['epoch_length'])

            mClose, MClose = min(n_window_close), max(n_window_close)
            mOpens, MOpens = min(n_window_open), max(n_window_open)
            mHighs, MHighs = min(n_window_high), max(n_window_high)
            mLows, MLows = min(n_window_low), max(n_window_low)

            close_norm = (np.array(p_window_close) - mClose) / (MClose - mClose)
            high_norm = (np.array(p_window_high) - mHighs) / (MHighs - mHighs)
            open_norm = (np.array(p_window_open) - mOpens) / (MOpens - mOpens)
            low_norm = (np.array(p_window_low) - mLows) / (MLows - mLows)

            final_input = np.array([
                close_norm, high_norm, open_norm, low_norm
            ]).T

            prediction = self.model.predict(np.expand_dims(final_input, axis=0), verbose=0)[0]
            pred_unnorm = mClose + prediction * (MClose - mClose)

            if not self.getposition():
                if pred_unnorm > currprice:
                    sl = currprice * (1 - self.sl_ratio)
                    tp = currprice * (1 + self.sl_ratio) * self.tp_ratio
                    size = self.risk * self.broker.getvalue() / (currprice - sl)
                    self.buy_bracket(
                        size=size,
                        price=None,
                        stopprice=sl,
                        limitprice=tp
                    )
                elif pred_unnorm < currprice:
                    sl = currprice * (1 + self.sl_ratio)
                    tp = currprice * (1 - self.sl_ratio) * self.tp_ratio
                    size = self.risk * self.broker.getvalue() / (sl - currprice)
                    self.sell_bracket(
                        size=size,
                        price=None, # market
                        stopprice=sl,
                        limitprice=tp
                    )

    def notify_trade(self, trade):
        if trade.justopened:
            self.open_trades[trade.ref] = {
                'size': trade.size,
                'price_open': trade.price,
                'dt_open': bt.num2date(trade.dtopen),
                'bar_open': trade.baropen,
            }

        if trade.isclosed:
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

            print(f"Opened at {dt_open}, size: {size}, closed at {dt_close}, PnL: {gross_pnl}")

            self.trade_log.append({
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




df = find_relevant_csv(stocks_path, 'AAPL')

for model in models:
        
    cerebro = bt.Cerebro()

    df.index = pd.to_datetime(df.index)
    cerebro.adddata(bt.feeds.PandasData(dataname=df, name='AAPL'))
    
    cerebro.addstrategy(MyStrategy, model=model)

    cerebro.broker.setcash(100000)

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

