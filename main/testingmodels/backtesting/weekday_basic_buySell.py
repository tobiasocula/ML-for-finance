import keras
from backtesting import Backtest, Strategy
import pandas as pd
import numpy as np
import sys, os, json, time
from ...datacode.find_relevant_csv import find_relevant_csv

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(os.path.dirname(script_dir), 'models')
data_path = os.path.join(os.path.dirname(script_dir), 'data', 'stocks_5m')

model_name = "DaySin_OHCL_model_7"
model = keras.models.load_model(os.path.join(model_path, model_name + '.keras'))

normalization_window_length = 50

df = find_relevant_csv(data_path, 'AAPL')
df.index = pd.to_datetime(df.index)

with open(os.path.join(model_path, 'hist.json'), 'r') as f:
    json_data = json.load(f)

model_json = json_data[model_name]
params = model_json['params']




class BT(Strategy):
    def init(self):
        self.risk = 0.01

    def next(self):
        
        if len(self.data.df) >= params['epoch_length']:
            data = self.data.df.drop(['Volume'], axis=1)
            data['DaySin'] = np.sin(2*np.pi*data.index.weekday/4)
            
            curr_price = data["Close"].iloc[-1]

            normalization_window = data.iloc[len(data)-params['epoch_length']:,:] # (random, features)
            batch_length = params['window_length']
            pred_window = data.iloc[len(data)-batch_length:,:] # (wl, features), unnormalized

            # floats
            close_min = normalization_window['Close'].min()
            close_max = normalization_window['Close'].max()


            # 4-length np array
            min_ohcl = normalization_window.reset_index().drop(['Datetime', 'DaySin'], axis=1).min(axis=0)
            max_ohcl = normalization_window.reset_index().drop(['Datetime', 'DaySin'], axis=1).max(axis=0)

            pred_window = (pred_window.reset_index().drop(['DaySin', 'Datetime'], axis=1) - min_ohcl) / (max_ohcl - min_ohcl + 1e-08) # normalize
            pred_window['DaySin'] = data.iloc[len(data)-batch_length:,:]['DaySin'].values

            input_array = pred_window.values.astype(np.float32)
            prediction = model.predict(np.expand_dims(input_array, axis=0), verbose=0)[0][0]

            pred_unnorm = close_min + prediction * (close_max - close_min + 1e-8)


            if pred_unnorm >= curr_price:
                self.buy(size=self.risk)
            else:
                self.sell(size=self.risk)


bt = Backtest(df, BT, cash=100_000, exclusive_orders=True, commission=0, spread=0)
results = bt.run()
print(results)
bt.plot()