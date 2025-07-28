import itertools
import keras
import pandas as pd
import sys
import numpy as np
import json
from multiprocessing import Pool
import os
from ..datacode.get_correct_date_range import correct_csvs
from datetime import datetime
from pathlib import Path
import tensorflow as tf
from keras.saving import register_keras_serializable

root = Path.cwd()
model_path = root/'main'/'models'
data_path = root/'main'/'data'

def with_last_flag(iterable):
    """Yields (item, is_last) for each item in iterable."""
    it = iter(iterable)
    try:
        prev = next(it)
    except StopIteration:
        return
    for curr in it:
        yield prev, False
        prev = curr
    yield prev, True

@register_keras_serializable()
class TemporalAttention(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.W = keras.layers.Dense(units)
        self.V = keras.layers.Dense(1)

    def call(self, inputs):
        # Step 1: Project each time step to a hidden representation
        score = tf.nn.tanh(self.W(inputs))  
        # shape: (batch_size, time_steps, units)

        # Step 2: Compute attention scores (1 scalar per time step)
        attention_weights = tf.nn.softmax(self.V(score), axis=1)  
        # shape: (batch_size, time_steps, 1)
        # softmax ensures all weights sum to 1 across time steps

        # Step 3: Multiply weights by original inputs (scale each time step)
        context_vector = attention_weights * inputs  
        # shape: (batch_size, time_steps, hidden_size)

        # Step 4: Sum over all time steps (weighted sum)
        context_vector = tf.reduce_sum(context_vector, axis=1)  
        # shape: (batch_size, hidden_size)
        return context_vector
    
    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


def run(X_train, Y_train, X_val, Y_val, window_length, n_neurons, epochs, batch_size, ptd, fh, le):
    inputs = keras.Input(shape=(window_length, 29))

    x = keras.layers.GRU(n_neurons, return_sequences=True)(inputs)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.GRU(n_neurons, return_sequences=True)(x)
    x = keras.layers.Dropout(0.2)(x)
    x = TemporalAttention(units=64)(x)

    outputs = keras.layers.Dense(7)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    optimizer = keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='mse')
    hist = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,
                     validation_data=(X_val, Y_val))
    return hist.history, model

def main():

    with open(os.path.join(model_path, 'model_info.json'), "r") as f:
        json_data = json.load(f)

    window_length = [100]
    pct_train_data = [0.9]
    forecast_horizon = [1]
    len_epochs = [700]
    n_neurons = [64]
    epochs = [25, 35, 45]
    batch_size = [24, 36, 48]

    stock_symbols = ['AAPL', 'GOOG', 'MSFT', 'TSLA']
    index_symbols = ['^DJI', '^GSPC', 'IXIC']
    symbols = stock_symbols + index_symbols

    model_name = "multiple_stocks_model_5min_temporal"

    model_desc = """Trained on AAPL, MSFT, GOOG, TSLA as well as some indices (
    '^DJI', '^GSPC', 'IXIC') on 5min candle.
    Predict one value for each stock (total of 4 here) as well as indices (total 7).
    Improved version: now supports Dropout and TemporalAttention layer(s)"""
    
    N_SUBPROCESSES = 10
    session_vars = []
    begin_dfs = correct_csvs(symbols, [
        os.path.join(data_path, dn)
        for dn in (len(stock_symbols) * ['stocks_5m']) + len(index_symbols) * ['indices_5m']
    ])
    len_df = len(begin_dfs[0])

    counter = 0
    model_counter = 0

    
    for params, is_last in with_last_flag(itertools.product(
        window_length, pct_train_data, forecast_horizon,
        len_epochs, n_neurons, epochs, batch_size)):

        wl, ptd, fh, le, nn, ep, bs = params
        
        X, Y = [], []

        for i in range(le, len_df - wl - fh + 1):

            final_X = None
            final_Y = []
            
            for symbname, data in zip(symbols, begin_dfs):
                normalization_range = data.iloc[i - le:i].copy()

                min_vals = normalization_range.min(axis=0)
                max_vals = normalization_range.max(axis=0)

                # Normalize input features for the window (including DaySin)
                X_window = data.iloc[i:i+wl].copy()
                X_window_normalized = (X_window - min_vals) / (max_vals - min_vals + 1e-8)
                X_window_normalized.index = pd.to_datetime(X_window_normalized.index)
                X_window_normalized.index = X_window_normalized.index.astype(np.int64)
                X_window_normalized = X_window_normalized.reset_index()
                
                if final_X is None:
                    final_X = pd.DataFrame(X_window_normalized.values, columns=[
                        'Datetime', f"{symbname}-Open", f"{symbname}-High", f"{symbname}-Low", f"{symbname}-Close"
                    ])
                else:
                    final_X = pd.concat([final_X, pd.DataFrame(
                        X_window_normalized.drop(['Datetime'], axis=1).values, columns=[
                        f"{symbname}-Open", f"{symbname}-High", f"{symbname}-Low", f"{symbname}-Close",
                    ])], axis=1)

                # Normalize the target Close price at forecast horizon (scalar)
                target_idx = i + wl + fh - 1  # single future index
                target_close = data.loc[data.index[target_idx], 'Close']

                # Normalize using same min/max from normalization window (Close column)
                target_close_norm = (target_close - min_vals['Close']) / (max_vals['Close'] - min_vals['Close'] + 1e-8)
                
                final_Y.append(target_close_norm) # at end: (num_symbols) long

            X.append(final_X.values)
            Y.append(final_Y)

        Y = np.expand_dims(Y, axis=2)

        X, Y = np.array(X), np.array(Y)
        #print(X.shape, Y.shape) # (2215, 100, 29) (2215, 8)
        
        X_train = X[:int(X.shape[0]*ptd),:,:]
        X_val = X[int(X.shape[0]*ptd):,:,:]
        Y_train = Y[:int(Y.shape[0]*ptd),:]
        Y_val = Y[int(Y.shape[0]*ptd):,:]

        print(X_train.shape, X_val.shape, Y_train.shape, Y_val.shape)
        # (1993, 100, 29) (222, 100, 29) (1993, 7, 1) (222, 7, 1) -> (n_windows, len_window, features)

        session_vars.append((X_train, Y_train, X_val, Y_val, wl, nn, ep, bs, ptd, fh, le))

        if counter % N_SUBPROCESSES == 0 and i != 0 or is_last:
            with Pool(N_SUBPROCESSES) as pool:
                results = pool.starmap(run, session_vars)
            for r in results:
                hist, mdl = r[0], r[1]
                
                mdl.save(os.path.join(model_path, f"{model_name}_{model_counter}.keras"))
                json_data[f"{model_name}_{model_counter}"] = {
                    "params": {
                        "n_neurons": nn,
                        "window_length": wl,
                        "pct_train_data": ptd,
                        "epochs": ep,
                        "batch_size": bs,
                        "forecast_horizon": fh,
                        "epoch_length": le
                    },
                    "training_loss_per_epoch": hist['loss'],
                    "validation_loss_per_epoch": hist["val_loss"],
                    "description": model_desc
                }
                model_counter += 1

            session_vars = []

        counter += 1

    
    with open(os.path.join(data_path, 'model_info.json'), 'w') as f:
        print('saving json to')
        json.dump(json_data, f)



    
if __name__ == '__main__':
    main()
