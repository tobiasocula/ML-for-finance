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


script_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(os.path.dirname(script_dir), 'models')
data_path = os.path.join(os.path.dirname(script_dir), 'data')


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


def run(X_train, Y_train, X_val, Y_val, window_length, n_neurons, epochs, batch_size, ptd, fh, le):
    model = keras.Sequential([
        keras.Input((window_length, 29)), # 29 = 7*4 + 1 (7 symbs + 1 timestamp)
        keras.layers.GRU(n_neurons, activation='tanh', return_sequences=True),
        keras.layers.GRU(n_neurons, activation='tanh'),
        keras.layers.Dense(7)  # Output layer for regression (predicting price) (7 tickers)
    ])
    optimizer = keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='mse')
    hist = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,
                     validation_data=(X_val, Y_val))
    return hist.history, model

def normalize(X, min, max):
    denominator = (max - min)
    denominator[denominator == 0] = 1e-8
    return (X - min) / denominator



def main():

    with open(os.path.join(save_path, 'model_info.json'), "r") as f:
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

    model_name = "multiple_stocks_model_5min"

    model_desc = """Trained on AAPL, MSFT, GOOG, TSLA as well as some indices (
    '^DJI', '^GSPC', 'IXIC') on 5min candle.
    Predict one value for each stock (total of 4 here) as well as indices (total 7)."""
    
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
            # final_Y = [
            #     (pd.to_datetime(begin_dfs[0].index[i + wl + fh - 1]) - datetime(1970,1,1)).total_seconds()
            #     ] # contains only valid timestamp (for prediction)
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
        # (1993, 100, 29) (222, 100, 29) (1993, 8, 1) (222, 8, 1)


        session_vars.append((X_train, Y_train, X_val, Y_val, wl, nn, ep, bs, ptd, fh, le))

        if counter % N_SUBPROCESSES == 0 and i != 0 or is_last:
            with Pool(N_SUBPROCESSES) as pool:
                results = pool.starmap(run, session_vars)
            for r in results:
                hist, mdl = r[0], r[1]
                
                mdl.save(os.path.join(save_path, f"{model_name}_{model_counter}.keras"))
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

    
    with open(os.path.join(save_path, 'model_info.json'), 'w') as f:
        json.dump(json_data, f)



    
if __name__ == '__main__':
    main()