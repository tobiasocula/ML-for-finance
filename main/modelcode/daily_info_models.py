import itertools
import keras
import pandas as pd
import sys
import numpy as np
import json
from multiprocessing import Pool
import os
from ..datacode.find_relevant_csv import find_relevant_csv

script_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(os.path.dirname(script_dir), 'models')

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
        keras.Input((window_length, 5)),
        keras.layers.GRU(n_neurons, activation='tanh', return_sequences=True),
        keras.layers.GRU(n_neurons, activation='tanh'),
        keras.layers.Dense(1)  # Output layer for regression (predicting price)
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

    window_length = [100]
    pct_train_data = [0.9]
    forecast_horizon = [1]
    len_epochs = [700] # approx 2 years (normalization)
    # for training
    n_neurons = [64]
    epochs = [25, 35, 45]
    batch_size = [24, 36, 48]

    model_name = "DaySin_OHCL_model"

    N_SUBPROCESSES = 10
    session_vars = []
    data = find_relevant_csv(os.path.join('data', 'stocks_5m'), 'AAPL')
    data.index = pd.to_datetime(data.index)

    data['DaySin'] = np.sin(2*np.pi*data.index.weekday/4)
    
    counter = 0
    dct = {} # will store json
    model_counter = 0


    for params, is_last in with_last_flag(itertools.product(
        window_length, pct_train_data, forecast_horizon,
        len_epochs, n_neurons, epochs, batch_size)):

        wl, ptd, fh, le, nn, ep, bs = params
        
        X, Y = [], []

        for i in range(le, len(data) - wl - fh + 1):

            # Extract normalization range for min/max scaling (excluding DaySin)
            normalization_range = data[i - le:i].drop(['DaySin'], axis=1)
            
            min_vals = normalization_range.min(axis=0)
            max_vals = normalization_range.max(axis=0)

            # Normalize input features for the window (including DaySin)
            X_window = data.iloc[i:i+wl].copy()
            X_window_normalized = (X_window.drop(['DaySin'], axis=1) - min_vals) / (max_vals - min_vals + 1e-8)
            X_window_normalized['DaySin'] = X_window['DaySin']  # add back DaySin

            X.append(X_window_normalized.values)  # shape (wl, features)

            # Normalize the target Close price at forecast horizon (scalar)
            target_idx = i + wl + fh - 1  # single future index
            target_close = data.loc[data.index[target_idx], 'Close']

            # Normalize using same min/max from normalization window (Close column)
            target_close_norm = (target_close - min_vals['Close']) / (max_vals['Close'] - min_vals['Close'] + 1e-8)
            # just a number

            Y.append(target_close_norm)  # shape (1,), scalar wrapped as list for consistent shape

        X, Y = np.array(X), np.array(Y) # (num_candles, wl, features), (num_candles, 1, features)
        #print(X.shape, Y.shape)
        
        X_train = X[:int(X.shape[0]*ptd),:,:]
        X_val = X[int(X.shape[0]*ptd):,:,:]
        Y_train = Y[:int(Y.shape[0]*ptd)]
        Y_val = Y[int(Y.shape[0]*ptd):]

        #print(X_train.shape, X_val.shape, Y_train.shape, Y_val.shape)
        # (9397, 100, 5) (1045, 100, 5) (9397,) (1045,)

        session_vars.append((X_train, Y_train, X_val, Y_val, wl, nn, ep, bs, ptd, fh, le))

        if counter % N_SUBPROCESSES == 0 and i != 0 or is_last:
            with Pool(N_SUBPROCESSES) as pool:
                results = pool.starmap(run, session_vars) # list of tuples

            for session_params, r in zip(session_vars, results):

                hist, mdl = r[0], r[1]
                _, _, _, _, sp_wl, sp_nn, sp_ep, sp_bs, sp_ptd, sp_fh, sp_le = session_params

                mdl.save(os.path.join(save_path, f"{model_name}_{model_counter}.keras"))

                dct[f"{model_name}_{model_counter}"] = {
                    "params": {
                        "n_neurons": sp_nn,
                        "window_length": sp_wl,
                        "pct_train_data": sp_ptd,
                        "epochs": sp_ep,
                        "batch_size": sp_bs,
                        "forecast_horizon": sp_fh,
                        "epoch_length": sp_le
                    },
                    "training_loss_per_epoch": hist["loss"],
                    "validation_loss_per_epoch": hist["val_loss"]
                }

                print('UPDATING dct:', dct[f"{model_name}_{model_counter}"])

                model_counter += 1

            session_vars = []

        counter += 1

    with open(os.path.join(save_path, "hist.json"), "w") as f:
        print('SAVING DCT')
        json.dump(dct, f)


            
            

if __name__ == '__main__':
    main()