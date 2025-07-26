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
        keras.Input((window_length, 6)),
        keras.layers.GRU(n_neurons, activation='tanh', return_sequences=True),
        keras.layers.GRU(n_neurons, activation='tanh'),
        keras.layers.Dense(12)  # Output layer for regression (predicting price)
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

    with open(os.path.join(save_path, 'hist.json'), "r") as f:
        json_data = json.load(f)

    pct_train_data = [0.9]
    # for training
    n_neurons = [64]
    epochs = [25, 35, 45]
    batch_size = [24, 36, 48]

    le = 462
    fh = 12
    wl = 30

    model_name = "mult_candle_model"

    model_desc = "Tries to predict price between 12 - 13h based on price between 9:30 - 12h. Trained on AAPL."

    N_SUBPROCESSES = 10
    session_vars = []
    data = find_relevant_csv(os.path.join('data', 'stocks_5m'), 'AAPL')
    data.index = pd.to_datetime(data.index)

    data['DaySin'] = np.sin(2*np.pi*data.index.weekday/4)
    data['Session'] = np.where((9 <= data.index.hour) & (data.index.hour <= 12), 0, 1)
    
    counter = 0
    dct = {} # will store json
    model_counter = 0

    for params, is_last in with_last_flag(itertools.product(
        pct_train_data, n_neurons, epochs, batch_size)):

        ptd, nn, ep, bs = params

        X, Y = [], []

        for i, (idx, row) in enumerate(data.iloc[le:,:].iterrows(), le):
            if idx.hour == 12 and idx.minute == 0:
                print('TEST')
                normalization_range = data[i - le:i].drop(['Session', 'DaySin'], axis=1)

                min_vals = normalization_range.min(axis=0)
                max_vals = normalization_range.max(axis=0)

                X_window = data.iloc[i-wl:i].copy()
                Y_window = data.iloc[i:i+fh].copy()['Close']

                X_window_normalized = (X_window.drop(['Session', 'DaySin'], axis=1) - min_vals) / (max_vals - min_vals + 1e-8)
                Y_window_normalized = (Y_window- min_vals['Close']) / (max_vals['Close'] - min_vals['Close'] + 1e-8)

                print(X_window_normalized.shape) # (30, 4)
                print(Y_window_normalized.shape) # (12, 4)
                
                X_window_normalized['Session'] = X_window['Session']
                X_window_normalized['DaySin'] = X_window['DaySin']

                print(X_window_normalized.shape)
                X.append(X_window_normalized.values)
                Y.append(Y_window_normalized.values)

        X, Y = np.array(X), np.array(Y)
            
        #print(Y.shape) # (33, 12, 6)
        #print(X.shape) # (33, 30, 6)

        X_train = X[:int(X.shape[0]*ptd),:,:]
        X_val = X[int(X.shape[0]*ptd):,:,:]
        Y_train = Y[:int(Y.shape[0]*ptd)]
        Y_val = Y[int(Y.shape[0]*ptd):]

        
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
        
    with open(os.path.join(save_path, 'hist.json'), 'w') as f:
        json.dump(json_data, f)

if __name__ == '__main__':
    main()