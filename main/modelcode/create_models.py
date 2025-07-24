import itertools
import keras
import pandas as pd
import sys
import numpy as np
import math
import json
from multiprocessing import Pool
from more_itertools import peekable

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

def normalize(X, min, max):
    # min, max: shape (1, 4)
    # X: shape (n_windows, window_len, 4)
    return (X - min) / (max - min + 1e-08)

def run(X_train, Y_train, X_val, Y_val, window_length, n_neurons, epochs, batch_size):
    model = keras.Sequential([
        keras.Input((window_length, 4)),
        keras.layers.GRU(n_neurons, activation='tanh', return_sequences=True),
        keras.layers.GRU(n_neurons, activation='tanh'),
        keras.layers.Dense(1)  # Output layer for regression (predicting price)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size)
    hist = model.evaluate(X_val, Y_val) # is float
    return hist, model

def main():

    window_length = [100]
    pct_train_data = [0.9]
    forecast_horizon = [1]
    len_epochs = [700] # approx 2 years (normalization)
    # for training
    n_neurons = [64]
    epochs = [25]
    batch_size = [24]

    N_SUBPROCESSES = 10
    session_vars = []
    data = pd.read_csv('data/AAPL.csv')[['Open', 'High', 'Low', 'Close']].values
    errors = []
    counter = 0
    dct = {} # will store json

    model_name = 'OHCL_model'


    for params, is_last in with_last_flag(itertools.product(
        window_length, pct_train_data, forecast_horizon,
        len_epochs, n_neurons, epochs, batch_size)):

        wl, ptd, fh, le, nn, ep, bs = params
        
        X, Y = [], []

        for i in range(le, len(data) - wl - fh + 1):
        
            d = data[i-le:i]
            m = d.min(axis=0) # size (4, 1)
            M = d.max(axis=0) # size (4, 1)
            X.append(normalize(data[i:i+wl,:], m, M))
            Y.append(normalize(data[i+wl:i+wl+fh,:], m, M))

        X, Y = np.array(X), np.array(Y)

        X_train = X[:int(X.shape[0]*ptd),:,:]
        X_val = X[int(X.shape[0]*ptd):,:,:]
        Y_train = Y[:int(Y.shape[0]*ptd),:,:]
        Y_val = Y[int(Y.shape[0]*ptd):,:,:]

        # print(X_train.shape, X_val.shape, Y_train.shape, Y_val.shape) ->
        # (5049, 100, 4) (562, 100, 4) (5049, 1, 4) (562, 1, 4)

        session_vars.append((X_train, Y_train, X_val, Y_val, wl, nn, ep, bs))

        if counter % N_SUBPROCESSES == 0 and i != 0 or is_last:
            with Pool(N_SUBPROCESSES) as pool:
                results = pool.starmap(run, session_vars) # list of errors
            for r in results:
                err, mdl = r[0], r[1]
                errors.append(err)
                mdl.save(f"{model_name}_{counter}.keras")

            session_vars = []

        counter += 1

        dct[f"{model_name}_{counter}"] = {
            "params": {
                "n_neurons": nn,
                "window_length": wl,
                "pct_train_data": ptd,
                "epochs": ep,
                "batch_size": bs,
                "forecast_horizon": fh,
                "epoch_length": le
            },
            "errors": errors
        }
        errors = []

    with open("hist.json", "w") as f:
        json.dump(dct, f)
            
if __name__ == "__main__":
    main()