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

    pct_train_data = [0.9]
    # for training
    n_neurons = [64]
    epochs = [25, 35, 45]
    batch_size = [24, 36, 48]

    le = 462
    fh = 12
    wl = 30

    model_name = "mult_candle_model"

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
                normalization_range = data[i - le:i].drop(['Session', 'DaySin'], axis=1)

                min_vals = normalization_range.min(axis=0)
                max_vals = normalization_range.max(axis=0)

                X_window = data.iloc[i-wl:i].copy()
                Y_window = data.iloc[i:i+fh].copy()

                print(X_window)
                print(Y_window)
                sys.exit()
                




if __name__ == '__main__':
    main()