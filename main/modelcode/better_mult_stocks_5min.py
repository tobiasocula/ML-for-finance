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

root = Path.cwd()


class TemporalAttention(keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
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


def run(X_train, Y_train, X_val, Y_val, window_length, n_neurons, epochs, batch_size, ptd, fh, le):
    inputs = keras.Input(shape=(window_length, 29))
    x = keras.layers.GRU(n_neurons, return_sequences=True)(inputs)
    x = TemporalAttention(units=64)(x)
    outputs = keras.layers.Dense(7)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    optimizer = keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='mse')
    hist = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,
                     validation_data=(X_val, Y_val))
    return hist.history, model
