This directory contains all the created and trained keras models.
All models might get generated with multiple configurations (see Python files). Examples include variable window length, amount of neurons, epoch and batch length, etc.

-5minTFsimpleModelWithSessions
These models were trained on AAPL 5-minute candle data.

-5minDailySessionModels
Similar to the previous, except with added day-of-the-week information column added to the dataframe.

-multCandleModels
Models trained on AAPL, but made to predict larger price movements (including multiple candles in the future). I didn't expect this model to perform very well, and it did not really in the end. Also trained on 5-min candle data.

-MultTickers5min
Models trained on multiple tickers (see Python file).

-multTickers5minWithTemporalAttention
Similar to the previous, however these ones use a more complex layer structure. Also trained on 5-min candle data.

model_info.json:
contains information about all of the generated models (name, description, parameters, numerical error data).