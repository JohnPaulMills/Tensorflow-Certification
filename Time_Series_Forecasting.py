# Import required libraries
import tensorflow as tf
import pandas as pd
import numpy as np
# Create dataset (x and y into the model)
insurance = pd.read_csv('https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/extras/BTC_USD_2013-10-01_2021-05-18-CoinDesk.csv',
                        parse_dates=["Date"],
                        index_col=["Date"])  # parse the date column (tell pandas column 1 is a datetime)
# Univariate time series
# Only want closing price for each day
bitcoin_prices = pd.DataFrame(insurance["Closing Price (USD)"]).rename(columns={"Closing Price (USD)": "Price"})
HORIZON = 1 # predict 1 step at a time
WINDOW_SIZE = 7 # use a week worth of timesteps to predict the horizon
# Make a copy of the Bitcoin historical data
bitcoin_prices_windowed = bitcoin_prices.copy()
# Add windowed columns
for i in range(WINDOW_SIZE): # Shift values for each step in WINDOW_SIZE
  bitcoin_prices_windowed[f"Price+{i+1}"] = bitcoin_prices_windowed["Price"].shift(periods=i+1)
# Let's create X & y, remove the NaN's and convert to float32 to prevent TensorFlow errors
X = bitcoin_prices_windowed.dropna().drop("Price", axis=1).astype(np.float32)
y = bitcoin_prices_windowed.dropna()["Price"].astype(np.float32)
# Make train and test sets
split_size = int(len(X) * 0.8)
train_windows, train_labels = X[:split_size], y[:split_size]
test_windows, test_labels = X[split_size:], y[split_size:]

# Multivariate time series
# Block reward values
block_reward_1 = 50 # 3 January 2009 (2009-01-03) - this block reward isn't in our dataset (it starts from 01 October 2013)
block_reward_2 = 25 # 28 November 2012
block_reward_3 = 12.5 # 9 July 2016
block_reward_4 = 6.25 # 11 May 2020
# Block reward dates (datetime form of the above date stamps)
block_reward_2_datetime = np.datetime64("2012-11-28")
block_reward_3_datetime = np.datetime64("2016-07-09")
block_reward_4_datetime = np.datetime64("2020-05-11")
# Get date indexes for when to add in different block dates
block_reward_2_days = (block_reward_3_datetime - bitcoin_prices.index[0]).days
block_reward_3_days = (block_reward_4_datetime - bitcoin_prices.index[0]).days
# Add block_reward column
bitcoin_prices_block = bitcoin_prices.copy()
bitcoin_prices_block["block_reward"] = None
# Set values of block_reward column (it's the last column hence -1 indexing on iloc)
bitcoin_prices_block.iloc[:block_reward_2_days, -1] = block_reward_2
bitcoin_prices_block.iloc[block_reward_2_days:block_reward_3_days, -1] = block_reward_3
bitcoin_prices_block.iloc[block_reward_3_days:, -1] = block_reward_4
# Make a copy of the Bitcoin historical data with block reward feature
bitcoin_prices_block_windowed = bitcoin_prices_block.copy()
# Add windowed columns
for i in range(WINDOW_SIZE): # Shift values for each step in WINDOW_SIZE
  bitcoin_prices_block_windowed[f"Price+{i+1}"] = bitcoin_prices_block_windowed["Price"].shift(periods=i+1)
# Let's create X & y, remove the NaN's and convert to float32 to prevent TensorFlow errors
X_multi = bitcoin_prices_block_windowed.dropna().drop("Price", axis=1).astype(np.float32)
y_multi = bitcoin_prices_block_windowed.dropna()["Price"].astype(np.float32)
# Make train and test sets
split_size = int(len(X) * 0.8)
train_block_windows, train_block_labels = X_multi[:split_size], y_multi[:split_size]
test_block_windows, test_block_labels = X_multi[split_size:], y_multi[split_size:]

# Build a model
model_dense = tf.keras.Sequential([
    tf.keras.layers.Dense(
        units=128,
        activation='relu'
    ),
    tf.keras.layers.Dense(
        units=HORIZON,
        activation='linear' # linear activation is the same as having no activation
    )
])
model_dense.compile(
    loss='mae',
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['mae'] # do not necessarily need this when the loss function is already MAE
)
model_dense.fit(
    x=train_windows,
    y=train_labels,
    epochs=100,
    batch_size=128,
    validation_data=(test_windows, test_labels)
)
# Create model with conv1d layer
model_conv1d = tf.keras.Sequential([
  # Create Lambda layer to reshape inputs, without this layer, the model will error
  tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1)), # resize the inputs to adjust for window size / Conv1D 3D input requirements
  tf.keras.layers.Conv1D(filters=128, kernel_size=5, padding="causal", activation="relu"),
  tf.keras.layers.Dense(HORIZON)
])
model_conv1d.compile(loss="mae",
                optimizer=tf.keras.optimizers.Adam()
)
model_conv1d.fit(train_block_windows,
                 train_block_labels,
                 batch_size=128,
                 epochs=100,
                 validation_data=(test_block_windows, test_block_labels)
)
# Create model with LSTM layer
inputs = tf.keras.layers.Input(shape=(WINDOW_SIZE+1)) #+1 because input is window+block
x = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(inputs) # expand input dimension to be compatible with LSTM
# x = tf.keras.layers.LSTM(128, activation="relu", return_sequences=True)(x) # this layer will error if the inputs are not the right shape
x = tf.keras.layers.LSTM(128, activation="relu")(x)
# x = tf.keras.layers.Dense(32, activation="relu")(x)
output = tf.keras.layers.Dense(HORIZON)(x)
model_lstm = tf.keras.Model(inputs=inputs, outputs=output)
model_lstm.compile(loss="mae",
                   optimizer=tf.keras.optimizers.Adam()
)
model_lstm.fit(train_block_windows,
               train_block_labels,
               epochs=100,
               batch_size=128,
               validation_data=(test_block_windows, test_block_labels)
)
# Make predictions using model_dense on the test dataset and view the results
model_dense_preds = tf.squeeze(model_dense.predict(test_windows))