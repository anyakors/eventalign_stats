import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

import tensorflow as tf
import keras

from keras.layers import Dense, Conv1D, BatchNormalization, Activation
from keras.layers import Input, Cropping1D
from keras.models import Model

from utils import *


def simple_regression(inputs_shape):

    model = keras.Sequential([
        keras.layers.Dense(16, activation='relu', input_shape=inputs_shape),
        keras.layers.Flatten(),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    #optimizer = tf.keras.optimizers.RMSprop(0.01)
    optimizer = keras.optimizers.Adam(learning_rate=0.01)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  #optimizer='adam',
                  metrics=['mae', 'mse'])

    return model


stats_table = pd.read_csv('./tables/stats_RNA3.csv', index_col=0)

# %%%%%%%%%%%%%%%%%%%%%%% TRAIN %%%%%%%%%%%%%%%%%%%%%%%

train_data = transform_input(stats_table)
print(train_data.shape)

labels = np.zeros(len(train_data))
# have to assign scores manually for now
labels[15:19] = 0.5
labels[19:36] = 1.0
labels[36:41] = 0.5
print(labels.shape)

model = simple_regression(train_data.shape[1:])
print(model.summary())

model.fit(train_data, labels, epochs=25, batch_size=8, shuffle=True)

# %%%%%%%%%%%%%%%%%%%%%%% PREDICT %%%%%%%%%%%%%%%%%%%%%%%

stats_table_test = pd.read_csv('./tables/stats_RNA5.csv', index_col=0)
test_data = transform_input(stats_table_test)

print("Predicting labels from the trained model...")
y_pred = add_flanks(np.array(model.predict(test_data)).flatten())

# %%%%%%%%%%%%%%%%%%%%%%% DECISION FUNCTION PERFORMANCE %%%%%%%%%%%%%%%%%%%%%%%

ax = plt.subplot(211)
ax.plot(y_pred, label='prediction')
ax.plot(stats_table_test['KS'], label='KS')
ax.legend()
ax.set_title("Decision function prediction on RNA5")

# re-predict for the train data for comparison
test_data = train_data
y_pred = add_flanks(np.array(model.predict(test_data)).flatten())

ax = plt.subplot(212)
ax.plot(y_pred, label='prediction')
ax.plot(stats_table['KS'], label='KS')
ax.legend()
ax.set_title("Decision function re-prediction on RNA3")

plt.show()