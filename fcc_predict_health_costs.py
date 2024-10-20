import pprint
from types import FunctionType
from typing import Any, Dict, Hashable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_docs.modeling
import tensorflow_docs.plots
from tensorflow import keras
from tensorflow.keras import layers

pretty_printer = pprint.PrettyPrinter(indent=2)
pp = (lambda x: pretty_printer.pprint(x))  # noqa: E731

dataset = pd.read_csv('insurance.csv')
dataset.dropna()

categorical = dataset.select_dtypes(include=['object', 'category']).columns.tolist()
numerical = dataset.select_dtypes(include=['int64', 'float64']).columns.tolist()

for col in categorical:
    dataset[col] = dataset[col].astype('category').cat.codes

feature_columns = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']

split = int(0.8 * len(dataset))

# One-liner for splitting using the split index
train, test = dataset[:split], dataset[split:]

# Better for randomization of split elts
train_df = dataset.sample(frac=0.8, random_state=0)
test_df = dataset.drop(train_df.index)

def df_to_ds(src):
    features = src.copy()
    labels = src.pop('expenses')
    return features, labels

train_dataset, train_labels = df_to_ds(train_df)
test_dataset, test_labels= df_to_ds(test_df)

pp(train_dataset.shape)

normalizer = layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_dataset))
pp(normalizer.mean.shape)

model = keras.Sequential([
    normalizer,
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1)
    ])

model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.MeanAbsoluteError(),
              metrics=['mae', 'mse'])

model.fit(train_dataset, train_labels, epochs=200, verbose=1)

# RUN THIS CELL TO TEST YOUR MODEL. DO NOT MODIFY CONTENTS.
# Test model by checking how well the model generalizes using the test set.
result = model.evaluate(test_dataset, test_labels, verbose=2)
pp(result)

# Plot predictions.
test_predictions = model.predict(test_dataset).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True values (expenses)')
plt.ylabel('Predictions (expenses)')
lims = [0, 50000]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims,lims)
