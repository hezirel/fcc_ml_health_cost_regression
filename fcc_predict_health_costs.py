import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots
from tensorflow import keras
from tensorflow.keras import layers

dataset = pd.read_csv('insurance.csv')
dataset.sample(10)

categorical = dataset.select_dtypes(include=['object', 'category']).columns.tolist()
numerical = dataset.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(categorical)
print(numerical)

split = int(0.8 * len(dataset))

# One-liner for splitting using the split index
train, test = dataset[:split], dataset[split:]

# Better for randomization of split elts
train_ds = dataset.sample(frac=0.8, random_state=0)
test_ds = dataset.drop(train_ds.index)

print(len(train_ds))
print(len(test_ds))

def df_to_dataset(src: pd.DataFrame , shuffle=True, batch_size=5):
  df = src.copy()
  labels = df.pop('expenses')
  ds = {key: value.values[:,tf.newaxis] for key, value in src.items()}
  ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(src))
  ds = ds.batch(batch_size)
  ds = ds.prefetch(batch_size)
  return ds

train_ds = df_to_dataset(train, batch_size=5)
test_ds = df_to_dataset(test, shuffle=False)

[(train_features, label_batch)] = train_ds.take(1)
print('Every feature:', list(train_features.keys()))
print('A batch of ages:', train_features['age'])
print('A batch of targets:', label_batch )

  # layer, so you can use them, or include them in the Keras Functional model later.
inputs = []
encoded_features = []

for header in numerical:
    num_col = tf.keras.Input(shape=(1,), name=header)
    normalization_layer = get_normalization_layer(header, train_ds)
    encoded_num_col = normalization_layer(num_col)
    inputs.append(num_col)
    encoded_features.append(encoded_num_col)

for header in categorical:
    cat_col = tf.keras.Input(shape=(1,), name=header, dtype='string')
    encoding_layer = get_category_encoding_layer(header, train_ds, dtype='string')
    encoded_cat_col = encoding_layer(cat_col)
    inputs.append(cat_col)
    encoded_features.append(encoded_cat_col)

features = tf.keras.layers.concatenate(encoded_features)

def build_and_compile_model(norm):
  model = keras.Sequential([
      norm,
      layers.Dense(64, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(1)
  ])

  model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))
  return model

# Output regression predictions.
model = build_and_compile_model(input)
model.summary

model = None
if model is None:
    exit
# RUN THIS CELL TO TEST YOUR MODEL. DO NOT MODIFY CONTENTS.
# Test model by checking how well the model generalizes using the test set.
loss, mae, mse = model.evaluate(test_ds, test_labels, verbose=2)

print("Testing set Mean Abs Error: {:5.2f} expenses".format(mae))

if mae < 3500:
  print("You passed the challenge. Great job!")
else:
  print("The Mean Abs Error must be less than 3500. Keep trying.")

# Plot predictions.
test_predictions = model.predict(test_ds).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True values (expenses)')
plt.ylabel('Predictions (expenses)')
lims = [0, 50000]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims,lims)
