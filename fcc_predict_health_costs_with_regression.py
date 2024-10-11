# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% id="1rRo8oNqZ-Rj"
# Import libraries. You may or may not use all of these.
# !pip install -q git+https://github.com/tensorflow/docs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# try:
#    # %tensorflow_version only exists in Colab.
#    %tensorflow_version 2.x
# except Exception:
#   pass
import tensorflow as tf
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots
from tensorflow import keras
from tensorflow.keras import layers

# %% id="CiX2FI4gZtTt"
# Import data
# !rm insurance.csv
# !wget https://cdn.freecodecamp.org/project-data/health-costs/insurance.csv


# %% id="KxywZJFCNa5W"
dataset = pd.read_csv('insurance.csv')
y = dataset.pop('expenses')
dataset.sample(10)

# %% id="bdIuVF2eGXfX"
categorical = dataset.select_dtypes(include=['object', 'category']).columns.tolist()
numerical = dataset.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(categorical)
print(numerical)

# %% id="Xe7RXH3N3CWU"
# RUN THIS CELL TO TEST YOUR MODEL. DO NOT MODIFY CONTENTS.
# Test model by checking how well the model generalizes using the test set.
loss, mae, mse = model.evaluate(test_dataset, test_labels, verbose=2)

print("Testing set Mean Abs Error: {:5.2f} expenses".format(mae))

if mae < 3500:
  print("You passed the challenge. Great job!")
else:
  print("The Mean Abs Error must be less than 3500. Keep trying.")

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
