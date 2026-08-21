import csv
import numpy as np
import pandas as pd

# https://archive.ics.uci.edu/ml/datasets/spambase
FILE_NAME = "spambase"

# load the data with csv
with open(FILE_NAME, "r") as f:
    data = list(csv.reader(f, delimiter=','))
data = np.array(data, dtype=np.float32)
print(data.shape)

# load the data with loadtxt from numpy
data = np.loadtxt(FILE_NAME, delimiter=',', dtype=np.float32)
print(data.shape, data.dtype)

# load the data with np.genfromTxt
data = np.genfromtxt(FILE_NAME, delimiter=',', dtype=np.float32)
print(data.shape)

n_samples, n_features = data.shape
n_features -= 1

X = data[:, 0:n_features]
y = data[:, n_features]

print(X.shape, y.shape)
print(X[0, 0:5])

# load data with panda
df = pd.read_csv(FILE_NAME, header=None, skiprows=0, dtype=np.float32)
df = df.fillna(0.0)

# dataframe to numpy
data = df.to_numpy()
print(data[4, 0:5])

data = np.asarray(data, dtype=np.float32)
print(data.dtype)

