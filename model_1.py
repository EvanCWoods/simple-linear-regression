# Import libraries
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Create a linear dataset
x = np.arange(0, 26000, 5)
y = x + 5
print(x[:5])
print(y[:5])

print(len(x) * 0.2)
print(len(x) * 0.8)

# Create the data subsets
train_data = x[:4160]
test_data = x[4160:]
train_labels = y[:4160]
test_labels = x [4160:]
