import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0, 26000, 5)
y = x + 5
print(x[:5])
print(y[:5])

print(len(x) * 0.2)
print(len(x) * 0.8)

train_data = x[:4160]
train_labels = y[:4160]
test_data = x[4160:]
test_labels = y [4160:]

tf.random.set_seed(42)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
])

model.compile(loss='mae',
            optimizer=tf.keras.optimizers.Adam(lr=0.0001),
            metrics=['mae'])

history = model.fit(train_data, 
                    train_labels,
                    epochs=100,
                    validation_data=(test_data,test_labels)
)


eval = model.evaluate(test_data, test_labels)
print(eval)

