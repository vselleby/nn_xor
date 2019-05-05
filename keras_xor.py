import tensorflow as tf
import numpy as np

X = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]
])

y = np.array([
    [0],
    [1],
    [1],
    [0]
])

model = tf.keras.Sequential([
    tf.keras.layers.Dense(2, activation=tf.nn.sigmoid, use_bias=True),
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid, use_bias=True)
])

model.compile(optimizer=tf.train.GradientDescentOptimizer(learning_rate=1),
              loss=tf.keras.losses.mean_squared_error,
              metrics=['accuracy'])

model.fit(X, y, epochs=20000)

predictions = model.predict(X)
print(predictions)

