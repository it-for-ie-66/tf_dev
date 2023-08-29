import numpy as np # Import NumPy - package for working with arrays in Python.
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

nPoint = 100
X = np.linspace(0,4*np.pi, nPoint)
y = np.sin(X) + np.random.random((nPoint,))/2 + X*3/4 + np.cos(3/2*X)

df = pd.DataFrame({'x': X, 'y': y})
xs = df['x'].values.reshape(-1, 1)
ys = df['y'].values.reshape(-1, 1)

# fig, ax = plt.subplots()
# ax.plot(X,y,'*--')
# ax.set_title('Actual Data')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# plt.show()
#fig.savefig('Fig-actual.png', dpi=300)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.callbacks import EarlyStopping
# Clear session
tf.keras.backend.clear_session()

normalizer = Normalization(axis=-1)
normalizer.adapt(xs)

# Create a new dense layer with 1 unit, and input shape of [1].
model = Sequential([
    InputLayer(input_shape=(1,), name='input'),
    normalizer,
    Dense(units=64, activation='sigmoid', kernel_initializer='normal', name='hidden1'),
    Dense(units=64, activation='sigmoid', kernel_initializer='normal', name='hidden2'),
    Dense(units=1,  kernel_initializer='normal', name='output'),
])

# Inspect model
model.summary()
plot_model(model, "model.png", show_shapes=True, show_layer_names=True)
plt.show()

# Compile the model using stochastic gradient descent as optimiser and the mean squared error loss function.
# model.compile(optimizer='adam', loss='mean_squared_error')

earlyStoppingCallback = EarlyStopping(monitor='loss', patience=5000, min_delta=0)

model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error', metrics=['mean_absolute_percentage_error'])

# Train the model
model.fit(xs, ys, epochs=100000, verbose=1, callbacks=[earlyStoppingCallback ])

# Generate multiple test values
arr = np.linspace(0,np.max(df['x'].values), 30)
xpst = tf.convert_to_tensor(arr)
xps = tf.reshape(xpst, (-1, 1))
print(xps)

# Prediction
yps = model.predict(xps)

# Visualization
xps_sq = np.squeeze(xps)
yps_sq = np.squeeze(yps)
dfp = pd.DataFrame({"x":xps_sq, "y":yps_sq})
fig, ax = plt.subplots(figsize=(6, 4))
sns.scatterplot(data=df, x='x', y='y', ax=ax)
sns.lineplot(data=dfp, x='x', y='y', color='g', ax=ax)
plt.legend(['Data','Prediction'])
plt.show()