import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, InputLayer, BatchNormalization
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.callbacks import EarlyStopping
# Read data
iris = datasets.load_iris()

# Extract the last 2 columns
X = iris.data
y = iris.target

# Standardization
# sc = StandardScaler()
# sc.fit(X_train)
#X_train_std = sc.transform(X_train)
#X_test_std = sc.transform(X_test)

xs = X


tf.keras.backend.clear_session()

normalizer = Normalization(axis=-1)
normalizer.adapt(xs)

model = tf.keras.Sequential([
    InputLayer(input_shape=(4,), name='input'),
    BatchNormalization(),
    Dense(64, activation='relu', name='hidden1', kernel_initializer='normal'),
    Dense(32, activation='relu', name='hidden2', kernel_initializer='normal'),
    Dense(16, activation='relu', name='hidden3', kernel_initializer='normal'),
    Dense(7, name='output', activation='softmax')
])

model.summary()
plot_model(model, "model.png", show_shapes=True, show_layer_names=True)

earlyStoppingCallback = EarlyStopping(monitor='loss', patience=30, min_delta=0)

model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.01),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

history = model.fit(xs, y,
                    batch_size=200,
                    epochs=1000,
                    verbose=1, callbacks=[earlyStoppingCallback])
pass
