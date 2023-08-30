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

df = pd.read_csv('./data_sampled.csv')

replace_dict = {
    'XXS': 0,
    'S': 1,
    'M': 2,
    'L': 3,
    'XL': 4,
    'XXL': 5,
    'XXXL': 6
}

df['size_num'] = df['size'].replace(replace_dict)

X = df[['weight', 'age', 'height']]
y = df[['size_num']]

# Standardization
sc = StandardScaler()
sc.fit(X)

#xs = sc.transform(X)
xs = X

tf.keras.backend.clear_session()

normalizer = Normalization(axis=-1)
normalizer.adapt(xs)

model = tf.keras.Sequential([
    InputLayer(input_shape=(3,), name='input'),
    BatchNormalization(),
    Dense(64, activation='relu', name='hidden1', kernel_initializer='normal'),
    Dense(32, activation='relu', name='hidden2', kernel_initializer='normal'),
    Dense(16, activation='relu', name='hidden3', kernel_initializer='normal'),
    Dense(7, name='output', activation='softmax')
])

model.summary()
plot_model(model, "model.png", show_shapes=True, show_layer_names=True)

earlyStoppingCallback = EarlyStopping(monitor='loss', patience=500, min_delta=0)

model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.01),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

history = model.fit(xs, y,
                    batch_size=200,
                    epochs=10000,
                    verbose=1, callbacks=[earlyStoppingCallback])
pass
