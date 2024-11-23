import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

def train_model(stock_name, days, epochs):
    X_train = np.load(f"data/processed/{stock_name}_X_train.npy")
    y_train = np.load(f"data/processed/{stock_name}_y_train.npy")
    
    model = Sequential()
    model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(days, 1), padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(128, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(64, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling1D(2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=epochs, batch_size=32)
    model.save(f"models/{stock_name}_model.h5")
