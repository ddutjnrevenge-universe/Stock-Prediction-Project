import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from sklearn.preprocessing import MinMaxScaler
from vnstock import stock_historical_data
import pandas as pd
import base64


def build_model(window_size):
    model = Sequential()
    model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(window_size, 1), padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(128, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(64, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling1D(2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1))
    return model


# Streamlit App
st.title("Vietnam Stock Price Prediction")
st.markdown("Enter stock details to predict future prices:")

# Input Form
with st.form("stock_form"):
    stock_name = st.text_input("Stock Name (Ticker Symbol)", value="VIC", help="E.g., VIC for Vingroup")
    epochs = st.number_input("Epochs", min_value=1, value=50, help="Number of training epochs")
    ahead = st.number_input("Days Ahead to Predict", min_value=1, value=10, help="Number of days to predict ahead")
    days = st.number_input("Window Size (Days)", min_value=1, value=30, help="Days of data used for prediction")
    submit = st.form_submit_button("Predict")

if submit:
    try:
        # Fetch Stock Data
        st.write("Fetching stock data...")
        stock_data = stock_historical_data(stock_name, "2005-01-01", "2024-10-24")

        # Data Preparation
        date_column = 'date' if 'date' in stock_data.columns else stock_data.columns[0]
        stock_data['Date'] = pd.to_datetime(stock_data[date_column], format='%Y-%m-%d', errors='coerce')
        stock_data.dropna(subset=['Date'], inplace=True)
        stock_data.set_index('Date', inplace=True)

        df = stock_data[['close']].copy()
        n = int(df.shape[0] * 0.8)
        training_set = df.iloc[:n].values
        test_set = df.iloc[n:].values

        scaler = MinMaxScaler(feature_range=(0, 1))
        training_set_scaled = scaler.fit_transform(training_set)

        X_train, y_train = [], []
        for i in range(days, n - ahead):
            X_train.append(training_set_scaled[i - days:i, 0])
            y_train.append(training_set_scaled[i + ahead, 0])
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

        # Build and Train Model
        model = build_model(days)
        model.compile(optimizer='adam', loss='mean_squared_error')
        st.write("Training the model...")
        model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)

        # Prediction
        dataset_total = pd.concat((df.iloc[:n], df.iloc[n:]), axis=0)
        inputs = dataset_total[len(dataset_total) - len(test_set) - days:].values

        if len(inputs) <= days:
            st.error("Not enough data to create sequences with specified 'Days'.")
        else:
            inputs = inputs.reshape(-1, 1)
            inputs = scaler.transform(inputs)

            X_test = []
            for i in range(days, inputs.shape[0]):
                X_test.append(inputs[i - days:i, 0])

            if len(X_test) == 0:
                st.error("Insufficient data to create test sequences.")
            else:
                X_test = np.array(X_test).reshape(len(X_test), days, 1)
                predicted_stock_price = model.predict(X_test)
                predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

                # Plot Results
                fig, ax = plt.subplots(figsize=(14, 7))
                ax.plot(df.index[n:], test_set, color='gold', label='Actual Price')
                ax.plot(df.index[n:], predicted_stock_price, color='blue', label='Predicted Price')
                ax.set_title(f"Stock Price Prediction for {stock_name}")
                ax.set_xlabel("Date")
                ax.set_ylabel("Price")
                ax.legend()
                ax.tick_params(axis='x', rotation=45)
                st.pyplot(fig)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
