from flask import Flask, render_template, request, jsonify
from flask_restful import Api, Resource
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from sklearn.preprocessing import MinMaxScaler
from vnstock import stock_historical_data
import pandas as pd
import requests
app = Flask(__name__)
api = Api(app)

predicted_stock_prices = []
dates = []
stock_name = ""
total_prediction_day = 0

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

@app.route('/', methods=['GET', 'POST'])
def form():
    global predicted_stock_prices, dates, stock_name, total_prediction_day
    predicted_stock_prices = []
    dates = []
    stock_name = ""
    total_prediction_day = 0

    if request.method == 'POST':
        stock_name = request.form['Name']
        ep = int(request.form['Epochs'])
        ahead = int(request.form['Ahead'])
        d = int(request.form['Days'])
        total_prediction_day = d + ahead

        try:
            stock_data = stock_historical_data(stock_name, "2005-01-01", "2024-11-1")
        except Exception as e:
            return render_template("form.html", error=f"Error fetching data for {stock_name}: {str(e)}")
        
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
        for i in range(d, n - ahead):
            X_train.append(training_set_scaled[i - d:i, 0])
            y_train.append(training_set_scaled[i + ahead, 0])
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

        model = build_model(d)
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=ep, batch_size=32)

        dataset_total = pd.concat((df.iloc[:n], df.iloc[n:]), axis=0)
        inputs = dataset_total[len(dataset_total) - len(test_set) - d:].values

        if len(inputs) <= d:
            return render_template("form.html", error="Not enough data to create sequences with specified 'Days'.")

        inputs = inputs.reshape(-1, 1)
        inputs = scaler.transform(inputs)

        X_test = []
        for i in range(d, inputs.shape[0]):
            X_test.append(inputs[i - d:i, 0])

        if len(X_test) == 0:
            return render_template("form.html", error="Insufficient data to create test sequences. Adjust 'Days' or 'Ahead' values.")

        X_test = np.array(X_test).reshape(len(X_test), d, 1)

        predicted_stock_price = model.predict(X_test)
        predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

        # Store predictions and dates for /predict endpoint
        predicted_stock_prices = predicted_stock_price.flatten().tolist()
        dates = df.index[n:].strftime('%Y-%m-%d').tolist()

        plt.figure(figsize=(14, 7))
        plt.plot(df.index[n:], test_set, color='gold', label='Actual Price')
        plt.plot(df.index[n:], predicted_stock_price, color='blue', label='Predicted Price')
        plt.title(f'Stock Price Prediction for {stock_name}')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()

        STOCK = BytesIO()
        plt.savefig(STOCK, format="png")
        STOCK.seek(0)
        plot_url = base64.b64encode(STOCK.getvalue()).decode('utf8')

        return render_template("form.html", plot_url=plot_url, stock_name=stock_name, total_prediction_day=total_prediction_day)

    return render_template('form.html')

@app.route('/predict', methods=['GET'])
def predict():
    if not predicted_stock_prices:
        return "No predictions available. Please use the main page to generate predictions first."
    return render_template(
        "predictions.html",
        stock_name=stock_name,
        total_prediction_day=total_prediction_day,
        predictions=zip(dates, predicted_stock_prices)
    )
    
if __name__ == "__main__":
    app.run(debug=True)
