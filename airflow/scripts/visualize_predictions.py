import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def visualize_predictions(stock_name):
    predictions = np.load(f"data/predictions/{stock_name}_predictions.npy")
    data = pd.read_csv(f"data/raw/{stock_name}_data.csv")
    n = int(data.shape[0] * 0.8)
    test_set = data.iloc[n:]['close'].values
    
    plt.figure(figsize=(14, 7))
    plt.plot(test_set, color='gold', label='Actual Price')
    plt.plot(predictions, color='blue', label='Predicted Price')
    plt.title(f'Stock Price Prediction for {stock_name}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig(f"data/predictions/{stock_name}_plot.png")
