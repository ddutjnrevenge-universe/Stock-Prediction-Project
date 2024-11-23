import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import pickle

def make_predictions(stock_name, days):
    model = load_model(f"models/{stock_name}_model.h5")
    with open(f"data/processed/{stock_name}_scaler.pkl", 'rb') as f:
        scaler = pickle.load(f)
    
    data = pd.read_csv(f"data/raw/{stock_name}_data.csv")
    df = data[['close']]
    n = int(df.shape[0] * 0.8)
    test_set = df.iloc[n:].values
    inputs = df.iloc[len(df) - len(test_set) - days:].values
    inputs = scaler.transform(inputs)
    
    X_test = []
    for i in range(days, inputs.shape[0]):
        X_test.append(inputs[i-days:i, 0])
    
    X_test = np.array(X_test).reshape(len(X_test), days, 1)
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    np.save(f"data/predictions/{stock_name}_predictions.npy", predictions)
