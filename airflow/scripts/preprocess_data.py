import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle

def preprocess_data(stock_name, days):
    data = pd.read_csv(f"data/raw/{stock_name}_data.csv")
    data['Date'] = pd.to_datetime(data['time'], errors='coerce')
    data.set_index('Date', inplace=True)
    df = data[['close']]
    
    n = int(df.shape[0] * 0.8)
    training_set = df.iloc[:n].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = scaler.fit_transform(training_set)
    
    X_train, y_train = [], []
    for i in range(days, n):
        X_train.append(training_set_scaled[i - days:i, 0])
        y_train.append(training_set_scaled[i, 0])
    
    X_train, y_train = np.array(X_train), np.array(y_train)
    np.save(f"data/processed/{stock_name}_X_train.npy", X_train)
    np.save(f"data/processed/{stock_name}_y_train.npy", y_train)
    
    with open(f"data/processed/{stock_name}_scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
