import pandas as pd
from vnstock import stock_historical_data
import os
import logging

def fetch_data(stock_name):
    # Ensure the 'data/raw' directory exists
    raw_data_path = "data/raw"
    os.makedirs(raw_data_path, exist_ok=True)
    logging.info(f"Directory {raw_data_path} created or already exists.")
    
    # Fetch and save data
    data = stock_historical_data(stock_name, "2005-01-01", "2024-11-1")
    data.to_csv(f"{raw_data_path}/{stock_name}_data.csv", index=False)
