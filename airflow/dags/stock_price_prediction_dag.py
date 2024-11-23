from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup
from datetime import datetime, timedelta
import os

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'stock_price_prediction',
    default_args=default_args,
    description='Automate Stock Price Prediction Workflow for Multiple Stocks',
    schedule_interval='@daily',
    start_date=datetime(2024, 11, 1),
    catchup=False,
)

stock_codes=[
    'AAT', 'APH', 'ASG', 'ASM', 'ASP', 'BCG', 'BVH', 'CKG', 'CMG', 'DAG',
    'DAH', 'DPG', 'DBC', 'DGC', 'DLG', 'DXG', 'EVG', 'FIT', 'GEX', 'HAP',
    'HBC', 'HDG', 'HPG', 'HSG', 'KDC', 'KHG', 'MSN', 'NVL', 'NSC', 'OGC',
    'PAN', 'PC1', 'PLX', 'TLG', 'TLH', 'TNI', 'TNT', 'TTB', 'TTF', 'VIC',
    'GVR', 'YEG'
]

def fetch_data_task(stock_name, **kwargs):
    import sys
    sys.path.append('/external_files/scripts')
    from fetch_data import fetch_data
    fetch_data(stock_name)

def preprocess_data_task(stock_name, days, **kwargs):
    import sys
    sys.path.append('/external_files/scripts')
    from preprocess_data import preprocess_data
    preprocess_data(stock_name, days)

def train_model_task(stock_name, days, epochs, **kwargs):
    import sys
    sys.path.append('/external_files/scripts')
    from train_model import train_model
    train_model(stock_name, days, epochs)

def make_predictions_task(stock_name, days, **kwargs):
    import sys
    sys.path.append('/external_files/scripts')
    from make_predictions import make_predictions
    make_predictions(stock_name, days)

def visualize_predictions_task(stock_name, **kwargs):
    import sys
    sys.path.append('/external_files/scripts')
    from visualize_predictions import visualize_predictions
    visualize_predictions(stock_name)

# Create task groups dynamically for each stock ticker
for stock in stock_codes:
    with TaskGroup(group_id=f'{stock}_workflow', dag=dag) as stock_group:
        fetch_task = PythonOperator(
            task_id='fetch_data',
            python_callable=fetch_data_task,
            op_kwargs={'stock_name': stock},
            dag=dag,  # Explicitly associate the task with the DAG
        )

        preprocess_task = PythonOperator(
            task_id='preprocess_data',
            python_callable=preprocess_data_task,
            op_kwargs={'stock_name': stock, 'days': 60},
            dag=dag,  # Explicitly associate the task with the DAG
        )

        train_task = PythonOperator(
            task_id='train_model',
            python_callable=train_model_task,
            op_kwargs={'stock_name': stock, 'days': 60, 'epochs': 20},
            dag=dag,  # Explicitly associate the task with the DAG
        )

        predict_task = PythonOperator(
            task_id='make_predictions',
            python_callable=make_predictions_task,
            op_kwargs={'stock_name': stock, 'days': 60},
            dag=dag,  # Explicitly associate the task with the DAG
        )

        visualize_task = PythonOperator(
            task_id='visualize_predictions',
            python_callable=visualize_predictions_task,
            op_kwargs={'stock_name': stock},
            dag=dag,  # Explicitly associate the task with the DAG
        )

        # Define task dependencies within the group
        fetch_task >> preprocess_task >> train_task >> predict_task >> visualize_task
