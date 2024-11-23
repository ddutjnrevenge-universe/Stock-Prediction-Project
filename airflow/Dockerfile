FROM apache/airflow:2.9.1

# Install required Python packages from requirements.txt
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Set environment variables for Airflow
ENV AIRFLOW_HOME=/opt/airflow

WORKDIR $AIRFLOW_HOME
