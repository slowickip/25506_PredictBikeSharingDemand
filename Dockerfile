FROM apache/airflow:2.10.3-python3.10

USER airflow
COPY requirements.txt .
RUN pip install -r requirements.txt