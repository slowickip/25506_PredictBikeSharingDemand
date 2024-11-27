import json
import logging
import os
from datetime import datetime, timedelta

import gspread
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from dotenv import load_dotenv
from google.oauth2.service_account import Credentials
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load environment variables from .env file
load_dotenv()

# Constants
GOOGLE_SHEETS_CREDENTIALS_ENV = "GOOGLE_SHEETS_CREDENTIALS"
SPREADSHEET_ID = "1ByNYGAETRPbHuB3oRQzljcOf7sGnk814pqLWiOHIxvo"
TRAIN_SHEET_NAME = "Train"
PROCESSED_TRAIN_SHEET_NAME = "Processed_Train"

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    dag_id='data_processing_2',
    default_args=default_args,
    description='A DAG to process data from Google Sheets',
    schedule_interval=timedelta(days=1),
)

def download_data_from_gsheets():
    credentials_info = os.getenv(GOOGLE_SHEETS_CREDENTIALS_ENV)

    if credentials_info is None:
        raise ValueError("GOOGLE_SHEETS_CREDENTIALS is empty.")

    creds_dict = json.loads(credentials_info)
    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)

    client = gspread.authorize(creds)

    logging.info("Spreadsheet ID: {}".format(SPREADSHEET_ID))
    spreadsheet = client.open_by_key(SPREADSHEET_ID)
    trainSheet = spreadsheet.worksheet(TRAIN_SHEET_NAME)

    data = trainSheet.get_all_values()
    headers = data.pop(0)
    df = pd.DataFrame(data, columns=headers)
    df.to_csv('/tmp/train.csv', index=False)

def clean_data():
    df = pd.read_csv('/tmp/train.csv')
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df.to_csv('/tmp/cleaned_train.csv', index=False)

def standardize_and_normalize_data():
    df = pd.read_csv('/tmp/cleaned_train.csv')
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    df[numerical_columns] = MinMaxScaler().fit_transform(df[numerical_columns])
    df.to_csv('/tmp/processed_train.csv', index=False)

def upload_processed_data_to_gsheets():
    credentials_info = os.getenv(GOOGLE_SHEETS_CREDENTIALS_ENV)

    if credentials_info is None:
        raise ValueError("GOOGLE_SHEETS_CREDENTIALS is empty.")

    creds_dict = json.loads(credentials_info)
    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)

    client = gspread.authorize(creds)

    logging.info("Spreadsheet ID: {}".format(SPREADSHEET_ID))
    spreadsheet = client.open_by_key(SPREADSHEET_ID)
    processedTrainSheet = spreadsheet.worksheet(PROCESSED_TRAIN_SHEET_NAME)

    processed_train = pd.read_csv('/tmp/processed_train.csv')
    processed_train = processed_train.astype(str)

    processedTrainSheet.clear()
    processedTrainSheet.update(values=[processed_train.columns.values.tolist()] + processed_train.values.tolist(), range_name='')

# Define the tasks
download_task = PythonOperator(
    task_id='download_data_from_gsheets',
    python_callable=download_data_from_gsheets,
    dag=dag,
)

clean_task = PythonOperator(
    task_id='clean_data',
    python_callable=clean_data,
    dag=dag,
)

standardize_normalize_task = PythonOperator(
    task_id='standardize_and_normalize_data',
    python_callable=standardize_and_normalize_data,
    dag=dag,
)

upload_task = PythonOperator(
    task_id='upload_processed_data_to_gsheets',
    python_callable=upload_processed_data_to_gsheets,
    dag=dag,
)

# Set the task dependencies
download_task >> clean_task >> standardize_normalize_task >> upload_task