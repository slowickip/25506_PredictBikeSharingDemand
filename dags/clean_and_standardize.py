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
TEST_SHEET_NAME = "Test"
PROCESSED_TEST_SHEET_NAME = "Processed_Test"

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    dag_id='clean_and_standardize_data',
    default_args=default_args,
    description='A DAG to process data from Google Sheets',
    schedule_interval=None,
)

def download_data_from_gsheets(sheet_name, output_path):
    credentials_info = os.getenv(GOOGLE_SHEETS_CREDENTIALS_ENV)

    if credentials_info is None:
        raise ValueError("GOOGLE_SHEETS_CREDENTIALS is empty.")

    creds_dict = json.loads(credentials_info)
    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)

    client = gspread.authorize(creds)

    logging.info("Spreadsheet ID: {}".format(SPREADSHEET_ID))
    spreadsheet = client.open_by_key(SPREADSHEET_ID)
    sheet = spreadsheet.worksheet(sheet_name)

    data = sheet.get_all_values()
    headers = data.pop(0)
    df = pd.DataFrame(data, columns=headers)
    df.to_csv(output_path, index=False)

def clean_data(input_path, output_path):
    df = pd.read_csv(input_path)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df.to_csv(output_path, index=False)

def standardize_and_normalize_data(input_path, output_path):
    df = pd.read_csv(input_path)
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    df[numerical_columns] = MinMaxScaler().fit_transform(df[numerical_columns])
    df.to_csv(output_path, index=False)

def upload_processed_data_to_gsheets(input_path, sheet_name):
    credentials_info = os.getenv(GOOGLE_SHEETS_CREDENTIALS_ENV)

    if credentials_info is None:
        raise ValueError("GOOGLE_SHEETS_CREDENTIALS is empty.")

    creds_dict = json.loads(credentials_info)
    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)

    client = gspread.authorize(creds)

    logging.info("Spreadsheet ID: {}".format(SPREADSHEET_ID))
    spreadsheet = client.open_by_key(SPREADSHEET_ID)
    sheet = spreadsheet.worksheet(sheet_name)

    df = pd.read_csv(input_path)
    df = df.astype(str)

    sheet.clear()
    sheet.update(values=[df.columns.values.tolist()] + df.values.tolist(), range_name='')

# Define the tasks for train data
download_train_task = PythonOperator(
    task_id='download_train_data_from_gsheets',
    python_callable=download_data_from_gsheets,
    op_args=[TRAIN_SHEET_NAME, '/tmp/train.csv'],
    dag=dag,
)

clean_train_task = PythonOperator(
    task_id='clean_train_data',
    python_callable=clean_data,
    op_args=['/tmp/train.csv', '/tmp/cleaned_train.csv'],
    dag=dag,
)

standardize_normalize_train_task = PythonOperator(
    task_id='standardize_and_normalize_train_data',
    python_callable=standardize_and_normalize_data,
    op_args=['/tmp/cleaned_train.csv', '/tmp/processed_train.csv'],
    dag=dag,
)

upload_train_task = PythonOperator(
    task_id='upload_processed_train_data_to_gsheets',
    python_callable=upload_processed_data_to_gsheets,
    op_args=['/tmp/processed_train.csv', PROCESSED_TRAIN_SHEET_NAME],
    dag=dag,
)

# Define the tasks for test data
download_test_task = PythonOperator(
    task_id='download_test_data_from_gsheets',
    python_callable=download_data_from_gsheets,
    op_args=[TEST_SHEET_NAME, '/tmp/test.csv'],
    dag=dag,
)

clean_test_task = PythonOperator(
    task_id='clean_test_data',
    python_callable=clean_data,
    op_args=['/tmp/test.csv', '/tmp/cleaned_test.csv'],
    dag=dag,
)

standardize_normalize_test_task = PythonOperator(
    task_id='standardize_and_normalize_test_data',
    python_callable=standardize_and_normalize_data,
    op_args=['/tmp/cleaned_test.csv', '/tmp/processed_test.csv'],
    dag=dag,
)

upload_test_task = PythonOperator(
    task_id='upload_processed_test_data_to_gsheets',
    python_callable=upload_processed_data_to_gsheets,
    op_args=['/tmp/processed_test.csv', PROCESSED_TEST_SHEET_NAME],
    dag=dag,
)

# Set the task dependencies
download_train_task >> clean_train_task >> standardize_normalize_train_task >> upload_train_task
download_test_task >> clean_test_task >> standardize_normalize_test_task >> upload_test_task