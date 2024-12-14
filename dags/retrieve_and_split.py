import json
import logging
import os
import zipfile
from datetime import datetime, timedelta

import gspread
import pandas as pd
import requests
from airflow import DAG
from airflow.operators.python import PythonOperator
from dotenv import load_dotenv
from google.oauth2.service_account import Credentials
from sklearn.model_selection import train_test_split

# Load environment variables from .env file
load_dotenv()

# Constants
DATASET_URL = 'https://www.kaggle.com/api/v1/datasets/download/bwandowando/tomtom-traffic-data-55-countries-387-cities'
LOCAL_ZIP_PATH = '/tmp/archive.zip'
EXTRACT_PATH = '/tmp/dataset'
LOCAL_CSV_PATH = '/tmp/dataset/ForExport.csv'
SAMPLE_SIZE = 50000
TEST_SIZE = 0.3
RANDOM_STATE = 42
TRAIN_CSV_PATH = '/tmp/train.csv'
TEST_CSV_PATH = '/tmp/test.csv'
GOOGLE_SHEETS_CREDENTIALS_ENV = "GOOGLE_SHEETS_CREDENTIALS"
SPREADSHEET_ID = "1ByNYGAETRPbHuB3oRQzljcOf7sGnk814pqLWiOHIxvo"
TRAIN_SHEET_NAME = "Train"
TEST_SHEET_NAME = "Test"

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
    dag_id='retrieve_and_split_data',
    default_args=default_args,
    description='A DAG to download, split, and upload data',
    schedule_interval=None,
)

def download_dataset():
    # Download the dataset
    response = requests.get(DATASET_URL, stream=True)
    with open(LOCAL_ZIP_PATH, 'wb') as file:
        for chunk in response.iter_content(chunk_size=128):
            file.write(chunk)

    # Extract the dataset
    with zipfile.ZipFile(LOCAL_ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_PATH)

    # Clean up the zip file
    os.remove(LOCAL_ZIP_PATH)

def split_dataset():
    df = pd.read_csv(LOCAL_CSV_PATH)
    logging.info("Total rows count: {}".format(df.shape[0]))
    df = df.sample(n=SAMPLE_SIZE)
    train, test = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train.to_csv(TRAIN_CSV_PATH, index=False)
    test.to_csv(TEST_CSV_PATH, index=False)

def upload_to_gsheets():
    credentials_info = os.getenv(GOOGLE_SHEETS_CREDENTIALS_ENV)

    if credentials_info is None or credentials_info == "":
        raise ValueError("GOOGLE_SHEETS_CREDENTIALS jest pusty.")

    creds_dict = json.loads(credentials_info)
    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)

    client = gspread.authorize(creds)

    logging.info("ID arkusza: {}".format(SPREADSHEET_ID))
    spreadsheet = client.open_by_key(SPREADSHEET_ID)
    trainSheet = spreadsheet.worksheet(TRAIN_SHEET_NAME)
    testSheet = spreadsheet.worksheet(TEST_SHEET_NAME)

    try:
        train = pd.read_csv(TRAIN_CSV_PATH)
        test = pd.read_csv(TEST_CSV_PATH)
        logging.info("Pliki CSV wczytane pomyślnie.")
    except FileNotFoundError as e:
        logging.error(f"Plik CSV nie został znaleziony: {e}")
        raise

    train = train.astype(str)
    test = test.astype(str)

    trainSheet.clear()  # Wyczyszczenie arkusza przed zapisem nowych danych
    trainSheet.update(values=[train.columns.values.tolist()] + train.values.tolist(), range_name='')  # Zapisz dane do arkusza

    testSheet.clear()
    testSheet.update(values=[test.columns.values.tolist()] + test.values.tolist(), range_name='')

    logging.info("Dane zostały przesłane do Google Sheets pomyślnie.")

# Define the tasks
download_task = PythonOperator(
    task_id='download_dataset',
    python_callable=download_dataset,
    dag=dag,
)

split_task = PythonOperator(
    task_id='split_dataset',
    python_callable=split_dataset,
    dag=dag,
)

upload_task = PythonOperator(
    task_id='upload_to_gsheets',
    python_callable=upload_to_gsheets,
    dag=dag,
)

# Set the task dependencies
download_task >> split_task >> upload_task