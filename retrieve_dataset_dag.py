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
    'data_processing',
    default_args=default_args,
    description='A DAG to download, split, and upload data',
    schedule_interval=timedelta(days=1),
)

def download_dataset():
    url = 'https://www.kaggle.com/api/v1/datasets/download/bwandowando/tomtom-traffic-data-55-countries-387-cities'
    local_zip_path = '/tmp/archive.zip'
    extract_path = '/tmp/dataset'

    # Download the dataset
    response = requests.get(url, stream=True)
    with open(local_zip_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=128):
            file.write(chunk)

    # Extract the dataset
    with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    # Clean up the zip file
    os.remove(local_zip_path)

def split_dataset():
    local_path = '/tmp/dataset/ForExport.csv'
    df = pd.read_csv(local_path)
    logging.info("Total rows count: {}".format(df.shape[0]))
    df = df.sample(n=50000)
    train, test = train_test_split(df, test_size=0.3, random_state=42)
    train.to_csv('/tmp/train.csv', index=False)
    test.to_csv('/tmp/test.csv', index=False)

def upload_to_gsheets():
    # Pobranie danych uwierzytelniających z sekretów środowiskowych
    credentials_info = os.getenv("GOOGLE_SHEETS_CREDENTIALS")

    if credentials_info is None:
        raise ValueError("GOOGLE_SHEETS_CREDENTIALS jest pusty.")

    # Utwórz obiekt poświadczeń z JSON
    creds_dict = json.loads(credentials_info)
    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)

    # Autoryzacja i połączenie z Google Sheets
    client = gspread.authorize(creds)

    # Otwórz arkusz Google Sheets
    spreadsheet_id = "1ByNYGAETRPbHuB3oRQzljcOf7sGnk814pqLWiOHIxvo"
    logging.info("ID arkusza: {}".format(spreadsheet_id))
    spreadsheet = client.open_by_key(spreadsheet_id)
    trainSheet = spreadsheet.worksheet("Train")
    testSheet = spreadsheet.worksheet("Test")

    # Wczytaj dane z plików CSV
    try:
        train = pd.read_csv('/tmp/train.csv')
        test = pd.read_csv('/tmp/test.csv')
        logging.info("Pliki CSV wczytane pomyślnie.")
    except FileNotFoundError as e:
        logging.error(f"Plik CSV nie został znaleziony: {e}")
        raise

    # Konwersja danych na typ str
    train = train.astype(str)
    test = test.astype(str)

    # Zapisz dane do Google Sheets
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
# download_task >> split_task >> upload_task

download_dataset()
split_dataset()
upload_to_gsheets()