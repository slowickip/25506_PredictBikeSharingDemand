import os
import json
import pandas as pd
from tpot import TPOTRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import gspread
from google.oauth2.service_account import Credentials
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

# Constants
SPREADSHEET_ID = "1ByNYGAETRPbHuB3oRQzljcOf7sGnk814pqLWiOHIxvo"  # Replace with your actual ID
PROCESSED_TRAIN_SHEET_NAME = "Processed_Train"
PROCESSED_TEST_SHEET_NAME = "Processed_Test"


def fetch_train_test_data():
    # Retrieve credentials from environment variable
    creds_json = os.getenv("GOOGLE_SHEETS_CREDENTIALS")
    if not creds_json:
        raise ValueError("Environment variable 'GOOGLE_SHEETS_CREDENTIALS' is not set or empty.")
    creds_dict = json.loads(creds_json)

    # Authenticate and initialize Google Sheets client
    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    gc = gspread.authorize(creds)

    # Fetch training data
    spreadsheet = gc.open_by_key(SPREADSHEET_ID)
    train_worksheet = spreadsheet.worksheet(PROCESSED_TRAIN_SHEET_NAME)
    train_data = pd.DataFrame(train_worksheet.get_all_records())

    # Fetch testing data
    test_worksheet = spreadsheet.worksheet(PROCESSED_TEST_SHEET_NAME)
    test_data = pd.DataFrame(test_worksheet.get_all_records())

    # Save train and test data locally
    os.makedirs('/tmp/airflow/processed_data', exist_ok=True)
    train_data.to_csv('/tmp/airflow/processed_data/train_data.csv', index=False)
    test_data.to_csv('/tmp/airflow/processed_data/test_data.csv', index=False)


def train_with_tpot():
    # Load train and test data
    train_data = pd.read_csv('/tmp/airflow/processed_data/train_data.csv')
    test_data = pd.read_csv('/tmp/airflow/processed_data/test_data.csv')

    target_column = "target_column"  # Replace with your actual target column
    X_train = train_data.drop(target_column, axis=1)
    y_train = train_data[target_column]
    X_test = test_data.drop(target_column, axis=1)
    y_test = test_data[target_column]

    # Initialize TPOTRegressor
    tpot = TPOTRegressor(
        generations=5,
        population_size=20,
        verbosity=2,
        random_state=42,
        n_jobs=-1
    )

    # Fit TPOT to training data
    tpot.fit(X_train, y_train)

    # Evaluate the best pipeline on test data
    y_pred = tpot.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Save the trained pipeline
    os.makedirs('/tmp/airflow/models', exist_ok=True)
    tpot.export('/tmp/airflow/models/best_pipeline.py')  # Saves the Python script for the best pipeline
    with open('/tmp/airflow/models/best_model.pkl', 'wb') as f:
        pickle.dump(tpot.fitted_pipeline_, f)

    # Save evaluation report
    os.makedirs('/tmp/airflow/reports', exist_ok=True)
    with open('/tmp/airflow/reports/evaluation_report.txt', 'w') as f:
        f.write(f"Mean Squared Error: {mse}\n")
        f.write(f"Mean Absolute Error: {mae}\n")
        f.write(f"R^2 Score: {r2}\n")


# Define the DAG
with DAG(
        dag_id="tpot_regression_training_dag",
        start_date=datetime(2024, 12, 1),
        schedule_interval=None,
) as dag:
    fetch_data_task = PythonOperator(
        task_id="fetch_train_test_data",
        python_callable=fetch_train_test_data,
    )

    train_model_task = PythonOperator(
        task_id="train_with_tpot",
        python_callable=train_with_tpot,
    )

    fetch_data_task >> train_model_task