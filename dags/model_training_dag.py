import os
import json
import pandas as pd
from tpot import TPOTRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import pickle
import gspread
from google.oauth2.service_account import Credentials
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
SPREADSHEET_ID = "1ByNYGAETRPbHuB3oRQzljcOf7sGnk814pqLWiOHIxvo"  # Replace with your actual ID
PROCESSED_TRAIN_SHEET_NAME = "Processed_Train"
PROCESSED_TEST_SHEET_NAME = "Processed_Test"
TARGET_COLUMN = "MinsDelay"  # Update based on your target variable

def fetch_train_test_data():
    logging.info("Fetching train and test data from Google Sheets.")
    creds_json = os.getenv("GOOGLE_SHEETS_CREDENTIALS")
    if not creds_json:
        logging.error("Environment variable 'GOOGLE_SHEETS_CREDENTIALS' is not set or empty.")
        raise ValueError("Environment variable 'GOOGLE_SHEETS_CREDENTIALS' is not set or empty.")
    creds_dict = json.loads(creds_json)

    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    credentials = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    gc = gspread.authorize(credentials)

    train_worksheet = gc.open_by_key(SPREADSHEET_ID).worksheet(PROCESSED_TRAIN_SHEET_NAME)
    train_data = pd.DataFrame(train_worksheet.get_all_records())
    test_worksheet = gc.open_by_key(SPREADSHEET_ID).worksheet(PROCESSED_TEST_SHEET_NAME)
    test_data = pd.DataFrame(test_worksheet.get_all_records())

    os.makedirs('/tmp/airflow/processed_data', exist_ok=True)
    train_data.to_csv('/tmp/airflow/processed_data/train_data.csv', index=False)
    test_data.to_csv('/tmp/airflow/processed_data/test_data.csv', index=False)
    logging.info("Train and test data saved locally.")

def train_with_tpot():
    logging.info("Starting model training with TPOT.")
    train_data = pd.read_csv('/tmp/airflow/processed_data/train_data.csv')
    test_data = pd.read_csv('/tmp/airflow/processed_data/test_data.csv')

    X_train = train_data.drop(columns=[TARGET_COLUMN])
    y_train = train_data[TARGET_COLUMN]
    X_test = test_data.drop(columns=[TARGET_COLUMN])
    y_test = test_data[TARGET_COLUMN]

    for df in [X_train, X_test]:
        df['hour'] = pd.to_datetime(df['UpdateTimeUTC']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['UpdateTimeUTC']).dt.dayofweek
        df['weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    X_train.drop(columns=['UpdateTimeUTC', 'UpdateTimeUTCWeekAgo'], inplace=True)
    X_test.drop(columns=['UpdateTimeUTC', 'UpdateTimeUTCWeekAgo'], inplace=True)

    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test)

    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

    X_train = X_train.apply(pd.to_numeric, errors='coerce')
    X_test = X_test.apply(pd.to_numeric, errors='coerce')

    X_train.fillna(X_train.mean(), inplace=True)
    X_test.fillna(X_test.mean(), inplace=True)

    tpot = TPOTRegressor(
        generations=5,
        population_size=20,
        verbosity=2,
        random_state=42,
        n_jobs=-1
    )

    tpot.fit(X_train, y_train)

    y_pred = tpot.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    os.makedirs('/tmp/airflow/models', exist_ok=True)
    tpot.export('/tmp/airflow/models/best_pipeline.py')
    with open('/tmp/airflow/models/best_model.pkl', 'wb') as f:
        pickle.dump(tpot.fitted_pipeline_, f)

    os.makedirs('/tmp/airflow/reports', exist_ok=True)
    with open('/tmp/airflow/reports/evaluation_report.txt', 'w') as f:
        f.write(f'Mean Squared Error: {mse}\n')
        f.write(f'Mean Absolute Error: {mae}\n')
        f.write(f'Mean Absolute Percentage Error: {mape}\n')
        f.write(f'R^2 Score: {r2}\n')

    logging.info("Model training completed. Evaluation report saved.")

with DAG(
    dag_id="model_training_dag",
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