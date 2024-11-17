import logging
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularPredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths to files
extracted_path = 'dataset'
output_train = 'dataset/train_data.csv'
output_test = 'dataset/test_data.csv'

logger.info("Searching for CSV file in the extracted directory.")
csv_file = None
for file in os.listdir(extracted_path):
    if file.endswith('.csv'):
        csv_file = os.path.join(extracted_path, file)
        break

# Check if the CSV file was found
if csv_file is None:
    raise FileNotFoundError(f"No CSV file found in {extracted_path} directory.")
logger.info(f"CSV file found: {csv_file}")

# Load the CSV file
data = pd.read_csv(csv_file)

logger.info("Displaying basic information about the data.")
logger.info(data.describe())

logger.info("Analyzing missing values.")
missing_values = data.isnull().sum()
logger.info(missing_values[missing_values > 0])

# Select relevant columns for prediction
data = data[['Country', 'City', 'UpdateTimeUTC', 'TrafficIndexLive']]

logger.info("Converting UpdateTimeUTC to datetime and extracting useful features.")
data['UpdateTimeUTC'] = pd.to_datetime(data['UpdateTimeUTC'])
data['Hour'] = data['UpdateTimeUTC'].dt.hour
data['DayOfWeek'] = data['UpdateTimeUTC'].dt.dayofweek
data['Month'] = data['UpdateTimeUTC'].dt.month
data['IsWeekend'] = data['UpdateTimeUTC'].dt.dayofweek >= 5
data['TimeOfDaySegment'] = pd.cut(data['Hour'], bins=[0, 6, 12, 18, 24], labels=['Night', 'Morning', 'Afternoon', 'Evening'], right=False)
data = data.drop('UpdateTimeUTC', axis=1)

logger.info("Converting categorical columns to numeric using one-hot encoding.")
data = pd.get_dummies(data, columns=['Country', 'City', 'DayOfWeek', 'Month', 'TimeOfDaySegment'])

logger.info("Splitting the data into training and testing sets.")
X = data.drop('TrafficIndexLive', axis=1)
y = data['TrafficIndexLive']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

logger.info("Initializing AutoGluon AutoML tool.")
predictor = TabularPredictor(label='TrafficIndexLive', eval_metric='r2').fit(train_data=pd.concat([X_train, y_train], axis=1), time_limit=600)

logger.info("Evaluating the model.")
performance = predictor.evaluate(pd.concat([X_test, y_test], axis=1))
logger.info(f"Best model score: {performance}")

logger.info("Exporting the best model.")
predictor.save('autogluon_best_model')

logger.info("AutoGluon recommended models:")
logger.info(predictor.leaderboard(pd.concat([X_test, y_test], axis=1), silent=True))

logger.info("Best model pipeline:")
logger.info(predictor.model_best)