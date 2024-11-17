import logging
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularPredictor
import multiprocessing

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

num_cpus = multiprocessing.cpu_count()
logger.info(f"Number of CPU cores available: {num_cpus}")

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
    logger.error(f"No CSV file found in {extracted_path} directory.")
    raise FileNotFoundError(f"No CSV file found in {extracted_path} directory.")
else:
    logger.info(f"CSV file found: {csv_file}")

# Load the CSV file
try:
    data = pd.read_csv(csv_file)
    logger.info(f"Data loaded successfully. Shape: {data.shape}")
except Exception as e:
    logger.error(f"Error loading data: {e}")
    raise

logger.info("Displaying basic information about the data.")
logger.debug(f"Data columns: {data.columns.tolist()}")
logger.debug(f"Data head:\n{data.head()}")

logger.info("Analyzing missing values.")
missing_values = data.isnull().sum()
missing_values = missing_values[missing_values > 0]
if not missing_values.empty:
    logger.warning(f"Missing values found:\n{missing_values}")
else:
    logger.info("No missing values found.")

logger.info("Selecting relevant columns for prediction.")
data = data[['Country', 'City', 'UpdateTimeUTC', 'TrafficIndexLive']]

logger.info("Converting UpdateTimeUTC to datetime and extracting features.")
data['UpdateTimeUTC'] = pd.to_datetime(data['UpdateTimeUTC'], errors='coerce')
if data['UpdateTimeUTC'].isnull().any():
    logger.warning("Some dates could not be converted and resulted in NaT.")

data['Hour'] = data['UpdateTimeUTC'].dt.hour
data['DayOfWeek'] = data['UpdateTimeUTC'].dt.dayofweek
data['Month'] = data['UpdateTimeUTC'].dt.month
data['IsWeekend'] = data['DayOfWeek'] >= 5
data['TimeOfDaySegment'] = pd.cut(
    data['Hour'],
    bins=[0, 6, 12, 18, 24],
    labels=['Night', 'Morning', 'Afternoon', 'Evening'],
    right=False,
    include_lowest=True
)
data = data.drop('UpdateTimeUTC', axis=1)

logger.info("Converting categorical columns to numeric using one-hot encoding.")
data = pd.get_dummies(
    data,
    columns=['Country', 'City', 'DayOfWeek', 'Month', 'TimeOfDaySegment'],
    drop_first=True  # Avoid multicollinearity
)

logger.info("Splitting the data into training and testing sets.")
X = data.drop('TrafficIndexLive', axis=1)
y = data['TrafficIndexLive']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
logger.info(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

logger.info("Initializing AutoGluon AutoML tool.")
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)
predictor = TabularPredictor(label='TrafficIndexLive', eval_metric='r2')

logger.info("Starting model training.")
predictor.fit(train_data=train_data, time_limit=3600, ag_args_fit={'num_cpus': num_cpus})

logger.info("Evaluating the model.")
performance = predictor.evaluate(test_data)
logger.info(f"Model performance: {performance}")

logger.info("Exporting the best model.")
predictor.save('autogluon_best_model')

logger.info("AutoGluon recommended models:")
leaderboard = predictor.leaderboard(test_data, silent=True)
logger.info(f"\n{leaderboard}")

logger.info("Best model pipeline:")
best_model = predictor.get_model_best()
logger.info(best_model)

leaderboard = predictor.leaderboard(test_data, silent=True)
leaderboard_file = 'autogluon_leaderboard.csv'
leaderboard.to_csv(leaderboard_file, index=False)
logger.info(f"Leaderboard saved to {leaderboard_file}")