import os
import pandas as pd
from ydata_profiling import ProfileReport

DATASET_SOURCE = 'dataset'
DATASET_PROFILING_REPORT_DESTINATION = "dataset_profiling_report.html"


def generate_data_profile(data: pd.DataFrame):
    profile = ProfileReport(data, title="Pandas Profiling Report", explorative=True)
    profile.to_file(DATASET_PROFILING_REPORT_DESTINATION)

if __name__ == "__main__":
    
    # Search for the CSV file in the extracted directory
    csv_file = None
    for file in os.listdir(DATASET_SOURCE):
        if file.endswith('.csv'):
            csv_file = os.path.join(DATASET_SOURCE, file)
            break

    # Check if the CSV file was found
    if csv_file is None:
        raise FileNotFoundError(f"No CSV file found in {DATASET_SOURCE} directory.")

    # Load the CSV file
    data = pd.read_csv(csv_file)

    # Generate Pandas Profiling report
    generate_data_profile(data)