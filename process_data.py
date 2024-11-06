import zipfile
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Ścieżki do plików
zip_path = 'data/archive.zip'
extracted_path = 'data/dataset'
output_train = 'data/train_data.csv'
output_test = 'data/test_data.csv'

# Sprawdzenie, czy plik zip istnieje
if not os.path.exists(zip_path):
    raise FileNotFoundError(f"File {zip_path} not found.")

# Rozpakowanie archiwum
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_path)

# Wyszukiwanie pliku CSV w rozpakowanym katalogu
csv_file = None
for file in os.listdir(extracted_path):
    if file.endswith('.csv'):
        csv_file = os.path.join(extracted_path, file)
        break

# Sprawdzenie, czy plik CSV został znaleziony
if csv_file is None:
    raise FileNotFoundError(f"No CSV file found in {extracted_path} directory.")

# Wczytanie pliku CSV
data = pd.read_csv(csv_file)

# Podział danych na zestaw treningowy i testowy
train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

# Zapisz podzielone dane
train_data.to_csv(output_train, index=False)
test_data.to_csv(output_test, index=False)

print("Data processing complete. Training and testing data saved.")
