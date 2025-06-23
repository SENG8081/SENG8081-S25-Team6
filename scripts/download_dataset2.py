import kaggle
import os

dataset = "nelgiriyewithana/credit-card-fraud-detection-dataset-2023"
download_path = "data/dataset2/"
os.makedirs(download_path, exist_ok=True)
kaggle.api.dataset_download_files(dataset, path=download_path, unzip=True)
print(f"Dataset downloaded to {download_path}")
