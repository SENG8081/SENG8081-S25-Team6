# SENG8081-S25-Team6
# Credit Card Fraud Detection Using Big Data Tools

## Project Overview
This project demonstrates the use of open-source data science tools to build a practical and reproducible fraud detection pipeline for credit card transactions. Using two real-world datasets from Kaggle, we apply supervised machine learning models to identify fraudulent transactions and visualize results with Tableau dashboards.

The goal is to simulate a real-world financial fraud detection system that can analyze large transaction volumes and provide actionable insights for fraud prevention.

## Project Team Members
1. Vikas Manchala - Visualization and Reporting
2. Roshan Bartaula - Data Engineer
3. Phani Mallampati - Team Lead & ETL Pipeline Engineer
4. Satyam Patel - ML Engineer

## Prerequisites
1. Python 3.8+
2. Install dependencies: pip install -r requirements.txt
3. Tableau Desktop (for dashboard visualization)

## Dataset

This project uses 2 data sets.

Data Set 1 - **Credit Card Fraud Detection dataset** from Kaggle:
- Source: [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Description**: The dataset contains transactions made by European cardholders in September 2013, with 284,807 transactions and 492 labeled frauds.
- For size and convenience, the dataset is included in compressed format:  `data/creditcard.csv.zip`
- Before using the data, unzip it:
  ```bash
    unzip data/creditcard.csv.zip

 Data Set 2 - **Credit Card Fraud Detection Dataset 2023** from Kaggle:
- Sourse: [Kaggle – Credit Card Fraud Detection 2023](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023)
- Due to file size, this dataset is not stored directly in the repo.
- Setup kaggle API by following below steps before running the script

  ### Kaggle API Setup

        To download the dataset, you need your own Kaggle API credentials.
        
        1. Go to your Kaggle account settings and click "Create New API Token".
        2. This will download a file called `kaggle.json`.
        3. Place `kaggle.json` in a folder named `.kaggle` in your home directory (e.g., `~/.kaggle/kaggle.json`).
        4. **Do not share or commit your `kaggle.json` file.**

   
- To download, run the following script
  ```bash
   python scripts/download_dataset2.py

## Project Structure
  ```bash
    ├── data/
    │   ├── creditcard.csv.zip
    │   ├── [other raw data files]
    │   └── processed/
    │
    ├── scripts/
    │   ├── download_dataset2.py
    │   ├── data_cleaning.py
    │   ├── model_training.py
    │   ├── feature_importance_extraction.py
    │   ├── test_set_predictions.py
    │   └── [other utility scripts]
    │
    ├── plots/
    │   ├── baseline/
    │   ├── model_prep/
    │   └── tuning/
    │
    ├── results/
    │   ├── feature_importance_creditcard_cleaned.csv
    │   ├── feature_importance_credit_card_fraud_2023_cleaned.csv
    │   ├── test_set_predictions_creditcard_cleaned.csv
    │   ├── test_set_predictions_credit_card_fraud_2023_cleaned.csv
    │   └── [other result CSVs]
    │
    ├── tableau/
    │   └── [Tableau workbook files, e.g., dashboards.twbx]
    │
    ├── requirements.txt
    ├── README.md
    ├── Project Report - Team 6.docx
    ├── SENG8081 Field Project Template.docx
    ├── SENG8081-25S-Project.docx
    ├── Project-Updated.docx
    └── [other documentation or supporting files]


## How to Run
    Install dependencies: pip install -r requirements.txt
    Download and extract datasets as described above.
    Run data preparation and model scripts in the scripts/ directory.
    View results and visualizations:
    Plots are saved in the plots/ directory.
    Exported CSVs for Tableau are in the results/ directory.
    Open Tableau dashboards to explore model performance and feature importances.

## Results
    Machine learning models (Logistic Regression, Random Forest) were trained and evaluated on both datasets.
    Model performance metrics, confusion matrices, ROC curves, and feature importances are visualized in the plots/ directory and Tableau dashboards.

## References
Source: [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
Sourse: [Kaggle – Credit Card Fraud Detection 2023](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023)
