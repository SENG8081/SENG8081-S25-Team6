# SENG8081-S25-Team6
# Credit Card Fraud Detection Using Big Data Tools

## Project Overview
This project demonstrates the use of the big data and integration tools to build a scalable and real-time fraud detection pipeline for credit card transactions.
It utilizes distributed data storage - HDFS, batch processing tools - Hive, Pig, real-time data streaming - Kafka, and machine learning to detect fraudulent behavior in credit card usage.

The goal is to simulate a real-world financial fraud detection system that can analyze massive transaction volumes and raise alerts in real time.

## Project Team Members
1. Vikas Manchala - Visualization and Reporting
2. Roshan Bartaula - Data Engineer
3. Phani Mallampati - Team Lead & ETL Pipeline Engineer
4. Satyam Patel - ML Engineer

## Prerequisites
1. Install dependencies: `pip install -r requirements.txt`

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
- To download, run the following script
  ```bash
   pip install -r requirements.txt
   python scripts/download_dataset2.py


