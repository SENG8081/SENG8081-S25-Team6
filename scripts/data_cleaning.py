
"""
Credit Card Fraud Detection - Data Cleanup Script
SENG8081-S25-Team6

This script cleans and preprocesses two credit card fraud detection datasets:
1. MLG-ULB Credit Card Fraud Dataset (from Kaggle)
2. Credit Card Fraud Detection Dataset 2023 (from Kaggle)
"""

import pandas as pd
import numpy as np
import os
import zipfile
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class CreditCardDataCleaner:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

    def load_dataset1(self, file_path):
        """Load the MLG-ULB Credit Card Fraud Dataset"""
        try:
            # Check if it's a zip file
            if file_path.endswith('.zip'):
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall('data/extracted/')
                    # Find the CSV file
                    for file in os.listdir('data/extracted/'):
                        if file.endswith('.csv'):
                            df = pd.read_csv(f'data/extracted/{file}')
                            break
            else:
                df = pd.read_csv(file_path)

            print(f"Dataset 1 loaded: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading dataset 1: {e}")
            return None

    def load_dataset2(self, file_path):
        """Load the Credit Card Fraud Detection Dataset 2023"""
        try:
            df = pd.read_csv(file_path)
            print(f"Dataset 2 loaded: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading dataset 2: {e}")
            return None

    def explore_data(self, df, dataset_name):
        """Explore dataset structure and quality"""
        print(f"\n=== {dataset_name} Exploration ===")
        print(f"Shape: {df.shape}")
        print(f"\nColumns: {list(df.columns)}")
        print(f"\nData types:\n{df.dtypes}")
        print(f"\nMissing values:\n{df.isnull().sum()}")
        print(f"\nDuplicate rows: {df.duplicated().sum()}")

        # Check for target variable
        potential_targets = ['Class', 'class', 'fraud', 'Fraud', 'is_fraud', 'target']
        target_col = None
        for col in potential_targets:
            if col in df.columns:
                target_col = col
                break

        if target_col:
            print(f"\nTarget variable '{target_col}' distribution:")
            print(df[target_col].value_counts())
            print(f"Fraud percentage: {(df[target_col].sum() / len(df)) * 100:.2f}%")

        return target_col

    def clean_dataset1(self, df):
        """Clean MLG-ULB dataset (typically has V1-V28 PCA features)"""
        print("\n=== Cleaning Dataset 1 (MLG-ULB) ===")

        # Create a copy
        df_clean = df.copy()

        # Handle missing values
        missing_before = df_clean.isnull().sum().sum()
        df_clean = df_clean.dropna()
        missing_after = df_clean.isnull().sum().sum()
        print(f"Removed {missing_before - missing_after} rows with missing values")

        # Remove duplicates
        duplicates_before = df_clean.duplicated().sum()
        df_clean = df_clean.drop_duplicates()
        duplicates_after = df_clean.duplicated().sum()
        print(f"Removed {duplicates_before - duplicates_after} duplicate rows")

        # Handle outliers in Amount column if it exists
        if 'Amount' in df_clean.columns:
            Q1 = df_clean['Amount'].quantile(0.25)
            Q3 = df_clean['Amount'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers_before = len(df_clean)
            df_clean = df_clean[(df_clean['Amount'] >= lower_bound) & (df_clean['Amount'] <= upper_bound)]
            outliers_after = len(df_clean)
            print(f"Removed {outliers_before - outliers_after} outliers based on Amount")

        # Normalize Time column if it exists
        if 'Time' in df_clean.columns:
            df_clean['Time_normalized'] = (df_clean['Time'] - df_clean['Time'].min()) / (df_clean['Time'].max() - df_clean['Time'].min())

        print(f"Final shape after cleaning: {df_clean.shape}")
        return df_clean

    def clean_dataset2(self, df):
        """Clean 2023 Credit Card Fraud dataset"""
        print("\n=== Cleaning Dataset 2 (2023) ===")

        # Create a copy
        df_clean = df.copy()

        # Handle missing values
        missing_before = df_clean.isnull().sum().sum()

        # Fill numerical missing values with median
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)

        # Fill categorical missing values with mode
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)

        missing_after = df_clean.isnull().sum().sum()
        print(f"Handled {missing_before - missing_after} missing values")

        # Remove duplicates
        duplicates_before = df_clean.duplicated().sum()
        df_clean = df_clean.drop_duplicates()
        duplicates_after = df_clean.duplicated().sum()
        print(f"Removed {duplicates_before - duplicates_after} duplicate rows")

        # Encode categorical variables
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col not in ['Class', 'class', 'fraud', 'Fraud', 'is_fraud', 'target']]

        for col in categorical_cols:
            if df_clean[col].nunique() < 50:  # Only encode if reasonable number of categories
                df_clean[f'{col}_encoded'] = self.label_encoder.fit_transform(df_clean[col].astype(str))
                print(f"Encoded categorical column: {col}")

        # Handle outliers for numerical columns
        for col in numerical_cols:
            if col not in ['Class', 'class', 'fraud', 'Fraud', 'is_fraud', 'target']:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers_count = len(df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)])
                if outliers_count > 0:
                    # Cap outliers instead of removing them
                    df_clean[col] = np.where(df_clean[col] < lower_bound, lower_bound, df_clean[col])
                    df_clean[col] = np.where(df_clean[col] > upper_bound, upper_bound, df_clean[col])
                    print(f"Capped {outliers_count} outliers in column: {col}")

        print(f"Final shape after cleaning: {df_clean.shape}")
        return df_clean

    def feature_engineering(self, df, dataset_name):
        """Create additional features"""
        print(f"\n=== Feature Engineering for {dataset_name} ===")

        df_features = df.copy()

        # Time-based features if Time column exists
        if 'Time' in df_features.columns:
            df_features['Hour'] = (df_features['Time'] / 3600) % 24
            df_features['Day'] = (df_features['Time'] / (3600 * 24)) % 7
            print("Created time-based features: Hour, Day")

        # Amount-based features if Amount column exists
        if 'Amount' in df_features.columns:
            df_features['Amount_log'] = np.log1p(df_features['Amount'])
            df_features['Amount_sqrt'] = np.sqrt(df_features['Amount'])
            print("Created amount-based features: Amount_log, Amount_sqrt")

        # Statistical features for PCA components (V1-V28)
        v_columns = [col for col in df_features.columns if col.startswith('V')]
        if len(v_columns) > 0:
            df_features['V_mean'] = df_features[v_columns].mean(axis=1)
            df_features['V_std'] = df_features[v_columns].std(axis=1)
            df_features['V_sum'] = df_features[v_columns].sum(axis=1)
            print(f"Created statistical features from {len(v_columns)} V columns")

        return df_features

    def save_cleaned_data(self, df, filename):
        """Save cleaned dataset"""
        output_path = f'data/cleaned/{filename}'
        os.makedirs('data/cleaned', exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Saved cleaned data to: {output_path}")
        return output_path

    def generate_data_quality_report(self, original_df, cleaned_df, dataset_name):
        """Generate data quality report"""
        print(f"\n=== Data Quality Report for {dataset_name} ===")

        report = {
            'original_shape': original_df.shape,
            'cleaned_shape': cleaned_df.shape,
            'rows_removed': original_df.shape[0] - cleaned_df.shape[0],
            'columns_added': cleaned_df.shape[1] - original_df.shape[1],
            'missing_values_original': original_df.isnull().sum().sum(),
            'missing_values_cleaned': cleaned_df.isnull().sum().sum(),
            'duplicates_original': original_df.duplicated().sum(),
            'duplicates_cleaned': cleaned_df.duplicated().sum()
        }

        for key, value in report.items():
            print(f"{key}: {value}")

        return report

def main():
    """Main execution function"""
    print("Credit Card Fraud Detection - Data Cleanup Pipeline")
    print("=" * 60)

    # Initialize cleaner
    cleaner = CreditCardDataCleaner()

    # Dataset paths (update these paths according to your file structure)
    dataset1_path = 'data/creditcard.csv.zip'  # MLG-ULB dataset
    dataset2_path = 'data/creditcard_2023.csv'  # 2023 dataset

    # Process Dataset 1
    if os.path.exists(dataset1_path):
        print("\nProcessing Dataset 1 ...")
        df1_original = cleaner.load_dataset1(dataset1_path)
        if df1_original is not None:
            target1 = cleaner.explore_data(df1_original, "Dataset 1")
            df1_cleaned = cleaner.clean_dataset1(df1_original)
            df1_final = cleaner.feature_engineering(df1_cleaned, "Dataset 1")
            cleaner.save_cleaned_data(df1_final, 'creditcard_cleaned.csv')
            cleaner.generate_data_quality_report(df1_original, df1_final, "Dataset 1")
    else:
        print(f"Dataset 1 not found at: {dataset1_path}")

    # Process Dataset 2
    if os.path.exists(dataset2_path):
        print("\nProcessing Dataset 2 (2023)...")
        df2_original = cleaner.load_dataset2(dataset2_path)
        if df2_original is not None:
            target2 = cleaner.explore_data(df2_original, "Dataset 2")
            df2_cleaned = cleaner.clean_dataset2(df2_original)
            df2_final = cleaner.feature_engineering(df2_cleaned, "Dataset 2")
            cleaner.save_cleaned_data(df2_final, 'creditcard_2023_cleaned.csv')
            cleaner.generate_data_quality_report(df2_original, df2_final, "Dataset 2")
    else:
        print(f"Dataset 2 not found at: {dataset2_path}")

    print("\nData cleanup pipeline completed!")

if __name__ == "__main__":
    main()
