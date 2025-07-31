
"""
Credit Card Fraud Detection - Model Preparation Script
SENG8081-S25-Team6

This script prepares both datasets for machine learning modeling:
1. MLG-ULB Credit Card Fraud Dataset (PCA-transformed features)
2. Credit Card Fraud Detection Dataset 2023

Features:
- Automatic dataset detection and handling
- Feature selection and engineering
- Class imbalance handling (multiple strategies)
- Train/validation/test splits
- Feature scaling
- Data export for modeling
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTETomek, SMOTEENN
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class ModelPreparation:
    def __init__(self):
        self.scaler = None
        self.label_encoders = {}
        self.feature_selector = None
        self.balancing_method = None

        # Create output directories
        os.makedirs('data/processed', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        os.makedirs('plots/model_prep', exist_ok=True)

    def load_and_identify_dataset(self, file_path):
        """Load dataset and identify its type"""
        try:
            df = pd.read_csv(file_path)
            print(f"Dataset loaded: {df.shape}")

            # Identify dataset type
            v_columns = [col for col in df.columns if col.startswith('V')]
            if len(v_columns) > 20:
                dataset_type = "MLG-ULB"
                print("Dataset identified as: MLG-ULB (PCA-transformed)")
            else:
                dataset_type = "2023"
                print("Dataset identified as: 2023 Credit Card Fraud")

            return df, dataset_type
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None, None

    def identify_target_column(self, df):
        """Identify the target column"""
        target_candidates = ['Class', 'class', 'fraud', 'Fraud', 'is_fraud', 'target', 'label']

        for col in target_candidates:
            if col in df.columns:
                # Verify it's binary
                unique_vals = df[col].nunique()
                if unique_vals == 2:
                    print(f"Target column identified: '{col}'")
                    return col

        print("Warning: Could not identify target column automatically")
        return None

    def prepare_features_mlg_ulb(self, df, target_col, dataset_name):
        """Prepare features for MLG-ULB dataset"""
        print("\nPreparing features for {dataset_name} MLG-ULB dataset...")

        # Get V columns (PCA features)
        v_columns = [col for col in df.columns if col.startswith('V')]
        print(f"Found {len(v_columns)} PCA features (V1-V{len(v_columns)})")

        # Base features
        feature_columns = v_columns.copy()

        # Add Amount if present
        if 'Amount' in df.columns:
            feature_columns.append('Amount')
            # Create Amount-based features
            df['Amount_log'] = np.log1p(df['Amount'])
            df['Amount_sqrt'] = np.sqrt(df['Amount'])
            feature_columns.extend(['Amount_log', 'Amount_sqrt'])
            print("Added Amount and derived features")

        # Add Time-based features if present
        if 'Time' in df.columns:
            df['Hour'] = ((df['Time'] / 3600) % 24).astype(int)
            df['Day'] = ((df['Time'] / (3600 * 24)) % 7).astype(int)
            df['Time_normalized'] = (df['Time'] - df['Time'].min()) / (df['Time'].max() - df['Time'].min())
            feature_columns.extend(['Hour', 'Day', 'Time_normalized'])
            print("Added time-based features")

        # Statistical features from V columns
        if len(v_columns) > 0:
            df['V_mean'] = df[v_columns].mean(axis=1)
            df['V_std'] = df[v_columns].std(axis=1)
            df['V_sum'] = df[v_columns].sum(axis=1)
            df['V_max'] = df[v_columns].max(axis=1)
            df['V_min'] = df[v_columns].min(axis=1)
            feature_columns.extend(['V_mean', 'V_std', 'V_sum', 'V_max', 'V_min'])
            print("Added statistical features from V columns")

        print(f"Total features prepared: {len(feature_columns)}")
        return df, feature_columns

    def prepare_features_2023(self, df, target_col, dataset_name):
        """Prepare features for 2023 dataset"""
        print("\nPreparing features for {dataset_name} 2023 dataset...")

        # Identify feature types
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

        # Remove target from feature lists
        if target_col in numerical_cols:
            numerical_cols.remove(target_col)
        if target_col in categorical_cols:
            categorical_cols.remove(target_col)

        print(f"Numerical features: {len(numerical_cols)}")
        print(f"Categorical features: {len(categorical_cols)}")

        feature_columns = numerical_cols.copy()

        # Encode categorical variables
        for col in categorical_cols:
            if df[col].nunique() < 50:  # Only encode if reasonable number of categories
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
                feature_columns.append(f'{col}_encoded')
                print(f"Encoded categorical feature: {col}")

        # Create interaction features for important numerical columns
        if 'amount' in [col.lower() for col in numerical_cols]:
            amount_col = [col for col in numerical_cols if col.lower() == 'amount'][0]
            df[f'{amount_col}_log'] = np.log1p(df[amount_col])
            df[f'{amount_col}_sqrt'] = np.sqrt(df[amount_col])
            feature_columns.extend([f'{amount_col}_log', f'{amount_col}_sqrt'])
            print(f"Added derived features for {amount_col}")

        # Time-based features if time column exists
        time_cols = [col for col in numerical_cols if 'time' in col.lower() or 'hour' in col.lower()]
        for time_col in time_cols:
            if df[time_col].max() > 24:  # Likely not already in hours
                df[f'{time_col}_hour'] = (df[time_col] % 24).astype(int)
                feature_columns.append(f'{time_col}_hour')
                print(f"Added hour feature from {time_col}")

        print(f"Total features prepared: {len(feature_columns)}")
        return df, feature_columns

    def feature_selection(self, X, y, method='mutual_info', k=20):
        """Select top k features using specified method"""
        print(f"\nPerforming feature selection using {method}...")

        if method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=min(k, X.shape[1]))
        else:  # f_classif
            selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))

        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()

        print(f"Selected {len(selected_features)} features out of {X.shape[1]}")
        print(f"Selected features: {selected_features[:10]}{'...' if len(selected_features) > 10 else ''}")

        self.feature_selector = selector
        return X_selected, selected_features

    def handle_class_imbalance(self, X, y, method='smote'):
        """Handle class imbalance using various techniques"""
        print(f"\nHandling class imbalance using {method}...")

        original_counts = pd.Series(y).value_counts()
        print(f"Original class distribution: {dict(original_counts)}")

        if method == 'smote':
            sampler = SMOTE(random_state=42)
        elif method == 'adasyn':
            sampler = ADASYN(random_state=42)
        elif method == 'borderline_smote':
            sampler = BorderlineSMOTE(random_state=42)
        elif method == 'smote_tomek':
            sampler = SMOTETomek(random_state=42)
        elif method == 'smote_enn':
            sampler = SMOTEENN(random_state=42)
        elif method == 'undersample':
            sampler = RandomUnderSampler(random_state=42)
        else:
            print("No balancing applied")
            return X, y

        try:
            X_balanced, y_balanced = sampler.fit_resample(X, y)
            balanced_counts = pd.Series(y_balanced).value_counts()
            print(f"Balanced class distribution: {dict(balanced_counts)}")

            self.balancing_method = method
            return X_balanced, y_balanced
        except Exception as e:
            print(f"Error in balancing: {e}")
            print("Returning original data")
            return X, y

    def create_train_val_test_split(self, X, y, test_size=0.2, val_size=0.2, random_state=42):
        """Create train/validation/test splits"""
        print(f"\nCreating train/validation/test splits...")
        print(f"Test size: {test_size}, Validation size: {val_size}")

        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Second split: separate train and validation from remaining data
        val_size_adjusted = val_size / (1 - test_size)  # Adjust val_size for remaining data
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
        )

        print(f"Train set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")

        # Print class distributions
        print(f"\nClass distributions:")
        print(f"Train: {dict(pd.Series(y_train).value_counts())}")
        print(f"Validation: {dict(pd.Series(y_val).value_counts())}")
        print(f"Test: {dict(pd.Series(y_test).value_counts())}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def scale_features(self, X_train, X_val, X_test, method='standard'):
        """Scale features using specified method"""
        print(f"\nScaling features using {method} scaler...")

        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            print("No scaling applied")
            return X_train, X_val, X_test

        # Fit on training data only
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        self.scaler = scaler
        print("Feature scaling completed")

        return X_train_scaled, X_val_scaled, X_test_scaled

    def visualize_class_distribution(self, y_original, y_balanced, dataset_name):
        """Visualize class distribution before and after balancing"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Original distribution
        original_counts = pd.Series(y_original).value_counts()
        axes[0].pie(original_counts.values, labels=['Non-Fraud', 'Fraud'], autopct='%1.2f%%')
        axes[0].set_title(f'Original Class Distribution - {dataset_name}')

        # Balanced distribution
        balanced_counts = pd.Series(y_balanced).value_counts()
        axes[1].pie(balanced_counts.values, labels=['Non-Fraud', 'Fraud'], autopct='%1.2f%%')
        axes[1].set_title(f'Balanced Class Distribution - {dataset_name}')

        plt.tight_layout()
        plt.savefig(f'plots/model_prep/class_distribution_{dataset_name.lower().replace("-", "_")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def save_prepared_data(self, X_train, X_val, X_test, y_train, y_val, y_test, 
                          feature_names, dataset_name, file_path):
        """Save prepared data for modeling"""
        print(f"\nSaving prepared data for {dataset_name}...")

        # Create dataset-specific directory
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        dataset_dir = f'data/processed/{base_name.lower()}'
        os.makedirs(dataset_dir, exist_ok=True)

        # Save feature data
        pd.DataFrame(X_train, columns=feature_names).to_csv(
            f'{dataset_dir}/X_train.csv', index=False)
        pd.DataFrame(X_val, columns=feature_names).to_csv(
            f'{dataset_dir}/X_val.csv', index=False)
        pd.DataFrame(X_test, columns=feature_names).to_csv(
            f'{dataset_dir}/X_test.csv', index=False)

        # Save target data
        pd.Series(y_train, name='target').to_csv(
            f'{dataset_dir}/y_train.csv', index=False)
        pd.Series(y_val, name='target').to_csv(
            f'{dataset_dir}/y_val.csv', index=False)
        pd.Series(y_test, name='target').to_csv(
            f'{dataset_dir}/y_test.csv', index=False)

        # Save feature names
        pd.Series(feature_names, name='feature').to_csv(
            f'{dataset_dir}/feature_names.csv', index=False)

        # Save metadata
        metadata = {
            'dataset_type': dataset_name,
            'n_features': len(feature_names),
            'train_samples': len(y_train),
            'val_samples': len(y_val),
            'test_samples': len(y_test),
            'balancing_method': self.balancing_method,
            'scaling_method': 'standard' if self.scaler else 'none'
        }

        pd.Series(metadata).to_csv(f'{dataset_dir}/metadata.csv')

        print(f"Data saved to: {dataset_dir}/")
        return dataset_dir

    def generate_preparation_report(self, dataset_name, dataset_type, original_shape, 
                                    final_shape, feature_names, y_original, y_balanced):
        """Generate comprehensive preparation report"""
        print(f"\n{'='*60}")
        print(f"MODEL PREPARATION REPORT - {dataset_name}")
        print(f"{'='*60}")

        print(f"Dataset: {dataset_name} ({dataset_type}")
        print(f"Original shape: {original_shape}")
        print(f"Final shape: {final_shape}")
        print(f"Features engineered: {final_shape[1] - (original_shape[1] - 1)}")  # -1 for target

        print(f"\nClass Distribution:")
        original_counts = pd.Series(y_original).value_counts()
        balanced_counts = pd.Series(y_balanced).value_counts()

        print(f"Original - Non-fraud: {original_counts[0]:,}, Fraud: {original_counts[1]:,}")
        print(f"Balanced - Non-fraud: {balanced_counts[0]:,}, Fraud: {balanced_counts[1]:,}")

        fraud_rate_original = (original_counts[1] / original_counts.sum()) * 100
        fraud_rate_balanced = (balanced_counts[1] / balanced_counts.sum()) * 100

        print(f"Fraud rate - Original: {fraud_rate_original:.4f}%, Balanced: {fraud_rate_balanced:.4f}%")

        print(f"\nFeature Engineering Summary:")
        print(f"Total features: {len(feature_names)}")
        print(f"Feature types: {', '.join(set([f.split('_')[0] for f in feature_names[:10]]))}")

        print(f"\nData Processing:")
        print(f"Balancing method: {self.balancing_method or 'None'}")
        print(f"Scaling method: {'Standard' if self.scaler else 'None'}")
        print(f"Feature selection: {'Applied' if self.feature_selector else 'None'}")

        print(f"\nReady for modeling: ✓")

def main():
    """Main execution function"""
    print("Credit Card Fraud Detection - Model Preparation Pipeline")
    print("=" * 70)

    # Initialize preparation class
    prep = ModelPreparation()

    # Dataset paths
    datasets = [
        'data/cleaned/creditcard_cleaned.csv',
        'data/cleaned/credit_card_fraud_2023_cleaned.csv'
    ]

    for dataset_path in datasets:
        if os.path.exists(dataset_path):
            print(f"\n\nProcessing: {dataset_path}")
            print("=" * 80)

            # Extract base name for unique identification
            base_name = os.path.splitext(os.path.basename(dataset_path))[0]

            # Load and identify dataset
            df, dataset_type = prep.load_and_identify_dataset(dataset_path)

            if df is not None:
                original_shape = df.shape

                # Identify target column
                target_col = prep.identify_target_column(df)

                if target_col is None:
                    print("Skipping dataset - no target column found")
                    continue

                # Prepare features based on dataset type
                if dataset_type == "MLG-ULB":
                    df, feature_columns = prep.prepare_features_mlg_ulb(df, target_col, base_name)
                else:
                    df, feature_columns = prep.prepare_features_2023(df, target_col, base_name)

                # Extract features and target
                X = df[feature_columns]
                y = df[target_col]

                print(f"\nFeature matrix shape: {X.shape}")
                print(f"Target distribution: {dict(y.value_counts())}")

                # Create train/val/test splits
                X_train, X_val, X_test, y_train, y_val, y_test = prep.create_train_val_test_split(X, y)

                # Handle class imbalance on training data only
                X_train_balanced, y_train_balanced = prep.handle_class_imbalance(
                    X_train, y_train, method='smote')

                # Scale features
                X_train_scaled, X_val_scaled, X_test_scaled = prep.scale_features(
                    X_train_balanced, X_val, X_test, method='standard')

                # Visualize class distribution
                prep.visualize_class_distribution(y_train, y_train_balanced, base_name)

                # Save prepared data
                output_dir = prep.save_prepared_data(
                    X_train_scaled, X_val_scaled, X_test_scaled,
                    y_train_balanced, y_val, y_test,
                    feature_columns, base_name, dataset_path
                )

                # Generate report
                prep.generate_preparation_report(
                    base_name, dataset_type, original_shape, X_train_scaled.shape,
                    feature_columns, y_train, y_train_balanced
                )

        else:
            print(f"Dataset not found: {dataset_path}")

    print(f"\n\nModel preparation completed!")
    print("Next steps:")
    print("1. Review the prepared data in 'data/processed/' directories")
    print("2. Check class distribution plots in 'plots/model_prep/'")
    print("3. Proceed with model training using the prepared datasets")

if __name__ == "__main__":
    main()
