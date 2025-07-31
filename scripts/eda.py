
"""
Credit Card Fraud Detection - Exploratory Data Analysis (EDA) Script
SENG8081-S25-Team6

This script performs comprehensive EDA on cleaned credit card fraud detection datasets.
Handles both MLG-ULB and 2023 datasets with automatic detection and appropriate analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy import stats
import os

warnings.filterwarnings('ignore')

class FraudDetectionEDA:
    def __init__(self):
        # Set plotting style
        plt.style.use('default')
        sns.set_palette("husl")

        # Create output directory for plots
        os.makedirs('plots', exist_ok=True)

    def load_and_identify_dataset(self, file_path):
        """Load dataset and identify its type"""
        try:
            df = pd.read_csv(file_path)
            print(f"Dataset loaded successfully: {df.shape}")

            # Identify dataset type based on columns
            if any(col.startswith('V') for col in df.columns) and len([col for col in df.columns if col.startswith('V')]) > 20:
                dataset_type = "MLG-ULB"
                print("Dataset identified as: MLG-ULB (PCA-transformed features)")
            else:
                dataset_type = "2023"
                print("Dataset identified as: 2023 Credit Card Fraud Dataset")

            return df, dataset_type
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None, None

    def basic_info(self, df, dataset_name):
        """Display basic dataset information"""
        print(f"\n{'='*60}")
        print(f"BASIC INFORMATION - {dataset_name}")
        print(f"{'='*60}")

        print(f"Dataset Shape: {df.shape}")
        print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        print(f"\nColumn Information:")
        print(f"Total Columns: {len(df.columns)}")
        print(f"Numerical Columns: {len(df.select_dtypes(include=[np.number]).columns)}")
        print(f"Categorical Columns: {len(df.select_dtypes(include=['object']).columns)}")

        print(f"\nMissing Values:")
        missing = df.isnull().sum()
        if missing.sum() == 0:
            print("No missing values found!")
        else:
            print(missing[missing > 0])

        print(f"\nDuplicate Rows: {df.duplicated().sum()}")

        # Identify target column
        target_cols = ['Class', 'class', 'fraud', 'Fraud', 'is_fraud', 'target']
        target_col = None
        for col in target_cols:
            if col in df.columns:
                target_col = col
                break

        if target_col:
            print(f"\nTarget Variable: '{target_col}'")
            print(f"Target Distribution:")
            print(df[target_col].value_counts())
            fraud_rate = (df[target_col].sum() / len(df)) * 100
            print(f"Fraud Rate: {fraud_rate:.4f}%")

        return target_col

    def plot_class_distribution(self, df, target_col, dataset_name):
        """Plot class distribution"""
        plt.figure(figsize=(10, 6))

        # Count plot
        plt.subplot(1, 2, 1)
        counts = df[target_col].value_counts()
        plt.pie(counts.values, labels=['Non-Fraud', 'Fraud'], autopct='%1.2f%%', startangle=90)
        plt.title(f'Class Distribution - {dataset_name}')

        # Bar plot
        plt.subplot(1, 2, 2)
        sns.countplot(data=df, x=target_col)
        plt.title(f'Class Counts - {dataset_name}')
        plt.xlabel('Class (0=Non-Fraud, 1=Fraud)')
        plt.ylabel('Count')

        # Add count labels on bars
        ax = plt.gca()
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', 
                       (p.get_x() + p.get_width()/2., p.get_height()),
                       ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(f'plots/class_distribution_{dataset_name.lower().replace("-", "_")}.png', dpi=300, bbox_inches='tight')
        plt.close()

    def analyze_amount_feature(self, df, target_col, dataset_name):
        """Analyze Amount feature if present"""
        if 'Amount' not in df.columns:
            print("Amount column not found in dataset")
            return

        print(f"\n{'='*60}")
        print(f"AMOUNT ANALYSIS - {dataset_name}")
        print(f"{'='*60}")

        # Basic statistics
        print("Amount Statistics:")
        print(df['Amount'].describe())

        # Statistics by class
        fraud_amounts = df[df[target_col] == 1]['Amount']
        normal_amounts = df[df[target_col] == 0]['Amount']

        print(f"\nAmount Statistics by Class:")
        print(f"Fraud - Mean: ${fraud_amounts.mean():.2f}, Median: ${fraud_amounts.median():.2f}")
        print(f"Normal - Mean: ${normal_amounts.mean():.2f}, Median: ${normal_amounts.median():.2f}")

        # Statistical test
        stat, p_value = stats.mannwhitneyu(fraud_amounts, normal_amounts, alternative='two-sided')
        print(f"\nMann-Whitney U test p-value: {p_value:.2e}")
        print("Significant difference in amounts" if p_value < 0.05 else "No significant difference in amounts")

        # Visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Distribution of amounts
        axes[0, 0].hist(df['Amount'], bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title(f'Amount Distribution - {dataset_name}')
        axes[0, 0].set_xlabel('Amount ($)')
        axes[0, 0].set_ylabel('Frequency')

        # Log-scale distribution
        axes[0, 1].hist(np.log1p(df['Amount']), bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title(f'Amount Distribution (Log Scale) - {dataset_name}')
        axes[0, 1].set_xlabel('Log(Amount + 1)')
        axes[0, 1].set_ylabel('Frequency')

        # Box plot by class
        sns.boxplot(data=df, x=target_col, y='Amount', ax=axes[1, 0])
        axes[1, 0].set_title(f'Amount by Class - {dataset_name}')
        axes[1, 0].set_xlabel('Class (0=Non-Fraud, 1=Fraud)')
        axes[1, 0].set_ylabel('Amount ($)')

        # Violin plot by class
        sns.violinplot(data=df, x=target_col, y='Amount', ax=axes[1, 1])
        axes[1, 1].set_title(f'Amount Distribution by Class - {dataset_name}')
        axes[1, 1].set_xlabel('Class (0=Non-Fraud, 1=Fraud)')
        axes[1, 1].set_ylabel('Amount ($)')

        plt.tight_layout()
        plt.savefig(f'plots/amount_analysis_{dataset_name.lower().replace("-", "_")}.png', dpi=300, bbox_inches='tight')
        plt.close()

    def analyze_time_feature(self, df, target_col, dataset_name):
        """Analyze Time feature if present"""
        if 'Time' not in df.columns:
            print("Time column not found in dataset")
            return

        print(f"\n{'='*60}")
        print(f"TIME ANALYSIS - {dataset_name}")
        print(f"{'='*60}")

        # Convert time to hours
        df['Hour'] = ((df['Time'] / 3600) % 24).astype(int)
        
        # Time statistics
        print("Time Statistics:")
        print(df['Time'].describe())

        # Fraud by hour analysis
        fraud_by_hour = df.groupby('Hour')[target_col].agg(['count', 'sum', 'mean']).reset_index()
        fraud_by_hour.columns = ['Hour', 'Total_Transactions', 'Fraud_Count', 'Fraud_Rate']

        print(f"\nFraud patterns by hour:")
        print(fraud_by_hour.head(10))

        # Visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Time distribution
        axes[0, 0].hist(df['Time'], bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title(f'Time Distribution - {dataset_name}')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Frequency')

        # Hourly transaction volume
        axes[0, 1].bar(fraud_by_hour['Hour'], fraud_by_hour['Total_Transactions'])
        axes[0, 1].set_title(f'Transactions by Hour - {dataset_name}')
        axes[0, 1].set_xlabel('Hour of Day')
        axes[0, 1].set_ylabel('Number of Transactions')

        # Fraud count by hour
        axes[1, 0].bar(fraud_by_hour['Hour'], fraud_by_hour['Fraud_Count'], color='red', alpha=0.7)
        axes[1, 0].set_title(f'Fraud Count by Hour - {dataset_name}')
        axes[1, 0].set_xlabel('Hour of Day')
        axes[1, 0].set_ylabel('Fraud Count')

        # Fraud rate by hour
        axes[1, 1].plot(fraud_by_hour['Hour'], fraud_by_hour['Fraud_Rate'], marker='o', color='orange')
        axes[1, 1].set_title(f'Fraud Rate by Hour - {dataset_name}')
        axes[1, 1].set_xlabel('Hour of Day')
        axes[1, 1].set_ylabel('Fraud Rate')

        plt.tight_layout()
        plt.savefig(f'plots/time_analysis_{dataset_name.lower().replace("-", "_")}.png', dpi=300, bbox_inches='tight')
        plt.close()

    def analyze_pca_features(self, df, target_col, dataset_name):
        """Analyze PCA features (V1-V28) for MLG-ULB dataset"""
        v_columns = [col for col in df.columns if col.startswith('V')]

        if len(v_columns) == 0:
            print("No PCA features (V columns) found in dataset")
            return

        print(f"\n{'='*60}")
        print(f"PCA FEATURES ANALYSIS - {dataset_name}")
        print(f"{'='*60}")

        print(f"Found {len(v_columns)} PCA features: {v_columns[:5]}...{v_columns[-5:]}")

        # Calculate feature importance based on correlation with target
        correlations = df[v_columns + [target_col]].corr()[target_col].abs().sort_values(ascending=False)
        correlations = correlations.drop(target_col)

        print(f"\nTop 10 features by correlation with fraud:")
        print(correlations.head(10))

        # Visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Feature correlations with target
        top_features = correlations.head(15)
        axes[0, 0].barh(range(len(top_features)), top_features.values)
        axes[0, 0].set_yticks(range(len(top_features)))
        axes[0, 0].set_yticklabels(top_features.index)
        axes[0, 0].set_title(f'Top 15 Features by Correlation with Fraud - {dataset_name}')
        axes[0, 0].set_xlabel('Absolute Correlation')

        # Distribution of top correlated feature
        top_feature = correlations.index[0]
        fraud_values = df[df[target_col] == 1][top_feature]
        normal_values = df[df[target_col] == 0][top_feature]

        axes[0, 1].hist(normal_values, bins=50, alpha=0.7, label='Normal', density=True)
        axes[0, 1].hist(fraud_values, bins=50, alpha=0.7, label='Fraud', density=True)
        axes[0, 1].set_title(f'Distribution of {top_feature} - {dataset_name}')
        axes[0, 1].set_xlabel(top_feature)
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].legend()

        # Correlation heatmap of top features
        top_10_features = correlations.head(10).index.tolist()
        corr_matrix = df[top_10_features].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 0])
        axes[1, 0].set_title(f'Correlation Heatmap - Top 10 Features - {dataset_name}')

        # Feature statistics comparison
        feature_stats = []
        for feature in top_10_features:
            fraud_mean = df[df[target_col] == 1][feature].mean()
            normal_mean = df[df[target_col] == 0][feature].mean()
            feature_stats.append(abs(fraud_mean - normal_mean))

        axes[1, 1].bar(range(len(feature_stats)), feature_stats)
        axes[1, 1].set_xticks(range(len(feature_stats)))
        axes[1, 1].set_xticklabels(top_10_features, rotation=45)
        axes[1, 1].set_title(f'Mean Difference (|Fraud - Normal|) - {dataset_name}')
        axes[1, 1].set_ylabel('Absolute Mean Difference')

        plt.tight_layout()
        plt.savefig(f'plots/pca_features_analysis_{dataset_name.lower().replace("-", "_")}.png', dpi=300, bbox_inches='tight')
        plt.close()

    def correlation_analysis(self, df, target_col, dataset_name):
        """Perform correlation analysis"""
        print(f"\n{'='*60}")
        print(f"CORRELATION ANALYSIS - {dataset_name}")
        print(f"{'='*60}")

        # Select numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Remove target from numerical columns for correlation matrix
        if target_col in numerical_cols:
            numerical_cols.remove(target_col)

        # Limit to reasonable number of features for visualization
        if len(numerical_cols) > 20:
            # Use top correlated features with target
            correlations_with_target = df[numerical_cols + [target_col]].corr()[target_col].abs().sort_values(ascending=False)
            top_features = correlations_with_target.head(20).index.tolist()
            if target_col in top_features:
                top_features.remove(target_col)
            numerical_cols = top_features[:19]  # Keep 19 + target = 20 total

        # Create correlation matrix
        corr_matrix = df[numerical_cols + [target_col]].corr()

        # Plot correlation heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title(f'Feature Correlation Matrix - {dataset_name}')
        plt.tight_layout()
        plt.savefig(f'plots/correlation_matrix_{dataset_name.lower().replace("-", "_")}.png', dpi=300, bbox_inches='tight')
        plt.close()

        # High correlation pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:  # High correlation threshold
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))

        if high_corr_pairs:
            print(f"\nHigh correlation pairs (|r| > 0.7):")
            for pair in high_corr_pairs:
                print(f"{pair[0]} - {pair[1]}: {pair[2]:.3f}")
        else:
            print("\nNo high correlation pairs found (|r| > 0.7)")

    def generate_summary_report(self, df, target_col, dataset_name):
        """Generate summary report"""
        print(f"\n{'='*60}")
        print(f"SUMMARY REPORT - {dataset_name}")
        print(f"{'='*60}")

        # Dataset overview
        fraud_count = df[target_col].sum()
        total_count = len(df)
        fraud_rate = (fraud_count / total_count) * 100

        print(f"Dataset: {dataset_name}")
        print(f"Total Transactions: {total_count:,}")
        print(f"Fraudulent Transactions: {fraud_count:,}")
        print(f"Fraud Rate: {fraud_rate:.4f}%")
        print(f"Class Imbalance Ratio: {(total_count - fraud_count) / fraud_count:.1f}:1")

        # Feature summary
        numerical_features = len(df.select_dtypes(include=[np.number]).columns)
        categorical_features = len(df.select_dtypes(include=['object']).columns)

        print(f"\nFeature Summary:")
        print(f"Numerical Features: {numerical_features}")
        print(f"Categorical Features: {categorical_features}")
        print(f"Total Features: {len(df.columns)}")

        # Data quality
        missing_values = df.isnull().sum().sum()
        duplicate_rows = df.duplicated().sum()

        print(f"\nData Quality:")
        print(f"Missing Values: {missing_values}")
        print(f"Duplicate Rows: {duplicate_rows}")
        print(f"Data Quality Score: {((total_count - missing_values - duplicate_rows) / total_count) * 100:.2f}%")

        # Key insights
        print(f"\nKey Insights:")
        if 'Amount' in df.columns:
            fraud_avg_amount = df[df[target_col] == 1]['Amount'].mean()
            normal_avg_amount = df[df[target_col] == 0]['Amount'].mean()
            print(f"- Average fraud amount: ${fraud_avg_amount:.2f}")
            print(f"- Average normal amount: ${normal_avg_amount:.2f}")
            print(f"- Amount difference: {abs(fraud_avg_amount - normal_avg_amount):.2f}")

        # Feature importance (top 5)
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numerical_cols:
            numerical_cols.remove(target_col)

        if len(numerical_cols) > 0:
            correlations = df[numerical_cols + [target_col]].corr()[target_col].abs().sort_values(ascending=False)
            correlations = correlations.drop(target_col)
            print(f"\n- Top 5 most correlated features with fraud:")
            for i, (feature, corr) in enumerate(correlations.head(5).items()):
                print(f"  {i+1}. {feature}: {corr:.4f}")

def main():
    """Main execution function"""
    print("Credit Card Fraud Detection - Exploratory Data Analysis")
    print("=" * 60)

    # Initialize EDA class
    eda = FraudDetectionEDA()

    # Dataset paths (update these according to your file structure)
    datasets = [
        'data/cleaned/creditcard_cleaned.csv',
        'data/cleaned/credit_card_fraud_2023_cleaned.csv'
    ]

    for dataset_path in datasets:
        if os.path.exists(dataset_path):
            print(f"\n\nProcessing: {dataset_path}")
            print("=" * 80)

            # Load and identify dataset
            df, dataset_type = eda.load_and_identify_dataset(dataset_path)

            if df is not None:
                # Basic information
                target_col = eda.basic_info(df, dataset_type)

                if target_col:
                    # Perform EDA
                    eda.plot_class_distribution(df, target_col, dataset_type)
                    eda.analyze_amount_feature(df, target_col, dataset_type)
                    eda.analyze_time_feature(df, target_col, dataset_type)
                    eda.analyze_pca_features(df, target_col, dataset_type)
                    eda.correlation_analysis(df, target_col, dataset_type)
                    eda.generate_summary_report(df, target_col, dataset_type)
                else:
                    print("Target column not found. Please check your dataset.")
        else:
            print(f"Dataset not found: {dataset_path}")

    print(f"\n\nEDA completed! Check the 'plots' directory for saved visualizations.")

if __name__ == "__main__":
    main()
