"""
Credit Card Fraud Detection - Feature Importance Extraction
This script extracts feature importances from trained Random Forest models
and exports them to CSV files for use in Tableau dashboards and reports.

Features:
- Automatic model and dataset detection
- Feature importance extraction from Random Forest models
- CSV export for Tableau integration
- Summary statistics and insights
"""

import pandas as pd
import numpy as np
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

class FeatureImportanceExtractor:
    def __init__(self):
        self.feature_importances = {}

        # Create output directory
        os.makedirs('results/feature_importance', exist_ok=True)

    def load_model_and_features(self, dataset_name):
        """Load trained Random Forest model and feature names"""
        try:
            # Load Random Forest model
            model_path = f'models/baseline/{dataset_name.lower()}/random_forest.pkl'

            if not os.path.exists(model_path):
                print(f"Model not found: {model_path}")
                return None, None

            with open(model_path, 'rb') as f:
                rf_model = pickle.load(f)

            # Load feature names
            features_path = f'data/processed/{dataset_name.lower()}/feature_names.csv'

            if not os.path.exists(features_path):
                print(f"Feature names not found: {features_path}")
                return None, None

            feature_names = pd.read_csv(features_path)['feature'].tolist()

            print(f"Loaded Random Forest model and {len(feature_names)} features for {dataset_name}")

            return rf_model, feature_names

        except Exception as e:
            print(f"Error loading model/features for {dataset_name}: {e}")
            return None, None

    def extract_feature_importance(self, model, feature_names, dataset_name):
        """Extract and process feature importances"""
        try:
            # Get feature importances from Random Forest
            importances = model.feature_importances_

            # Create DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances,
                'importance_percent': importances * 100
            })

            # Sort by importance (descending)
            importance_df = importance_df.sort_values('importance', ascending=False)

            # Add rank
            importance_df['rank'] = range(1, len(importance_df) + 1)

            # Add cumulative importance
            importance_df['cumulative_importance'] = importance_df['importance'].cumsum()
            importance_df['cumulative_percent'] = importance_df['cumulative_importance'] * 100

            # Store results
            self.feature_importances[dataset_name] = importance_df

            print(f"Extracted feature importances for {dataset_name}")
            print(f"Top 5 most important features:")
            for i, row in importance_df.head().iterrows():
                print(f"  {row['rank']}. {row['feature']}: {row['importance_percent']:.2f}%")

            return importance_df

        except Exception as e:
            print(f"Error extracting feature importance for {dataset_name}: {e}")
            return None

    def analyze_feature_importance(self, importance_df, dataset_name):
        """Analyze feature importance patterns"""
        print(f"\n{'='*60}")
        print(f"FEATURE IMPORTANCE ANALYSIS - {dataset_name}")
        print(f"{'='*60}")

        total_features = len(importance_df)

        # Top features analysis
        top_10_importance = importance_df.head(10)['importance'].sum()
        top_20_importance = importance_df.head(20)['importance'].sum()

        print(f"\nFeature Concentration Analysis:")
        print(f"  Total features: {total_features}")
        print(f"  Top 10 features contribute: {top_10_importance*100:.1f}% of total importance")
        print(f"  Top 20 features contribute: {top_20_importance*100:.1f}% of total importance")

        # Find features contributing to 80% of importance
        features_80_percent = len(importance_df[importance_df['cumulative_percent'] <= 80])
        print(f"  Features needed for 80% importance: {features_80_percent} ({features_80_percent/total_features*100:.1f}%)")

        # Low importance features
        low_importance_threshold = 0.001  # 0.1%
        low_importance_count = len(importance_df[importance_df['importance'] < low_importance_threshold])
        print(f"  Features with <0.1% importance: {low_importance_count} ({low_importance_count/total_features*100:.1f}%)")

        # Feature categories (if applicable)
        print(f"\nTop 10 Most Important Features:")
        print(f"{'Rank':<5} {'Feature':<30} {'Importance':<12} {'Cumulative':<12}")
        print("-" * 60)
        for i, row in importance_df.head(10).iterrows():
            print(f"{row['rank']:<5} {row['feature']:<30} {row['importance_percent']:<11.2f}% {row['cumulative_percent']:<11.1f}%")

    def export_to_csv(self, importance_df, dataset_name):
        """Export feature importances to CSV for Tableau"""
        try:
            # Full feature importance data
            full_csv_path = f'results/feature_importance/feature_importance_{dataset_name.lower()}.csv'
            importance_df.to_csv(full_csv_path, index=False)
            print(f"\nExported full feature importance: {full_csv_path}")

            # Top 20 features for focused analysis
            top_20_csv_path = f'results/feature_importance/top_20_features_{dataset_name.lower()}.csv'
            importance_df.head(20).to_csv(top_20_csv_path, index=False)
            print(f"Exported top 20 features: {top_20_csv_path}")

            # Summary statistics
            summary_data = {
                'dataset': [dataset_name],
                'total_features': [len(importance_df)],
                'top_feature': [importance_df.iloc[0]['feature']],
                'top_feature_importance': [importance_df.iloc[0]['importance_percent']],
                'top_10_cumulative': [importance_df.head(10)['importance'].sum() * 100],
                'features_for_80_percent': [len(importance_df[importance_df['cumulative_percent'] <= 80])],
                'low_importance_features': [len(importance_df[importance_df['importance'] < 0.001])]
            }

            summary_df = pd.DataFrame(summary_data)
            summary_csv_path = f'results/feature_importance/summary_{dataset_name.lower()}.csv'
            summary_df.to_csv(summary_csv_path, index=False)
            print(f"Exported summary statistics: {summary_csv_path}")

            return full_csv_path, top_20_csv_path, summary_csv_path

        except Exception as e:
            print(f"Error exporting CSV for {dataset_name}: {e}")
            return None, None, None

    def create_combined_analysis(self):
        """Create combined analysis across all datasets"""
        if len(self.feature_importances) < 2:
            print("\nNeed at least 2 datasets for combined analysis")
            return

        print(f"\n{'='*60}")
        print("COMBINED FEATURE IMPORTANCE ANALYSIS")
        print(f"{'='*60}")

        # Get common features across datasets
        all_features = set()
        dataset_features = {}

        for dataset_name, importance_df in self.feature_importances.items():
            features = set(importance_df['feature'].tolist())
            all_features.update(features)
            dataset_features[dataset_name] = features

        # Find common features
        common_features = set.intersection(*dataset_features.values())
        print(f"\nFeature Overlap Analysis:")
        print(f"  Total unique features across datasets: {len(all_features)}")
        print(f"  Common features across all datasets: {len(common_features)}")

        # Compare top features across datasets
        print(f"\nTop 10 Features Comparison:")
        comparison_data = []

        for dataset_name, importance_df in self.feature_importances.items():
            top_10 = importance_df.head(10)
            for i, row in top_10.iterrows():
                comparison_data.append({
                    'dataset': dataset_name,
                    'feature': row['feature'],
                    'rank': row['rank'],
                    'importance_percent': row['importance_percent']
                })

        comparison_df = pd.DataFrame(comparison_data)

        # Export combined analysis
        combined_csv_path = 'results/feature_importance/combined_feature_analysis.csv'
        comparison_df.to_csv(combined_csv_path, index=False)
        print(f"\nExported combined analysis: {combined_csv_path}")

        # Show top features that appear in multiple datasets
        feature_counts = comparison_df['feature'].value_counts()
        common_top_features = feature_counts[feature_counts > 1]

        if len(common_top_features) > 0:
            print(f"\nFeatures appearing in top 10 of multiple datasets:")
            for feature, count in common_top_features.items():
                print(f"  {feature}: appears in {count} datasets")
                # Show ranks across datasets
                feature_ranks = comparison_df[comparison_df['feature'] == feature][['dataset', 'rank', 'importance_percent']]
                for _, row in feature_ranks.iterrows():
                    print(f"    {row['dataset']}: Rank {row['rank']} ({row['importance_percent']:.2f}%)")

def main():
    """Main execution function"""
    print("Credit Card Fraud Detection - Feature Importance Extraction")
    print("=" * 70)

    # Initialize extractor
    extractor = FeatureImportanceExtractor()

    # Find available models
    models_dir = 'models/baseline'

    if not os.path.exists(models_dir):
        print(f"Error: Models directory not found: {models_dir}")
        print("Please run the baseline model training script first.")
        return

    # Get all dataset directories with Random Forest models
    dataset_dirs = []
    for d in os.listdir(models_dir):
        model_path = os.path.join(models_dir, d, 'random_forest.pkl')
        if os.path.exists(model_path):
            dataset_dirs.append(d)

    if not dataset_dirs:
        print("No Random Forest models found. Please train models first.")
        return

    print(f"Found Random Forest models for {len(dataset_dirs)} datasets: {dataset_dirs}")

    # Process each dataset
    for dataset_name in dataset_dirs:
        print(f"\n\n{'='*80}")
        print(f"PROCESSING DATASET: {dataset_name.upper()}")
        print(f"{'='*80}")

        # Load model and features
        rf_model, feature_names = extractor.load_model_and_features(dataset_name)

        if rf_model is None or feature_names is None:
            print(f"Skipping {dataset_name} - could not load model or features")
            continue

        # Extract feature importance
        importance_df = extractor.extract_feature_importance(rf_model, feature_names, dataset_name)

        if importance_df is None:
            print(f"Skipping {dataset_name} - could not extract feature importance")
            continue

        # Analyze feature importance
        extractor.analyze_feature_importance(importance_df, dataset_name)

        # Export to CSV
        full_csv, top_20_csv, summary_csv = extractor.export_to_csv(importance_df, dataset_name)

    # Create combined analysis if multiple datasets
    if len(extractor.feature_importances) > 1:
        extractor.create_combined_analysis()

    print(f"\n\n{'='*80}")
    print("FEATURE IMPORTANCE EXTRACTION COMPLETED")
    print(f"{'='*80}")
    print("\nFiles created for Tableau:")
    print("  - results/feature_importance/feature_importance_<dataset>.csv (full data)")
    print("  - results/feature_importance/top_20_features_<dataset>.csv (top features)")
    print("  - results/feature_importance/summary_<dataset>.csv (summary stats)")
    if len(extractor.feature_importances) > 1:
        print("  - results/feature_importance/combined_feature_analysis.csv (comparison)")

    print("\nReady for Tableau dashboard creation!")
    print("\nNext steps:")
    print("  1. Load CSV files into Tableau")
    print("  2. Create feature importance visualizations")
    print("  3. Build interactive dashboard")
    print("  4. Prepare project report and presentation")

if __name__ == "__main__":
    main()
