"""
Credit Card Fraud Detection - Test Set Predictions
This script loads trained models and runs predictions on test set data,
creating detailed prediction results for visualization in Tableau dashboards.

Features:
- Loads trained Random Forest models
- Runs predictions on test set data
- Calculates prediction probabilities and confidence scores
- Exports detailed results for Tableau visualization
- Creates prediction summary statistics
- Handles multiple datasets automatically
"""

import pandas as pd
import numpy as np
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)

class FraudPredictor:
    def __init__(self):
        self.prediction_results = {}

        # Create output directories
        os.makedirs('results/predictions', exist_ok=True)

    def load_model_and_test_data(self, dataset_name):
        """Load trained model and test data"""
        try:
            # Load Random Forest model
            model_path = f'models/baseline/{dataset_name.lower()}/random_forest.pkl'

            if not os.path.exists(model_path):
                print(f"Model not found: {model_path}")
                return None, None, None, None

            with open(model_path, 'rb') as f:
                rf_model = pickle.load(f)

            # Load test data
            data_path = f'data/processed/{dataset_name.lower()}'
            X_test = pd.read_csv(f'{data_path}/X_test.csv')
            y_test = pd.read_csv(f'{data_path}/y_test.csv')['target']

            # Load feature names for reference
            feature_names = pd.read_csv(f'{data_path}/feature_names.csv')['feature'].tolist()

            print(f"Loaded model and test data for {dataset_name}")
            print(f"  Test samples: {len(X_test)}")
            print(f"  Features: {len(feature_names)}")
            print(f"  Actual fraud cases: {y_test.sum()} ({y_test.mean()*100:.2f}%)")

            return rf_model, X_test, y_test, feature_names

        except Exception as e:
            print(f"Error loading model/data for {dataset_name}: {e}")
            return None, None, None, None

    def run_predictions(self, model, X_test, y_test, dataset_name):
        """Run predictions and calculate detailed results"""
        try:
            print(f"\nRunning predictions for {dataset_name}...")

            # Get predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)

            # Extract probabilities for each class
            prob_non_fraud = y_pred_proba[:, 0]  # Probability of class 0 (non-fraud)
            prob_fraud = y_pred_proba[:, 1]      # Probability of class 1 (fraud)

            # Calculate confidence scores
            confidence = np.max(y_pred_proba, axis=1)  # Highest probability

            # Create detailed results DataFrame
            results_df = pd.DataFrame({
                'transaction_id': range(1, len(X_test) + 1),
                'actual_label': y_test.values,
                'predicted_label': y_pred,
                'prob_non_fraud': prob_non_fraud,
                'prob_fraud': prob_fraud,
                'confidence': confidence,
                'prediction_correct': (y_test.values == y_pred).astype(int)
            })

            # Add prediction categories
            results_df['prediction_category'] = results_df.apply(
                lambda row: self.categorize_prediction(row), axis=1
            )

            # Add confidence levels
            results_df['confidence_level'] = pd.cut(
                results_df['confidence'], 
                bins=[0, 0.6, 0.8, 0.9, 1.0], 
                labels=['Low', 'Medium', 'High', 'Very High']
            )

            # Add fraud probability bins for analysis
            results_df['fraud_prob_bin'] = pd.cut(
                results_df['prob_fraud'],
                bins=[0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                labels=['0-10%', '10-30%', '30-50%', '50-70%', '70-90%', '90-100%']
            )

            # Store results
            self.prediction_results[dataset_name] = results_df

            # Calculate performance metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, prob_fraud)

            print(f"Prediction Results:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            print(f"  ROC-AUC: {roc_auc:.4f}")

            return results_df

        except Exception as e:
            print(f"Error running predictions for {dataset_name}: {e}")
            return None

    def categorize_prediction(self, row):
        """Categorize each prediction for analysis"""
        actual = row['actual_label']
        predicted = row['predicted_label']

        if actual == 1 and predicted == 1:
            return 'True Positive (Fraud Caught)'
        elif actual == 0 and predicted == 0:
            return 'True Negative (Correct Non-Fraud)'
        elif actual == 0 and predicted == 1:
            return 'False Positive (False Alarm)'
        else:  # actual == 1 and predicted == 0
            return 'False Negative (Fraud Missed)'

    def analyze_predictions(self, results_df, dataset_name):
        """Analyze prediction patterns and create insights"""
        print(f"\n{'='*60}")
        print(f"PREDICTION ANALYSIS - {dataset_name}")
        print(f"{'='*60}")

        total_transactions = len(results_df)

        # Prediction category breakdown
        category_counts = results_df['prediction_category'].value_counts()
        print(f"\nPrediction Breakdown:")
        for category, count in category_counts.items():
            percentage = (count / total_transactions) * 100
            print(f"  {category}: {count} ({percentage:.2f}%)")

        # Confidence level analysis
        print(f"\nConfidence Level Distribution:")
        confidence_counts = results_df['confidence_level'].value_counts()
        for level, count in confidence_counts.items():
            percentage = (count / total_transactions) * 100
            print(f"  {level} Confidence: {count} ({percentage:.2f}%)")

        # Fraud probability distribution
        print(f"\nFraud Probability Distribution:")
        prob_counts = results_df['fraud_prob_bin'].value_counts().sort_index()
        for prob_bin, count in prob_counts.items():
            percentage = (count / total_transactions) * 100
            print(f"  {prob_bin}: {count} ({percentage:.2f}%)")

        # High-risk transactions (fraud probability > 70%)
        high_risk = results_df[results_df['prob_fraud'] > 0.7]
        print(f"\nHigh-Risk Transactions (>70% fraud probability):")
        print(f"  Count: {len(high_risk)} ({len(high_risk)/total_transactions*100:.2f}%)")
        if len(high_risk) > 0:
            actual_fraud_in_high_risk = high_risk['actual_label'].sum()
            print(f"  Actual fraud in high-risk: {actual_fraud_in_high_risk} ({actual_fraud_in_high_risk/len(high_risk)*100:.1f}%)")

        # Low confidence predictions (might need review)
        low_confidence = results_df[results_df['confidence'] < 0.6]
        print(f"\nLow Confidence Predictions (<60% confidence):")
        print(f"  Count: {len(low_confidence)} ({len(low_confidence)/total_transactions*100:.2f}%)")
        if len(low_confidence) > 0:
            low_conf_accuracy = low_confidence['prediction_correct'].mean()
            print(f"  Accuracy of low confidence predictions: {low_conf_accuracy:.3f}")

    def create_fraud_cases_detail(self, results_df, X_test, feature_names, dataset_name):
        """Create detailed view of predicted fraud cases"""
        # Get predicted fraud cases
        fraud_predictions = results_df[results_df['predicted_label'] == 1].copy()

        if len(fraud_predictions) == 0:
            print(f"No fraud cases predicted for {dataset_name}")
            return None

        print(f"\nCreating detailed fraud cases for {dataset_name}...")
        print(f"Predicted fraud cases: {len(fraud_predictions)}")

        # Add original feature data for fraud cases
        fraud_indices = fraud_predictions.index
        fraud_features = X_test.iloc[fraud_indices].copy()

        # Combine prediction results with features
        fraud_detail = pd.concat([
            fraud_predictions.reset_index(drop=True),
            fraud_features.reset_index(drop=True)
        ], axis=1)

        # Sort by fraud probability (highest first)
        fraud_detail = fraud_detail.sort_values('prob_fraud', ascending=False)

        return fraud_detail

    def export_results(self, results_df, dataset_name, fraud_detail=None):
        """Export prediction results to CSV files for Tableau"""
        try:
            # Main prediction results
            main_csv_path = f'results/predictions/predictions_{dataset_name.lower()}.csv'
            results_df.to_csv(main_csv_path, index=False)
            print(f"\nExported main predictions: {main_csv_path}")

            # Summary statistics
            summary_data = {
                'dataset': [dataset_name],
                'total_transactions': [len(results_df)],
                'predicted_fraud': [results_df['predicted_label'].sum()],
                'actual_fraud': [results_df['actual_label'].sum()],
                'accuracy': [results_df['prediction_correct'].mean()],
                'high_confidence_predictions': [len(results_df[results_df['confidence'] > 0.8])],
                'high_risk_transactions': [len(results_df[results_df['prob_fraud'] > 0.7])],
                'avg_fraud_probability': [results_df['prob_fraud'].mean()],
                'max_fraud_probability': [results_df['prob_fraud'].max()]
            }

            summary_df = pd.DataFrame(summary_data)
            summary_csv_path = f'results/predictions/prediction_summary_{dataset_name.lower()}.csv'
            summary_df.to_csv(summary_csv_path, index=False)
            print(f"Exported prediction summary: {summary_csv_path}")

            # Fraud cases detail (if available)
            if fraud_detail is not None and len(fraud_detail) > 0:
                fraud_csv_path = f'results/predictions/fraud_cases_{dataset_name.lower()}.csv'
                fraud_detail.to_csv(fraud_csv_path, index=False)
                print(f"Exported fraud cases detail: {fraud_csv_path}")

            # Confusion matrix data
            cm_data = []
            categories = results_df['prediction_category'].value_counts()
            for category, count in categories.items():
                cm_data.append({
                    'dataset': dataset_name,
                    'category': category,
                    'count': count,
                    'percentage': (count / len(results_df)) * 100
                })

            cm_df = pd.DataFrame(cm_data)
            cm_csv_path = f'results/predictions/confusion_matrix_{dataset_name.lower()}.csv'
            cm_df.to_csv(cm_csv_path, index=False)
            print(f"Exported confusion matrix data: {cm_csv_path}")

            return main_csv_path, summary_csv_path, fraud_csv_path if fraud_detail is not None else None, cm_csv_path

        except Exception as e:
            print(f"Error exporting results for {dataset_name}: {e}")
            return None, None, None, None

    def create_combined_predictions(self):
        """Create combined analysis across all datasets"""
        if len(self.prediction_results) < 2:
            print("\nNeed at least 2 datasets for combined analysis")
            return

        print(f"\n{'='*60}")
        print("COMBINED PREDICTION ANALYSIS")
        print(f"{'='*60}")

        # Combine all prediction results
        combined_data = []

        for dataset_name, results_df in self.prediction_results.items():
            dataset_results = results_df.copy()
            dataset_results['dataset'] = dataset_name
            combined_data.append(dataset_results)

        combined_df = pd.concat(combined_data, ignore_index=True)

        # Overall statistics
        print(f"\nOverall Prediction Statistics:")
        print(f"  Total transactions analyzed: {len(combined_df)}")
        print(f"  Total predicted fraud: {combined_df['predicted_label'].sum()}")
        print(f"  Total actual fraud: {combined_df['actual_label'].sum()}")
        print(f"  Overall accuracy: {combined_df['prediction_correct'].mean():.4f}")

        # Dataset comparison
        print(f"\nDataset Comparison:")
        dataset_stats = combined_df.groupby('dataset').agg({
            'prediction_correct': 'mean',
            'prob_fraud': 'mean',
            'confidence': 'mean',
            'predicted_label': 'sum',
            'actual_label': 'sum'
        }).round(4)

        print(dataset_stats)

        # Export combined results
        combined_csv_path = 'results/predictions/combined_predictions.csv'
        combined_df.to_csv(combined_csv_path, index=False)
        print(f"\nExported combined predictions: {combined_csv_path}")

        return combined_df

def main():
    """Main execution function"""
    print("Credit Card Fraud Detection - Test Set Predictions")
    print("=" * 70)

    # Initialize predictor
    predictor = FraudPredictor()

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

        # Load model and test data
        rf_model, X_test, y_test, feature_names = predictor.load_model_and_test_data(dataset_name)

        if rf_model is None:
            print(f"Skipping {dataset_name} - could not load model or data")
            continue

        # Run predictions
        results_df = predictor.run_predictions(rf_model, X_test, y_test, dataset_name)

        if results_df is None:
            print(f"Skipping {dataset_name} - could not run predictions")
            continue

        # Analyze predictions
        predictor.analyze_predictions(results_df, dataset_name)

        # Create fraud cases detail
        fraud_detail = predictor.create_fraud_cases_detail(results_df, X_test, feature_names, dataset_name)

        # Export results
        main_csv, summary_csv, fraud_csv, cm_csv = predictor.export_results(results_df, dataset_name, fraud_detail)

    # Create combined analysis if multiple datasets
    if len(predictor.prediction_results) > 1:
        combined_df = predictor.create_combined_predictions()

    print(f"\n\n{'='*80}")
    print("PREDICTION ANALYSIS COMPLETED")
    print(f"{'='*80}")
    print("\nFiles created for Tableau:")
    print("  - results/predictions/predictions_<dataset>.csv (detailed predictions)")
    print("  - results/predictions/prediction_summary_<dataset>.csv (summary stats)")
    print("  - results/predictions/fraud_cases_<dataset>.csv (predicted fraud details)")
    print("  - results/predictions/confusion_matrix_<dataset>.csv (confusion matrix)")
    if len(predictor.prediction_results) > 1:
        print("  - results/predictions/combined_predictions.csv (all datasets)")

    print("\nTableau Visualization Ideas:")
    print("  1. Fraud probability distribution (histogram)")
    print("  2. Prediction categories (pie chart)")
    print("  3. Confidence levels (bar chart)")
    print("  4. High-risk transactions (table/filter)")
    print("  5. Model performance comparison (dashboard)")
    print("  6. Fraud cases with feature details (detailed table)")

    print("\nReady for Tableau dashboard creation!")

if __name__ == "__main__":
    main()
