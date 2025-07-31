
"""
Credit Card Fraud Detection - Baseline Model Training Script
This script trains and evaluates baseline machine learning models for fraud detection:
- Logistic Regression
- Random Forest
- Gradient Boosting (XGBoost)
- Support Vector Machine
- Naive Bayes

Features:
- Automatic dataset detection and loading
- Multiple baseline algorithms
- Comprehensive evaluation metrics
- Cross-validation
- Model comparison and ranking
- Visualization of results
- Model persistence
- Performance reports
"""

import pandas as pd
import numpy as np
import os
import pickle
import json
from datetime import datetime
import warnings
import numbers
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Evaluation
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

def to_serializable(val):
    if isinstance(val, np.ndarray):
        return val.tolist()
    elif isinstance(val, (np.integer, np.int64, np.int32)):
        return int(val)
    elif isinstance(val, (np.floating, np.float64, np.float32)):
        return float(val)
    elif isinstance(val, numbers.Number):
        return val
    else:
        return val

# Advanced models (if available)

class BaselineModelTrainer:
    def __init__(self):
        self.models = {}
        self.results = {}
        self.trained_models = {}

        # Create output directories
        os.makedirs('models/baseline', exist_ok=True)
        os.makedirs('results/baseline', exist_ok=True)
        os.makedirs('plots/baseline', exist_ok=True)

        # Initialize baseline models
        self.initialize_models()

    def initialize_models(self):
        """Initialize baseline models with optimized parameters"""
        print("Initializing baseline models...")

        self.models = {
            'Logistic_Regression': LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced',
                solver='liblinear'
            ),
            'Random_Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1,
                max_depth=10
            ),
            # 'Gradient_Boosting': GradientBoostingClassifier(
            #     n_estimators=100,
            #     random_state=42,
            #     learning_rate=0.1,
            #     max_depth=6
            # ),
            # 'SVM': SVC(
            #     random_state=42,
            #     class_weight='balanced',
            #     probability=True,
            #     kernel='rbf',
            #     C=1.0
            # ),
            # 'Naive_Bayes': GaussianNB(),
            # 'KNN': KNeighborsClassifier(
            #     n_neighbors=5,
            #     n_jobs=-1
            # ),
            # 'Decision_Tree': DecisionTreeClassifier(
            #     random_state=42,
            #     class_weight='balanced',
            #     max_depth=10
            # )
        }

        # # Add XGBoost if available
        # if XGBOOST_AVAILABLE:
        #     self.models['XGBoost'] = xgb.XGBClassifier(
        #         random_state=42,
        #         eval_metric='logloss',
        #         use_label_encoder=False,
        #         n_estimators=100,
        #         learning_rate=0.1,
        #         max_depth=6
        #     )

        # # Add LightGBM if available
        # if LIGHTGBM_AVAILABLE:
        #     self.models['LightGBM'] = lgb.LGBMClassifier(
        #         random_state=42,
        #         n_estimators=100,
        #         learning_rate=0.1,
        #         max_depth=6,
        #         verbose=-1
        #     )

        print(f"Initialized {len(self.models)} baseline models")

    def load_processed_data(self, dataset_path):
        """Load processed data from model preparation step"""
        try:
            print(f"Loading processed data from: {dataset_path}")

            # Load feature data
            X_train = pd.read_csv(f'{dataset_path}/X_train.csv')
            X_val = pd.read_csv(f'{dataset_path}/X_val.csv')
            X_test = pd.read_csv(f'{dataset_path}/X_test.csv')

            # Load target data
            y_train = pd.read_csv(f'{dataset_path}/y_train.csv')['target']
            y_val = pd.read_csv(f'{dataset_path}/y_val.csv')['target']
            y_test = pd.read_csv(f'{dataset_path}/y_test.csv')['target']

            # Load feature names
            feature_names = pd.read_csv(f'{dataset_path}/feature_names.csv')['feature'].tolist()

            # Load metadata
            metadata = pd.read_csv(f'{dataset_path}/metadata.csv', index_col=0)['0'].to_dict()

            print(f"Data loaded successfully:")
            print(f"  Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
            print(f"  Validation: {X_val.shape[0]} samples")
            print(f"  Test: {X_test.shape[0]} samples")
            print(f"  Features: {len(feature_names)}")

            return X_train, X_val, X_test, y_train, y_val, y_test, feature_names, metadata

        except Exception as e:
            print(f"Error loading processed data: {e}")
            return None, None, None, None, None, None, None, None

    def train_model(self, model_name, model, X_train, y_train, X_val, y_val):
        """Train a single model and return validation predictions"""
        print(f"Training {model_name}...")

        try:
            # Train the model
            start_time = datetime.now()
            model.fit(X_train, y_train)
            training_time = (datetime.now() - start_time).total_seconds()

            # Make predictions
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)

            # Get prediction probabilities
            if hasattr(model, 'predict_proba'):
                y_train_proba = model.predict_proba(X_train)[:, 1]
                y_val_proba = model.predict_proba(X_val)[:, 1]
            else:
                y_train_proba = y_train_pred
                y_val_proba = y_val_pred

            # Store trained model
            self.trained_models[model_name] = model

            print(f"  ✓ {model_name} trained in {training_time:.2f} seconds")

            return {
                'model': model,
                'y_train_pred': y_train_pred,
                'y_val_pred': y_val_pred,
                'y_train_proba': y_train_proba,
                'y_val_proba': y_val_proba,
                'training_time': training_time
            }

        except Exception as e:
            print(f"  ✗ Error training {model_name}: {e}")
            return None

    def evaluate_model(self, model_name, y_true, y_pred, y_proba, dataset_type='validation'):
        """Evaluate model performance with comprehensive metrics"""

        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary')
        recall = recall_score(y_true, y_pred, average='binary')
        f1 = f1_score(y_true, y_pred, average='binary')

        # ROC-AUC (if probabilities available)
        try:
            roc_auc = roc_auc_score(y_true, y_proba)
        except:
            roc_auc = roc_auc_score(y_true, y_pred)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value

        # Fraud detection specific metrics
        fraud_detection_rate = recall  # Same as recall for fraud class
        false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'specificity': specificity,
            'npv': npv,
            'fraud_detection_rate': fraud_detection_rate,
            'false_alarm_rate': false_alarm_rate,
            'confusion_matrix': cm,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
        }

    def cross_validate_model(self, model_name, model, X, y, cv_folds=5):
        """Perform cross-validation for more robust evaluation"""
        print(f"Cross-validating {model_name}...")

        try:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

            # Cross-validate different metrics
            cv_accuracy = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
            cv_precision = cross_val_score(model, X, y, cv=cv, scoring='precision', n_jobs=-1)
            cv_recall = cross_val_score(model, X, y, cv=cv, scoring='recall', n_jobs=-1)
            cv_f1 = cross_val_score(model, X, y, cv=cv, scoring='f1', n_jobs=-1)
            cv_roc_auc = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)

            return {
                'cv_accuracy_mean': cv_accuracy.mean(),
                'cv_accuracy_std': cv_accuracy.std(),
                'cv_precision_mean': cv_precision.mean(),
                'cv_precision_std': cv_precision.std(),
                'cv_recall_mean': cv_recall.mean(),
                'cv_recall_std': cv_recall.std(),
                'cv_f1_mean': cv_f1.mean(),
                'cv_f1_std': cv_f1.std(),
                'cv_roc_auc_mean': cv_roc_auc.mean(),
                'cv_roc_auc_std': cv_roc_auc.std()
            }

        except Exception as e:
            print(f"  ✗ Error in cross-validation for {model_name}: {e}")
            return {}

    def train_all_models(self, X_train, X_val, X_test, y_train, y_val, y_test, dataset_name):
        """Train all baseline models and evaluate performance"""
        print(f"\n{'='*60}")
        print(f"TRAINING BASELINE MODELS - {dataset_name}")
        print(f"{'='*60}")

        model_results = {}

        for model_name, model in self.models.items():
            print(f"\n--- {model_name} ---")

            # Train model
            training_result = self.train_model(model_name, model, X_train, y_train, X_val, y_val)

            if training_result is None:
                continue

            # Evaluate on training set
            train_metrics = self.evaluate_model(
                model_name, y_train, training_result['y_train_pred'], 
                training_result['y_train_proba'], 'training'
            )

            # Evaluate on validation set
            val_metrics = self.evaluate_model(
                model_name, y_val, training_result['y_val_pred'], 
                training_result['y_val_proba'], 'validation'
            )

            # Cross-validation
            cv_metrics = self.cross_validate_model(model_name, model, X_train, y_train)

            # Test set evaluation (final evaluation)
            y_test_pred = model.predict(X_test)
            if hasattr(model, 'predict_proba'):
                y_test_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_test_proba = y_test_pred

            test_metrics = self.evaluate_model(
                model_name, y_test, y_test_pred, y_test_proba, 'test'
            )

            # Store all results
            model_results[model_name] = {
                'training_time': training_result['training_time'],
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'test_metrics': test_metrics,
                'cv_metrics': cv_metrics,
                'predictions': {
                    'y_val_pred': training_result['y_val_pred'],
                    'y_val_proba': training_result['y_val_proba'],
                    'y_test_pred': y_test_pred,
                    'y_test_proba': y_test_proba
                }
            }

            # Print summary
            print(f"  Training Accuracy: {train_metrics['accuracy']:.4f}")
            print(f"  Validation Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
            print(f"  Test F1-Score: {test_metrics['f1_score']:.4f}")
            print(f"  Test ROC-AUC: {test_metrics['roc_auc']:.4f}")

        return model_results

    def create_model_comparison(self, model_results, dataset_name):
        """Create comprehensive model comparison"""
        print(f"\nCreating model comparison for {dataset_name}...")

        # Prepare comparison data
        comparison_data = []

        for model_name, results in model_results.items():
            test_metrics = results['test_metrics']
            cv_metrics = results.get('cv_metrics', {})

            comparison_data.append({
                'Model': model_name,
                'Test_Accuracy': test_metrics['accuracy'],
                'Test_Precision': test_metrics['precision'],
                'Test_Recall': test_metrics['recall'],
                'Test_F1': test_metrics['f1_score'],
                'Test_ROC_AUC': test_metrics['roc_auc'],
                'Test_Specificity': test_metrics['specificity'],
                'False_Alarm_Rate': test_metrics['false_alarm_rate'],
                'CV_F1_Mean': cv_metrics.get('cv_f1_mean', 0),
                'CV_F1_Std': cv_metrics.get('cv_f1_std', 0),
                'CV_ROC_AUC_Mean': cv_metrics.get('cv_roc_auc_mean', 0),
                'Training_Time': results['training_time']
            })

        comparison_df = pd.DataFrame(comparison_data)

        # Sort by Test F1-Score (most important for fraud detection)
        comparison_df = comparison_df.sort_values('Test_F1', ascending=False)

        # Save comparison
        comparison_df.to_csv(f'results/baseline/model_comparison_{dataset_name.lower()}.csv', index=False)

        return comparison_df

    def visualize_results(self, model_results, y_val, y_test, dataset_name):
        """Create comprehensive visualizations"""
        print(f"Creating visualizations for {dataset_name}...")

        # 1. Model Performance Comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        models = list(model_results.keys())
        test_f1 = [model_results[m]['test_metrics']['f1_score'] for m in models]
        test_roc_auc = [model_results[m]['test_metrics']['roc_auc'] for m in models]
        test_precision = [model_results[m]['test_metrics']['precision'] for m in models]
        test_recall = [model_results[m]['test_metrics']['recall'] for m in models]

        # F1-Score comparison
        axes[0, 0].bar(models, test_f1, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Test F1-Score Comparison')
        axes[0, 0].set_ylabel('F1-Score')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # ROC-AUC comparison
        axes[0, 1].bar(models, test_roc_auc, color='lightgreen', alpha=0.7)
        axes[0, 1].set_title('Test ROC-AUC Comparison')
        axes[0, 1].set_ylabel('ROC-AUC')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # Precision vs Recall
        axes[1, 0].scatter(test_recall, test_precision, s=100, alpha=0.7)
        for i, model in enumerate(models):
            axes[1, 0].annotate(model, (test_recall[i], test_precision[i]), 
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_title('Precision vs Recall')
        axes[1, 0].grid(True, alpha=0.3)

        # Training time comparison
        training_times = [model_results[m]['training_time'] for m in models]
        axes[1, 1].bar(models, training_times, color='orange', alpha=0.7)
        axes[1, 1].set_title('Training Time Comparison')
        axes[1, 1].set_ylabel('Time (seconds)')
        axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(f'plots/baseline/model_comparison_{dataset_name.lower()}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()

        # 2. ROC Curves
        plt.figure(figsize=(10, 8))

        for model_name in models:
            y_proba = model_results[model_name]['predictions']['y_test_proba']
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc_score = model_results[model_name]['test_metrics']['roc_auc']
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})')

        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves - {dataset_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'plots/baseline/roc_curves_{dataset_name.lower()}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Confusion Matrices for Top 3 Models
        top_models = sorted(models, key=lambda x: model_results[x]['test_metrics']['f1_score'], reverse=True)[:3]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for i, model_name in enumerate(top_models):
            cm = model_results[model_name]['test_metrics']['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(f'{model_name}\nF1: {model_results[model_name]["test_metrics"]["f1_score"]:.3f}')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')

        plt.tight_layout()
        plt.savefig(f'plots/baseline/confusion_matrices_{dataset_name.lower()}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def save_models(self, dataset_name):
        """Save trained models"""
        print(f"Saving trained models for {dataset_name}...")

        model_dir = f'models/baseline/{dataset_name.lower()}'
        os.makedirs(model_dir, exist_ok=True)

        for model_name, model in self.trained_models.items():
            model_path = f'{model_dir}/{model_name.lower()}.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)

        print(f"Models saved to: {model_dir}/")

    def generate_report(self, model_results, comparison_df, dataset_name, metadata):
        """Generate comprehensive training report"""
        print(f"\n{'='*80}")
        print(f"BASELINE MODEL TRAINING REPORT - {dataset_name}")
        print(f"{'='*80}")

        print(f"\nDataset Information:")
        print(f"Dataset: {dataset_name}")
        print(f"Features: {metadata.get('n_features', 'N/A')}")
        print(f"Training samples: {metadata.get('train_samples', 'N/A')}")
        print(f"Test samples: {metadata.get('test_samples', 'N/A')}")
        print(f"Balancing method: {metadata.get('balancing_method', 'N/A')}")

        print(f"\nModel Performance Summary (Test Set):")
        print("=" * 100)
        print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'ROC-AUC':<10} {'Time(s)':<10}")
        print("=" * 100)

        for _, row in comparison_df.iterrows():
            print(f"{row['Model']:<20} {row['Test_Accuracy']:<10.4f} {row['Test_Precision']:<10.4f} "
                  f"{row['Test_Recall']:<10.4f} {row['Test_F1']:<10.4f} {row['Test_ROC_AUC']:<10.4f} "
                  f"{row['Training_Time']:<10.2f}")

        print("=" * 100)

        # Best model analysis
        best_model = comparison_df.iloc[0]
        print(f"\nBest Performing Model: {best_model['Model']}")
        print(f"  Test F1-Score: {best_model['Test_F1']:.4f}")
        print(f"  Test ROC-AUC: {best_model['Test_ROC_AUC']:.4f}")
        print(f"  False Alarm Rate: {best_model['False_Alarm_Rate']:.4f}")
        print(f"  Training Time: {best_model['Training_Time']:.2f} seconds")

        # Fraud detection insights
        print(f"\nFraud Detection Insights:")
        best_model_name = best_model['Model']
        best_results = model_results[best_model_name]
        test_metrics = best_results['test_metrics']

        print(f"  True Positives (Frauds Caught): {test_metrics['tp']}")
        print(f"  False Negatives (Frauds Missed): {test_metrics['fn']}")
        print(f"  False Positives (False Alarms): {test_metrics['fp']}")
        print(f"  True Negatives (Correct Non-Fraud): {test_metrics['tn']}")

        fraud_catch_rate = (test_metrics['tp'] / (test_metrics['tp'] + test_metrics['fn'])) * 100
        print(f"  Fraud Detection Rate: {fraud_catch_rate:.2f}%")

        print(f"\nRecommendations:")
        if best_model['Test_F1'] > 0.8:
            print("  ✓ Excellent performance - ready for production consideration")
        elif best_model['Test_F1'] > 0.6:
            print("  ⚠ Good performance - consider hyperparameter tuning")
        else:
            print("  ⚠ Moderate performance - feature engineering or advanced models recommended")

        if best_model['False_Alarm_Rate'] > 0.1:
            print("  ⚠ High false alarm rate - consider adjusting decision threshold")

        print(f"\nNext Steps:")
        print("  1. Hyperparameter tuning for top-performing models")
        print("  2. Feature importance analysis")
        print("  3. Ensemble methods")
        print("  4. Advanced algorithms (Neural Networks, etc.)")
        print("  5. Production deployment preparation")

def main():
    """Main execution function"""
    print("Credit Card Fraud Detection - Baseline Model Training")
    print("=" * 70)

    # Initialize trainer
    trainer = BaselineModelTrainer()

    # Find processed datasets
    processed_dir = 'data/processed'

    if not os.path.exists(processed_dir):
        print(f"Error: Processed data directory not found: {processed_dir}")
        print("Please run the model preparation script first.")
        return

    # Get all dataset directories
    dataset_dirs = [d for d in os.listdir(processed_dir) 
                   if os.path.isdir(os.path.join(processed_dir, d))]

    if not dataset_dirs:
        print("No processed datasets found. Please run model preparation first.")
        return

    print(f"Found {len(dataset_dirs)} processed datasets: {dataset_dirs}")

    # Process each dataset
    for dataset_name in dataset_dirs:
        dataset_path = os.path.join(processed_dir, dataset_name)

        print(f"\n\n{'='*80}")
        print(f"PROCESSING DATASET: {dataset_name.upper()}")
        print(f"{'='*80}")

        # Load processed data
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names, metadata = trainer.load_processed_data(dataset_path)

        if X_train is None:
            print(f"Skipping {dataset_name} - could not load data")
            continue

        # Train all models
        model_results = trainer.train_all_models(
            X_train, X_val, X_test, y_train, y_val, y_test, dataset_name
        )

        if not model_results:
            print(f"No models trained successfully for {dataset_name}")
            continue

        # Create model comparison
        comparison_df = trainer.create_model_comparison(model_results, dataset_name)

        # Create visualizations
        trainer.visualize_results(model_results, y_val, y_test, dataset_name)

        # Save models
        trainer.save_models(dataset_name)

        # Generate report
        trainer.generate_report(model_results, comparison_df, dataset_name, metadata)

        # Save detailed results
        results_path = f'results/baseline/detailed_results_{dataset_name.lower()}.json'
        with open(results_path, 'w') as f:
            json_results = {}
            for model_name, results in model_results.items():
                json_results[model_name] = {
                    'training_time': float(results['training_time']),
                    'train_metrics': {k: to_serializable(v) for k, v in results['train_metrics'].items()},
                    'val_metrics': {k: to_serializable(v) for k, v in results['val_metrics'].items()},
                    'test_metrics': {k: to_serializable(v) for k, v in results['test_metrics'].items()},
                    'cv_metrics': {k: to_serializable(v) for k, v in results['cv_metrics'].items()}
        }
            json.dump(json_results, f, indent=2)

    print(f"\n\n{'='*80}")
    print("BASELINE MODEL TRAINING COMPLETED")
    print(f"{'='*80}")
    print("Results saved to:")
    print("  - models/baseline/ (trained models)")
    print("  - results/baseline/ (performance metrics)")
    print("  - plots/baseline/ (visualizations)")
    print("\nReady for hyperparameter tuning and advanced modeling!")

if __name__ == "__main__":
    main()
