"""
Model evaluation and explainability for sentiment analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import shap
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    """
    Comprehensive model evaluation for sentiment analysis
    """
    
    def __init__(self):
        self.results = {}
        self.shap_values = {}
    
    def evaluate_model(self, model, X_test: Any, y_test: pd.Series, 
                      model_name: str = "Model") -> Dict[str, float]:
        """
        Evaluate a single model with comprehensive metrics
        
        Args:
            model: Trained model with predict and predict_proba methods
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model for results storage
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        # For binary classification, get positive class probability
        if len(y_proba.shape) > 1 and y_proba.shape[1] > 1:
            y_proba = y_proba[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'roc_auc': roc_auc_score(y_test, y_proba),
        }
        
        # Store results
        self.results[model_name] = {
            'metrics': metrics,
            'predictions': y_pred,
            'probabilities': y_proba,
            'true_labels': y_test.values
        }
        
        return metrics
    
    def cross_validate_model(self, model, X: Any, y: pd.Series, 
                           cv_folds: int = 5, model_name: str = "Model") -> Dict[str, List[float]]:
        """
        Perform cross-validation on a model
        
        Args:
            model: Model to evaluate
            X: Features
            y: Labels
            cv_folds: Number of cross-validation folds
            model_name: Name of the model
            
        Returns:
            Dictionary of cross-validation results
        """
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # For models that need different preprocessing, we'll need to handle them separately
        if hasattr(model, 'prepare_data'):  # LSTM or DistilBERT
            # This is a simplified version - in practice, you'd need more sophisticated handling
            scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        else:
            scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        
        cv_results = {
            'accuracy_scores': scores.tolist(),
            'mean_accuracy': scores.mean(),
            'std_accuracy': scores.std()
        }
        
        return cv_results
    
    def compare_models(self, models: Dict[str, Any], X_test: Any, y_test: pd.Series) -> pd.DataFrame:
        """
        Compare multiple models
        
        Args:
            models: Dictionary of {model_name: model}
            X_test: Test features
            y_test: Test labels
            
        Returns:
            DataFrame with comparison results
        """
        comparison_results = []
        
        for name, model in models.items():
            metrics = self.evaluate_model(model, X_test, y_test, name)
            comparison_results.append({
                'Model': name,
                **metrics
            })
        
        return pd.DataFrame(comparison_results)
    
    def plot_confusion_matrix(self, model_name: str, figsize: Tuple[int, int] = (8, 6)):
        """Plot confusion matrix for a model"""
        if model_name not in self.results:
            raise ValueError(f"Model {model_name} not found in results")
        
        y_true = self.results[model_name]['true_labels']
        y_pred = self.results[model_name]['predictions']
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
    
    def plot_roc_curve(self, model_name: str, figsize: Tuple[int, int] = (8, 6)):
        """Plot ROC curve for a model"""
        if model_name not in self.results:
            raise ValueError(f"Model {model_name} not found in results")
        
        y_true = self.results[model_name]['true_labels']
        y_proba = self.results[model_name]['probabilities']
        
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        
        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_precision_recall_curve(self, model_name: str, figsize: Tuple[int, int] = (8, 6)):
        """Plot precision-recall curve for a model"""
        if model_name not in self.results:
            raise ValueError(f"Model {model_name} not found in results")
        
        y_true = self.results[model_name]['true_labels']
        y_proba = self.results[model_name]['probabilities']
        
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        
        plt.figure(figsize=figsize)
        plt.plot(recall, precision, label=model_name)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def generate_classification_report(self, model_name: str) -> str:
        """Generate detailed classification report"""
        if model_name not in self.results:
            raise ValueError(f"Model {model_name} not found in results")
        
        y_true = self.results[model_name]['true_labels']
        y_pred = self.results[model_name]['predictions']
        
        return classification_report(y_true, y_pred, target_names=['Negative', 'Positive'])

class SHAPExplainer:
    """
    SHAP explainability for NLP models
    """
    
    def __init__(self):
        self.explainers = {}
        self.shap_values = {}
    
    def explain_naive_bayes(self, model, X_test: pd.DataFrame, feature_names: List[str] = None):
        """
        Explain Naive Bayes model using SHAP
        
        Args:
            model: Trained Naive Bayes model
            X_test: Test features (TF-IDF matrix)
            feature_names: Names of features
        """
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X_test.shape[1])]
        
        # Create explainer
        explainer = shap.LinearExplainer(model.model, X_test)
        shap_values = explainer.shap_values(X_test)
        
        self.explainers['naive_bayes'] = explainer
        self.shap_values['naive_bayes'] = shap_values
        
        return explainer, shap_values
    
    def explain_bert_model(self, model, texts: List[str], max_samples: int = 100):
        """
        Explain BERT model using SHAP
        
        Args:
            model: Trained BERT model
            texts: List of texts to explain
            max_samples: Maximum number of samples to explain
        """
        # Limit samples for computational efficiency
        if len(texts) > max_samples:
            texts = texts[:max_samples]
        
        # Create explainer
        explainer = shap.Explainer(model.predict, model.tokenizer)
        shap_values = explainer(texts)
        
        self.explainers['bert'] = explainer
        self.shap_values['bert'] = shap_values
        
        return explainer, shap_values
    
    def plot_shap_summary(self, model_type: str, max_display: int = 20):
        """Plot SHAP summary for a model"""
        if model_type not in self.shap_values:
            raise ValueError(f"SHAP values for {model_type} not found")
        
        shap_values = self.shap_values[model_type]
        
        if model_type == 'naive_bayes':
            # For Naive Bayes, plot feature importance
            shap.summary_plot(shap_values, max_display=max_display)
        elif model_type == 'bert':
            # For BERT, plot token importance
            shap.plots.text(shap_values)
    
    def plot_shap_waterfall(self, model_type: str, sample_idx: int = 0):
        """Plot SHAP waterfall plot for a specific sample"""
        if model_type not in self.shap_values:
            raise ValueError(f"SHAP values for {model_type} not found")
        
        shap_values = self.shap_values[model_type]
        
        if model_type == 'naive_bayes':
            shap.waterfall_plot(shap_values[sample_idx])
        elif model_type == 'bert':
            shap.plots.text(shap_values[sample_idx])

def analyze_sentiment_drivers(model, X_test: pd.DataFrame, y_test: pd.Series, 
                            feature_names: List[str] = None, top_n: int = 20) -> pd.DataFrame:
    """
    Analyze the main drivers of sentiment predictions
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        feature_names: Names of features
        top_n: Number of top features to return
        
    Returns:
        DataFrame with feature importance analysis
    """
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X_test.shape[1])]
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # For binary classification, get positive class probability
    if len(y_proba.shape) > 1 and y_proba.shape[1] > 1:
        y_proba = y_proba[:, 1]
    
    # Calculate feature importance using correlation
    feature_importance = []
    for i, feature in enumerate(feature_names):
        correlation = np.corrcoef(X_test.iloc[:, i], y_proba)[0, 1]
        feature_importance.append({
            'feature': feature,
            'correlation': correlation,
            'abs_correlation': abs(correlation)
        })
    
    # Sort by absolute correlation
    importance_df = pd.DataFrame(feature_importance)
    importance_df = importance_df.sort_values('abs_correlation', ascending=False).head(top_n)
    
    return importance_df

def create_evaluation_report(evaluator: ModelEvaluator, output_file: str = "evaluation_report.txt"):
    """
    Create a comprehensive evaluation report
    
    Args:
        evaluator: ModelEvaluator instance with results
        output_file: Output file path
    """
    with open(output_file, 'w') as f:
        f.write("SENTIMENT ANALYSIS MODEL EVALUATION REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        for model_name, results in evaluator.results.items():
            f.write(f"MODEL: {model_name}\n")
            f.write("-" * 30 + "\n")
            
            metrics = results['metrics']
            f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"Precision: {metrics['precision']:.4f}\n")
            f.write(f"Recall: {metrics['recall']:.4f}\n")
            f.write(f"F1-Score: {metrics['f1_score']:.4f}\n")
            f.write(f"ROC-AUC: {metrics['roc_auc']:.4f}\n\n")
            
            # Classification report
            f.write("Detailed Classification Report:\n")
            f.write(evaluator.generate_classification_report(model_name))
            f.write("\n\n")
    
    print(f"Evaluation report saved to {output_file}")

if __name__ == "__main__":
    # Test the evaluator with sample data
    from preprocessing import load_sample_data, TextPreprocessor
    from models import NaiveBayesModel
    
    # Load and preprocess sample data
    data = load_sample_data()
    preprocessor = TextPreprocessor()
    processed_texts = preprocessor.preprocess_batch(data['review_text'].tolist())
    
    # Create TF-IDF features
    vectorizer, X_tfidf = preprocessor.create_tfidf_features(processed_texts)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, data['sentiment'], test_size=0.2, random_state=42
    )
    
    # Train and evaluate model
    nb_model = NaiveBayesModel()
    nb_model.train(X_train, y_train, vectorizer)
    
    # Evaluate
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_model(nb_model, X_test, y_test, "Naive Bayes")
    
    print("Evaluation Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Generate report
    create_evaluation_report(evaluator) 