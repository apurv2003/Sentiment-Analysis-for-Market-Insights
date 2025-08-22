"""
Standalone model evaluation script
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing import TextPreprocessor, load_sample_data
from models import NaiveBayesModel, LSTMModel, DistilBERTModel
from evaluation import ModelEvaluator, create_evaluation_report

def main():
    """
    Evaluate trained models
    """
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    # Load data
    print("Loading data...")
    data = load_sample_data()
    
    # Preprocess data
    print("Preprocessing data...")
    preprocessor = TextPreprocessor()
    processed_texts = preprocessor.preprocess_batch(data['review_text'].tolist())
    vectorizer, tfidf_features = preprocessor.create_tfidf_features(processed_texts)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        processed_texts, data['sentiment'], test_size=0.2, random_state=42, stratify=data['sentiment']
    )
    
    X_train_tfidf, X_test_tfidf = train_test_split(
        tfidf_features, test_size=0.2, random_state=42, stratify=data['sentiment']
    )
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Load and evaluate models
    models_to_evaluate = {}
    
    # Try to load Naive Bayes
    try:
        nb_model = NaiveBayesModel()
        nb_model.load_model('models/naive_bayes_model.pkl')
        models_to_evaluate['naive_bayes'] = nb_model
        print("✅ Loaded Naive Bayes model")
    except Exception as e:
        print(f"❌ Could not load Naive Bayes model: {e}")
    
    # Try to load LSTM
    try:
        lstm_model = LSTMModel()
        lstm_model.load_model('models/lstm_model.pkl')
        models_to_evaluate['lstm'] = lstm_model
        print("✅ Loaded LSTM model")
    except Exception as e:
        print(f"❌ Could not load LSTM model: {e}")
    
    # Try to load BERT
    try:
        bert_model = DistilBERTModel()
        bert_model.load_model('models/bert_model')
        models_to_evaluate['bert'] = bert_model
        print("✅ Loaded BERT model")
    except Exception as e:
        print(f"❌ Could not load BERT model: {e}")
    
    if not models_to_evaluate:
        print("❌ No models found to evaluate!")
        return
    
    # Evaluate models
    print(f"\nEvaluating {len(models_to_evaluate)} models...")
    
    for model_name, model in models_to_evaluate.items():
        print(f"\nEvaluating {model_name}...")
        
        if model_name == 'naive_bayes':
            metrics = evaluator.evaluate_model(model, X_test_tfidf, y_test, model_name)
        else:
            # For LSTM and BERT, we need to prepare the data
            if model_name == 'lstm':
                X_test_processed = model.prepare_data(X_test)
            else:
                X_test_processed = X_test
            
            y_pred = model.predict(X_test_processed)
            y_proba = model.predict_proba(X_test_processed)
            
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1_score': f1_score(y_test, y_pred, average='weighted'),
                'roc_auc': roc_auc_score(y_test, y_proba if len(y_proba.shape) == 1 else y_proba[:, 1])
            }
        
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    
    # Generate evaluation report
    print("\nGenerating evaluation report...")
    create_evaluation_report(evaluator, "evaluation_report.txt")
    
    print("\n✅ Evaluation completed!")
    print("📄 Report saved to: evaluation_report.txt")

if __name__ == "__main__":
    main() 