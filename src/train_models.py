"""
Main training script for sentiment analysis models
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing import TextPreprocessor, load_sample_data
from models import NaiveBayesModel, LSTMModel, DistilBERTModel
from evaluation import ModelEvaluator, SHAPExplainer, analyze_sentiment_drivers
from visualization import SentimentVisualizer

class SentimentAnalysisPipeline:
    """
    Complete pipeline for sentiment analysis model training and evaluation
    """
    
    def __init__(self, data_path: str = None, use_sample_data: bool = True):
        """
        Initialize the pipeline
        
        Args:
            data_path: Path to the dataset
            use_sample_data: Whether to use sample data for testing
        """
        self.data_path = data_path
        self.use_sample_data = use_sample_data
        self.preprocessor = TextPreprocessor()
        self.evaluator = ModelEvaluator()
        self.visualizer = SentimentVisualizer()
        self.models = {}
        self.results = {}
        
        # Create output directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        os.makedirs('plots', exist_ok=True)
    
    def load_data(self) -> pd.DataFrame:
        """
        Load and prepare the dataset
        
        Returns:
            DataFrame with the loaded data
        """
        if self.use_sample_data or self.data_path is None:
            print("Loading sample data...")
            data = load_sample_data()
        else:
            print(f"Loading data from {self.data_path}...")
            # Add support for different file formats
            if self.data_path.endswith('.csv'):
                data = pd.read_csv(self.data_path)
            elif self.data_path.endswith('.json'):
                data = pd.read_json(self.data_path)
            else:
                raise ValueError("Unsupported file format. Please use CSV or JSON.")
        
        print(f"Loaded {len(data)} samples")
        return data
    
    def preprocess_data(self, data: pd.DataFrame) -> tuple:
        """
        Preprocess the data
        
        Args:
            data: Raw data DataFrame
            
        Returns:
            Tuple of (processed_texts, labels, vectorizer, tfidf_features)
        """
        print("Preprocessing data...")
        
        # Extract text and labels
        if 'review_text' in data.columns:
            texts = data['review_text'].tolist()
        elif 'text' in data.columns:
            texts = data['text'].tolist()
        else:
            raise ValueError("No text column found. Expected 'review_text' or 'text'")
        
        if 'sentiment' in data.columns:
            labels = data['sentiment'].values
        elif 'label' in data.columns:
            labels = data['label'].values
        else:
            raise ValueError("No sentiment column found. Expected 'sentiment' or 'label'")
        
        # Preprocess texts
        processed_texts = self.preprocessor.preprocess_batch(texts)
        
        # Create TF-IDF features
        vectorizer, tfidf_features = self.preprocessor.create_tfidf_features(processed_texts)
        
        print(f"Preprocessed {len(processed_texts)} texts")
        return processed_texts, labels, vectorizer, tfidf_features
    
    def train_naive_bayes(self, X_train: pd.DataFrame, y_train: pd.Series, 
                         vectorizer) -> NaiveBayesModel:
        """
        Train Naive Bayes model
        
        Args:
            X_train: Training features
            y_train: Training labels
            vectorizer: TF-IDF vectorizer
            
        Returns:
            Trained Naive Bayes model
        """
        print("Training Naive Bayes model...")
        
        nb_model = NaiveBayesModel()
        nb_model.train(X_train, y_train, vectorizer)
        
        # Save model
        nb_model.save_model('models/naive_bayes_model.pkl')
        
        self.models['naive_bayes'] = nb_model
        print("Naive Bayes model trained and saved!")
        
        return nb_model
    
    def train_lstm(self, X_train: list, y_train: np.ndarray, 
                   X_val: list = None, y_val: np.ndarray = None) -> LSTMModel:
        """
        Train LSTM model
        
        Args:
            X_train: Training texts
            y_train: Training labels
            X_val: Validation texts
            y_val: Validation labels
            
        Returns:
            Trained LSTM model
        """
        print("Training LSTM model...")
        
        lstm_model = LSTMModel()
        
        # Prepare data
        X_train_lstm = lstm_model.prepare_data(X_train)
        if X_val is not None:
            X_val_lstm = lstm_model.prepare_data(X_val)
        else:
            X_val_lstm = None
            y_val = None
        
        # Train model
        history = lstm_model.train(
            X_train_lstm, y_train,
            X_val_lstm, y_val,
            epochs=10,
            batch_size=32
        )
        
        # Save model
        lstm_model.save_model('models/lstm_model.pkl')
        
        self.models['lstm'] = lstm_model
        print("LSTM model trained and saved!")
        
        return lstm_model
    
    def train_bert(self, X_train: list, y_train: np.ndarray,
                   X_val: list = None, y_val: np.ndarray = None) -> DistilBERTModel:
        """
        Train DistilBERT model
        
        Args:
            X_train: Training texts
            y_train: Training labels
            X_val: Validation texts
            y_val: Validation labels
            
        Returns:
            Trained DistilBERT model
        """
        print("Training DistilBERT model...")
        
        bert_model = DistilBERTModel()
        
        # Train model
        bert_model.train(
            X_train, y_train,
            X_val, y_val,
            epochs=3,
            batch_size=16,
            learning_rate=2e-5
        )
        
        # Save model
        bert_model.save_model('models/bert_model')
        
        self.models['bert'] = bert_model
        print("DistilBERT model trained and saved!")
        
        return bert_model
    
    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series, 
                       X_test_texts: list = None) -> pd.DataFrame:
        """
        Evaluate all trained models
        
        Args:
            X_test: Test features (for Naive Bayes)
            y_test: Test labels
            X_test_texts: Test texts (for LSTM and BERT)
            
        Returns:
            DataFrame with evaluation results
        """
        print("Evaluating models...")
        
        evaluation_results = []
        
        for model_name, model in self.models.items():
            print(f"Evaluating {model_name}...")
            
            if model_name == 'naive_bayes':
                # Evaluate Naive Bayes
                metrics = self.evaluator.evaluate_model(model, X_test, y_test, model_name)
                evaluation_results.append({
                    'Model': model_name,
                    **metrics
                })
                
            elif model_name in ['lstm', 'bert'] and X_test_texts is not None:
                # Evaluate LSTM and BERT
                if model_name == 'lstm':
                    X_test_processed = model.prepare_data(X_test_texts)
                else:
                    X_test_processed = X_test_texts
                
                y_pred = model.predict(X_test_processed)
                y_proba = model.predict_proba(X_test_processed)
                
                # Calculate metrics
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
                
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='weighted'),
                    'recall': recall_score(y_test, y_pred, average='weighted'),
                    'f1_score': f1_score(y_test, y_pred, average='weighted'),
                    'roc_auc': roc_auc_score(y_test, y_proba if len(y_proba.shape) == 1 else y_proba[:, 1])
                }
                
                evaluation_results.append({
                    'Model': model_name,
                    **metrics
                })
        
        results_df = pd.DataFrame(evaluation_results)
        
        # Save results
        results_df.to_csv('results/model_evaluation_results.csv', index=False)
        
        # Create visualization
        fig = self.visualizer.plot_model_comparison(
            {row['Model']: row.to_dict() for _, row in results_df.iterrows()}
        )
        fig.write_html('plots/model_comparison.html')
        
        self.results = results_df
        print("Model evaluation completed!")
        
        return results_df
    
    def generate_insights(self, data: pd.DataFrame, X_test: pd.DataFrame, 
                         y_test: pd.Series, feature_names: list = None):
        """
        Generate business insights from the analysis
        
        Args:
            data: Original data
            X_test: Test features
            y_test: Test labels
            feature_names: Names of features
        """
        print("Generating business insights...")
        
        # Get best model
        best_model_name = self.results.loc[self.results['accuracy'].idxmax(), 'Model']
        best_model = self.models[best_model_name]
        
        print(f"Best performing model: {best_model_name}")
        
        # Analyze sentiment drivers
        if best_model_name == 'naive_bayes':
            drivers = analyze_sentiment_drivers(best_model, X_test, y_test, feature_names)
            
            # Save insights
            drivers.to_csv('results/sentiment_drivers.csv', index=False)
            
            # Create visualization
            fig = self.visualizer.plot_feature_importance(drivers)
            fig.savefig('plots/feature_importance.png', dpi=300, bbox_inches='tight')
            
            print("Top positive sentiment drivers:")
            positive_drivers = drivers[drivers['correlation'] > 0].head(10)
            for _, row in positive_drivers.iterrows():
                print(f"  {row['feature']}: {row['correlation']:.3f}")
            
            print("\nTop negative sentiment drivers:")
            negative_drivers = drivers[drivers['correlation'] < 0].head(10)
            for _, row in negative_drivers.iterrows():
                print(f"  {row['feature']}: {row['correlation']:.3f}")
        
        # Create comprehensive visualizations
        self.create_comprehensive_visualizations(data)
        
        print("Business insights generated!")
    
    def create_comprehensive_visualizations(self, data: pd.DataFrame):
        """
        Create comprehensive visualizations
        
        Args:
            data: Original data
        """
        print("Creating visualizations...")
        
        # Sentiment distribution
        fig = self.visualizer.plot_sentiment_distribution(data)
        fig.write_html('plots/sentiment_distribution.html')
        
        # Word clouds
        positive_texts = data[data['sentiment'] == 1]['review_text'].tolist()
        negative_texts = data[data['sentiment'] == 0]['review_text'].tolist()
        
        # Positive sentiment word cloud
        fig = self.visualizer.create_wordcloud(positive_texts, sentiment=1)
        fig.savefig('plots/positive_wordcloud.png', dpi=300, bbox_inches='tight')
        
        # Negative sentiment word cloud
        fig = self.visualizer.create_wordcloud(negative_texts, sentiment=0)
        fig.savefig('plots/negative_wordcloud.png', dpi=300, bbox_inches='tight')
        
        # Word frequency analysis
        fig = self.visualizer.plot_word_frequency(positive_texts, sentiment=1)
        fig.savefig('plots/positive_word_frequency.png', dpi=300, bbox_inches='tight')
        
        fig = self.visualizer.plot_word_frequency(negative_texts, sentiment=0)
        fig.savefig('plots/negative_word_frequency.png', dpi=300, bbox_inches='tight')
        
        # Dashboard summary
        fig = self.visualizer.create_dashboard_summary(data)
        fig.write_html('plots/dashboard_summary.html')
        
        print("Visualizations created!")
    
    def run_pipeline(self):
        """
        Run the complete sentiment analysis pipeline
        """
        print("=" * 60)
        print("SENTIMENT ANALYSIS PIPELINE")
        print("=" * 60)
        
        start_time = datetime.now()
        
        try:
            # 1. Load data
            data = self.load_data()
            
            # 2. Preprocess data
            processed_texts, labels, vectorizer, tfidf_features = self.preprocess_data(data)
            
            # 3. Split data
            X_train, X_test, y_train, y_test = train_test_split(
                processed_texts, labels, test_size=0.2, random_state=42, stratify=labels
            )
            
            # Split TF-IDF features
            X_train_tfidf, X_test_tfidf = train_test_split(
                tfidf_features, test_size=0.2, random_state=42, stratify=labels
            )
            
            # 4. Train models
            nb_model = self.train_naive_bayes(X_train_tfidf, y_train, vectorizer)
            
            # Train LSTM (with smaller dataset for demo)
            if len(X_train) > 1000:
                X_train_lstm = X_train[:1000]
                y_train_lstm = y_train[:1000]
                X_test_lstm = X_test[:200]
                y_test_lstm = y_test[:200]
            else:
                X_train_lstm = X_train
                y_train_lstm = y_train
                X_test_lstm = X_test
                y_test_lstm = y_test
            
            lstm_model = self.train_lstm(X_train_lstm, y_train_lstm, X_test_lstm, y_test_lstm)
            
            # Train BERT (with even smaller dataset for demo)
            if len(X_train) > 500:
                X_train_bert = X_train[:500]
                y_train_bert = y_train[:500]
                X_test_bert = X_test[:100]
                y_test_bert = y_test[:100]
            else:
                X_train_bert = X_train
                y_train_bert = y_train
                X_test_bert = X_test
                y_test_bert = y_test
            
            bert_model = self.train_bert(X_train_bert, y_train_bert, X_test_bert, y_test_bert)
            
            # 5. Evaluate models
            results = self.evaluate_models(X_test_tfidf, y_test, X_test)
            
            # 6. Generate insights
            self.generate_insights(data, X_test_tfidf, y_test, 
                                 vectorizer.get_feature_names_out())
            
            # 7. Save pipeline summary
            self.save_pipeline_summary(start_time)
            
            print("\n" + "=" * 60)
            print("PIPELINE COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print(f"Total time: {datetime.now() - start_time}")
            print("\nResults saved in:")
            print("- models/: Trained models")
            print("- results/: Evaluation results and insights")
            print("- plots/: Visualizations")
            
        except Exception as e:
            print(f"Error in pipeline: {str(e)}")
            raise
    
    def save_pipeline_summary(self, start_time: datetime):
        """
        Save pipeline summary
        
        Args:
            start_time: Pipeline start time
        """
        summary = {
            'pipeline_run_time': datetime.now().isoformat(),
            'duration_seconds': (datetime.now() - start_time).total_seconds(),
            'models_trained': list(self.models.keys()),
            'best_model': self.results.loc[self.results['accuracy'].idxmax(), 'Model'],
            'best_accuracy': self.results['accuracy'].max(),
            'total_samples': len(self.results) if hasattr(self, 'results') else 0
        }
        
        with open('results/pipeline_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

def main():
    """
    Main function to run the sentiment analysis pipeline
    """
    # You can specify your data path here
    data_path = None  # Set to your data file path if available
    
    # Create and run pipeline
    pipeline = SentimentAnalysisPipeline(data_path=data_path, use_sample_data=True)
    pipeline.run_pipeline()

if __name__ == "__main__":
    main() 