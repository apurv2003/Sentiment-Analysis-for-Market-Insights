"""
Sentiment analysis models: Naive Bayes, LSTM, and DistilBERT
"""

import numpy as np
import pandas as pd
import pickle
import json
from typing import Tuple, Dict, Any, Optional
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from transformers import pipeline
import torch
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

class NaiveBayesModel:
    """
    Naive Bayes model with TF-IDF features
    """
    
    def __init__(self):
        self.model = MultinomialNB()
        self.vectorizer = None
        self.is_trained = False
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, vectorizer=None):
        """
        Train the Naive Bayes model
        
        Args:
            X_train: Training features (TF-IDF matrix)
            y_train: Training labels
            vectorizer: Pre-fitted TF-IDF vectorizer
        """
        if vectorizer is not None:
            self.vectorizer = vectorizer
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict_proba(X)
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'is_trained': self.is_trained
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.vectorizer = model_data['vectorizer']
        self.is_trained = model_data['is_trained']

class LSTMModel:
    """
    LSTM model for sentiment analysis
    """
    
    def __init__(self, max_words: int = 10000, max_len: int = 100, embedding_dim: int = 100):
        self.max_words = max_words
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
        self.model = None
        self.is_trained = False
    
    def prepare_data(self, texts: list) -> np.ndarray:
        """Prepare text data for LSTM"""
        self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_len, padding='post', truncating='post')
        return padded_sequences
    
    def build_model(self, vocab_size: int):
        """Build the LSTM model architecture"""
        self.model = Sequential([
            Embedding(vocab_size, self.embedding_dim, input_length=self.max_len),
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.3),
            Bidirectional(LSTM(32)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              epochs: int = 10, batch_size: int = 32):
        """Train the LSTM model"""
        if self.model is None:
            self.build_model(len(self.tokenizer.word_index) + 1)
        
        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            verbose=1
        )
        
        self.is_trained = True
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        predictions = self.model.predict(X)
        return (predictions > 0.5).astype(int).flatten()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X).flatten()
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'tokenizer': self.tokenizer,
            'max_words': self.max_words,
            'max_len': self.max_len,
            'embedding_dim': self.embedding_dim,
            'is_trained': self.is_trained
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.tokenizer = model_data['tokenizer']
        self.max_words = model_data['max_words']
        self.max_len = model_data['max_len']
        self.embedding_dim = model_data['embedding_dim']
        self.is_trained = model_data['is_trained']

class DistilBERTModel:
    """
    DistilBERT model for sentiment analysis
    """
    
    def __init__(self, model_name: str = 'distilbert-base-uncased'):
        self.model_name = model_name
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.is_trained = False
    
    def prepare_data(self, texts: list, labels: list = None) -> Dict[str, torch.Tensor]:
        """Prepare data for DistilBERT"""
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        )
        
        if labels is not None:
            encodings['labels'] = torch.tensor(labels)
        
        return encodings
    
    def train(self, train_texts: list, train_labels: list,
              val_texts: list = None, val_labels: list = None,
              epochs: int = 3, batch_size: int = 16, learning_rate: float = 2e-5):
        """Train the DistilBERT model"""
        # Prepare training data
        train_encodings = self.prepare_data(train_texts, train_labels)
        train_dataset = SentimentDataset(train_encodings)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Prepare validation data if provided
        val_loader = None
        if val_texts is not None and val_labels is not None:
            val_encodings = self.prepare_data(val_texts, val_labels)
            val_dataset = SentimentDataset(val_encodings)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Training setup
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}')
        
        self.is_trained = True
    
    def predict(self, texts: list) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for text in texts:
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding=True,
                    max_length=512,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=1)
                pred = torch.argmax(probs, dim=1).item()
                predictions.append(pred)
        
        return np.array(predictions)
    
    def predict_proba(self, texts: list) -> np.ndarray:
        """Get prediction probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        self.model.eval()
        probabilities = []
        
        with torch.no_grad():
            for text in texts:
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding=True,
                    max_length=512,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=1)
                probabilities.append(probs.cpu().numpy()[0])
        
        return np.array(probabilities)
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        self.model.save_pretrained(filepath)
        self.tokenizer.save_pretrained(filepath)
        
        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'is_trained': self.is_trained
        }
        with open(f"{filepath}/metadata.json", 'w') as f:
            json.dump(metadata, f)
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        self.model = DistilBertForSequenceClassification.from_pretrained(filepath)
        self.tokenizer = DistilBertTokenizer.from_pretrained(filepath)
        self.model.to(self.device)
        
        # Load metadata
        with open(f"{filepath}/metadata.json", 'r') as f:
            metadata = json.load(f)
        
        self.is_trained = metadata['is_trained']

class SentimentDataset(Dataset):
    """Custom dataset for DistilBERT training"""
    
    def __init__(self, encodings):
        self.encodings = encodings
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item
    
    def __len__(self):
        return len(self.encodings.input_ids)

def create_model_ensemble(models: list, weights: list = None) -> Dict[str, Any]:
    """
    Create an ensemble of models
    
    Args:
        models: List of trained models
        weights: Optional weights for each model
        
    Returns:
        Ensemble configuration
    """
    if weights is None:
        weights = [1.0] * len(models)
    
    return {
        'models': models,
        'weights': weights
    }

def ensemble_predict(ensemble: Dict[str, Any], X: Any) -> np.ndarray:
    """
    Make ensemble predictions
    
    Args:
        ensemble: Ensemble configuration
        X: Input data
        
    Returns:
        Ensemble predictions
    """
    predictions = []
    weights = ensemble['weights']
    
    for i, model in enumerate(ensemble['models']):
        pred = model.predict_proba(X)
        if len(pred.shape) == 1:
            pred = pred.reshape(-1, 1)
        predictions.append(pred * weights[i])
    
    # Weighted average
    ensemble_pred = np.sum(predictions, axis=0) / np.sum(weights)
    
    # Convert to binary predictions
    return (ensemble_pred > 0.5).astype(int).flatten()

if __name__ == "__main__":
    # Test the models with sample data
    from preprocessing import load_sample_data, TextPreprocessor
    
    # Load sample data
    data = load_sample_data()
    preprocessor = TextPreprocessor()
    
    # Preprocess text
    processed_texts = preprocessor.preprocess_batch(data['review_text'].tolist())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        processed_texts, data['sentiment'], test_size=0.2, random_state=42
    )
    
    # Test Naive Bayes
    print("Testing Naive Bayes model...")
    vectorizer, X_train_tfidf = preprocessor.create_tfidf_features(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    nb_model = NaiveBayesModel()
    nb_model.train(X_train_tfidf, y_train, vectorizer)
    nb_pred = nb_model.predict(X_test_tfidf)
    nb_accuracy = accuracy_score(y_test, nb_pred)
    print(f"Naive Bayes Accuracy: {nb_accuracy:.4f}")
    
    # Test LSTM
    print("\nTesting LSTM model...")
    lstm_model = LSTMModel()
    X_train_lstm = lstm_model.prepare_data(X_train)
    X_test_lstm = lstm_model.prepare_data(X_test)
    
    lstm_model.train(X_train_lstm, y_train.values, epochs=5)
    lstm_pred = lstm_model.predict(X_test_lstm)
    lstm_accuracy = accuracy_score(y_test, lstm_pred)
    print(f"LSTM Accuracy: {lstm_accuracy:.4f}")
    
    print("\nModel testing completed!") 