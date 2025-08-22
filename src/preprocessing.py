"""
Text preprocessing utilities for sentiment analysis
"""

import re
import nltk
import spacy
import pandas as pd
from typing import List, Tuple
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class TextPreprocessor:
    """
    Comprehensive text preprocessing for sentiment analysis
    """
    
    def __init__(self, use_spacy: bool = True):
        """
        Initialize the text preprocessor
        
        Args:
            use_spacy: Whether to use spaCy for advanced preprocessing
        """
        self.use_spacy = use_spacy
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Add custom stop words for e-commerce context
        self.custom_stops = {
            'product', 'item', 'order', 'delivery', 'shipping', 'amazon',
            'buy', 'purchase', 'customer', 'review', 'rating', 'star'
        }
        self.stop_words.update(self.custom_stops)
        
        if use_spacy:
            try:
                self.nlp = spacy.load('en_core_web_sm')
            except OSError:
                print("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
                self.use_spacy = False
    
    def clean_text(self, text: str) -> str:
        """
        Basic text cleaning
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep apostrophes for contractions
        text = re.sub(r'[^a-zA-Z\s\']', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove stop words from tokenized text
        
        Args:
            tokens: List of tokens
            
        Returns:
            Tokens with stop words removed
        """
        return [token for token in tokens if token not in self.stop_words]
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize tokens
        
        Args:
            tokens: List of tokens
            
        Returns:
            Lemmatized tokens
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess_text(self, text: str) -> str:
        """
        Complete text preprocessing pipeline
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        # Clean text
        text = self.clean_text(text)
        
        if self.use_spacy:
            # Use spaCy for advanced preprocessing
            doc = self.nlp(text)
            
            # Extract tokens, remove stopwords, lemmatize
            tokens = [token.lemma_ for token in doc 
                     if not token.is_stop and not token.is_punct and len(token.text) > 2]
        else:
            # Use NLTK for basic preprocessing
            tokens = word_tokenize(text)
            tokens = self.remove_stopwords(tokens)
            tokens = self.lemmatize_tokens(tokens)
            tokens = [token for token in tokens if len(token) > 2]
        
        return ' '.join(tokens)
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Preprocess a batch of texts
        
        Args:
            texts: List of input texts
            
        Returns:
            List of preprocessed texts
        """
        return [self.preprocess_text(text) for text in texts]
    
    def create_tfidf_features(self, texts: List[str], max_features: int = 5000) -> Tuple[TfidfVectorizer, pd.DataFrame]:
        """
        Create TF-IDF features from preprocessed texts
        
        Args:
            texts: List of preprocessed texts
            max_features: Maximum number of features
            
        Returns:
            Tuple of (TfidfVectorizer, feature matrix)
        """
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=5,
            max_df=0.7
        )
        
        features = vectorizer.fit_transform(texts)
        feature_df = pd.DataFrame(
            features.toarray(),
            columns=vectorizer.get_feature_names_out()
        )
        
        return vectorizer, feature_df

def load_sample_data() -> pd.DataFrame:
    """
    Load sample data for testing (creates synthetic data if no real data available)
    
    Returns:
        DataFrame with sample reviews
    """
    sample_reviews = [
        "This product is amazing! Great quality and fast delivery.",
        "Terrible experience. Product arrived damaged and customer service was unhelpful.",
        "Good value for money. Would recommend to others.",
        "Disappointed with the quality. Not worth the price.",
        "Excellent service and product quality. Very satisfied!",
        "Poor packaging led to damaged items. Very frustrating.",
        "Love this product! Exceeds expectations.",
        "Average product, nothing special.",
        "Outstanding customer support and fast shipping!",
        "Product broke after a week. Waste of money."
    ]
    
    sample_labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 for positive, 0 for negative
    
    return pd.DataFrame({
        'review_text': sample_reviews,
        'sentiment': sample_labels
    })

if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = TextPreprocessor()
    
    # Test with sample data
    sample_data = load_sample_data()
    print("Original text:", sample_data['review_text'].iloc[0])
    
    processed_text = preprocessor.preprocess_text(sample_data['review_text'].iloc[0])
    print("Processed text:", processed_text)
    
    # Test batch processing
    processed_texts = preprocessor.preprocess_batch(sample_data['review_text'].tolist())
    print(f"\nProcessed {len(processed_texts)} texts successfully!") 