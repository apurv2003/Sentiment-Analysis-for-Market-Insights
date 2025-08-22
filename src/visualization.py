"""
Visualization utilities for sentiment analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
from collections import Counter
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style for matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SentimentVisualizer:
    """
    Comprehensive visualization for sentiment analysis results
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
    
    def plot_sentiment_distribution(self, data: pd.DataFrame, sentiment_col: str = 'sentiment',
                                  title: str = "Sentiment Distribution") -> go.Figure:
        """
        Plot sentiment distribution
        
        Args:
            data: DataFrame with sentiment data
            sentiment_col: Column name for sentiment labels
            title: Plot title
            
        Returns:
            Plotly figure
        """
        sentiment_counts = data[sentiment_col].value_counts()
        
        fig = go.Figure(data=[
            go.Bar(
                x=['Negative', 'Positive'],
                y=sentiment_counts.values,
                marker_color=['#ff6b6b', '#4ecdc4'],
                text=sentiment_counts.values,
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title="Sentiment",
            yaxis_title="Count",
            template="plotly_white",
            showlegend=False
        )
        
        return fig
    
    def plot_sentiment_over_time(self, data: pd.DataFrame, date_col: str, sentiment_col: str = 'sentiment',
                                freq: str = 'D') -> go.Figure:
        """
        Plot sentiment trends over time
        
        Args:
            data: DataFrame with date and sentiment data
            date_col: Column name for dates
            sentiment_col: Column name for sentiment labels
            freq: Frequency for resampling ('D' for daily, 'W' for weekly, etc.)
            
        Returns:
            Plotly figure
        """
        # Convert date column to datetime
        data[date_col] = pd.to_datetime(data[date_col])
        
        # Group by date and sentiment
        daily_sentiment = data.groupby([pd.Grouper(key=date_col, freq=freq), sentiment_col]).size().unstack(fill_value=0)
        
        fig = go.Figure()
        
        colors = ['#ff6b6b', '#4ecdc4']
        labels = ['Negative', 'Positive']
        
        for i, sentiment in enumerate([0, 1]):
            fig.add_trace(go.Scatter(
                x=daily_sentiment.index,
                y=daily_sentiment[sentiment],
                mode='lines+markers',
                name=labels[i],
                line=dict(color=colors[i], width=2),
                marker=dict(size=6)
            ))
        
        fig.update_layout(
            title="Sentiment Trends Over Time",
            xaxis_title="Date",
            yaxis_title="Number of Reviews",
            template="plotly_white",
            hovermode='x unified'
        )
        
        return fig
    
    def create_wordcloud(self, texts: List[str], sentiment: int = None, 
                        max_words: int = 200, figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Create word cloud from text data
        
        Args:
            texts: List of text documents
            sentiment: Sentiment label (0 for negative, 1 for positive, None for all)
            max_words: Maximum number of words to display
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Combine all texts
        combined_text = ' '.join(texts)
        
        # Create word cloud
        wordcloud = WordCloud(
            width=800,
            height=600,
            background_color='white',
            max_words=max_words,
            colormap='viridis' if sentiment is None else ('Reds' if sentiment == 0 else 'Greens'),
            contour_width=1,
            contour_color='steelblue'
        ).generate(combined_text)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        
        sentiment_label = "All" if sentiment is None else ("Negative" if sentiment == 0 else "Positive")
        ax.set_title(f'Word Cloud - {sentiment_label} Sentiment', fontsize=16, pad=20)
        
        return fig
    
    def plot_word_frequency(self, texts: List[str], sentiment: int = None, 
                           top_n: int = 20, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot word frequency analysis
        
        Args:
            texts: List of text documents
            sentiment: Sentiment label (0 for negative, 1 for positive, None for all)
            top_n: Number of top words to display
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Tokenize and count words
        all_words = []
        for text in texts:
            words = text.lower().split()
            all_words.extend(words)
        
        # Count word frequencies
        word_counts = Counter(all_words)
        
        # Get top words
        top_words = word_counts.most_common(top_n)
        words, counts = zip(*top_words)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = 'red' if sentiment == 0 else 'green' if sentiment == 1 else 'blue'
        sentiment_label = "All" if sentiment is None else ("Negative" if sentiment == 0 else "Positive")
        
        bars = ax.barh(range(len(words)), counts, color=colors, alpha=0.7)
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words)
        ax.set_xlabel('Frequency')
        ax.set_title(f'Top {top_n} Words - {sentiment_label} Sentiment', fontsize=16)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                   str(int(width)), ha='left', va='center')
        
        plt.tight_layout()
        return fig
    
    def plot_sentiment_by_category(self, data: pd.DataFrame, category_col: str, 
                                  sentiment_col: str = 'sentiment') -> go.Figure:
        """
        Plot sentiment distribution by category
        
        Args:
            data: DataFrame with category and sentiment data
            category_col: Column name for categories
            sentiment_col: Column name for sentiment labels
            
        Returns:
            Plotly figure
        """
        # Group by category and sentiment
        category_sentiment = data.groupby([category_col, sentiment_col]).size().unstack(fill_value=0)
        
        # Calculate percentages
        category_sentiment_pct = category_sentiment.div(category_sentiment.sum(axis=1), axis=0) * 100
        
        fig = go.Figure()
        
        colors = ['#ff6b6b', '#4ecdc4']
        labels = ['Negative', 'Positive']
        
        for i, sentiment in enumerate([0, 1]):
            fig.add_trace(go.Bar(
                name=labels[i],
                x=category_sentiment_pct.index,
                y=category_sentiment_pct[sentiment],
                marker_color=colors[i]
            ))
        
        fig.update_layout(
            title="Sentiment Distribution by Category",
            xaxis_title="Category",
            yaxis_title="Percentage (%)",
            template="plotly_white",
            barmode='stack'
        )
        
        return fig
    
    def plot_model_comparison(self, model_results: Dict[str, Dict[str, float]]) -> go.Figure:
        """
        Plot model performance comparison
        
        Args:
            model_results: Dictionary of {model_name: {metric: value}}
            
        Returns:
            Plotly figure
        """
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        models = list(model_results.keys())
        
        fig = go.Figure()
        
        for i, metric in enumerate(metrics):
            values = [model_results[model].get(metric, 0) for model in models]
            
            fig.add_trace(go.Bar(
                name=metric.replace('_', ' ').title(),
                x=models,
                y=values,
                text=[f'{v:.3f}' for v in values],
                textposition='auto'
            ))
        
        fig.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Models",
            yaxis_title="Score",
            template="plotly_white",
            barmode='group'
        )
        
        return fig
    
    def plot_confusion_matrix_heatmap(self, cm: np.ndarray, model_name: str = "Model",
                                    figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
        """
        Plot confusion matrix as heatmap
        
        Args:
            cm: Confusion matrix
            model_name: Name of the model
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'],
                   ax=ax)
        
        ax.set_title(f'Confusion Matrix - {model_name}')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        
        return fig
    
    def plot_feature_importance(self, feature_importance: pd.DataFrame, 
                               top_n: int = 20, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot feature importance analysis
        
        Args:
            feature_importance: DataFrame with feature importance data
            top_n: Number of top features to display
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Get top features
        top_features = feature_importance.head(top_n)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        bars = ax.barh(range(len(top_features)), top_features['abs_correlation'], 
                      color='skyblue', alpha=0.7)
        
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Absolute Correlation')
        ax.set_title(f'Top {top_n} Feature Importance', fontsize=16)
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{width:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        return fig
    
    def create_dashboard_summary(self, data: pd.DataFrame, sentiment_col: str = 'sentiment') -> go.Figure:
        """
        Create a comprehensive dashboard summary
        
        Args:
            data: DataFrame with sentiment data
            sentiment_col: Column name for sentiment labels
            
        Returns:
            Plotly figure with subplots
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Sentiment Distribution', 'Sentiment by Rating',
                          'Review Length Distribution', 'Top Words'),
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "histogram"}, {"type": "bar"}]]
        )
        
        # 1. Sentiment Distribution (Pie Chart)
        sentiment_counts = data[sentiment_col].value_counts()
        fig.add_trace(
            go.Pie(labels=['Negative', 'Positive'], values=sentiment_counts.values,
                   marker_colors=['#ff6b6b', '#4ecdc4']),
            row=1, col=1
        )
        
        # 2. Sentiment by Rating (if rating column exists)
        if 'rating' in data.columns:
            rating_sentiment = data.groupby('rating')[sentiment_col].mean()
            fig.add_trace(
                go.Bar(x=rating_sentiment.index, y=rating_sentiment.values,
                       marker_color='lightblue'),
                row=1, col=2
            )
        
        # 3. Review Length Distribution
        if 'review_text' in data.columns:
            review_lengths = data['review_text'].str.len()
            fig.add_trace(
                go.Histogram(x=review_lengths, nbinsx=30, marker_color='lightgreen'),
                row=2, col=1
            )
        
        # 4. Top Words (simplified)
        if 'review_text' in data.columns:
            # Simple word count for demonstration
            all_words = ' '.join(data['review_text'].astype(str)).lower().split()
            word_counts = Counter(all_words)
            top_words = word_counts.most_common(10)
            words, counts = zip(*top_words)
            
            fig.add_trace(
                go.Bar(x=list(words), y=list(counts), marker_color='orange'),
                row=2, col=2
            )
        
        fig.update_layout(
            title="Sentiment Analysis Dashboard",
            template="plotly_white",
            height=800
        )
        
        return fig

def create_interactive_wordcloud(texts: List[str], sentiment: int = None) -> go.Figure:
    """
    Create interactive word cloud using plotly
    
    Args:
        texts: List of text documents
        sentiment: Sentiment label
        
    Returns:
        Plotly figure
    """
    # Combine all texts
    combined_text = ' '.join(texts)
    
    # Create word cloud
    wordcloud = WordCloud(
        width=800,
        height=600,
        background_color='white',
        max_words=100,
        colormap='viridis' if sentiment is None else ('Reds' if sentiment == 0 else 'Greens')
    ).generate(combined_text)
    
    # Convert to plotly
    word_list = []
    freq_list = []
    pos_list = []
    
    for (word, freq), font_size, position, orientation in wordcloud.layout_:
        word_list.append(word)
        freq_list.append(freq)
        pos_list.append(position)
    
    # Create scatter plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=[pos[0] for pos in pos_list],
        y=[pos[1] for pos in pos_list],
        mode='text',
        text=word_list,
        textfont=dict(size=[freq/10 for freq in freq_list]),
        textposition='middle center',
        hoverinfo='text',
        hovertext=[f'{word}: {freq}' for word, freq in zip(word_list, freq_list)]
    ))
    
    sentiment_label = "All" if sentiment is None else ("Negative" if sentiment == 0 else "Positive")
    
    fig.update_layout(
        title=f'Interactive Word Cloud - {sentiment_label} Sentiment',
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        template="plotly_white"
    )
    
    return fig

if __name__ == "__main__":
    # Test the visualizer with sample data
    from preprocessing import load_sample_data
    
    # Load sample data
    data = load_sample_data()
    
    # Create visualizer
    visualizer = SentimentVisualizer()
    
    # Test sentiment distribution plot
    fig = visualizer.plot_sentiment_distribution(data)
    print("Sentiment distribution plot created successfully!")
    
    # Test word cloud
    fig = visualizer.create_wordcloud(data['review_text'].tolist())
    print("Word cloud created successfully!")
    
    # Test word frequency plot
    fig = visualizer.plot_word_frequency(data['review_text'].tolist())
    print("Word frequency plot created successfully!") 