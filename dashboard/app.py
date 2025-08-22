"""
Streamlit Dashboard for Sentiment Analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocessing import TextPreprocessor, load_sample_data
from models import NaiveBayesModel, LSTMModel, DistilBERTModel
from visualization import SentimentVisualizer, create_interactive_wordcloud

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load sample data"""
    return load_sample_data()

@st.cache_resource
def load_models():
    """Load trained models if available"""
    models = {}
    
    # Try to load Naive Bayes model
    try:
        nb_model = NaiveBayesModel()
        nb_model.load_model('models/naive_bayes_model.pkl')
        models['naive_bayes'] = nb_model
    except:
        pass
    
    # Try to load LSTM model
    try:
        lstm_model = LSTMModel()
        lstm_model.load_model('models/lstm_model.pkl')
        models['lstm'] = lstm_model
    except:
        pass
    
    # Try to load BERT model
    try:
        bert_model = DistilBERTModel()
        bert_model.load_model('models/bert_model')
        models['bert'] = bert_model
    except:
        pass
    
    return models

def main():
    """Main dashboard function"""
    
    # Header
    st.markdown('<h1 class="main-header">📊 Sentiment Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🎛️ Dashboard Controls")
    
    # Load data
    data = load_data()
    
    # Load models
    models = load_models()
    
    # Sidebar options
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["📈 Overview", "🔍 Data Explorer", "🤖 Model Predictions", "📊 Visualizations", "💡 Insights"]
    )
    
    if page == "📈 Overview":
        show_overview(data, models)
    elif page == "🔍 Data Explorer":
        show_data_explorer(data)
    elif page == "🤖 Model Predictions":
        show_model_predictions(data, models)
    elif page == "📊 Visualizations":
        show_visualizations(data)
    elif page == "💡 Insights":
        show_insights(data)

def show_overview(data, models):
    """Show overview page"""
    st.header("📈 Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Reviews",
            value=len(data),
            delta=None
        )
    
    with col2:
        positive_count = len(data[data['sentiment'] == 1])
        st.metric(
            label="Positive Reviews",
            value=positive_count,
            delta=f"{positive_count/len(data)*100:.1f}%"
        )
    
    with col3:
        negative_count = len(data[data['sentiment'] == 0])
        st.metric(
            label="Negative Reviews",
            value=negative_count,
            delta=f"{negative_count/len(data)*100:.1f}%"
        )
    
    with col4:
        st.metric(
            label="Models Available",
            value=len(models),
            delta=None
        )
    
    # Sentiment distribution
    st.subheader("Sentiment Distribution")
    
    sentiment_counts = data['sentiment'].value_counts()
    fig = px.pie(
        values=sentiment_counts.values,
        names=['Negative', 'Positive'],
        color_discrete_map={'Negative': '#ff6b6b', 'Positive': '#4ecdc4'},
        title="Distribution of Sentiment"
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Sample reviews
    st.subheader("Sample Reviews")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Positive Reviews:**")
        positive_samples = data[data['sentiment'] == 1]['review_text'].head(5)
        for i, review in enumerate(positive_samples, 1):
            st.write(f"{i}. {review}")
    
    with col2:
        st.write("**Negative Reviews:**")
        negative_samples = data[data['sentiment'] == 0]['review_text'].head(5)
        for i, review in enumerate(negative_samples, 1):
            st.write(f"{i}. {review}")

def show_data_explorer(data):
    """Show data explorer page"""
    st.header("🔍 Data Explorer")
    
    # Data info
    st.subheader("Dataset Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Dataset Shape:**")
        st.write(f"Rows: {data.shape[0]}")
        st.write(f"Columns: {data.shape[1]}")
    
    with col2:
        st.write("**Column Information:**")
        for col in data.columns:
            st.write(f"- {col}: {data[col].dtype}")
    
    # Data preview
    st.subheader("Data Preview")
    st.dataframe(data.head(10))
    
    # Filtering options
    st.subheader("Filter Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        sentiment_filter = st.selectbox(
            "Filter by Sentiment:",
            ["All", "Positive", "Negative"]
        )
    
    with col2:
        min_length = st.slider(
            "Minimum Review Length:",
            min_value=0,
            max_value=500,
            value=0
        )
    
    # Apply filters
    filtered_data = data.copy()
    
    if sentiment_filter == "Positive":
        filtered_data = filtered_data[filtered_data['sentiment'] == 1]
    elif sentiment_filter == "Negative":
        filtered_data = filtered_data[filtered_data['sentiment'] == 0]
    
    filtered_data = filtered_data[filtered_data['review_text'].str.len() >= min_length]
    
    st.write(f"**Filtered Results:** {len(filtered_data)} reviews")
    st.dataframe(filtered_data.head(10))
    
    # Download filtered data
    csv = filtered_data.to_csv(index=False)
    st.download_button(
        label="Download Filtered Data",
        data=csv,
        file_name=f"filtered_sentiment_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

def show_model_predictions(data, models):
    """Show model predictions page"""
    st.header("🤖 Model Predictions")
    
    if not models:
        st.warning("No trained models found. Please run the training pipeline first.")
        return
    
    # Model selection
    model_name = st.selectbox(
        "Select Model:",
        list(models.keys())
    )
    
    model = models[model_name]
    
    # Single text prediction
    st.subheader("Single Text Prediction")
    
    text_input = st.text_area(
        "Enter text for sentiment analysis:",
        height=100,
        placeholder="Enter your text here..."
    )
    
    if text_input:
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Analyze Sentiment"):
                with st.spinner("Analyzing..."):
                    # Preprocess text
                    preprocessor = TextPreprocessor()
                    processed_text = preprocessor.preprocess_text(text_input)
                    
                    # Make prediction
                    if model_name == 'naive_bayes':
                        # For Naive Bayes, we need TF-IDF features
                        vectorizer, features = preprocessor.create_tfidf_features([processed_text])
                        prediction = model.predict(features)[0]
                        probability = model.predict_proba(features)[0]
                    else:
                        # For LSTM and BERT
                        prediction = model.predict([text_input])[0]
                        probability = model.predict_proba([text_input])[0]
                    
                    # Display results
                    sentiment = "Positive" if prediction == 1 else "Negative"
                    confidence = probability[1] if len(probability) > 1 else probability
                    
                    st.success(f"**Prediction:** {sentiment}")
                    st.metric("Confidence", f"{confidence:.2%}")
        
        with col2:
            # Show prediction details
            if 'prediction' in locals():
                st.write("**Prediction Details:**")
                st.write(f"Model: {model_name}")
                st.write(f"Processed Text: {processed_text}")
    
    # Batch prediction
    st.subheader("Batch Prediction")
    
    uploaded_file = st.file_uploader(
        "Upload CSV file with 'text' column:",
        type=['csv']
    )
    
    if uploaded_file is not None:
        try:
            batch_data = pd.read_csv(uploaded_file)
            
            if 'text' in batch_data.columns:
                st.write("**Preview of uploaded data:**")
                st.dataframe(batch_data.head())
                
                if st.button("Run Batch Prediction"):
                    with st.spinner("Processing batch predictions..."):
                        # Preprocess texts
                        preprocessor = TextPreprocessor()
                        processed_texts = preprocessor.preprocess_batch(batch_data['text'].tolist())
                        
                        # Make predictions
                        if model_name == 'naive_bayes':
                            vectorizer, features = preprocessor.create_tfidf_features(processed_texts)
                            predictions = model.predict(features)
                            probabilities = model.predict_proba(features)
                        else:
                            predictions = model.predict(batch_data['text'].tolist())
                            probabilities = model.predict_proba(batch_data['text'].tolist())
                        
                        # Add predictions to dataframe
                        batch_data['predicted_sentiment'] = predictions
                        batch_data['sentiment_probability'] = [prob[1] if len(prob) > 1 else prob for prob in probabilities]
                        
                        st.write("**Prediction Results:**")
                        st.dataframe(batch_data)
                        
                        # Download results
                        csv = batch_data.to_csv(index=False)
                        st.download_button(
                            label="Download Results",
                            data=csv,
                            file_name=f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
            else:
                st.error("No 'text' column found in uploaded file.")
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

def show_visualizations(data):
    """Show visualizations page"""
    st.header("📊 Visualizations")
    
    # Create visualizer
    visualizer = SentimentVisualizer()
    
    # Visualization options
    viz_option = st.selectbox(
        "Choose Visualization:",
        ["Sentiment Distribution", "Word Clouds", "Word Frequency", "Review Length Analysis"]
    )
    
    if viz_option == "Sentiment Distribution":
        st.subheader("Sentiment Distribution")
        
        fig = visualizer.plot_sentiment_distribution(data)
        st.plotly_chart(fig, use_container_width=True)
        
        # Sentiment by review length
        data['review_length'] = data['review_text'].str.len()
        
        fig = px.scatter(
            data,
            x='review_length',
            y='sentiment',
            color='sentiment',
            title="Sentiment vs Review Length",
            color_discrete_map={0: '#ff6b6b', 1: '#4ecdc4'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_option == "Word Clouds":
        st.subheader("Word Clouds")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Positive Sentiment Word Cloud**")
            positive_texts = data[data['sentiment'] == 1]['review_text'].tolist()
            fig = visualizer.create_wordcloud(positive_texts, sentiment=1)
            st.pyplot(fig)
        
        with col2:
            st.write("**Negative Sentiment Word Cloud**")
            negative_texts = data[data['sentiment'] == 0]['review_text'].tolist()
            fig = visualizer.create_wordcloud(negative_texts, sentiment=0)
            st.pyplot(fig)
    
    elif viz_option == "Word Frequency":
        st.subheader("Word Frequency Analysis")
        
        sentiment_choice = st.selectbox(
            "Choose Sentiment:",
            ["All", "Positive", "Negative"]
        )
        
        if sentiment_choice == "All":
            texts = data['review_text'].tolist()
            sentiment = None
        elif sentiment_choice == "Positive":
            texts = data[data['sentiment'] == 1]['review_text'].tolist()
            sentiment = 1
        else:
            texts = data[data['sentiment'] == 0]['review_text'].tolist()
            sentiment = 0
        
        fig = visualizer.plot_word_frequency(texts, sentiment=sentiment)
        st.pyplot(fig)
    
    elif viz_option == "Review Length Analysis":
        st.subheader("Review Length Analysis")
        
        data['review_length'] = data['review_text'].str.len()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                data,
                x='review_length',
                color='sentiment',
                title="Review Length Distribution",
                color_discrete_map={0: '#ff6b6b', 1: '#4ecdc4'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Box plot
            fig = px.box(
                data,
                x='sentiment',
                y='review_length',
                title="Review Length by Sentiment",
                color='sentiment',
                color_discrete_map={0: '#ff6b6b', 1: '#4ecdc4'}
            )
            st.plotly_chart(fig, use_container_width=True)

def show_insights(data):
    """Show insights page"""
    st.header("💡 Business Insights")
    
    # Key insights
    st.subheader("Key Findings")
    
    # Sentiment distribution insights
    positive_pct = len(data[data['sentiment'] == 1]) / len(data) * 100
    negative_pct = len(data[data['sentiment'] == 0]) / len(data) * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"**Positive Reviews:** {positive_pct:.1f}%")
        if positive_pct > 60:
            st.success("✅ High customer satisfaction")
        elif positive_pct > 40:
            st.warning("⚠️ Moderate customer satisfaction")
        else:
            st.error("❌ Low customer satisfaction")
    
    with col2:
        st.info(f"**Negative Reviews:** {negative_pct:.1f}%")
        if negative_pct < 20:
            st.success("✅ Low customer dissatisfaction")
        elif negative_pct < 40:
            st.warning("⚠️ Moderate customer dissatisfaction")
        else:
            st.error("❌ High customer dissatisfaction")
    
    # Review length insights
    st.subheader("Review Length Insights")
    
    data['review_length'] = data['review_text'].str.len()
    
    avg_length_positive = data[data['sentiment'] == 1]['review_length'].mean()
    avg_length_negative = data[data['sentiment'] == 0]['review_length'].mean()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Avg Length (Positive)", f"{avg_length_positive:.0f} chars")
    
    with col2:
        st.metric("Avg Length (Negative)", f"{avg_length_negative:.0f} chars")
    
    # Word analysis
    st.subheader("Top Words Analysis")
    
    # Get top words for each sentiment
    from collections import Counter
    
    positive_texts = ' '.join(data[data['sentiment'] == 1]['review_text'].tolist()).lower()
    negative_texts = ' '.join(data[data['sentiment'] == 0]['review_text'].tolist()).lower()
    
    positive_words = Counter(positive_texts.split())
    negative_words = Counter(negative_texts.split())
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Top Positive Words:**")
        for word, count in positive_words.most_common(10):
            st.write(f"- {word}: {count}")
    
    with col2:
        st.write("**Top Negative Words:**")
        for word, count in negative_words.most_common(10):
            st.write(f"- {word}: {count}")
    
    # Recommendations
    st.subheader("Business Recommendations")
    
    if positive_pct > 60:
        st.success("""
        **Recommendations for High Satisfaction:**
        - Continue current practices that drive positive sentiment
        - Focus on maintaining quality and customer service
        - Consider expanding successful product lines
        """)
    elif positive_pct > 40:
        st.warning("""
        **Recommendations for Moderate Satisfaction:**
        - Identify and address common pain points
        - Improve customer service response times
        - Consider product quality improvements
        """)
    else:
        st.error("""
        **Recommendations for Low Satisfaction:**
        - Conduct detailed customer feedback analysis
        - Implement immediate customer service improvements
        - Review product quality and delivery processes
        - Consider pricing strategy adjustments
        """)

if __name__ == "__main__":
    main() 