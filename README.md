<<<<<<< HEAD
# Sentiment Analysis for Customer Reviews

## 📌 Project Overview
This project applies NLP to analyze customer feedback and extract actionable business insights.
It uses *ML, Deep Learning, and Transformers (BERT)* for sentiment classification.

## 🚀 Key Features
- Text preprocessing: tokenization, lemmatization, stopword removal
- Models: Naive Bayes + TF-IDF, LSTM, DistilBERT fine-tuning
- Word clouds & frequency plots of positive vs negative sentiment
- Streamlit dashboard to explore sentiment by category, product, and region
- SHAP explainability for NLP models

## 📂 Project Structure
```
sentiment-analysis/
├── data/                   # Dataset storage
├── models/                 # Trained model files
├── notebooks/              # Jupyter notebooks for exploration
├── src/                    # Source code
│   ├── preprocessing.py    # Text preprocessing utilities
│   ├── models.py          # Model implementations
│   ├── evaluation.py      # Model evaluation metrics
│   └── visualization.py   # Plotting and visualization
├── dashboard/              # Streamlit dashboard
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## ⚙ Tech Stack
- Python (NLTK, spaCy, scikit-learn)
- TensorFlow/Keras (LSTM)
- HuggingFace Transformers (DistilBERT)
- Streamlit / Plotly for dashboard

## 🛠 Installation & Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd sentiment-analysis
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download spaCy model**
```bash
python -m spacy download en_core_web_sm
```

5. **Download NLTK data**
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

## 📊 Dataset
- [Amazon Reviews dataset](https://www.kaggle.com/bittlingmayer/amazonreviews)
- Place the dataset in the `data/` directory

## 🚀 Usage

### Training Models
```bash
python src/train_models.py
```

### Running Dashboard
```bash
streamlit run dashboard/app.py
```

### Model Evaluation
```bash
python src/evaluate_models.py
```

## 📊 Results
- DistilBERT achieved highest accuracy (91%)
- Top negative sentiment drivers: delivery delays, poor packaging
- Positive sentiment driven by product quality & affordability

## 💡 Business Impact
Helped uncover *3 key areas of improvement* in customer experience → logistics, packaging, customer support.

## 📝 License
MIT License 
=======
# Sentiment-Analysis-for-Market-Insights
# Sentiment Analysis for Customer Reviews  

## 📌 Project Overview  
This project applies NLP to analyze customer feedback and extract actionable business insights.  
It uses *ML, Deep Learning, and Transformers (BERT)* for sentiment classification.  

## 🚀 Key Features  
- Text preprocessing: tokenization, lemmatization, stopword removal  
- Models: Naive Bayes + TF-IDF, LSTM, DistilBERT fine-tuning  
- Word clouds & frequency plots of positive vs negative sentiment  
- Streamlit dashboard to explore sentiment by category, product, and region  
- SHAP explainability for NLP models  

## 📂 Dataset  
- [Amazon Reviews dataset](https://www.kaggle.com/bittlingmayer/amazonreviews)  

## ⚙ Tech Stack  
- Python (NLTK, spaCy, scikit-learn)  
- TensorFlow/Keras (LSTM)  
- HuggingFace Transformers (DistilBERT)  
- Streamlit / Plotly for dashboard  

## 📊 Results  
- DistilBERT achieved highest accuracy (91%)  
- Top negative sentiment drivers: delivery delays, poor packaging  
- Positive sentiment driven by product quality & affordability  

## 💡 Business Impact  
Helped uncover *3 key areas of improvement* in customer experience → logistics, packaging, customer support.  

---
>>>>>>> 7a4038dc9cc5921eea15237bd1518e63a298a8d4
