import streamlit as st
import pandas as pd
import numpy as np
import re
import string
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec
import nltk
import pickle
import os

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    pass

# Initialize lemmatizer
wnl = WordNetLemmatizer()

# Preprocessing functions
def remove_html(text):
    CLEANR = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    return CLEANR.sub(r'', text)

def remove_punctuation(text):
    ex = string.punctuation
    for i in ex:
        text = text.replace(i, '')
    return text

def chat_removal(text):
    chat_dict = {
        'AFAIK': 'As Far As I Know', 'AFK': 'Away From Keyboard', 'ASAP': 'As Soon As Possible',
        'ATM': 'At The Moment', 'BRB': 'Be Right Back', 'BTW': 'By The Way', 'CU': 'See You',
        'FYI': 'For Your Information', 'LOL': 'Laughing Out Loud', 'ROFL': 'Rolling On Floor Laughing',
        'TTYL': 'Talk To You Later', 'OMG': 'Oh My God', 'WTF': 'What The Heck', 'IMO': 'In My Opinion'
    }
    
    for key, value in chat_dict.items():
        text = re.sub(rf'\b{key}\b', value, text, flags=re.IGNORECASE)
    return text

def preprocess_text(text):
    text = text.lower()
    text = remove_html(text)
    text = remove_punctuation(text)
    text = chat_removal(text)
    tokens = word_tokenize(text)
    tokens = [wnl.lemmatize(word) for word in tokens]
    return tokens

def review_to_vectors(tokens, model_wv):
    return [model_wv.wv[word] for word in tokens if word in model_wv.wv]

def predict_sentiment(review, model_wv, model, max_len=150):
    # Preprocess the text
    tokens = preprocess_text(review)
    
    # Convert to vectors
    vectors = review_to_vectors(tokens, model_wv)
    
    if len(vectors) == 0:
        return "Unable to process", 0.5
    
    # Pad sequence
    padded = pad_sequences([vectors], maxlen=max_len, dtype='float32', 
                          padding='post', truncating='post', value=0.0)
    
    # Predict
    prediction = model.predict(padded, verbose=0)[0][0]
    sentiment = "Positive 😊" if prediction >= 0.5 else "Negative 😞"
    
    return sentiment, float(prediction)

# Streamlit app configuration
st.set_page_config(
    page_title="Movie Review Sentiment Analyzer",
    page_icon="🎬",
    layout="wide"
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
    .sentiment-positive {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .sentiment-negative {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .confidence-bar {
        width: 100%;
        background-color: #e0e0e0;
        border-radius: 25px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.markdown('<h1 class="main-header">🎬 Movie Review Sentiment Analyzer</h1>', unsafe_allow_html=True)
st.markdown("### Analyze the sentiment of movie reviews using Deep Learning (LSTM + Word2Vec)")

# Sidebar with project information
with st.sidebar:
    st.header("📊 Project Details")
    st.write("**Model Architecture:**")
    st.write("- LSTM Neural Network")
    st.write("- Word2Vec Embeddings (300D)")
    st.write("- Sequence Length: 150")
    st.write("")
    st.write("**Dataset:**")
    st.write("- 50,000 Movie Reviews")
    st.write("- Binary Classification")
    st.write("- Balanced Dataset")
    st.write("")
    st.write("**Accuracy:** ~86%")
    
    st.header("🔗 Links")
    st.write("📧 Contact: your.email@example.com")
    st.write("💼 LinkedIn: your-profile")
    st.write("🐱 GitHub: your-repo")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Enter a Movie Review")
    
    # Sample reviews for quick testing
    sample_reviews = {
        "Positive Example": "This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout. Highly recommended!",
        "Negative Example": "This was one of the worst movies I've ever seen. The plot made no sense and the acting was terrible. Complete waste of time.",
        "Mixed Example": "The movie had some good moments but overall it was just okay. The special effects were impressive but the story was lacking."
    }
    
    selected_sample = st.selectbox("Try a sample review:", ["Custom Input"] + list(sample_reviews.keys()))
    
    if selected_sample != "Custom Input":
        default_text = sample_reviews[selected_sample]
    else:
        default_text = ""
    
    user_review = st.text_area(
        "Movie Review Text:",
        value=default_text,
        height=150,
        placeholder="Enter your movie review here... For example: 'This movie was amazing! Great storyline and excellent acting.'"
    )
    
    analyze_button = st.button("🔍 Analyze Sentiment", type="primary")

with col2:
    st.subheader("How it works")
    st.write("""
    1. **Text Preprocessing**: Clean and tokenize the input
    2. **Word Embeddings**: Convert words to 300D vectors using Word2Vec
    3. **LSTM Processing**: Sequential analysis of the review
    4. **Prediction**: Binary classification (Positive/Negative)
    """)

# Create placeholder models for demonstration (replace with your actual models)
@st.cache_resource
def load_models():
    # For demonstration - replace these with your actual model loading
    # model = load_model("lstm_sentiment_model.h5")
    # model_wv = Word2Vec.load("word2vec_model.model")
    
    # Placeholder - in real implementation, load your trained models
    model = None  # Your LSTM model
    model_wv = None  # Your Word2Vec model
    
    return model, model_wv

# Results section
if analyze_button and user_review.strip():
    with st.spinner("Analyzing sentiment..."):
        try:
            # Load models (replace with actual model loading)
            model, model_wv = load_models()
            
            if model is None or model_wv is None:
                st.warning("⚠️ Demo Mode: Models not loaded. This is a UI demonstration.")
                # Simulate prediction for demo
                import random
                demo_prediction = random.random()
                sentiment = "Positive 😊" if demo_prediction >= 0.5 else "Negative 😞"
                confidence = demo_prediction
            else:
                # Real prediction with loaded models
                sentiment, confidence = predict_sentiment(user_review, model_wv, model)
            
            # Display results
            st.markdown("---")
            st.subheader("📊 Analysis Results")
            
            # Create two columns for results
            res_col1, res_col2 = st.columns([1, 1])
            
            with res_col1:
                if "Positive" in sentiment:
                    st.markdown(f'<div class="sentiment-positive"><h3>{sentiment}</h3></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="sentiment-negative"><h3>{sentiment}</h3></div>', unsafe_allow_html=True)
            
            with res_col2:
                st.metric("Confidence Score", f"{confidence:.3f}")
                
                # Confidence bar
                confidence_percentage = confidence * 100
                if confidence >= 0.5:
                    color = "#28a745"
                    bar_width = confidence_percentage
                else:
                    color = "#dc3545"
                    bar_width = (1 - confidence) * 100
                
                st.markdown(f"""
                <div class="confidence-bar">
                    <div style="width: {bar_width}%; background-color: {color}; height: 20px; 
                               text-align: center; line-height: 20px; color: white; font-weight: bold;">
                        {bar_width:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Additional insights
            st.subheader("💡 Insights")
            
            if confidence > 0.8:
                st.success("🎯 High confidence prediction!")
            elif confidence > 0.6:
                st.info("👍 Good confidence level")
            else:
                st.warning("⚠️ Low confidence - the review might be neutral or ambiguous")
            
            # Word analysis (simplified demonstration)
            tokens = preprocess_text(user_review)
            st.subheader("🔤 Processed Tokens")
            st.write(f"**Token count:** {len(tokens)}")
            st.write("**First 10 tokens:**", tokens[:10] if tokens else "No tokens")
            
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
            st.write("Please check your input and try again.")

elif analyze_button:
    st.warning("⚠️ Please enter a movie review to analyze.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>Built with Streamlit | Deep Learning Sentiment Analysis Project</p>
    <p>This model was trained on 50,000 movie reviews using LSTM neural networks and Word2Vec embeddings</p>
</div>
""", unsafe_allow_html=True)

# Demo data section (expandable)
with st.expander("📈 View Model Training Results"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Training Accuracy", "89.02%")
    with col2:
        st.metric("Validation Accuracy", "86.50%")
    with col3:
        st.metric("Test Accuracy", "86.33%")
    
    st.write("**Training Details:**")
    st.write("- Epochs: 5")
    st.write("- Batch Size: 32")
    st.write("- LSTM Units: 128, 64")
    st.write("- Dropout: 0.1")
    st.write("- Optimizer: Adam")
    st.write("- Loss Function: Binary Crossentropy")