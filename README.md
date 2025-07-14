# Movie Review Sentiment Analyzer 🎬

A deep learning web application that analyzes the sentiment of movie reviews using LSTM neural networks and Word2Vec embeddings.

## 🚀 Live Demo

[Your Deployed App URL]

## 📊 Features

- **Real-time sentiment analysis** of movie reviews
- **Deep Learning Model**: LSTM + Word2Vec embeddings
- **Interactive web interface** built with Streamlit
- **86%+ accuracy** on test data
- **Confidence scoring** for predictions
- **Text preprocessing** pipeline

## 🧠 Model Architecture

- **Word2Vec**: 300-dimensional word embeddings
- **LSTM**: Sequential neural network with 128 and 64 units
- **Training Data**: 50,000 movie reviews
- **Accuracy**: 86.33% on test set

## 🛠️ Technical Stack

- **Framework**: Streamlit
- **Deep Learning**: TensorFlow/Keras
- **NLP**: NLTK, Gensim
- **Data Processing**: Pandas, NumPy
- **Deployment**: Streamlit Cloud / Heroku

## 📁 Project Structure

```
sentiment-analysis/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── models/               # Trained models (not included in repo)
│   ├── lstm_model.h5
│   └── word2vec_model.model
├── notebooks/            # Jupyter notebooks
│   ├── data_preprocessing.ipynb
│   └── model_training.ipynb
└── README.md
```

## 🔧 Local Setup

1. **Clone the repository**
```bash
git clone [your-repo-url]
cd sentiment-analysis
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run app.py
```

## 📈 Model Performance

| Metric | Score |
|--------|-------|
| Training Accuracy | 89.02% |
| Validation Accuracy | 86.50% |
| Test Accuracy | 86.33% |

## 🚀 Deployment

### Streamlit Cloud (Recommended)
1. Push your code to GitHub
2. Connect your GitHub repo to [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy with one click!

### Heroku
1. Install Heroku CLI
2. Create a `Procfile` with: `web: sh setup.sh && streamlit run app.py`
3. Deploy to Heroku

## 📊 Data & Training

- **Dataset**: 50,000 movie reviews from IMDB
- **Preprocessing**: HTML removal, punctuation cleaning, tokenization, lemmatization
- **Word Embeddings**: Word2Vec (300 dimensions, window=10)
- **Model**: Bidirectional LSTM with dropout
- **Training**: 5 epochs, Adam optimizer

## 🤝 Contributing

Feel free to fork this project and submit pull requests!

## 📧 Contact

- **Email**: your.email@example.com
- **LinkedIn**: [Your LinkedIn Profile]
- **GitHub**: [Your GitHub Profile]

## 📄 License

This project is licensed under the MIT License.