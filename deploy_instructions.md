# Deployment Instructions 🚀

## Option 1: Streamlit Cloud (Easiest & Free)

1. **Upload to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit - Sentiment Analysis App"
   git branch -M main
   git remote add origin https://github.com/yourusername/sentiment-analysis-app.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Connect your GitHub repository
   - Select your repository and branch
   - Set main file path to `app.py`
   - Click "Deploy"

3. **Your app will be live** at: `https://yourusername-sentiment-analysis-app-app-xyz.streamlit.app/`

## Option 2: Heroku (Free Tier Available)

1. **Create Heroku Account**: Sign up at [heroku.com](https://heroku.com)

2. **Install Heroku CLI**: Download from [devcenter.heroku.com](https://devcenter.heroku.com/articles/heroku-cli)

3. **Deploy**:
   ```bash
   heroku login
   heroku create your-sentiment-app
   git add .
   git commit -m "Deploy to Heroku"
   git push heroku main
   ```

## Option 3: Railway (Modern Alternative)

1. **Create Railway Account**: Sign up at [railway.app](https://railway.app)
2. **Connect GitHub**: Link your repository
3. **Deploy**: Railway will automatically detect your Python app

## Option 4: Local Sharing (For Testing)

```bash
streamlit run app.py --server.port 8501
# Use ngrok for temporary public URL
ngrok http 8501
```

## Model Files Setup

**Important**: Your trained models need to be accessible. Options:

1. **Upload to GitHub LFS** (for files < 100MB)
2. **Use cloud storage** (Google Drive, AWS S3, etc.)
3. **Include model loading logic** in your app

## Environment Variables

For production deployment, set these environment variables:
- `MODEL_PATH`: Path to your trained LSTM model
- `WORD2VEC_PATH`: Path to your Word2Vec model

## Post-Deployment Checklist

✅ App loads without errors  
✅ Sample predictions work  
✅ UI is responsive  
✅ All features functional  
✅ README updated with live URL  
✅ Added to resume/portfolio  

## Resume/Portfolio Integration

**For your resume, include**:
- 🔗 **Live Demo**: [Your App URL]
- 🐱 **Source Code**: [GitHub Repository]
- 📊 **Accuracy**: 86%+ on test data
- ⚡ **Tech Stack**: Python, TensorFlow, LSTM, Word2Vec, Streamlit

**Project Description Example**:
*"Developed a deep learning web application for movie review sentiment analysis using LSTM neural networks and Word2Vec embeddings. Achieved 86%+ accuracy on 50,000 movie reviews. Built interactive demo with Streamlit and deployed to cloud platform."*