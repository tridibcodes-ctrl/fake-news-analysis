# Fake News Detector

A Streamlit application that uses machine learning and AI to detect fake news articles and provide analysis.

## Features

- Real-time fake news detection using a TensorFlow model
- AI-powered analysis using Google's Gemini model
- Wikipedia image integration for visual context
- Confidence scoring and detailed analysis

## Local Development

1. Clone the repository:
```bash
git clone <your-repository-url>
cd <repository-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:
```bash
streamlit run app.py
```

## Deployment on Streamlit Cloud

1. Push your code to GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Sign in with your GitHub account
4. Click "New app"
5. Select your repository, branch, and main file (app.py)
6. Click "Deploy"

## Required Files

Make sure these files are in your repository:
- `app.py` - The main Streamlit application
- `fake_news_model.h5` - The trained TensorFlow model
- `tokenizer.pkl` - The text tokenizer
- `requirements.txt` - Python dependencies

## Environment Variables

The application uses the following environment variables:
- `GOOGLE_API_KEY` - Your Google Gemini API key

## Note

Make sure your model files (`fake_news_model.h5` and `tokenizer.pkl`) are tracked using Git LFS if they're large files. 