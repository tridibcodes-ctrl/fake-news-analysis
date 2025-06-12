import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from google import genai
import requests
import re
import nltk
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# Configure Gemini API
GOOGLE_API_KEY = "AIzaSyC5YwiSnkg8gj1DNPLVv7gZCrsv7vIk4V0"
client = genai.Client(api_key=GOOGLE_API_KEY)

# Load the saved model and tokenizer
@st.cache_resource
def load_model():
    ml_model = tf.keras.models.load_model('fake_news_model.h5')
    with open('tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return ml_model, tokenizer

ml_model, tokenizer = load_model()
MAXLEN = 1000  # Same as in the training script

# Function to extract a keyword (proper noun or first two words)
def extract_keyword(text):
    words = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(words)
    for word, tag in tagged:
        if tag in ('NNP', 'NNPS'):
            return word
    # fallback: first two words
    return ' '.join(words[:2])

# Function to get image from Wikipedia
def get_wikipedia_image(query):
    search_url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "prop": "pageimages",
        "titles": query,
        "pithumbsize": 400
    }
    try:
        response = requests.get(search_url, params=params, timeout=5).json()
        pages = response.get("query", {}).get("pages", {})
        for page in pages.values():
            if "thumbnail" in page:
                return page["thumbnail"]["source"]
    except Exception:
        pass
    return None

def get_gemini_analysis(text):
    try:
        prompt = f"""Analyze this news text briefly:
        {text}
        
        Provide a concise response in HTML format with:
        1. Key fact check (1-2 sentences)
        2. Reliability score (0-100)
        
        Keep the response under 100 words.
        Use HTML tags for formatting instead of markdown."""
        
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        html = response.text.strip()
        # Remove code block markers if present
        html = re.sub(r'^```html[\r\n]*', '', html)
        html = re.sub(r'```$', '', html)
        # Add Wikipedia image
        keyword = extract_keyword(text)
        image_url = get_wikipedia_image(keyword)
        if image_url:
            html += f"<br><img src='{image_url}' alt='Related image' width='200'>"
        return html.strip()
    except Exception as e:
        return f"Error in Gemini analysis: {str(e)}"

# Streamlit UI
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

st.title("📰 Fake News Detector")
st.write("Enter a news article or text to analyze its authenticity.")

# Text input
text_input = st.text_area("Enter the news text here:", height=200)

if st.button("Analyze"):
    if text_input:
        with st.spinner("Analyzing..."):
            # Tokenize and predict
            sequence = tokenizer.texts_to_sequences([text_input.lower()])
            if sequence and sequence[0]:
                padded = pad_sequences(sequence, maxlen=MAXLEN)
                prediction = ml_model.predict(padded)
                confidence = float(prediction[0][0])
                is_fake = confidence < 0.5
                confidence = max(0.0, min(1.0, confidence))
                confidence_percentage = round((1 - confidence) * 100, 2) if is_fake else round(confidence * 100, 2)

                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Prediction Results")
                    if is_fake:
                        st.error(f"⚠️ This news appears to be FAKE")
                    else:
                        st.success(f"✅ This news appears to be REAL")
                    st.metric("Confidence", f"{confidence_percentage}%")

                with col2:
                    st.subheader("AI Analysis")
                    gemini_analysis = get_gemini_analysis(text_input)
                    st.markdown(gemini_analysis, unsafe_allow_html=True)
            else:
                st.error("Unable to process the input text. Please try again with different text.")
    else:
        st.warning("Please enter some text to analyze.")

# Add footer
st.markdown("---")
st.markdown("Built with Streamlit • Powered by TensorFlow and Google Gemini")