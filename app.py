from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from google import genai
import os
import requests
import re
import nltk
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

app = Flask(__name__)

# Configure Gemini API
GOOGLE_API_KEY = "AIzaSyC5YwiSnkg8gj1DNPLVv7gZCrsv7vIk4V0"
client = genai.Client(api_key=GOOGLE_API_KEY)

# Load the saved model and tokenizer
ml_model = tf.keras.models.load_model('fake_news_model.h5')
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

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
        print("Keyword for Wikipedia search:", keyword)
        print("Wikipedia image URL:", image_url)
        if image_url:
            html += f"<br><img src='{image_url}' alt='Related image' width='200'>"
        print("Final HTML sent to frontend:", html)
        return html.strip()
    except Exception as e:
        return f"Error in Gemini analysis: {str(e)}"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400

        text = data['text'].strip().lower()
        if not text:
            return jsonify({'error': 'Empty text provided'}), 400

        sequence = tokenizer.texts_to_sequences([text])
        if not sequence or not sequence[0]:
            return jsonify({'error': 'Unable to tokenize input'}), 400

        padded = pad_sequences(sequence, maxlen=MAXLEN)
        prediction = ml_model.predict(padded)
        confidence = float(prediction[0][0])
        is_fake = confidence < 0.5
        # Ensure confidence is between 0 and 1 before converting to percentage
        confidence = max(0.0, min(1.0, confidence))
        confidence_percentage = round((1 - confidence) * 100, 2) if is_fake else round(confidence * 100, 2)

        gemini_analysis = get_gemini_analysis(text)

        return jsonify({
            'is_fake': is_fake,
            'confidence': confidence_percentage,
            'text': text,
            'gemini_analysis': gemini_analysis
        })
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True)