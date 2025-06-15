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
from datetime import datetime

app = Flask(__name__)

# Configure Gemini API
GOOGLE_API_KEY = "AIzaSyC5YwiSnkg8gj1DNPLVv7gZCrsv7vIk4V0"
client = genai.Client(api_key=GOOGLE_API_KEY)

# Load the saved model and tokenizer
ml_model = tf.keras.models.load_model('fake_news_model.h5')
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

MAXLEN = 1000

def extract_keyword(text):
    words = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(words)
    for word, tag in tagged:
        if tag in ('NNP', 'NNPS'):
            return word
    return ' '.join(words[:2])

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
        prompt = f"""Analyze this news text and provide a comprehensive fact check using your knowledge:
        {text}
        
        Provide a detailed response in HTML format with:
        1. Key Facts (bullet points with verification status)
        2. Historical Context (if relevant)
        3. Source Verification
        4. Overall Reliability Assessment
        
        Format:
        - Use <ul> and <li> tags for bullet points
        - Keep each fact concise but informative
        - Maximum 5 key facts
        - Total response should be under 150 words
        - Use <strong> tags for important terms
        - Use <span style='color: var(--success)'> for verified facts
        - Use <span style='color: var(--danger)'> for disputed facts"""
        
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        html = response.text.strip()
        html = re.sub(r'^```html[\r\n]*', '', html)
        html = re.sub(r'```$', '', html)
        
        # Add Wikipedia image if available
        keyword = extract_keyword(text)
        image_url = get_wikipedia_image(keyword)
        if image_url:
            html += f"<br><img src='{image_url}' alt='Related image' class='img-fluid rounded mt-3' style='max-width: 300px;'>"
        
        return html.strip()
    except Exception as e:
        return f"Error in Gemini analysis: {str(e)}"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400

        text = data['text'].strip()
        if not text:
            return jsonify({'error': 'Empty text provided'}), 400

        # ML Model Analysis
        sequence = tokenizer.texts_to_sequences([text.lower()])
        if not sequence or not sequence[0]:
            return jsonify({'error': 'Unable to tokenize input'}), 400

        padded = pad_sequences(sequence, maxlen=MAXLEN)
        prediction = ml_model.predict(padded)
        confidence = float(prediction[0][0])
        is_fake = confidence < 0.5
        confidence = max(0.0, min(1.0, confidence))
        confidence_percentage = round((1 - confidence) * 100, 2) if is_fake else round(confidence * 100, 2)

        # Gemini Analysis
        gemini_analysis = get_gemini_analysis(text)

        # Extract key entities and topics
        keyword = extract_keyword(text)
        
        return jsonify({
            'is_fake': is_fake,
            'confidence': confidence_percentage,
            'text': text,
            'gemini_analysis': gemini_analysis,
            'keyword': keyword,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True)