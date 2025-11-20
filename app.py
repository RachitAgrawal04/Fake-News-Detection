from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import re
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Initialize App
app = Flask(__name__)

# Load Model and Vectorizer
model = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Ensure NLTK data is downloaded
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('omw-1.4', quiet=True)

def preprocess_text(text):
    """
    Preprocess the input text exactly as done in the notebook.
    Steps: lowercase -> remove punctuation/links -> tokenize -> lemmatize -> remove stopwords -> join
    """
    # 1. Lowercase
    text = str(text).lower()
    
    # 2. Remove punctuation/links (Regex from your notebook)
    text = re.sub(r'[^\w\s,]', '', text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = text.replace(' ,', ',').replace(', ', ',')
    
    # 3. Tokenize
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    
    # 4. Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(y) for y in tokens]
    
    # 5. Stopwords
    stop = stopwords.words('english')
    tokens = [item for item in tokens if item not in stop]
    
    # 6. Join back
    return ' '.join(tokens)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Please provide "text" field in JSON body'}), 400
        
        user_text = data['text']
        
        # Preprocess
        clean_text = preprocess_text(user_text)
        
        # Vectorize
        # Note: toarray() is needed because your model was trained on dense arrays
        vectorized_text = vectorizer.transform([clean_text]).toarray()
        
        # Predict
        prediction = model.predict(vectorized_text)
        proba = model.predict_proba(vectorized_text)
        
        # Map result (1 = True/Mostly True, 0 = False)
        result = "Real/True" if prediction[0] == 1 else "Fake/False"
        confidence = float(max(proba[0]))
        
        return jsonify({
            'prediction': result, 
            'class': int(prediction[0]),
            'confidence': round(confidence * 100, 2),
            'probabilities': {
                'fake': round(float(proba[0][0]) * 100, 2),
                'real': round(float(proba[0][1]) * 100, 2)
            },
            'preprocessed_text': clean_text[:100] + '...' if len(clean_text) > 100 else clean_text
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Fake News Detector running at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
