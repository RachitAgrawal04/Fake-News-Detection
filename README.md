# Fake News Detector

AI-powered fake news detection using Machine Learning (Gaussian Naive Bayes classifier).

## Quick Start

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run the app:**
```bash
python app.py
```

3. **Open in browser:**
```
http://localhost:5000
```

## Features

- 🔍 Real-time fake news detection
- 🎯 ~60% accuracy using Gaussian Naive Bayes
- 🎨 Modern, responsive web UI
- 📊 Confidence scores and probability breakdown

## API Usage

**POST /predict**
```json
{
  "text": "Your news statement here"
}
```

**Response:**
```json
{
  "prediction": "Real/True",
  "class": 1,
  "confidence": "85.32%",
  "preprocessed_text": "..."
}
```

## Dataset

LIAR dataset with 12,788 political statements labeled by PolitiFact.

## Model

- **Algorithm:** Gaussian Naive Bayes
- **Features:** Bag of Words (2000 max features)
- **Preprocessing:** Lowercase, regex cleaning, tokenization, lemmatization, stopword removal
