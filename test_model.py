import joblib
from app import preprocess_text

# Load model and vectorizer
model = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Test cases
tests = [
    "Earth is round",
    "The moon is made of cheese",
    "Scientists confirm climate change is real",
    "Aliens invaded New York yesterday"
]

print("=" * 80)
print("MODEL DIAGNOSIS")
print("=" * 80)
print(f"Model classes: {model.classes_}")
print()

for text in tests:
    clean = preprocess_text(text)
    vec = vectorizer.transform([clean]).toarray()
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    
    print(f"Text: {text}")
    print(f"  Prediction: {pred}")
    print(f"  Probabilities: {proba}")
    print(f"  proba[0] (class {model.classes_[0]}): {proba[0]:.4f}")
    print(f"  proba[1] (class {model.classes_[1]}): {proba[1]:.4f}")
    print()
