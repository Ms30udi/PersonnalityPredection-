from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Download NLTK data (only needed once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Global variables for model and preprocessing
model = None
emb_model = None
label_encoder = None
stop_words = None
lemmatizer = None

def clean_text(text):
    """Clean and preprocess text"""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

def load_and_train_model():
    """Load data and train the model"""
    global model, emb_model, label_encoder, stop_words, lemmatizer
    
    print("🔄 Loading model...")
    
    # Initialize preprocessing tools
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    # Load and clean data
    df = pd.read_csv("PersonalityPredict_texts.csv")
    df.columns = df.columns.str.strip()
    df['text'] = df['text'].astype(str).apply(clean_text)
    df = df[df['text'].str.strip() != ""]
    
    # Load embedding model
    emb_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Create embeddings
    X = emb_model.encode(df['text'].tolist(), show_progress_bar=True)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['personality'])
    
    # Train model
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    model = LogisticRegression(max_iter=2000, solver='lbfgs', multi_class='auto')
    model.fit(X_train, y_train)
    
    print("✅ Model trained successfully!")
    print(f"📊 Personality types: {list(label_encoder.classes_)}")

# Load model on startup
load_and_train_model()

@app.route('/')
def home():
    """Health check endpoint"""
    return jsonify({
        "status": "running",
        "message": "Personality Prediction API is running",
        "version": "1.0.0"
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict personality from text"""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                "error": "Missing 'text' field in request body"
            }), 400
        
        user_text = data['text']
        
        # Validate input
        if not user_text or user_text.strip() == "":
            return jsonify({
                "error": "Text cannot be empty"
            }), 400
        
        # Clean and encode text
        clean_input = clean_text(user_text)
        
        if not clean_input or clean_input.strip() == "":
            return jsonify({
                "error": "Text is too short or contains no valid words"
            }), 400
        
        user_vector = emb_model.encode([clean_input])
        
        # Predict
        pred_num = model.predict(user_vector)[0]
        pred_label = label_encoder.inverse_transform([pred_num])[0]
        
        # Get probabilities for all classes
        probabilities = model.predict_proba(user_vector)[0]
        confidence = float(max(probabilities) * 100)
        
        # Create response with all personality scores
        all_predictions = []
        for i, personality in enumerate(label_encoder.classes_):
            all_predictions.append({
                "personality": personality,
                "confidence": float(probabilities[i] * 100)
            })
        
        # Sort by confidence
        all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        return jsonify({
            "success": True,
            "prediction": {
                "personality": pred_label,
                "confidence": confidence
            },
            "all_scores": all_predictions,
            "cleaned_text": clean_input
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

@app.route('/api/personalities', methods=['GET'])
def get_personalities():
    """Get list of all personality types"""
    try:
        return jsonify({
            "success": True,
            "personalities": list(label_encoder.classes_)
        })
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Detailed health check"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "embedding_model_loaded": emb_model is not None,
        "personalities_count": len(label_encoder.classes_) if label_encoder else 0
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Starting Personality Prediction API Server")
    print("="*60)
    print("📍 API running at: http://localhost:5000")
    print("📍 Prediction endpoint: http://localhost:5000/api/predict")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)