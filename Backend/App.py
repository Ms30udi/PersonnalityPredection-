# ============================================
# 🧠 PERSONALITY PREDICTOR - WITH FULL COMMENTS
# ============================================

# Import libraries for data manipulation and preprocessing
import pandas as pd                 # For handling data in table format (CSV files)
import re                           # For text cleaning using regular expressions
import nltk                         # Natural Language Toolkit for text processing
from nltk.corpus import stopwords   # List of common words to ignore (like "the", "is", etc.)
from nltk.tokenize import word_tokenize  # Split text into words (tokens)
from nltk.stem import WordNetLemmatizer  # Reduces words to their base form (e.g., "running" → "run")

# Import libraries for machine learning
from sklearn.preprocessing import LabelEncoder          # Converts text labels into numbers
from sklearn.model_selection import train_test_split    # Splits data into training and testing sets
from sklearn.linear_model import LogisticRegression     # Simple but powerful ML algorithm
from sklearn.metrics import accuracy_score, classification_report  # For model evaluation

# Import SentenceTransformer to turn text into numerical embeddings
from sentence_transformers import SentenceTransformer

# Download necessary NLTK data (only runs once per environment)
nltk.download('punkt')        # For tokenizing text
nltk.download('stopwords')    # For removing common English words
nltk.download('wordnet')      # For lemmatization (word root extraction)

# ============================================
# STEP 3: LOAD AND INSPECT DATA
# ============================================

# Load dataset from CSV file
df = pd.read_csv("PersonalityPredict_texts.csv")

# Remove any unwanted spaces from column names
df.columns = df.columns.str.strip()

# Display confirmation and basic dataset info
print("✅ Data loaded successfully!")
print(f"Total samples: {len(df)}")     # Print total number of rows
print(df.head())                       # Show first few rows for inspection

# ============================================
# STEP 4: TEXT CLEANING FUNCTION
# ============================================

# Load English stopwords into a set for quick lookup
stop_words = set(stopwords.words('english'))

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Define a custom function to clean text
def clean_text(text):
    """
    Clean the input text by performing:
    - Lowercasing
    - Removing URLs, numbers, punctuation
    - Tokenizing
    - Removing stopwords
    - Lemmatizing words
    """
    text = text.lower()                                 # Convert all text to lowercase
    text = re.sub(r"http\S+|www\S+", "", text)          # Remove URLs
    text = re.sub(r"[^a-z\s]", "", text)                # Remove punctuation, numbers, symbols
    tokens = word_tokenize(text)                        # Split text into words
    tokens = [lemmatizer.lemmatize(word)                # Convert each word to its base form
              for word in tokens if word not in stop_words]  # Exclude stopwords
    return " ".join(tokens)                             # Join cleaned words back into one string

# Apply cleaning to all text data
df['text'] = df['text'].astype(str).apply(clean_text)

# Remove rows that become empty after cleaning
df = df[df['text'].str.strip() != ""]
print(f"✅ Samples after cleaning: {len(df)}")

# ============================================
# STEP 5: TEXT TO EMBEDDINGS (SentenceTransformer)
# ============================================

# Load a pre-trained model to convert sentences into numerical vectors
emb_model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode all cleaned text into embeddings (numerical format for ML)
X = emb_model.encode(df['text'], show_progress_bar=True)

# Encode target labels (personality types) into numeric form
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['personality'])

# Show which personality types were found
print(f"Personality types found: {list(label_encoder.classes_)}")

# ============================================
# STEP 6: TRAIN / TEST SPLIT
# ============================================

# Split the dataset into training (70%) and testing (30%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y  # stratify ensures balanced class distribution
)

# ============================================
# STEP 7: TRAIN LOGISTIC REGRESSION MODEL
# ============================================

# Initialize logistic regression model with high iteration limit
model = LogisticRegression(max_iter=2000, solver='lbfgs', multi_class='auto')

# Train the model on training data
model.fit(X_train, y_train)
print("✅ Model trained successfully!")

# ============================================
# STEP 8: EVALUATE MODEL
# ============================================

# Predict personality labels for test data
preds = model.predict(X_test)

# Calculate accuracy score
accuracy = accuracy_score(y_test, preds)
print(f"\n📊 Model Accuracy: {accuracy * 100:.2f}%")

# Print detailed performance metrics for each class
print("\n📈 Classification Report:")
print(classification_report(y_test, preds, target_names=label_encoder.classes_))

# ============================================
# STEP 9: USER INPUT PREDICTION FUNCTION
# ============================================

def predict_personality(user_text):
    """
    Predict the personality type of a user based on their written text.
    """
    # Check for empty input
    if not user_text or user_text.strip() == "":
        return "❌ Cannot predict: Empty text. Please describe yourself."
    
    # Clean the input text
    clean_input = clean_text(user_text)
    
    # Convert cleaned text into embedding
    user_vector = emb_model.encode([clean_input])
    
    # Predict class (numerical label)
    pred_num = model.predict(user_vector)[0]
    
    # Convert number back into actual label (e.g., 'introvert')
    pred_label = label_encoder.inverse_transform([pred_num])[0]
    
    # Get confidence score (probability of prediction)
    confidence = model.predict_proba(user_vector)[0]
    max_conf = max(confidence) * 100
    
    # Return formatted prediction with confidence
    return f"✅ Predicted Personality Type: {pred_label} (Confidence: {max_conf:.1f}%)"

# ============================================
# STEP 10: INTERACTIVE TEST
# ============================================

print("\n" + "="*60)
print("🌟 PERSONALITY MIRROR - Predict Your Personality Type 🌟")
print("="*60)

# Ask user to enter a short paragraph
user_input = input("\n✍️  Enter a short paragraph about yourself:\n> ")

# Predict and display the result
result = predict_personality(user_input)

print("\n" + "-"*60)
print(result)
print("-"*60)
