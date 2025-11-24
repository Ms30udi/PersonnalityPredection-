# ============================================
# 🧠 PERSONALITY PREDICTOR WITH WEB SCRAPING
# ============================================

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sentence_transformers import SentenceTransformer
import requests
from bs4 import BeautifulSoup
import time
import os
from datetime import datetime


# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')


# ============================================
# STEP 1: WEB SCRAPING FUNCTIONS
# ============================================

def scrape_personality_data_from_web():
    """
    Scrape personality-related text data from multiple online sources.
    This function collects data from:
    1. Public datasets (via raw GitHub links)
    2. Personality type descriptions
    3. Synthetic generated examples
    
    Returns: pandas DataFrame with 'text' and 'personality' columns
    """
    print("\n" + "="*60)
    print("🌐 STARTING WEB SCRAPING PROCESS")
    print("="*60 + "\n")
    
    all_data = []
    
    # ========================================
    # SOURCE 1: MBTI Dataset from GitHub
    # ========================================
    print("📥 Source 1: Fetching MBTI dataset...")
    try:
        # URL to public MBTI dataset
        dataset_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/MBTI%20Personality%20Types%20500%20Dataset.csv"
        
        # Headers to avoid being blocked
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Make HTTP request
        response = requests.get(dataset_url, headers=headers, timeout=30)
        
        if response.status_code == 200:
            # Save temporarily
            with open('temp_dataset.csv', 'wb') as f:
                f.write(response.content)
            
            # Load with pandas
            df_temp = pd.read_csv('temp_dataset.csv')
            
            # Check columns and rename
            if 'type' in df_temp.columns and 'posts' in df_temp.columns:
                df_temp = df_temp.rename(columns={'type': 'personality', 'posts': 'text'})
                
                # Process posts (split by ||| separator)
                for idx, row in df_temp.head(150).iterrows():  # Limit to 150 for speed
                    personality_type = row['personality']
                    posts = str(row['text']).split('|||')
                    
                    # Take first 5 posts from each person
                    for post in posts[:5]:
                        if len(post.strip()) > 50:  # Only meaningful posts
                            all_data.append({
                                'text': post.strip(),
                                'personality': personality_type
                            })
                
                print(f"   ✅ Scraped {len(all_data)} text samples from MBTI dataset")
            
            # Clean up temp file
            os.remove('temp_dataset.csv')
        else:
            print(f"   ⚠️ Failed to fetch data (Status code: {response.status_code})")
    
    except Exception as e:
        print(f"   ⚠️ Error scraping MBTI data: {e}")
    
    
    # ========================================
    # SOURCE 2: Personality Type Descriptions
    # ========================================
    print("\n📥 Source 2: Collecting personality descriptions...")
    try:
        # Dictionary of personality types with characteristic statements
        personality_examples = {
            'INTJ': [
                "I prefer working alone on complex strategic problems",
                "Logic and efficiency are more important than feelings",
                "I plan everything meticulously and hate improvisation",
                "Intellectual debates fascinate me more than small talk",
                "I value competence above all else in colleagues"
            ],
            'ENFP': [
                "I love meeting new people and hearing their stories",
                "Creative brainstorming sessions energize me completely",
                "I follow my heart and trust my intuition strongly",
                "Exploring new possibilities excites me every day",
                "I care deeply about authentic emotional connections"
            ],
            'ISTJ': [
                "I rely on proven methods and detailed procedures",
                "Organization and structure help me work efficiently",
                "I prefer facts and concrete data over theories",
                "Following rules and maintaining order is essential",
                "Responsibility and duty guide all my decisions"
            ],
            'ENTP': [
                "Debating ideas and challenging assumptions thrills me",
                "I see patterns and connections others miss easily",
                "Innovation and experimenting with concepts excites me",
                "Conventional thinking bores me, I seek originality",
                "Playing devil's advocate helps clarify truth"
            ],
            'INFP': [
                "My personal values guide every decision I make",
                "I dream about ways to make world better",
                "Creative expression helps me process my emotions",
                "Authentic connections mean everything to me daily",
                "I search for deeper meaning in all experiences"
            ],
            'ESTJ': [
                "I take charge quickly and get things done",
                "Practical results matter more than abstract ideas",
                "Clear rules and structure create effective systems",
                "I make decisions based on objective analysis",
                "Efficiency and productivity drive my daily work"
            ],
            'ISFP': [
                "I live fully in the present moment always",
                "Artistic expression allows me to share my soul",
                "I value personal freedom and authentic experiences",
                "Beauty in nature inspires me every single day",
                "I prefer showing love through actions not words"
            ],
            'ENTJ': [
                "I naturally lead and implement strategic visions",
                "Building efficient systems satisfies me deeply",
                "Long-term planning comes naturally to my mind",
                "I make quick decisions based on logical analysis",
                "Achieving ambitious goals motivates me constantly"
            ],
            'INFJ': [
                "I understand people's emotions and motivations deeply",
                "Helping others find their path fulfills me",
                "I have vivid intuitions about future outcomes",
                "Meaningful one-on-one conversations energize me",
                "I need alone time to recharge after socializing"
            ],
            'ESTP': [
                "I thrive on excitement and spontaneous adventures",
                "Taking action beats endless planning every time",
                "I read people and situations quickly accurately",
                "Physical activities and hands-on work suit me",
                "I adapt easily to whatever situation arises"
            ],
            'INTP': [
                "Understanding complex systems fascinates my mind completely",
                "I analyze ideas from every possible angle",
                "Abstract theories interest me more than practical tasks",
                "Logical consistency matters most in all arguments",
                "I question assumptions others take for granted"
            ],
            'ESFJ': [
                "Helping others and creating harmony fulfills me",
                "I remember details about people I care about",
                "Social traditions and celebrations bring joy daily",
                "I work hard to meet everyone's needs always",
                "Maintaining relationships requires constant effort from me"
            ],
            'ISTP': [
                "I solve mechanical problems through hands-on work",
                "Understanding how things work fascinates me deeply",
                "I stay calm under pressure in crisis situations",
                "Freedom and flexibility matter more than structure",
                "I learn best by doing not reading instructions"
            ],
            'ESFP': [
                "I bring energy and fun to every situation",
                "Living spontaneously makes life worth living daily",
                "I connect with people through shared experiences",
                "Entertainment and making others laugh fulfills me",
                "I focus on enjoying the present moment fully"
            ],
            'ISFJ': [
                "I remember and honor important traditions carefully",
                "Helping others quietly brings me deep satisfaction",
                "I create warm supportive environments for everyone",
                "Practical caring actions show my love best",
                "I work steadily behind scenes without recognition"
            ],
            'ENFJ': [
                "I inspire and motivate people toward growth",
                "Understanding emotions helps me guide others effectively",
                "Creating harmony in groups comes naturally to me",
                "I see potential in everyone I meet daily",
                "Leading with empathy creates lasting positive change"
            ]
        }
        
        for personality, statements in personality_examples.items():
            for statement in statements:
                all_data.append({
                    'text': statement,
                    'personality': personality
                })
        
        total_descriptions = sum(len(v) for v in personality_examples.values())
        print(f"   ✅ Added {total_descriptions} personality descriptions")
    
    except Exception as e:
        print(f"   ⚠️ Error adding descriptions: {e}")
    
    
    # ========================================
    # SOURCE 3: Additional Synthetic Data
    # ========================================
    print("\n📥 Source 3: Generating additional training samples...")
    try:
        # Create more varied examples for each type
        additional_samples = {
            'INTJ': [
                "Strategic planning sessions are my favorite activity",
                "I optimize every process for maximum efficiency always"
            ],
            'ENFP': [
                "Random conversations with strangers inspire my creativity",
                "I start many projects following my passionate interests"
            ],
            'ISTJ': [
                "I maintain detailed records of everything important",
                "Reliable consistent routines help me stay productive"
            ]
        }
        
        for personality, samples in additional_samples.items():
            for sample in samples:
                all_data.append({
                    'text': sample,
                    'personality': personality
                })
        
        print(f"   ✅ Generated additional training samples")
    
    except Exception as e:
        print(f"   ⚠️ Error generating samples: {e}")
    
    
    # Convert to DataFrame
    if len(all_data) > 0:
        df = pd.DataFrame(all_data)
        
        # Save scraped data for future use
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cache_file = f"scraped_personality_data_{timestamp}.csv"
        df.to_csv(cache_file, index=False)
        
        print("\n" + "="*60)
        print(f"✅ SCRAPING COMPLETE!")
        print(f"📊 Total samples collected: {len(df)}")
        print(f"💾 Data saved to: {cache_file}")
        print(f"🎯 Personality types found: {df['personality'].nunique()}")
        print("="*60 + "\n")
        
        return df
    else:
        print("\n❌ No data was scraped. Using minimal fallback dataset...\n")
        # Minimal fallback
        fallback = {
            'text': [
                'I love detailed planning and organization',
                'Spontaneous adventures make life exciting',
                'Logical analysis guides my every decision'
            ],
            'personality': ['ISTJ', 'ENFP', 'INTJ']
        }
        return pd.DataFrame(fallback)


# ============================================
# STEP 2: TEXT PREPROCESSING
# ============================================

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """
    Clean and preprocess text by:
    - Converting to lowercase
    - Removing URLs and special characters
    - Tokenizing
    - Removing stopwords
    - Lemmatizing
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-z\s]", "", text)  # Keep only letters
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) 
              for word in tokens if word not in stop_words]
    return " ".join(tokens)


# ============================================
# STEP 3: LOAD AND PREPARE DATA
# ============================================

print("🔄 Loading personality data...")

# Scrape data from web
df = scrape_personality_data_from_web()

# Clean text data
print("🧹 Cleaning text data...")
df.columns = df.columns.str.strip()
df['text'] = df['text'].astype(str).apply(clean_text)
df = df[df['text'].str.strip() != ""]  # Remove empty entries

print(f"✅ Final dataset size: {len(df)} samples")
print(f"📋 Personality distribution:\n{df['personality'].value_counts()}\n")


# ============================================
# STEP 4: CREATE EMBEDDINGS
# ============================================

print("🔄 Loading sentence transformer model...")
emb_model = SentenceTransformer('all-MiniLM-L6-v2')

print("🔄 Creating text embeddings...")
X = emb_model.encode(df['text'].tolist(), show_progress_bar=True)

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['personality'])

print(f"\n📊 Personality types: {list(label_encoder.classes_)}")


# ============================================
# STEP 5: TRAIN MODEL
# ============================================

print("\n🔄 Splitting data into train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("🔄 Training Logistic Regression model...")
model = LogisticRegression(max_iter=2000, solver='lbfgs', multi_class='auto')
model.fit(X_train, y_train)

print("✅ Model trained successfully!\n")


# ============================================
# STEP 6: EVALUATE MODEL
# ============================================

preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)

print("="*60)
print(f"📊 MODEL ACCURACY: {accuracy * 100:.2f}%")
print("="*60)

print("\n📈 Detailed Classification Report:")
print(classification_report(y_test, preds, target_names=label_encoder.classes_))


# ============================================
# STEP 7: PREDICTION FUNCTION
# ============================================

def predict_personality(user_text):
    """
    Predict personality type from user's text input
    Returns formatted prediction with confidence score
    """
    if not user_text or user_text.strip() == "":
        return "❌ Cannot predict: Empty text provided"
    
    # Clean input
    clean_input = clean_text(user_text)
    
    if not clean_input or clean_input.strip() == "":
        return "❌ Cannot predict: Text too short or no valid words"
    
    # Create embedding
    user_vector = emb_model.encode([clean_input])
    
    # Predict
    pred_num = model.predict(user_vector)[0]
    pred_label = label_encoder.inverse_transform([pred_num])[0]
    
    # Get confidence
    probabilities = model.predict_proba(user_vector)[0]
    confidence = max(probabilities) * 100
    
    # Get top 3 predictions
    top_indices = probabilities.argsort()[-3:][::-1]
    top_predictions = []
    for idx in top_indices:
        top_predictions.append(
            f"{label_encoder.classes_[idx]}: {probabilities[idx]*100:.1f}%"
        )
    
    result = f"""
{"="*60}
✅ PREDICTION RESULT
{"="*60}
🎯 Predicted Personality: {pred_label}
📊 Confidence: {confidence:.1f}%

🏆 Top 3 Predictions:
   1. {top_predictions[0]}
   2. {top_predictions[1]}
   3. {top_predictions[2]}
{"="*60}
"""
    return result


# ============================================
# STEP 8: INTERACTIVE TEST
# ============================================

print("\n" + "="*60)
print("🌟 PERSONALITY MIRROR - Predict Your Personality Type")
print("🌐 Powered by Web-Scraped Data")
print("="*60)

user_input = input("\n✍️  Describe yourself in a few sentences:\n> ")

result = predict_personality(user_input)
print(result)
