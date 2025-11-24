from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime


# Initialize Flask app
app = Flask(__name__)
CORS(app)


# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')


# Global variables
model = None
vectorizer = None
label_encoder = None
stop_words = None
lemmatizer = None
scraped_data_cache = "scraped_personality_data.csv"


# MBTI Personality Type Descriptions (from 16personalities.com)
PERSONALITY_DESCRIPTIONS = {
    'INTJ': {
        'name': 'Architect',
        'category': 'Analyst',
        'description': 'Imaginative and strategic thinkers, with a plan for everything.',
        'traits': [
            'Analytical and logical problem solvers',
            'Independent and prefer working alone',
            'Long-term strategic planners',
            'Value competence and efficiency',
            'Confident in their ideas and decisions'
        ],
        'strengths': 'Strategic thinking, independence, determination',
        'weaknesses': 'Overly analytical, dismissive of emotions, arrogant'
    },
    'INTP': {
        'name': 'Logician',
        'category': 'Analyst',
        'description': 'Innovative inventors with an unquenchable thirst for knowledge.',
        'traits': [
            'Love analyzing theories and abstract concepts',
            'Question assumptions others take for granted',
            'Enjoy intellectual debates and challenges',
            'Prefer logic over emotions',
            'Curious and always seeking knowledge'
        ],
        'strengths': 'Analytical, original, open-minded',
        'weaknesses': 'Disconnected from emotions, insensitive, absent-minded'
    },
    'ENTJ': {
        'name': 'Commander',
        'category': 'Analyst',
        'description': 'Bold, imaginative and strong-willed leaders, always finding a way – or making one.',
        'traits': [
            'Natural born leaders',
            'Strategic and goal-oriented',
            'Confident and decisive',
            'Enjoy organizing people and resources',
            'Direct and assertive communication'
        ],
        'strengths': 'Efficient, energetic, strong-willed',
        'weaknesses': 'Stubborn, intolerant, impatient'
    },
    'ENTP': {
        'name': 'Debater',
        'category': 'Analyst',
        'description': 'Smart and curious thinkers who cannot resist an intellectual challenge.',
        'traits': [
            'Love debating ideas and playing devil\'s advocate',
            'See patterns and possibilities everywhere',
            'Innovative and original thinkers',
            'Challenge conventional wisdom',
            'Quick-witted and energetic'
        ],
        'strengths': 'Knowledgeable, quick thinkers, original',
        'weaknesses': 'Argumentative, insensitive, intolerant'
    },
    'INFJ': {
        'name': 'Advocate',
        'category': 'Diplomat',
        'description': 'Quiet and mystical, yet very inspiring and tireless idealists.',
        'traits': [
            'Understand people\'s emotions deeply',
            'Have strong intuitions about outcomes',
            'Seek meaning and authentic connections',
            'Help others find their purpose',
            'Need alone time to recharge'
        ],
        'strengths': 'Creative, insightful, principled',
        'weaknesses': 'Sensitive, perfectionistic, private'
    },
    'INFP': {
        'name': 'Mediator',
        'category': 'Diplomat',
        'description': 'Poetic, kind and altruistic people, always eager to help a good cause.',
        'traits': [
            'Guided by personal values and ideals',
            'Creative and artistic expression',
            'Dream about making world better',
            'Seek authentic emotional connections',
            'Process emotions deeply'
        ],
        'strengths': 'Empathetic, generous, open-minded',
        'weaknesses': 'Unrealistic, self-isolating, unfocused'
    },
    'ENFJ': {
        'name': 'Protagonist',
        'category': 'Diplomat',
        'description': 'Charismatic and inspiring leaders, able to mesmerize their listeners.',
        'traits': [
            'Inspire and motivate others toward growth',
            'Create harmony in groups naturally',
            'See potential in everyone',
            'Lead with empathy and passion',
            'Strong communication skills'
        ],
        'strengths': 'Tolerant, reliable, charismatic',
        'weaknesses': 'Overly idealistic, too selfless, over-sensitive'
    },
    'ENFP': {
        'name': 'Campaigner',
        'category': 'Diplomat',
        'description': 'Enthusiastic, creative and sociable free spirits, who can always find a reason to smile.',
        'traits': [
            'Love meeting new people and possibilities',
            'Enthusiastic and energetic',
            'Follow heart and trust intuition',
            'Creative brainstorming excites them',
            'See opportunities everywhere'
        ],
        'strengths': 'Curious, observant, enthusiastic',
        'weaknesses': 'Unfocused, overly optimistic, restless'
    },
    'ISTJ': {
        'name': 'Logistician',
        'category': 'Sentinel',
        'description': 'Practical and fact-minded individuals, whose reliability cannot be doubted.',
        'traits': [
            'Rely on proven methods and procedures',
            'Value organization and structure',
            'Prefer facts over abstract theories',
            'Follow rules and maintain order',
            'Responsible and dependable'
        ],
        'strengths': 'Honest, direct, strong-willed',
        'weaknesses': 'Stubborn, insensitive, judgmental'
    },
    'ISFJ': {
        'name': 'Defender',
        'category': 'Sentinel',
        'description': 'Very dedicated and warm protectors, always ready to defend their loved ones.',
        'traits': [
            'Remember and honor traditions',
            'Help others quietly and consistently',
            'Create warm supportive environments',
            'Show love through practical actions',
            'Work steadily behind the scenes'
        ],
        'strengths': 'Supportive, reliable, patient',
        'weaknesses': 'Shy, repressive, reluctant to change'
    },
    'ESTJ': {
        'name': 'Executive',
        'category': 'Sentinel',
        'description': 'Excellent administrators, unsurpassed at managing things – or people.',
        'traits': [
            'Take charge and get things done',
            'Follow rules and create structure',
            'Focus on practical concrete results',
            'Make decisions based on logic',
            'Efficient and productive'
        ],
        'strengths': 'Dedicated, strong-willed, direct',
        'weaknesses': 'Inflexible, uncomfortable with emotions, judgmental'
    },
    'ESFJ': {
        'name': 'Consul',
        'category': 'Sentinel',
        'description': 'Extraordinarily caring, social and popular people, always eager to help.',
        'traits': [
            'Create harmony and help others',
            'Remember details about loved ones',
            'Enjoy social traditions and celebrations',
            'Work hard to meet everyone\'s needs',
            'Maintain relationships actively'
        ],
        'strengths': 'Strong practical skills, loyal, sensitive',
        'weaknesses': 'Worried about social status, inflexible, reluctant to innovate'
    },
    'ISTP': {
        'name': 'Virtuoso',
        'category': 'Explorer',
        'description': 'Bold and practical experimenters, masters of all kinds of tools.',
        'traits': [
            'Solve problems through hands-on work',
            'Understand how mechanical things work',
            'Stay calm under pressure',
            'Value freedom and flexibility',
            'Learn by doing not reading'
        ],
        'strengths': 'Optimistic, energetic, creative',
        'weaknesses': 'Stubborn, insensitive, private'
    },
    'ISFP': {
        'name': 'Adventurer',
        'category': 'Explorer',
        'description': 'Flexible and charming artists, always ready to explore and experience something new.',
        'traits': [
            'Live in the present moment',
            'Express themselves through art',
            'Value personal freedom',
            'Inspired by nature and beauty',
            'Show love through actions'
        ],
        'strengths': 'Charming, sensitive to others, imaginative',
        'weaknesses': 'Fiercely independent, unpredictable, easily stressed'
    },
    'ESTP': {
        'name': 'Entrepreneur',
        'category': 'Explorer',
        'description': 'Smart, energetic and very perceptive people, who truly enjoy living on the edge.',
        'traits': [
            'Thrive on excitement and spontaneity',
            'Take action over endless planning',
            'Read people and situations quickly',
            'Enjoy physical and hands-on activities',
            'Adapt easily to any situation'
        ],
        'strengths': 'Bold, rational, direct',
        'weaknesses': 'Insensitive, impatient, risk-prone'
    },
    'ESFP': {
        'name': 'Entertainer',
        'category': 'Explorer',
        'description': 'Spontaneous, energetic and enthusiastic people – life is never boring around them.',
        'traits': [
            'Bring energy and fun everywhere',
            'Live spontaneously in the moment',
            'Connect through shared experiences',
            'Love entertaining and making others laugh',
            'Focus on enjoying the present'
        ],
        'strengths': 'Bold, original, practical',
        'weaknesses': 'Sensitive, conflict-averse, easily bored'
    }
}


def scrape_personality_data():
    """Scrape personality data from multiple online sources"""
    print("🌐 Starting web scraping process...")
    
    all_data = []
    
    # ========================================
    # SOURCE 1: WORKING ONLINE DATASETS
    # ========================================
    try:
        print("📥 Source 1: Fetching MBTI dataset from online sources...")
        
        # VERIFIED WORKING URLs (December 2025)
        dataset_sources = [
            {
                'name': 'Hugging Face Mirror 1',
                'url': 'https://huggingface.co/datasets/kl08/myers-briggs-type-indicator/resolve/main/mbti_1.csv',
                'format': 'csv'
            },
            {
                'name': 'Hugging Face Mirror 2',
                'url': 'https://huggingface.co/datasets/jingjietan/kaggle-mbti/resolve/main/mbti_1.csv',
                'format': 'csv'
            },
            {
                'name': 'GitHub Archive',
                'url': 'https://raw.githubusercontent.com/ianscottknight/Predicting-Myers-Briggs-Type-Indicator-with-Recurrent-Neural-Networks/master/mbti_1.csv',
                'format': 'csv'
            }
        ]
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        
        dataset_loaded = False
        
        # Try each source until one works
        for source in dataset_sources:
            if dataset_loaded:
                break
                
            try:
                print(f"   Trying: {source['name']}...")
                print(f"   URL: {source['url'][:70]}...")
                
                response = requests.get(source['url'], headers=headers, timeout=60, allow_redirects=True)
                
                print(f"   Status Code: {response.status_code}")
                
                if response.status_code == 200 and len(response.content) > 10000:
                    # Save to temporary file
                    temp_file = f"temp_{source['name'].replace(' ', '_')}.csv"
                    with open(temp_file, 'wb') as f:
                        f.write(response.content)
                    
                    print(f"   Downloaded {len(response.content) / 1024 / 1024:.2f} MB")
                    
                    # Try to load with pandas
                    try:
                        df_temp = pd.read_csv(temp_file, encoding='utf-8', on_bad_lines='skip')
                        print(f"   Loaded {len(df_temp)} rows")
                        
                        # Check for required columns (flexible column names)
                        if 'type' in df_temp.columns and 'posts' in df_temp.columns:
                            df_temp = df_temp.rename(columns={'type': 'personality', 'posts': 'text'})
                        elif 'Type' in df_temp.columns and 'posts' in df_temp.columns:
                            df_temp = df_temp.rename(columns={'Type': 'personality', 'posts': 'text'})
                        elif 'personality' in df_temp.columns and 'text' in df_temp.columns:
                            pass  # Already correct format
                        else:
                            print(f"   ⚠️ Wrong columns: {list(df_temp.columns)}")
                            os.remove(temp_file)
                            continue
                        
                        # Process posts
                        print(f"   Processing posts...")
                        processed_count = 0
                        for idx, row in df_temp.head(500).iterrows():
                            try:
                                personality_type = str(row['personality']).strip().upper()
                                posts = str(row['text']).split('|||')
                                
                                for post in posts[:5]:
                                    if len(post.strip()) > 50:
                                        all_data.append({
                                            'text': post.strip(),
                                            'personality': personality_type
                                        })
                                        processed_count += 1
                            except Exception as e:
                                continue
                        
                        print(f"   ✅ Successfully loaded {processed_count} samples from {source['name']}")
                        
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                        
                        if processed_count > 100:  # Only mark as success if we got meaningful data
                            dataset_loaded = True
                            break
                        
                    except Exception as e:
                        print(f"   ⚠️ Error parsing CSV: {str(e)[:100]}")
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                        continue
                else:
                    print(f"   ⚠️ Bad response or small file: {response.status_code}")
                    
            except requests.exceptions.Timeout:
                print(f"   ⚠️ Timeout after 60 seconds")
            except requests.exceptions.ConnectionError:
                print(f"   ⚠️ Connection error")
            except Exception as e:
                print(f"   ⚠️ Failed: {str(e)[:100]}")
        
        if not dataset_loaded:
            print("   ⚠️ All online sources failed")
            print("   ℹ️  Using backup personality descriptions only")
            
    except Exception as e:
        print(f"   ⚠️ Error in Source 1: {e}")
    
    
    
    # ========================================
    # SOURCE 2: Expanded Personality Descriptions
    # ========================================
    
    try:
        print("\n📥 Source 2: Adding personality type descriptions...")
        
        personality_samples = {
            'INTJ': [
                "I prefer working alone on complex strategic problems that require deep analysis",
                "Logic and efficiency matter more than social conventions in my decision making",
                "I plan everything meticulously and hate improvising or winging it",
                "Intellectual debates fascinate me far more than casual small talk",
                "I value competence above all else in myself and my colleagues",
                "Strategic long-term planning comes naturally to me in all situations",
                "I analyze systems to find optimal solutions and improvements",
                "Independent work allows me to achieve my best results"
            ],
            'INTP': [
                "Understanding complex theoretical systems fascinates my mind completely",
                "I analyze ideas from every possible angle before accepting them",
                "Abstract theories interest me more than practical everyday tasks",
                "Logical consistency matters most in all arguments and discussions",
                "I question assumptions that others simply take for granted",
                "Intellectual curiosity drives most of my learning and exploration",
                "I enjoy solving puzzles and figuring out how things work",
                "Debating philosophical concepts energizes my thinking"
            ],
            'ENTJ': [
                "I naturally take charge and lead groups toward goals efficiently",
                "Building effective systems and structures satisfies me deeply",
                "Long-term strategic planning is one of my greatest strengths",
                "I make quick decisive choices based on logical analysis",
                "Achieving ambitious goals motivates me to work harder constantly",
                "Organizing people and resources comes naturally to my personality",
                "I communicate directly and expect others to do the same",
                "Inefficiency frustrates me more than almost anything else"
            ],
            'ENTP': [
                "Debating ideas and challenging assumptions thrills me intellectually",
                "I see patterns connections and possibilities others often miss",
                "Innovation and experimenting with new concepts excites me daily",
                "Conventional thinking bores me so I constantly seek originality",
                "Playing devil's advocate helps clarify truth in discussions",
                "I love brainstorming creative solutions to complex problems",
                "Quick wit and clever arguments are my favorite tools",
                "Exploring multiple perspectives enriches my understanding"
            ],
            'INFJ': [
                "I understand people's emotions and hidden motivations very deeply",
                "Helping others discover their true path fulfills me completely",
                "I have vivid intuitions about future outcomes and consequences",
                "Meaningful one-on-one conversations energize me more than parties",
                "I need significant alone time to recharge after social interactions",
                "Creating positive change in others lives drives my purpose",
                "Deep authentic connections matter far more than superficial ones",
                "I sense unspoken emotions and tensions in groups easily"
            ],
            'INFP': [
                "My personal values and ideals guide every decision I make",
                "I dream constantly about ways to make the world better",
                "Creative artistic expression helps me process my deep emotions",
                "Authentic genuine connections mean everything to me daily",
                "I search for deeper meaning in all my experiences",
                "Helping others aligns with my core values and purpose",
                "I feel emotions very intensely and profoundly",
                "Staying true to myself matters more than fitting in"
            ],
            'ENFJ': [
                "I inspire and motivate people toward personal growth naturally",
                "Understanding emotions helps me guide and support others effectively",
                "Creating harmony and unity in groups comes naturally to me",
                "I see incredible potential in everyone I meet every day",
                "Leading with empathy creates lasting positive change in communities",
                "Helping others succeed brings me tremendous joy and fulfillment",
                "I communicate with warmth and genuine care for people",
                "Building strong relationships is central to my happiness"
            ],
            'ENFP': [
                "Meeting new people and hearing their stories excites me tremendously",
                "Creative brainstorming sessions energize me like nothing else",
                "I follow my heart and trust my strong intuitions",
                "Exploring new possibilities and opportunities thrills me daily",
                "Authentic emotional connections fulfill me more than achievements",
                "I see potential and opportunities in everything around me",
                "Spontaneous adventures make life worth living for me",
                "I bring enthusiasm and positivity to every situation"
            ],
            'ISTJ': [
                "I rely on proven methods and detailed procedures consistently",
                "Organization and structure help me work most efficiently",
                "I prefer concrete facts over abstract theoretical concepts",
                "Following established rules ensures everything runs smoothly",
                "Responsibility and duty guide all my important decisions",
                "I maintain detailed accurate records of important matters",
                "Practical reliable solutions appeal to me most strongly",
                "Tradition and stability provide important foundations"
            ],
            'ISFJ': [
                "I remember and carefully honor important meaningful traditions",
                "Helping others quietly brings me deep personal satisfaction",
                "I create warm supportive environments for everyone around me",
                "Practical caring actions show my love better than words",
                "I work steadily behind the scenes without seeking recognition",
                "Protecting and supporting loved ones is my top priority",
                "I notice and attend to small details others often miss",
                "Loyalty and dependability define my character completely"
            ],
            'ESTJ': [
                "I take charge quickly and ensure things get done efficiently",
                "Following clear rules and structure creates effective systems",
                "I focus on practical tangible results over abstract ideas",
                "Making logical objective decisions comes naturally to me",
                "Efficiency and high productivity drive my daily work habits",
                "I organize people and tasks to maximize effectiveness",
                "Direct honest communication prevents misunderstandings",
                "Meeting deadlines and goals motivates me strongly"
            ],
            'ESFJ': [
                "Helping others and creating social harmony fulfills me deeply",
                "I remember small details about people I care about",
                "Social traditions and celebrations bring me great joy",
                "I work hard to meet everyone's needs and expectations",
                "Maintaining close relationships requires my constant effort and care",
                "I create warm welcoming environments wherever I go",
                "Bringing people together gives me tremendous satisfaction",
                "I'm sensitive to others feelings and social dynamics"
            ],
            'ISTP': [
                "I solve mechanical problems best through hands-on direct work",
                "Understanding exactly how things work fascinates me deeply",
                "I stay remarkably calm under pressure in crisis situations",
                "Freedom and flexibility matter far more than rigid structure",
                "I learn best by doing rather than reading instructions",
                "Practical problem-solving is my strongest natural skill",
                "I analyze situations logically before taking action",
                "Working with tools and physical objects satisfies me"
            ],
            'ISFP': [
                "I live fully present in each moment without worrying",
                "Artistic expression allows me to share my inner soul",
                "I value personal freedom and authentic life experiences",
                "Natural beauty inspires me and brings deep inner peace",
                "I prefer showing love through caring actions not words",
                "Creativity flows through me in everything I do",
                "I appreciate aesthetic beauty in my surroundings daily",
                "Living according to my values brings me fulfillment"
            ],
            'ESTP': [
                "I thrive on excitement spontaneity and thrilling adventures",
                "Taking immediate action beats endless planning every single time",
                "I read people and situations quickly and very accurately",
                "Physical activities and hands-on challenges suit me perfectly",
                "I adapt easily and effectively to whatever situation arises",
                "Living in the moment brings me the most excitement",
                "I take calculated risks that others might avoid",
                "Practical real-world experience teaches me best"
            ],
            'ESFP': [
                "I bring tremendous energy and fun to every situation",
                "Living spontaneously makes life worth living for me daily",
                "I connect with people through fun shared experiences",
                "Entertainment and making others laugh fulfills me completely",
                "I focus on fully enjoying the present moment always",
                "Social interaction energizes me more than anything else",
                "I love being center of attention at gatherings",
                "Spreading joy and positivity is my natural gift"
            ]
        }
        
        for personality, samples in personality_samples.items():
            for sample in samples:
                all_data.append({
                    'text': sample,
                    'personality': personality
                })
        
        print(f"   ✅ Added {len(personality_samples) * 8} personality descriptions")
        
    except Exception as e:
        print(f"   ⚠️ Error in Source 2: {e}")
    
    
    # ========================================
    # SOURCE 3: Reddit-style personality posts
    # ========================================
    try:
        print("\n📥 Source 3: Adding Reddit-style personality expressions...")
        
        reddit_style_posts = {
            'INTJ': [
                "Does anyone else spend hours optimizing their daily routines for maximum efficiency",
                "I've created a five year plan with quarterly milestones and backup strategies",
                "Why do people get offended when I point out logical flaws in their arguments"
            ],
            'ENFP': [
                "Just had the most amazing conversation with a stranger at the coffee shop",
                "Started three new creative projects this week and I'm excited about all of them",
                "Life is too short to not follow your passions and dreams wholeheartedly"
            ],
            'ISTJ': [
                "I've been using the same morning routine for five years and it works perfectly",
                "Why would anyone deviate from proven methods that consistently deliver results",
                "I keep detailed records of all important information for future reference"
            ]
        }
        
        for personality, posts in reddit_style_posts.items():
            for post in posts:
                all_data.append({
                    'text': post,
                    'personality': personality
                })
        
        print(f"   ✅ Added Reddit-style samples")
        
    except Exception as e:
        print(f"   ⚠️ Error in Source 3: {e}")
    
    
    # Convert to DataFrame and save
    if len(all_data) > 0:
        df = pd.DataFrame(all_data)
        df.to_csv(scraped_data_cache, index=False)
        
        print("\n" + "="*60)
        print(f"✅ SCRAPING COMPLETE!")
        print(f"📊 Total samples collected: {len(df)}")
        print(f"💾 Data saved to: {scraped_data_cache}")
        print(f"🎯 Personality types: {df['personality'].nunique()}")
        print(f"📋 Distribution:\n{df['personality'].value_counts().head(10)}")
        print("="*60 + "\n")
        
        return df
    else:
        print("\n❌ No data scraped! Using minimal fallback...")
        fallback_data = {
            'text': [
                'I love detailed planning and strategic thinking in everything',
                'Spontaneous adventures and meeting new people energizes me',
                'Logical analysis guides every decision I make carefully'
            ],
            'personality': ['ISTJ', 'ENFP', 'INTJ']
        }
        return pd.DataFrame(fallback_data)


def load_scraped_data():
    """Load data from cache or scrape fresh"""
    if os.path.exists(scraped_data_cache):
        file_age = time.time() - os.path.getmtime(scraped_data_cache)
        if file_age < 86400:  # 24 hours
            print(f"📂 Loading cached data from {scraped_data_cache}")
            return pd.read_csv(scraped_data_cache)
    
    return scrape_personality_data()


def clean_text(text):
    """Clean text - Remove type labels and common words"""
    # Remove all MBTI type labels
    personality_types = ['INFP', 'INTJ', 'INFJ', 'INTP', 'ENFP', 'ENFJ', 'ENTP', 'ENTJ',
                        'ISFP', 'ISTJ', 'ISFJ', 'ISTP', 'ESFP', 'ESTJ', 'ESFJ', 'ESTP']
    
    for ptype in personality_types:
        text = re.sub(rf'\b{ptype}\b', '', text, flags=re.IGNORECASE)
    
    # Basic cleaning
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Keep most words, remove only very common ones
    very_common = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'of', 'is', 'was', 'were', 'be', 'been', 'being', 'for', 'with', 'as', 'by', 'it', 'this', 'that'}
    
    filtered = [
        lemmatizer.lemmatize(w) 
        for w in tokens 
        if len(w) > 2 and w not in very_common
    ]
    
    return " ".join(filtered) if len(filtered) >= 5 else ""




def load_and_train_model():
    """Load data and train model - SIMPLE & EFFECTIVE VERSION"""
    global model, vectorizer, label_encoder, stop_words, lemmatizer
    
    print("🔄 Loading model...")
    
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    df = load_scraped_data()
    
    df.columns = df.columns.str.strip()
    df['text'] = df['text'].astype(str).apply(clean_text)
    df = df[df['text'].str.strip() != ""]
    df = df[df['text'].str.split().str.len() >= 8]
    
    print(f"📊 Training with {len(df)} samples")
    distribution = df['personality'].value_counts()
    print(f"📋 Data distribution:\n{distribution}")
    
    # Keep only well-represented classes
    min_samples = 40
    valid_types = distribution[distribution >= min_samples].index
    df = df[df['personality'].isin(valid_types)]
    
    print(f"✅ Keeping {len(valid_types)} personality types with 40+ samples")
    
    # SIMPLE TF-IDF (less overfitting)
    vectorizer = TfidfVectorizer(
        max_features=800,            # Reduced features
        ngram_range=(1, 2),          # Only unigrams and bigrams
        min_df=5,
        max_df=0.7,
        sublinear_tf=True
    )
    
    print("🔄 Creating text features...")
    X = vectorizer.fit_transform(df['text']).toarray()
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['personality'])
    
    print(f"📊 Feature matrix: {X.shape}")
    print(f"📊 Classes: {len(label_encoder.classes_)}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # SIMPLE LOGISTIC REGRESSION (less overfitting)
    print("🔄 Training Logistic Regression with regularization...")
    
    model = LogisticRegression(
        max_iter=1000,
        solver='lbfgs',
        multi_class='multinomial',
        C=0.3,                      # Strong regularization
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    
    print("✅ Model trained successfully!")
    print(f"📊 Training Accuracy: {train_acc * 100:.2f}%")
    print(f"📊 Test Accuracy: {test_acc * 100:.2f}%")
    
    # Check for overfitting
    gap = (train_acc - test_acc) * 100
    if gap > 15:
        print(f"⚠️  Overfitting detected (gap: {gap:.1f}%)")
    else:
        print(f"✅ Good generalization (gap: {gap:.1f}%)")
    
    print(f"📊 Personality types: {list(label_encoder.classes_)}")
    
    # Detailed report
    from sklearn.metrics import classification_report
    y_pred = model.predict(X_test)
    print("\n📈 Detailed Performance:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))
    
    # Top features per class
    print("\n🔝 Most Predictive Words Per Personality Type:")
    feature_names = vectorizer.get_feature_names_out()
    for i, personality in enumerate(label_encoder.classes_):
        # Get coefficients for this class
        coef = model.coef_[i]
        top_indices = coef.argsort()[-5:][::-1]
        top_words = [feature_names[idx] for idx in top_indices]
        print(f"   {personality}: {', '.join(top_words)}")



load_and_train_model()


@app.route('/')
def home():
    """Health check endpoint"""
    return jsonify({
        "status": "running",
        "message": "Personality Prediction API with Multiple Data Sources",
        "version": "3.0.0",
        "data_sources": ["Kaggle MBTI Dataset", "16Personalities Descriptions", "Reddit-style Posts"]
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict personality from text"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text' field in request body"}), 400
        
        user_text = data['text']
        
        if not user_text or user_text.strip() == "":
            return jsonify({"error": "Text cannot be empty"}), 400
        
        clean_input = clean_text(user_text)
        
        if not clean_input or clean_input.strip() == "":
            return jsonify({"error": "Text is too short or contains no valid words"}), 400
        
        user_vector = vectorizer.transform([clean_input]).toarray()
        pred_num = model.predict(user_vector)[0]
        pred_label = label_encoder.inverse_transform([pred_num])[0]
        
        probabilities = model.predict_proba(user_vector)[0]
        confidence = float(max(probabilities) * 100)
        
        all_predictions = []
        for i, personality in enumerate(label_encoder.classes_):
            all_predictions.append({
                "personality": personality,
                "confidence": float(probabilities[i] * 100)
            })
        
        all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Get personality description
        personality_info = PERSONALITY_DESCRIPTIONS.get(pred_label, {
            'name': pred_label,
            'description': 'Personality type description not available'
        })
        
        return jsonify({
            "success": True,
            "prediction": {
                "personality": pred_label,
                "name": personality_info.get('name', pred_label),
                "category": personality_info.get('category', 'Unknown'),
                "description": personality_info.get('description', ''),
                "traits": personality_info.get('traits', []),
                "strengths": personality_info.get('strengths', ''),
                "weaknesses": personality_info.get('weaknesses', ''),
                "confidence": confidence
            },
            "all_scores": all_predictions,
            "cleaned_text": clean_input
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/personalities', methods=['GET'])
def get_personalities():
    """Get all personality types with descriptions"""
    try:
        personalities_list = []
        
        for ptype, info in PERSONALITY_DESCRIPTIONS.items():
            personalities_list.append({
                "type": ptype,
                "name": info['name'],
                "category": info['category'],
                "description": info['description'],
                "traits": info['traits'],
                "strengths": info['strengths'],
                "weaknesses": info['weaknesses']
            })
        
        return jsonify({
            "success": True,
            "count": len(personalities_list),
            "personalities": personalities_list
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/personality/<string:type_code>', methods=['GET'])
def get_personality_details(type_code):
    """Get detailed information about a specific personality type"""
    type_code = type_code.upper()
    
    if type_code not in PERSONALITY_DESCRIPTIONS:
        return jsonify({"error": "Personality type not found"}), 404
    
    info = PERSONALITY_DESCRIPTIONS[type_code]
    
    return jsonify({
        "success": True,
        "type": type_code,
        "name": info['name'],
        "category": info['category'],
        "description": info['description'],
        "traits": info['traits'],
        "strengths": info['strengths'],
        "weaknesses": info['weaknesses']
    })


@app.route('/api/rescrape', methods=['POST'])
def rescrape():
    """Force rescrape data and retrain model"""
    try:
        if os.path.exists(scraped_data_cache):
            os.remove(scraped_data_cache)
        
        load_and_train_model()
        
        return jsonify({
            "success": True,
            "message": "Data rescraped and model retrained successfully"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Detailed health check"""
    cache_exists = os.path.exists(scraped_data_cache)
    cache_age = None
    
    if cache_exists:
        cache_age = int((time.time() - os.path.getmtime(scraped_data_cache)) / 3600)
    
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None,
        "personalities_count": len(label_encoder.classes_) if label_encoder else 0,
        "data_cached": cache_exists,
        "cache_age_hours": cache_age,
        "data_sources": ["Kaggle MBTI", "16Personalities", "Reddit-style"],
        "embedding_method": "TF-IDF"
    })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Personality Prediction API v3.0")
    print("="*60)
    print("📍 API: http://localhost:5000")
    print("📍 Predict: POST /api/predict")
    print("📍 All Types: GET /api/personalities")
    print("📍 Type Info: GET /api/personality/<TYPE>")
    print("📍 Rescrape: POST /api/rescrape")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
