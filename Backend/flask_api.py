from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
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


# URL filtering helpers
URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
DOMAIN_RE = re.compile(r"\b(youtube|youtu\.be|vimeo|soundcloud|imgur|reddit|twitter|instagram|watch|tumblr|media)\b", re.IGNORECASE)


def strip_urls_keep_text(s: str) -> str:
    """Remove URLs and domain words from text"""
    s = URL_RE.sub(" ", s)
    s = DOMAIN_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def looks_like_linkdump(s: str) -> bool:
    """Check if text is mostly URLs/links"""
    total = max(len(s), 1)
    url_cnt = len(URL_RE.findall(s))
    link_ratio = url_cnt / max(1, s.count(" ") + 1)
    letters = sum(ch.isalpha() for ch in s)
    nonletter_ratio = 1 - (letters / total)
    return (url_cnt >= 2) or (link_ratio > 0.15) or (nonletter_ratio > 0.6)


def scrape_personality_data():
    """Scrape personality data - Focus on quality over quantity"""
    print("🌐 Starting data collection process...")
    
    all_data = []
    web_samples = 0
    
    # ========================================
    # SOURCE 1: Try web scraping (limited samples)
    # ========================================
    try:
        print("📥 Source 1: Attempting to fetch online MBTI dataset...")
        
        url = 'https://huggingface.co/datasets/kl08/myers-briggs-type-indicator/resolve/main/mbti_1.csv'
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        try:
            print(f"   Trying: Hugging Face Mirror...")
            response = requests.get(url, headers=headers, timeout=60, allow_redirects=True)
            
            if response.status_code == 200 and len(response.content) > 10000:
                temp_file = 'temp_mbti.csv'
                with open(temp_file, 'wb') as f:
                    f.write(response.content)
                
                df_temp = pd.read_csv(temp_file, encoding='utf-8', on_bad_lines='skip')
                
                if 'type' in df_temp.columns and 'posts' in df_temp.columns:
                    df_temp = df_temp.rename(columns={'type': 'personality', 'posts': 'text'})
                    
                    # Take ONLY 80 users, 2 posts each (160 samples for diversity)
                    print(f"   Processing {min(80, len(df_temp))} users...")
                    for idx, row in df_temp.head(80).iterrows():
                        try:
                            personality_type = str(row['personality']).strip().upper()
                            posts = str(row['text']).split('|||')
                            
                            for post in posts[:2]:  # Only 2 posts per user
                                # Step 1: Skip if too short initially
                                if not post or len(post.strip()) < 50:
                                    continue
                                
                                # Step 2: Skip if it's mostly links/URLs
                                if looks_like_linkdump(post):
                                    continue
                                
                                # Step 3: Remove URLs and clean
                                post = strip_urls_keep_text(post)
                                
                                # Step 4: Skip if too short after URL removal
                                if len(post) < 80:
                                    continue
                                
                                # Step 5: Add to dataset
                                all_data.append({
                                    'text': post,
                                    'personality': personality_type
                                })
                                web_samples += 1
                        except Exception as e:
                            continue
                    
                    print(f"   ✅ Added {web_samples} diverse web samples")
                    os.remove(temp_file)
        except Exception as e:
            print(f"   ⚠️ Web scraping failed: {str(e)[:50]}")
                
    except Exception as e:
        print(f"   ⚠️ Error in web scraping: {e}")
    
    
    # ========================================
    # SOURCE 2: HIGH-QUALITY CURATED DATA (MAIN SOURCE)
    # ========================================
    print("\n📥 Source 2: Adding high-quality personality descriptions...")

    
    personality_samples = {
        'INTJ': [
            "I prefer working alone on complex strategic problems that require deep analysis and long-term planning",
            "Logic and efficiency matter more than social conventions in my decision making process",
            "I plan everything meticulously weeks in advance and hate improvising or winging situations",
            "Intellectual debates about theories and concepts fascinate me far more than casual small talk",
            "I value competence and expertise above all else in myself and my colleagues",
            "Strategic long-term planning comes naturally to me in all situations I encounter",
            "I analyze systems constantly to find optimal solutions and significant improvements",
            "Independent work allows me to achieve my best results without interference from others",
            "I trust data and logic more than intuition or emotions when making important decisions",
            "Inefficiency and incompetence frustrate me more than anything else in professional life",
            "I organize my thoughts into detailed frameworks and mental models constantly throughout day",
            "Abstract theoretical discussions energize me more than practical small details about implementation",
            "I question conventional wisdom and established methods to find better more efficient approaches",
            "My ideal day involves solving complex intellectual problems with minimal social interaction required",
            "I prefer direct honest communication over sugar-coating messages or emotional appeals",
            "Long-term consequences guide my choices more than immediate gratification or short term gains",
            "I spend hours researching topics that interest me until I completely master them",
            "Logical consistency in arguments matters more to me than maintaining social harmony",
            "I create detailed plans with contingencies for multiple possible future outcomes",
            "My strength lies in seeing patterns others miss and planning accordingly for success"
        ],
        'INTP': [
            "Understanding complex theoretical systems fascinates my mind completely every single day",
            "I analyze ideas from every possible angle before accepting them as truth",
            "Abstract theories interest me more than practical everyday tasks and implementations",
            "Logical consistency matters most in all arguments and discussions I participate in",
            "I question assumptions that others simply take for granted without thinking",
            "Intellectual curiosity drives most of my learning and exploration throughout life",
            "I enjoy solving puzzles and figuring out exactly how things work mechanically",
            "Debating philosophical concepts energizes my thinking more than anything else",
            "I prefer theoretical understanding over practical application in most situations",
            "My mind constantly generates new ideas and connects seemingly unrelated concepts",
            "I can spend hours exploring rabbit holes of interesting topics online",
            "Precision in language and definitions matters greatly to me in all communication",
            "I love analyzing systems to understand their underlying logical structure",
            "My curiosity leads me to learn about wide variety of subjects deeply",
            "I question everything including my own beliefs and thought processes regularly",
            "Abstract mathematical and logical problems fascinate me more than real world issues",
            "I enjoy finding flaws in arguments and strengthening logical reasoning",
            "My thinking process involves constant internal debate and analysis of ideas",
            "I prefer understanding why something works over just knowing that it works",
            "Intellectual independence and original thinking matter more than social conformity"
        ],
        'ENTJ': [
            "I naturally take charge and lead groups toward goals efficiently and effectively",
            "Building effective systems and structures satisfies me deeply on personal level",
            "Long-term strategic planning is one of my greatest natural strengths in life",
            "I make quick decisive choices based on logical analysis of available information",
            "Achieving ambitious goals motivates me to work harder constantly without stopping",
            "Organizing people and resources comes naturally to my personality and skillset",
            "I communicate directly and expect others to do same without sugarcoating",
            "Inefficiency and lack of direction frustrate me more than almost anything",
            "I see opportunities for improvement everywhere and implement changes immediately",
            "Leadership roles feel natural and comfortable to me in all situations",
            "I set high standards for myself and everyone around me daily",
            "My confidence in decisions comes from thorough logical analysis beforehand",
            "I enjoy organizing people toward common objectives and measuring progress",
            "Strategic thinking allows me to anticipate obstacles and plan around them",
            "I value competence and results over feelings and social niceties consistently",
            "My natural assertiveness helps me take control of chaotic situations quickly",
            "I thrive on challenges that require coordination and resource management",
            "Long-term vision guides my decisions more than short-term comfort",
            "I excel at seeing big picture and breaking it into actionable steps",
            "My strength lies in turning ideas into concrete plans and executing them"
        ],
        'ENTP': [
            "Debating ideas and challenging assumptions thrills me intellectually every single day",
            "I see patterns connections and possibilities that others often miss completely",
            "Innovation and experimenting with new concepts excites me more than anything",
            "Conventional thinking bores me so I constantly seek originality in all things",
            "Playing devil's advocate helps clarify truth and exposes weak arguments effectively",
            "I love brainstorming creative solutions to complex problems with others",
            "Quick wit and clever arguments are my favorite tools in discussions",
            "Exploring multiple perspectives enriches my understanding of every situation",
            "I enjoy intellectual sparring and pushing boundaries of conventional wisdom",
            "My mind constantly generates new ideas and unconventional approaches",
            "I question everything including widely accepted truths and common practices",
            "Rigid rules and procedures feel constraining to my creative thinking process",
            "I thrive on intellectual challenges that require innovative thinking",
            "My curiosity leads me to explore many different fields and subjects",
            "I enjoy finding loopholes and alternative interpretations in systems",
            "Entrepreneurial thinking comes naturally when I see opportunities",
            "I love connecting unrelated concepts to create novel solutions",
            "My communication style is direct and sometimes provocatively challenging",
            "I excel at seeing possibilities and potential in unexpected places",
            "Intellectual freedom matters more to me than security or stability"
        ],
        'INFJ': [
            "I understand people's emotions and hidden motivations very deeply intuitively",
            "Helping others discover their true path fulfills me completely on soul level",
            "I have vivid intuitions about future outcomes and consequences regularly",
            "Meaningful one-on-one conversations energize me more than large parties",
            "I need significant alone time to recharge after social interactions thoroughly",
            "Creating positive change in others lives drives my sense of purpose",
            "Deep authentic connections matter far more than superficial social relationships",
            "I sense unspoken emotions and tensions in groups easily and naturally",
            "My empathy allows me to feel others pain and joy deeply",
            "I see potential in people that they often cannot see themselves",
            "Complex human dynamics fascinate me more than simple social interactions",
            "I often know what people will say before they speak words",
            "My intuition guides me toward understanding deeper meanings in situations",
            "I feel called to help others find meaning and purpose in life",
            "Authentic self-expression matters more to me than social acceptance",
            "I pick up on subtle emotional cues that others completely miss",
            "My idealism drives me to envision better future for humanity",
            "I need deep connections with few people rather than many acquaintances",
            "My insight into human nature helps me counsel and guide others",
            "I feel others emotions so strongly they sometimes overwhelm me"
        ],
        'INFP': [
            "My personal values and ideals guide every single decision I make daily",
            "I dream constantly about ways to make world significantly better place",
            "Creative artistic expression helps me process very deep emotional experiences",
            "Authentic genuine connections mean everything to me in all relationships",
            "I search for deeper meaning and purpose in all my life experiences",
            "Helping others aligns perfectly with my core values and sense of purpose",
            "I feel emotions very intensely and profoundly throughout each day",
            "Staying true to myself matters far more than fitting in with crowds",
            "I am drawn to causes that champion underdogs and promote social justice",
            "My imagination runs wild with possibilities for creative projects and stories",
            "I need regular alone time to recharge and process complex emotions",
            "Shallow conversations drain me while deep discussions energize my spirit",
            "I see beauty and potential in people others might overlook completely",
            "My sensitivity to others emotions helps me be empathetic listener",
            "I express myself best through creative writing art or music",
            "Authenticity and sincerity matter more than popularity or social status",
            "I daydream about ideal futures and perfect scenarios constantly",
            "My values system guides me like internal moral compass every day",
            "I struggle with practical details while excelling at big picture thinking",
            "Creative expression allows me to share inner world with others"
        ],
        'ENFP': [
            "Meeting new people and hearing their stories excites me tremendously every day",
            "Creative brainstorming sessions energize me like nothing else in world",
            "I follow my heart and trust my strong intuitions in all decisions",
            "Exploring new possibilities and opportunities thrills me more than routine",
            "Authentic emotional connections fulfill me more than material achievements",
            "I see potential and opportunities in everything around me constantly",
            "Spontaneous adventures make life worth living for me every single day",
            "I bring enthusiasm and positivity to every situation I encounter",
            "My passion for ideas and people drives everything I do daily",
            "I love inspiring others to pursue their dreams and reach potential",
            "Conventional paths bore me so I create my own unique journey",
            "I connect with people quickly through genuine warmth and interest",
            "My energy and excitement are contagious to everyone around me",
            "I see life as adventure full of exciting possibilities waiting",
            "Deep conversations about values and dreams energize me completely",
            "I champion causes I believe in with passionate wholehearted enthusiasm",
            "My creativity flows constantly generating new ideas and projects",
            "I need variety and novelty to feel truly alive and engaged",
            "Authentic self-expression matters more than following social conventions",
            "I inspire others through my genuine enthusiasm for life and people"
        ],
        'ENFJ': [
            "I inspire and motivate people toward personal growth naturally every day",
            "Understanding emotions helps me guide and support others effectively always",
            "Creating harmony and unity in groups comes naturally to my personality",
            "I see incredible potential in everyone I meet every single day",
            "Leading with empathy creates lasting positive change in communities worldwide",
            "Helping others succeed brings me tremendous joy and fulfillment daily",
            "I communicate with warmth and genuine care for people around me",
            "Building strong relationships is central to my happiness and purpose",
            "I sense what people need emotionally before they express it verbally",
            "My natural charisma helps me connect with diverse groups of people",
            "I feel responsible for others wellbeing and happiness deeply",
            "Organizing people toward common good comes naturally to my skills",
            "I excel at seeing best in people and bringing it out",
            "My empathy allows me to understand different perspectives easily",
            "I create environments where people feel valued and understood",
            "Mentoring and coaching others toward success fulfills me completely",
            "I naturally take on role of peacemaker in conflicts",
            "My warmth and encouragement inspire others to be their best",
            "I prioritize others needs sometimes at expense of my own",
            "Building meaningful connections gives my life deep sense of purpose"
        ],
        'ISTJ': [
            "I rely on proven methods and detailed procedures consistently every day",
            "Organization and structure help me work most efficiently and productively",
            "I prefer concrete facts over abstract theoretical concepts always",
            "Following established rules ensures everything runs smoothly without problems",
            "Responsibility and duty guide all my important life decisions",
            "I maintain detailed accurate records of all important matters carefully",
            "Practical reliable solutions appeal to me most strongly in situations",
            "Tradition and stability provide important foundations for society",
            "I complete tasks thoroughly and accurately without cutting corners",
            "My dependability makes me reliable team member people trust",
            "I value concrete evidence over speculation and abstract theories",
            "Planning and preparation prevent problems before they occur",
            "I follow through on commitments no matter how difficult",
            "My attention to detail ensures nothing important gets overlooked",
            "I prefer tried and true methods over experimental new approaches",
            "Structure and routine help me maintain productivity and efficiency",
            "I take my obligations seriously and always fulfill them completely",
            "My practical mindset focuses on what works in real world",
            "I create order and organization wherever I go naturally",
            "Reliability and consistency define my character and work ethic"
        ],
        'ISFJ': [
            "I remember and carefully honor important meaningful traditions always",
            "Helping others quietly brings me deep personal satisfaction daily",
            "I create warm supportive environments for everyone around me",
            "Practical caring actions show my love better than words ever could",
            "I work steadily behind scenes without seeking recognition or praise",
            "Protecting and supporting loved ones is my top life priority",
            "I notice and attend to small details others often miss completely",
            "Loyalty and dependability define my character and relationships completely",
            "I anticipate others needs before they have to ask for help",
            "My patience allows me to support people through difficult times",
            "I preserve traditions that connect people to their heritage",
            "Practical service to others gives my life meaning and purpose",
            "I create stability and comfort for people I care about",
            "My sensitivity helps me understand what others need emotionally",
            "I work hard to meet everyone expectations without complaint",
            "Maintaining harmony in relationships matters greatly to me",
            "I show love through consistent reliable actions over time",
            "My nurturing nature makes people feel safe and cared for",
            "I remember small personal details about people I care about",
            "Creating peaceful supportive environments fulfills me deeply"
        ],
        'ESTJ': [
            "I take charge quickly and ensure things get done efficiently always",
            "Following clear rules and structure creates most effective systems",
            "I focus on practical tangible results over abstract theoretical ideas",
            "Making logical objective decisions comes naturally to my personality",
            "Efficiency and high productivity drive my daily work habits consistently",
            "I organize people and tasks to maximize overall effectiveness",
            "Direct honest communication prevents misunderstandings and confusion",
            "Meeting deadlines and goals motivates me strongly every day",
            "I implement proven procedures that ensure consistent quality results",
            "My leadership style emphasizes accountability and clear expectations",
            "I value tradition and established ways that have proven effective",
            "Structure and organization are essential for achieving any objective",
            "I make decisions quickly based on facts and logical analysis",
            "My no-nonsense approach gets things done without wasting time",
            "I hold myself and others to high standards consistently",
            "Practical problem solving is my greatest strength in work",
            "I create systems that maximize efficiency and minimize waste",
            "My direct communication style leaves no room for ambiguity",
            "I take responsibility seriously and always follow through completely",
            "Results and concrete achievements matter more than feelings"
        ],
        'ESFJ': [
            "Helping others and creating social harmony fulfills me deeply every day",
            "I remember small details about people I care about naturally",
            "Social traditions and celebrations bring me great joy and connection",
            "I work hard to meet everyone needs and expectations consistently",
            "Maintaining close relationships requires my constant effort and care",
            "I create warm welcoming environments wherever I go naturally",
            "Bringing people together gives me tremendous personal satisfaction",
            "I am sensitive to others feelings and social dynamics always",
            "My natural warmth makes people feel comfortable and valued",
            "I organize social gatherings that bring joy to everyone",
            "Helping friends and family brings me more joy than anything",
            "I notice when someone is upset and offer support immediately",
            "Traditional values and customs guide my approach to relationships",
            "I take pride in creating comfortable spaces for people",
            "My loyalty to loved ones is unwavering through all circumstances",
            "I express care through practical helpful actions every day",
            "Community involvement and service give my life deep meaning",
            "I work tirelessly to maintain harmony in all my relationships",
            "Social connections energize me and make me feel alive",
            "I show love by remembering what matters to each person"
        ],
        'ISTP': [
            "I solve mechanical problems best through hands-on direct work",
            "Understanding exactly how things work fascinates me deeply always",
            "I stay remarkably calm under pressure in crisis situations",
            "Freedom and flexibility matter far more than rigid structure rules",
            "I learn best by doing rather than reading instructions manuals",
            "Practical problem-solving is my strongest natural skill in life",
            "I analyze situations logically before taking any action",
            "Working with tools and physical objects satisfies me greatly",
            "My independence allows me to work effectively alone anytime",
            "I prefer action over talking and immediate results over planning",
            "Hands-on experience teaches me more than any theoretical lecture",
            "I troubleshoot problems systematically using logical deduction",
            "My calm demeanor helps me handle emergencies effectively",
            "I enjoy taking things apart to understand their mechanisms",
            "Physical challenges and adventures appeal to me strongly",
            "I trust my ability to adapt to situations as they arise",
            "My practical skills make me valuable in crisis situations",
            "I observe situations carefully before deciding how to act",
            "Freedom to explore and experiment matters greatly to me",
            "I solve problems through trial and error rather than theory"
        ],
        'ISFP': [
            "I live fully present in each moment without worrying about future",
            "Artistic expression allows me to share my inner soul with world",
            "I value personal freedom and authentic life experiences above all",
            "Natural beauty inspires me and brings deep inner peace daily",
            "I prefer showing love through caring actions not empty words",
            "Creativity flows through me in everything I do each day",
            "I appreciate aesthetic beauty in my surroundings constantly",
            "Living according to my values brings me true fulfillment",
            "I express myself through art music or other creative outlets",
            "My sensitivity to beauty enriches every experience I have",
            "I need freedom to follow my own path at my pace",
            "Authentic self-expression matters more than social expectations",
            "I connect with nature and find peace in natural settings",
            "My gentle approach to life reflects my peaceful inner nature",
            "I avoid conflict and seek harmony in all relationships",
            "Creating beautiful things brings me joy and satisfaction",
            "I live by my values even when others don't understand",
            "My artistic nature sees beauty others might miss completely",
            "I prefer showing rather than telling how I feel",
            "Freedom and authenticity define how I live my life"
        ],
        'ESTP': [
            "I thrive on excitement spontaneity and thrilling adventures daily",
            "Taking immediate action beats endless planning every single time",
            "I read people and situations quickly and very accurately always",
            "Physical activities and hands-on challenges suit me perfectly",
            "I adapt easily and effectively to whatever situation arises",
            "Living in moment brings me most excitement and satisfaction",
            "I take calculated risks that others might avoid completely",
            "Practical real-world experience teaches me best always",
            "My energy and enthusiasm are contagious to everyone around",
            "I solve problems through direct action rather than theory",
            "Quick thinking helps me seize opportunities others miss",
            "I enjoy pushing limits and testing boundaries regularly",
            "My confidence comes from knowing I can handle anything",
            "I prefer doing over planning and action over contemplation",
            "Physical challenges and competitions energize me greatly",
            "I read social dynamics quickly and navigate them easily",
            "My spontaneity makes life exciting and unpredictable daily",
            "I trust my instincts and act quickly on opportunities",
            "Living on edge makes me feel most alive and engaged",
            "I thrive in fast-paced environments that require quick thinking"
        ],
        'ESFP': [
            "I bring tremendous energy and fun to every situation I enter",
            "Living spontaneously makes life worth living for me every day",
            "I connect with people through fun shared experiences together",
            "Entertainment and making others laugh fulfills me completely",
            "I focus on fully enjoying present moment without worry",
            "Social interaction energizes me more than anything else possible",
            "I love being center of attention at gatherings and parties",
            "Spreading joy and positivity is my natural gift to others",
            "My enthusiasm for life is contagious to everyone around me",
            "I create fun and excitement wherever I go naturally",
            "Living in moment allows me to fully enjoy life experiences",
            "I love making people smile and feel good about themselves",
            "My energy and warmth draw people to me effortlessly",
            "I turn ordinary moments into memorable fun experiences",
            "Social events and gatherings are where I truly shine",
            "I express affection openly and physically with everyone",
            "My spontaneous nature makes every day an adventure",
            "I live for experiences that create joy and laughter",
            "Being around people energizes and excites me daily",
            "I help others let loose and enjoy themselves freely"
        ]
    }
    
    for personality, samples in personality_samples.items():
        for sample in samples:
            all_data.append({
                'text': sample,
                'personality': personality
            })
    
    print(f"   ✅ Added {len(personality_samples) * 20} high-quality descriptions (320 samples)")
    
    # Convert to DataFrame
    if len(all_data) > 0:
        df = pd.DataFrame(all_data)
        df.to_csv(scraped_data_cache, index=False)
        
        print("\n" + "="*60)
        print(f"✅ DATA COLLECTION COMPLETE!")
        print(f"📊 Total samples: {len(df)}")
        print(f"💎 Web samples: {len(all_data) - 320}")
        print(f"💎 Curated samples: 320")
        print(f"🎯 All 16 personality types included")
        print("="*60 + "\n")
        
        return df
    else:
        return pd.DataFrame({'text': ['placeholder'], 'personality': ['INTJ']})


def load_scraped_data():
    """Load data from cache or scrape fresh"""
    if os.path.exists(scraped_data_cache):
        file_age = time.time() - os.path.getmtime(scraped_data_cache)
        if file_age < 86400:  # 24 hours
            print(f"📂 Loading cached data from {scraped_data_cache}")
            return pd.read_csv(scraped_data_cache)
    
    return scrape_personality_data()


def clean_text(text):
    """Clean text - GENTLE VERSION that keeps more content"""
    # Remove MBTI type labels (prevent data leakage)
    personality_types = ['INFP', 'INTJ', 'INFJ', 'INTP', 'ENFP', 'ENFJ', 'ENTP', 'ENTJ',
                        'ISFP', 'ISTJ', 'ISFJ', 'ISTP', 'ESFP', 'ESTJ', 'ESFJ', 'ESTP']
    
    for ptype in personality_types:
        text = re.sub(rf'\b{ptype}\b', ' ', text, flags=re.IGNORECASE)
    
    # Strip URLs (already done in scraping, but safety check)
    text = strip_urls_keep_text(text)
    
    # Basic cleaning
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # MUCH LESS AGGRESSIVE - only remove these specific words
    minimal_stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'of', 'is', 'was', 'were'}
    
    # Keep almost everything
    filtered = [
        lemmatizer.lemmatize(w) 
        for w in tokens 
        if len(w) > 2 and w not in minimal_stopwords
    ]
    
    # CHANGED: Only require 5 words (not 8)
    return " ".join(filtered) if len(filtered) >= 5 else ""

def get_temperament(mbti_type):
    """Convert 16 MBTI types to 4 temperaments"""
    mbti_type = mbti_type.upper()
    
    if mbti_type in ['INTJ', 'INTP', 'ENTJ', 'ENTP']:
        return 'NT_Analyst'
    elif mbti_type in ['INFJ', 'INFP', 'ENFJ', 'ENFP']:
        return 'NF_Diplomat'
    elif mbti_type in ['ISTJ', 'ISFJ', 'ESTJ', 'ESFJ']:
        return 'SJ_Sentinel'
    elif mbti_type in ['ISTP', 'ISFP', 'ESTP', 'ESFP']:
        return 'SP_Explorer'
    else:
        return 'Unknown'



def load_and_train_model():
    """Load data and train model - MAXIMUM REGULARIZATION"""
    global model, vectorizer, label_encoder, stop_words, lemmatizer
    
    print("🔄 Loading model...")
    
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    df = load_scraped_data()
    
    df.columns = df.columns.str.strip()
    df['text'] = df['text'].astype(str).apply(clean_text)
    df = df[df['text'].str.strip() != ""]
    df = df[df['text'].str.split().str.len() >= 5]
    df['original_personality'] = df['personality']
    df['personality'] = df['personality'].apply(get_temperament)
    
    print(f"📊 Training with {len(df)} samples")
    distribution = df['personality'].value_counts()
    print(f"📋 Data distribution:\n{distribution}")
    
    # MINIMAL features - prevent overfitting
    vectorizer = TfidfVectorizer(
        max_features=150,           # VERY REDUCED (was 300)
        ngram_range=(1, 1),         # ONLY unigrams (no bigrams)
        min_df=3,                   # Must appear in 3+ documents
        max_df=0.7,                 # Ignore very common
        sublinear_tf=True
    )
    
    print("🔄 Creating text features...")
    X = vectorizer.fit_transform(df['text']).toarray()
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['personality'])
    
    print(f"📊 Feature matrix: {X.shape}")
    print(f"📊 Classes: {len(label_encoder.classes_)}")
    
    # EVEN LARGER test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.40, random_state=42, stratify=y  # 40% test
    )
    
    # EXTREME REGULARIZATION
    print("🔄 Training with EXTREME regularization...")
    
    model = LogisticRegression(
        max_iter=500,               # Fewer iterations
        solver='lbfgs',
        multi_class='multinomial',
        C=0.01,                     # EXTREME regularization (was 0.05)
        class_weight='balanced',
        penalty='l2',
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
    
    gap = (train_acc - test_acc) * 100
    if gap < 10:
        print(f"✅ Excellent generalization (gap: {gap:.1f}%)")
    elif gap < 20:
        print(f"✅ Good generalization (gap: {gap:.1f}%)")
    elif gap < 30:
        print(f"⚠️  Acceptable overfitting (gap: {gap:.1f}%)")
    else:
        print(f"❌ Severe overfitting (gap: {gap:.1f}%)")
    
    print(f"📊 Personality types: {list(label_encoder.classes_)}")
    
    # Detailed report
    from sklearn.metrics import classification_report
    y_pred = model.predict(X_test)
    print("\n📈 Detailed Performance:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))
    
    # Top features
    print("\n🔝 Most Predictive Words Per Personality Type:")
    feature_names = vectorizer.get_feature_names_out()
    for i, personality in enumerate(label_encoder.classes_):
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
def predict_personality():
    try:
        data = request.get_json()
        
        # Debug logging
        print(f"📥 Received request: {data}")
        
        text = data.get('text', '') if data else ''
        
        print(f"📝 Text length: {len(text)} characters")
        
        # More lenient validation - accept 30+ chars instead of 50
        if not text or len(text.strip()) < 30:
            print(f"❌ Text too short: {len(text.strip())} chars")
            return jsonify({
                'success': False,
                'error': f'Please provide at least 30 characters of text (you provided {len(text.strip())})'
            }), 400
        
        cleaned_text = clean_text(text)
        print(f"🧹 Cleaned text: '{cleaned_text[:100]}...' ({len(cleaned_text)} chars)")
        
        if not cleaned_text or len(cleaned_text.split()) < 3:
            print(f"❌ Text too short after cleaning")
            return jsonify({
                'success': False,
                'error': 'Text too short after processing. Please write more descriptive text with meaningful words.'
            }), 400
        
        # Prediction
        X = vectorizer.transform([cleaned_text]).toarray()
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        
        predicted_temperament = label_encoder.inverse_transform([prediction])[0]
        confidence = float(probabilities[prediction] * 100)
        
        print(f"✅ Prediction: {predicted_temperament} ({confidence:.1f}%)")
        
        # Map temperament back to MBTI types
        temperament_to_types = {
            'NT_Analyst': ['INTJ', 'INTP', 'ENTJ', 'ENTP'],
            'NF_Diplomat': ['INFJ', 'INFP', 'ENFJ', 'ENFP'],
            'SJ_Sentinel': ['ISTJ', 'ISFJ', 'ESTJ', 'ESFJ'],
            'SP_Explorer': ['ISTP', 'ISFP', 'ESTP', 'ESFP']
        }
        
        likely_types = temperament_to_types.get(predicted_temperament, [])
        most_likely = likely_types[0] if likely_types else 'INTJ'
        
        # Get all temperament scores
        all_scores = []
        for i, prob in enumerate(probabilities):
            temperament = label_encoder.inverse_transform([i])[0]
            all_scores.append({
                'temperament': temperament,
                'confidence': float(prob * 100)
            })
        all_scores.sort(key=lambda x: x['confidence'], reverse=True)
        
        return jsonify({
            'success': True,
            'prediction': {
                'temperament': predicted_temperament,
                'personality': most_likely,
                'likely_types': likely_types,
                'name': PERSONALITY_DESCRIPTIONS.get(most_likely, {}).get('name', 'Unknown'),
                'category': PERSONALITY_DESCRIPTIONS.get(most_likely, {}).get('category', 'Unknown'),
                'description': PERSONALITY_DESCRIPTIONS.get(most_likely, {}).get('description', ''),
                'traits': PERSONALITY_DESCRIPTIONS.get(most_likely, {}).get('traits', []),
                'strengths': PERSONALITY_DESCRIPTIONS.get(most_likely, {}).get('strengths', ''),
                'weaknesses': PERSONALITY_DESCRIPTIONS.get(most_likely, {}).get('weaknesses', ''),
                'confidence': confidence
            },
            'all_scores': all_scores
        })
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        }), 500



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
