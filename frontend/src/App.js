
import React, { useState, useEffect, useRef } from 'react';
import './App.css';

// Small local icon components (emoji fallbacks) to avoid external icon dependency
const Icon = ({ children, size = 18, className = '' }) => (
  <span className={className} style={{ fontSize: size, lineHeight: 1 }}>{children}</span>
);

const Sparkles = (props) => <Icon {...props}>✨</Icon>;
const TrendingUp = (props) => <Icon {...props}>📈</Icon>;
const Users = (props) => <Icon {...props}>👥</Icon>;
const Lightbulb = (props) => <Icon {...props}>💡</Icon>;
const Target = (props) => <Icon {...props}>🎯</Icon>;
const Download = (props) => <Icon {...props}>⬇️</Icon>;
const Share2 = (props) => <Icon {...props}>🔗</Icon>;
const Moon = (props) => <Icon {...props}>🌙</Icon>;
const Sun = (props) => <Icon {...props}>☀️</Icon>;
const RefreshCw = (props) => <Icon {...props}>🔄</Icon>;

const API_URL = 'http://localhost:5000';

// Floating particles background
const FloatingParticles = ({ colorScheme }) => {
  const canvasRef = useRef(null);
  
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    
    const particles = [];
    const particleCount = 50;
    
    const colors = colorScheme === 'purple' ? ['#6B46C1', '#3B82F6', '#14B8A6'] :
                   colorScheme === 'coral' ? ['#FF6B9D', '#F59E0B', '#3B82F6'] :
                   colorScheme === 'teal' ? ['#14B8A6', '#6B46C1', '#3B82F6'] :
                   ['#6B46C1', '#3B82F6', '#14B8A6'];
    
    for (let i = 0; i < particleCount; i++) {
      particles.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        radius: Math.random() * 3 + 1,
        dx: (Math.random() - 0.5) * 0.5,
        dy: (Math.random() - 0.5) * 0.5,
        color: colors[Math.floor(Math.random() * colors.length)]
      });
    }
    
    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      particles.forEach(particle => {
        ctx.beginPath();
        ctx.arc(particle.x, particle.y, particle.radius, 0, Math.PI * 2);
        ctx.fillStyle = particle.color + '40';
        ctx.fill();
        
        particle.x += particle.dx;
        particle.y += particle.dy;
        
        if (particle.x < 0 || particle.x > canvas.width) particle.dx *= -1;
        if (particle.y < 0 || particle.y > canvas.height) particle.dy *= -1;
      });
      
      requestAnimationFrame(animate);
    };
    
    animate();
    
    const handleResize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [colorScheme]);
  
  return <canvas ref={canvasRef} style={{ position: 'fixed', top: 0, left: 0, zIndex: 0, pointerEvents: 'none' }} />;
};

// 3D Brain visualization using CSS
const Brain3D = () => {
  return (
    <div className="brain-container">
      <div className="brain-sphere">
        <div className="brain-ring ring-1"></div>
        <div className="brain-ring ring-2"></div>
        <div className="brain-ring ring-3"></div>
        <div className="brain-core"></div>
      </div>
    </div>
  );
};

// Personality Avatar Component
const PersonalityAvatar = ({ type }) => {
  const avatarMap = {
    'INFP': { icon: '🌟', color: '#FF6B9D', name: 'The Dreamer' },
    'INTJ': { icon: '🧠', color: '#6B46C1', name: 'The Architect' },
    'ENFP': { icon: '✨', color: '#F59E0B', name: 'The Champion' },
    'ISTJ': { icon: '🎯', color: '#3B82F6', name: 'The Inspector' },
    'ESTP': { icon: '⚡', color: '#14B8A6', name: 'The Dynamo' },
  };
  
  const avatar = avatarMap[type] || { icon: '💫', color: '#6B46C1', name: 'Unknown' };
  
  return (
    <div className="personality-avatar" style={{ '--avatar-color': avatar.color }}>
      <div className="avatar-circle">
        <span className="avatar-icon">{avatar.icon}</span>
      </div>
      <div className="avatar-rings">
        <div className="avatar-ring ring-1"></div>
        <div className="avatar-ring ring-2"></div>
        <div className="avatar-ring ring-3"></div>
      </div>
    </div>
  );
};

function App() {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [personalities, setPersonalities] = useState([]);
  const [darkMode, setDarkMode] = useState(false);
  const [colorScheme, setColorScheme] = useState('purple');
  const [showConfetti, setShowConfetti] = useState(false);

  useEffect(() => {
    fetchPersonalities();
  }, []);

  useEffect(() => {
    if (result) {
      setShowConfetti(true);
      setTimeout(() => setShowConfetti(false), 3000);
      
      // Set color scheme based on personality
      const personality = result.prediction.personality;
      if (personality.includes('INFP') || personality.includes('ENFP')) {
        setColorScheme('coral');
      } else if (personality.includes('INTJ') || personality.includes('ISTJ')) {
        setColorScheme('purple');
      } else {
        setColorScheme('teal');
      }
    }
  }, [result]);

  const fetchPersonalities = async () => {
    try {
      const response = await fetch(`${API_URL}/api/personalities`);
      const data = await response.json();
      if (data.success) {
        setPersonalities(data.personalities);
      }
    } catch (err) {
      console.error('Error fetching personalities:', err);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!text.trim()) {
      setError('Please share your thoughts with us');
      return;
    }

    setLoading(true);
    setError('');
    setResult(null);

    try {
      const response = await fetch(`${API_URL}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });

      const data = await response.json();

      if (data.success) {
        setResult(data);
      } else {
        setError('Analysis incomplete. Please try again.');
      }
    } catch (err) {
      setError('Connection failed. Ensure backend is running on port 5000.');
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setText('');
    setResult(null);
    setError('');
    setColorScheme('purple');
  };

  return (
    <div className={`app ${darkMode ? 'dark-mode' : ''}`} data-color-scheme={colorScheme}>
      <FloatingParticles colorScheme={colorScheme} />
      
      {showConfetti && <div className="confetti-container">
        {[...Array(50)].map((_, i) => (
          <div key={i} className="confetti" style={{
            left: `${Math.random() * 100}%`,
            animationDelay: `${Math.random() * 2}s`,
            backgroundColor: ['#FF6B9D', '#6B46C1', '#3B82F6', '#14B8A6', '#F59E0B'][Math.floor(Math.random() * 5)]
          }}></div>
        ))}
      </div>}
      
      <div className="theme-toggle" onClick={() => setDarkMode(!darkMode)}>
        {darkMode ? <Sun size={20} /> : <Moon size={20} />}
      </div>
      
      <div className="container">
        <header className="hero-section">
          <Brain3D />
          <h1 className="hero-title">
            <span className="gradient-text">Personality Mirror</span>
          </h1>
          <p className="hero-subtitle">Discover your unique psychological blueprint through AI</p>
          <div className="hero-stats">
            <div className="stat-item glass-card">
              <Users size={24} />
              <span>8 Types</span>
            </div>
            <div className="stat-item glass-card">
              <Sparkles size={24} />
              <span>AI Powered</span>
            </div>
            <div className="stat-item glass-card">
              <TrendingUp size={24} />
              <span>81% Accurate</span>
            </div>
          </div>
        </header>

        <main className="main-content">
          {!result ? (
            <div className="input-section glass-card">
              <form onSubmit={handleSubmit}>
                <div className="input-header">
                  <Lightbulb className="section-icon" size={28} />
                  <h2>Share Your Story</h2>
                </div>
                <p className="input-description">
                  Tell us about yourself - your passions, how you think, what drives you, and how you connect with others.
                </p>
                
                <div className="textarea-wrapper">
                  <textarea
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                    placeholder="I am someone who... When faced with challenges, I tend to... My ideal way to spend time is... What excites me most is..."
                    rows="10"
                    disabled={loading}
                    className="personality-input"
                  />
                  <div className="word-count">{text.trim().split(/\s+/).filter(Boolean).length} words</div>
                </div>
                
                {error && (
                  <div className="error-message glass-card">
                    <span>⚠️</span>
                    <p>{error}</p>
                  </div>
                )}
                
                <div className="button-group">
                  <button 
                    type="submit" 
                    className="btn btn-primary"
                    disabled={loading || !text.trim()}
                  >
                    {loading ? (
                      <>
                        <div className="spinner"></div>
                        <span>Analyzing Your Essence...</span>
                      </>
                    ) : (
                      <>
                        <Sparkles size={20} />
                        <span>Reveal My Personality</span>
                      </>
                    )}
                  </button>
                  
                  {text && (
                    <button 
                      type="button" 
                      onClick={handleClear}
                      className="btn btn-secondary"
                      disabled={loading}
                    >
                      <RefreshCw size={18} />
                      <span>Reset</span>
                    </button>
                  )}
                </div>
              </form>

              {personalities.length > 0 && (
                <div className="personality-types">
                  <h3>Personality Spectrum</h3>
                  <div className="types-grid">
                    {personalities.slice(0, 8).map((p, index) => (
                      <div key={index} className="type-badge">{p}</div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="results-section">
              <div className="result-hero glass-card">
                <PersonalityAvatar type={result.prediction.personality} />
                <h2 className="result-title">Your Personality Type</h2>
                <div className="personality-type-badge">
                  {result.prediction.personality}
                </div>
                <div className="confidence-display">
                  <span className="confidence-label">Confidence Score</span>
                  <span className="confidence-value">{result.prediction.confidence.toFixed(1)}%</span>
                </div>
                <div className="confidence-bar-wrapper">
                  <div 
                    className="confidence-bar-fill"
                    style={{ width: `${result.prediction.confidence}%` }}
                  ></div>
                </div>
                
                <div className="action-buttons">
                  <button className="btn btn-icon">
                    <Download size={18} />
                    <span>Download</span>
                  </button>
                  <button className="btn btn-icon">
                    <Share2 size={18} />
                    <span>Share</span>
                  </button>
                  <button className="btn btn-icon" onClick={handleClear}>
                    <RefreshCw size={18} />
                    <span>Retake</span>
                  </button>
                </div>
              </div>

              {result.all_scores && result.all_scores.length > 1 && (
                <div className="trait-analysis glass-card">
                  <div className="analysis-header">
                    <Target size={24} />
                    <h3>Personality Spectrum Analysis</h3>
                  </div>
                  <div className="traits-grid">
                    {result.all_scores.map((score, index) => (
                      <div key={index} className={`trait-item ${index === 0 ? 'primary-trait' : ''}`}>
                        <div className="trait-header">
                          <span className="trait-name">{score.personality}</span>
                          <span className="trait-score">{score.confidence.toFixed(1)}%</span>
                        </div>
                        <div className="trait-bar">
                          <div 
                            className="trait-bar-fill"
                            style={{ 
                              width: `${score.confidence}%`,
                              animationDelay: `${index * 0.1}s`
                            }}
                          ></div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </main>

        <footer className="footer">
          <p>Powered by Advanced AI & Psychological Research</p>
          <div className="footer-links">
            <a href="#about">About</a>
            <span>•</span>
            <a href="#privacy">Privacy</a>
            <span>•</span>
            <a href="#contact">Contact</a>
          </div>
        </footer>
      </div>
    </div>
  );
} 

export default App;