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
const RefreshCw = (props) => <Icon {...props}>🔄</Icon>;
const Award = (props) => <Icon {...props}>🏆</Icon>;
const AlertCircle = (props) => <Icon {...props}>⚠️</Icon>;
const CheckCircle = (props) => <Icon {...props}>✅</Icon>;

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
    'INFP': { icon: '🌟', color: '#FF6B9D', name: 'The Mediator' },
    'INTJ': { icon: '🧠', color: '#6B46C1', name: 'The Architect' },
    'ENFP': { icon: '✨', color: '#F59E0B', name: 'The Campaigner' },
    'ISTJ': { icon: '🎯', color: '#3B82F6', name: 'The Logistician' },
    'ESTP': { icon: '⚡', color: '#14B8A6', name: 'The Entrepreneur' },
    'INFJ': { icon: '🔮', color: '#8B5CF6', name: 'The Advocate' },
    'INTP': { icon: '🔬', color: '#6366F1', name: 'The Logician' },
    'ENTJ': { icon: '👑', color: '#EF4444', name: 'The Commander' },
    'ENTP': { icon: '💭', color: '#10B981', name: 'The Debater' },
    'ISFJ': { icon: '🛡️', color: '#06B6D4', name: 'The Defender' },
    'ESFJ': { icon: '🤝', color: '#F59E0B', name: 'The Consul' },
    'ISFP': { icon: '🎨', color: '#EC4899', name: 'The Adventurer' },
    'ESFP': { icon: '🎭', color: '#F97316', name: 'The Entertainer' },
    'ESTJ': { icon: '⚖️', color: '#0EA5E9', name: 'The Executive' },
    'ISTP': { icon: '🔧', color: '#64748B', name: 'The Virtuoso' },
    'ENFJ': { icon: '💫', color: '#A855F7', name: 'The Protagonist' },
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
    <div className="app dark-mode" data-color-scheme={colorScheme}>
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
              <span>16 Types</span>
            </div>
            <div className="stat-item glass-card">
              <Sparkles size={24} />
              <span>AI Powered</span>
            </div>
            <div className="stat-item glass-card">
              <TrendingUp size={24} />
              <span>50% Accurate</span>
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
                    {[
                      { code: 'INTJ', name: 'The Architect' },
                      { code: 'INTP', name: 'The Logician' },
                      { code: 'ENTJ', name: 'The Commander' },
                      { code: 'ENTP', name: 'The Debater' },
                      { code: 'INFJ', name: 'The Advocate' },
                      { code: 'INFP', name: 'The Mediator' },
                      { code: 'ENFJ', name: 'The Protagonist' },
                      { code: 'ENFP', name: 'The Campaigner' },
                      { code: 'ISTJ', name: 'The Logistician' },
                      { code: 'ISFJ', name: 'The Defender' },
                      { code: 'ESTJ', name: 'The Executive' },
                      { code: 'ESFJ', name: 'The Consul' },
                      { code: 'ISTP', name: 'The Virtuoso' },
                      { code: 'ISFP', name: 'The Adventurer' },
                      { code: 'ESTP', name: 'The Entrepreneur' },
                      { code: 'ESFP', name: 'The Entertainer' }
                    ].map((type, index) => (
                      <div key={index} className="type-badge" title={type.name}>
                        <strong>{type.code}</strong>
                        <span style={{
                          display: 'block',
                          fontSize: '0.7rem',
                          fontWeight: '400',
                          marginTop: '0.25rem',
                          opacity: '0.8'
                        }}>
                          {type.name}
                        </span>
                      </div>
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

                {/* Display personality name */}
                {result.prediction.name && (
                  <h3 className="personality-name" style={{
                    fontSize: '1.8rem',
                    marginTop: '1rem',
                    color: '#6B46C1',
                    fontWeight: '700'
                  }}>
                    {result.prediction.name}
                  </h3>
                )}

                {/* Display category badge */}
                {result.prediction.category && (
                  <div className="personality-category" style={{
                    display: 'inline-block',
                    padding: '0.5rem 1rem',
                    background: 'linear-gradient(135deg, #6B46C1 0%, #3B82F6 100%)',
                    color: 'white',
                    borderRadius: '20px',
                    fontSize: '0.9rem',
                    marginTop: '0.5rem',
                    fontWeight: '600'
                  }}>
                    {result.prediction.category}
                  </div>
                )}

                {/* ONLY RETAKE BUTTON */}
                <div className="action-buttons" style={{ marginTop: '2rem' }}>
                  <button className="btn btn-icon" onClick={handleClear}>
                    <RefreshCw size={18} />
                    <span>Retake Test</span>
                  </button>
                </div>
              </div>

              {/* Personality Description - WHITE TEXT */}
              {result.prediction.description && (
                <div className="personality-description glass-card" style={{ marginTop: '2rem' }}>
                  <div className="analysis-header">
                    <Sparkles size={24} />
                    <h3>Your Personality Profile</h3>
                  </div>
                  <p style={{
                    fontSize: '1.1rem',
                    lineHeight: '1.8',
                    color: '#FFFFFF',
                    fontStyle: 'italic',
                    padding: '1rem',
                    background: 'rgba(59, 130, 246, 0.08)',
                    borderRadius: '12px',
                    borderLeft: '4px solid #3B82F6'
                  }}>
                    {result.prediction.description}
                  </p>
                </div>
              )}

              {/* Key Traits Section - WHITE TEXT */}
              {result.prediction.traits && result.prediction.traits.length > 0 && (
                <div className="personality-traits glass-card" style={{ marginTop: '2rem' }}>
                  <div className="analysis-header">
                    <CheckCircle size={24} />
                    <h3>Key Characteristics</h3>
                  </div>
                  <ul style={{
                    listStyle: 'none',
                    padding: 0,
                    display: 'grid',
                    gap: '1rem'
                  }}>
                    {result.prediction.traits.map((trait, index) => (
                      <li key={index} style={{
                        display: 'flex',
                        alignItems: 'flex-start',
                        padding: '1rem',
                        background: 'rgba(16, 185, 129, 0.08)',
                        borderRadius: '10px',
                        border: '1px solid rgba(16, 185, 129, 0.2)',
                        transition: 'all 0.3s ease',
                        animation: `fadeInUp 0.5s ease-out ${index * 0.1}s both`
                      }}>
                        <span style={{
                          fontSize: '1.5rem',
                          marginRight: '1rem',
                          color: '#10B981'
                        }}>✓</span>
                        <span style={{
                          fontSize: '1rem',
                          color: '#FFFFFF',
                          lineHeight: '1.6'
                        }}>{trait}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Strengths and Weaknesses Grid */}
              <div className="strengths-weaknesses-grid" style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
                gap: '2rem',
                marginTop: '2rem'
              }}>
                {/* Strengths */}
                {result.prediction.strengths && (
                  <div className="strengths-card glass-card" style={{
                    background: 'linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(52, 211, 153, 0.05) 100%)',
                    border: '2px solid rgba(16, 185, 129, 0.2)'
                  }}>
                    <div className="analysis-header">
                      <Award size={24} style={{ color: '#10B981' }} />
                      <h3 style={{ color: '#10B981' }}>Strengths</h3>
                    </div>
                    <p style={{
                      fontSize: '1rem',
                      lineHeight: '1.8',
                      color: '#065F46',
                      padding: '1rem'
                    }}>
                      {result.prediction.strengths}
                    </p>
                  </div>
                )}

                {/* Weaknesses */}
                {result.prediction.weaknesses && (
                  <div className="weaknesses-card glass-card" style={{
                    background: 'linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(251, 191, 36, 0.05) 100%)',
                    border: '2px solid rgba(245, 158, 11, 0.2)'
                  }}>
                    <div className="analysis-header">
                      <AlertCircle size={24} style={{ color: '#F59E0B' }} />
                      <h3 style={{ color: '#F59E0B' }}>Areas for Growth</h3>
                    </div>
                    <p style={{
                      fontSize: '1rem',
                      lineHeight: '1.8',
                      color: '#92400E',
                      padding: '1rem'
                    }}>
                      {result.prediction.weaknesses}
                    </p>
                  </div>
                )}
              </div>

              {/* FIXED: Personality Spectrum Analysis - Show Temperaments with proper formatting */}
              {result.all_scores && result.all_scores.length > 0 && (
                <div className="trait-analysis glass-card" style={{ marginTop: '2rem' }}>
                  <div className="analysis-header">
                    <Target size={24} />
                    <h3>Personality Spectrum Analysis</h3>
                  </div>
                  <p style={{
                    color: '#6B7280',
                    marginBottom: '1.5rem',
                    fontSize: '0.95rem'
                  }}>
                    Your personality shows traits from multiple temperaments. Here's how you align with each:
                  </p>
                  <div className="traits-grid">
                    {result.all_scores.map((score, index) => {
                      // Format temperament name properly
                      const formatTemperament = (temp) => {
                        const mapping = {
                          'NT_Analyst': 'Analysts (NT)',
                          'NF_Diplomat': 'Diplomats (NF)',
                          'SJ_Sentinel': 'Sentinels (SJ)',
                          'SP_Explorer': 'Explorers (SP)'
                        };
                        return mapping[temp] || temp;
                      };

                      const displayName = formatTemperament(score.temperament || score.personality);

                      return (
                        <div key={index} className={`trait-item ${index === 0 ? 'primary-trait' : ''}`}>
                          <div className="trait-header">
                            <span className="trait-name">{displayName}</span>
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
                      );
                    })}
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