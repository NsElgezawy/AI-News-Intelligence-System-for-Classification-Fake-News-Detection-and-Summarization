import streamlit as st
from transformers import pipeline
import torch
import re
from collections import Counter

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="NewsIntel AI",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================
# Device Setup
# =========================
DEVICE = 0 if torch.cuda.is_available() else -1

# =========================
# Load Models
# =========================
@st.cache_resource
def load_models():
    fake_detector = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        device=DEVICE
    )
    clf_distilbert = pipeline(
        "text-classification",
        model="nselgezawy/news-classifier-distilbert",
        device=DEVICE
    )
    return fake_detector, clf_distilbert

@st.cache_resource
def load_summarizer():
    return pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6",
        device=DEVICE
    )

fake_detector, clf_distilbert = load_models()
summarizer = load_summarizer()

# =========================
# NLP Functions
# =========================
def detect_fake(text):
    prompt = f"""
    Analyze the credibility of this news text.
    Return in this format:
    Label: Real/Fake/Suspicious
    Confidence: %
    News:
    {text}
    Don't include any explanations, just the label and confidence.
    """
    result = fake_detector(
        prompt,
        max_new_tokens=30
    )[0]["generated_text"]
    result_lower = result.lower()
    if "fake" in result_lower:
        label = "Fake ❌"
    elif "Real" in result_lower:
        label = "Real ✅"
    elif "Suspicious" in result_lower:
        label = "Suspicious ⚠️"
    else:        label = "Unknown ❓"

    match = re.search(r'(\d+)', result)
    if match:
        score = int(match.group(1)) / 100
    else:
        score = 0.75
    return label, score

def classify_text(text):
    text = text[:512]
    res2 = clf_distilbert(text)[0]
    label2, score2 = res2["label"], float(res2["score"])
    return label2, score2

def summarize_text(text):
    text = text[:1000]
    input_length = len(text.split())
    max_len = max(10, int(input_length * 0.6))
    min_len = max(5, int(input_length * 0.3))
    result = summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)
    return result[0]["summary_text"]

def generate_word_freq(text):
    stop_words = {
        'the','a','an','and','or','but','in','on','at','to','for','of','with',
        'is','are','was','were','be','been','have','has','had','this','that',
        'it','its','by','from','as','not','can','will','do','did','so','if',
        'we','they','he','she','his','her','their','our','i','you','your','my',
        'all','also','more','about','than','up','out','into','over','after',
        'new','said','says','would','could','should','may','just','been','one'
    }
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    filtered = [w for w in words if w not in stop_words]
    freq = Counter(filtered).most_common(30)
    return freq

# =========================
# Category Label Mapping
# =========================
id2label = {
    0: 'CRIME', 1: 'DIVORCE', 2: 'ENTERTAINMENT', 3: 'FOOD & DRINK',
    4: 'HOME & LIVING', 5: 'MONEY', 6: 'PARENTING', 7: 'POLITICS',
    8: 'QUEER VOICES', 9: 'SPORTS', 10: 'STYLE & BEAUTY', 11: 'TRAVEL',
    12: 'WEDDINGS', 13: 'WELLNESS', 14: 'WORLD NEWS'
}

category_icons = {
    'CRIME': '🚔', 'DIVORCE': '⚖️', 'ENTERTAINMENT': '🎬',
    'FOOD & DRINK': '🍽️', 'HOME & LIVING': '🏠', 'MONEY': '💰',
    'PARENTING': '👨‍👩‍👧', 'POLITICS': '🏛️', 'QUEER VOICES': '🏳️‍🌈',
    'SPORTS': '⚽', 'STYLE & BEAUTY': '💄', 'TRAVEL': '✈️',
    'WEDDINGS': '💍', 'WELLNESS': '🧘', 'WORLD NEWS': '🌍'
}

def decode_label(label):
    idx = int(label.split("_")[-1])
    return id2label[idx]

# =========================
# Session State
# =========================
if "history" not in st.session_state:
    st.session_state.history = []
if "text_input" not in st.session_state:
    st.session_state.text_input = ""
if "tasks" not in st.session_state:
    st.session_state.tasks = ["All"]
if "total_analyzed" not in st.session_state:
    st.session_state.total_analyzed = 0
if "fake_count" not in st.session_state:
    st.session_state.fake_count = 0
if "real_count" not in st.session_state:
    st.session_state.real_count = 0
if "top_categories" not in st.session_state:
    st.session_state.top_categories = []

# =========================
# CSS — Theme-Aware
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ==============================
   CSS VARIABLES (Dark Default)
============================== */
:root {
    --bg-primary: #020617;
    --bg-secondary: #0f172a;
    --bg-card: rgba(15, 23, 42, 0.7);
    --border-color: rgba(59, 130, 246, 0.15);
    --text-primary: #e2e8f0;
    --text-secondary: #94a3b8;
    --text-muted: #475569;
    --accent-blue: #3b82f6;
    --accent-cyan: #06b6d4;
    --accent-green: #22c55e;
    --accent-red: #ef4444;
    --accent-amber: #f59e0b;
    --glow-blue: rgba(59, 130, 246, 0.25);
    --glow-cyan: rgba(6, 182, 212, 0.2);
    --shadow-card: 0 8px 32px rgba(0, 0, 0, 0.4);
    --backdrop-blur: blur(16px);
    --ticker-bg: #0f172a;
    --wc-opacity: 0.05;
    --gradient-hero: linear-gradient(135deg, #020617 0%, #0f172a 50%, #020617 100%);
}

/* Light mode overrides via Streamlit's [data-theme] */
[data-theme="light"] {
    --bg-primary: #f8fafc;
    --bg-secondary: #f1f5f9;
    --bg-card: rgba(255, 255, 255, 0.85);
    --border-color: rgba(59, 130, 246, 0.18);
    --text-primary: #0f172a;
    --text-secondary: #475569;
    --text-muted: #94a3b8;
    --glow-blue: rgba(59, 130, 246, 0.12);
    --glow-cyan: rgba(6, 182, 212, 0.1);
    --shadow-card: 0 4px 24px rgba(0, 0, 0, 0.08);
    --ticker-bg: #1e293b;
    --wc-opacity: 0.04;
    --gradient-hero: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 50%, #f8fafc 100%);
}

/* ==============================
   BASE RESET
============================== */
.stApp {
    background: var(--gradient-hero) !important;
    font-family: 'DM Sans', sans-serif;
    min-height: 100vh;
}

.main .block-container {
    padding-top: 0 !important;
    max-width: 1200px !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}

/* ==============================
   WORD CLOUD BACKGROUND
============================== */
.wordcloud-bg {
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    pointer-events: none;
    z-index: 0;
    overflow: hidden;
}

.wordcloud-bg span {
    position: absolute;
    font-family: 'Playfair Display', serif;
    font-weight: 900;
    color: var(--text-primary);
    opacity: var(--wc-opacity);
    filter: blur(1.5px);
    user-select: none;
    white-space: nowrap;
}

/* ==============================
   NEWS TICKER
============================== */
.ticker-wrapper {
    width: 100%;
    background: var(--ticker-bg);
    border-bottom: 1px solid var(--border-color);
    padding: 10px 0;
    overflow: hidden;
    position: relative;
    z-index: 100;
    margin-bottom: 0;
}

.ticker-inner {
    display: flex;
    align-items: center;
    gap: 0;
}

.ticker-label {
    background: var(--accent-red);
    color: #fff;
    font-family: 'DM Sans', sans-serif;
    font-weight: 700;
    font-size: 11px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 4px 14px;
    white-space: nowrap;
    flex-shrink: 0;
    z-index: 2;
    position: relative;
    margin-right: 12px;
    border-radius: 3px;
}

.ticker-label::after {
    content: '';
    position: absolute;
    right: -8px;
    top: 0;
    width: 0;
    height: 100%;
    border-left: 8px solid var(--accent-red);
    border-top: 12px solid transparent;
    border-bottom: 12px solid transparent;
}

.ticker-track {
    display: flex;
    animation: ticker-scroll 35s linear infinite;
    white-space: nowrap;
    overflow: hidden;
    flex: 1;
}

.ticker-text {
    color: var(--text-secondary);
    font-size: 13px;
    font-weight: 400;
    letter-spacing: 0.3px;
    padding-right: 80px;
    white-space: nowrap;
}

.ticker-text .dot {
    color: var(--accent-cyan);
    margin: 0 8px;
    font-weight: 700;
}

@keyframes ticker-scroll {
    0% { transform: translateX(0); }
    100% { transform: translateX(-50%); }
}

/* ==============================
   HEADER / MASTHEAD
============================== */
.masthead {
    padding: 40px 0 20px 0;
    text-align: center;
    position: relative;
    z-index: 10;
}

.masthead-eyebrow {
    display: inline-block;
    font-family: 'DM Sans', sans-serif;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--accent-cyan);
    margin-bottom: 12px;
    padding: 4px 16px;
    border: 1px solid rgba(6, 182, 212, 0.3);
    border-radius: 20px;
    background: rgba(6, 182, 212, 0.08);
}

.masthead-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(36px, 5vw, 64px);
    font-weight: 900;
    color: var(--text-primary);
    line-height: 1.05;
    margin: 0 0 12px 0;
    letter-spacing: -1px;
}

.masthead-title span {
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.masthead-subtitle {
    font-family: 'DM Sans', sans-serif;
    font-size: 16px;
    font-weight: 300;
    color: var(--text-secondary);
    max-width: 520px;
    margin: 0 auto;
    line-height: 1.6;
}

.masthead-divider {
    width: 60px;
    height: 2px;
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan));
    margin: 20px auto 0;
    border-radius: 2px;
}

/* ==============================
   STATS STRIP
============================== */
.stats-strip {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin: 32px 0;
    position: relative;
    z-index: 10;
}

.stat-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 14px;
    padding: 20px 18px;
    backdrop-filter: var(--backdrop-blur);
    -webkit-backdrop-filter: var(--backdrop-blur);
    box-shadow: var(--shadow-card);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    position: relative;
    overflow: hidden;
}

.stat-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    border-radius: 14px 14px 0 0;
}

.stat-card.blue::before { background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan)); }
.stat-card.red::before { background: var(--accent-red); }
.stat-card.green::before { background: var(--accent-green); }
.stat-card.amber::before { background: var(--accent-amber); }

.stat-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 16px 40px rgba(0, 0, 0, 0.3), 0 0 20px var(--glow-blue);
}

.stat-icon {
    font-size: 22px;
    margin-bottom: 10px;
    display: block;
}

.stat-value {
    font-family: 'Playfair Display', serif;
    font-size: 32px;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1;
    margin-bottom: 4px;
}

.stat-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 6px;
}

.stat-desc {
    font-size: 12px;
    color: var(--text-secondary);
    line-height: 1.4;
}

/* ==============================
   SECTION DIVIDER
============================== */
.section-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 36px 0 20px 0;
    position: relative;
    z-index: 10;
}

.section-header h2 {
    font-family: 'Playfair Display', serif;
    font-size: 22px;
    font-weight: 700;
    color: var(--text-primary) !important;
    margin: 0;
}

.section-line {
    flex: 1;
    height: 1px;
    background: var(--border-color);
}

/* ==============================
   FEATURE CARDS
============================== */
.feature-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 40px;
    position: relative;
    z-index: 10;
}

.feature-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 16px;
    padding: 24px 20px;
    backdrop-filter: var(--backdrop-blur);
    -webkit-backdrop-filter: var(--backdrop-blur);
    box-shadow: var(--shadow-card);
    transition: all 0.25s ease;
    cursor: pointer;
    text-align: center;
    position: relative;
    overflow: hidden;
}

.feature-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 3px;
    opacity: 0;
    transition: opacity 0.25s ease;
}

.feature-card.fc-red::after { background: var(--accent-red); }
.feature-card.fc-blue::after { background: var(--accent-blue); }
.feature-card.fc-cyan::after { background: var(--accent-cyan); }
.feature-card.fc-green::after { background: var(--accent-green); }

.feature-card:hover {
    transform: translateY(-5px);
    border-color: rgba(59, 130, 246, 0.4);
    box-shadow: 0 20px 48px rgba(0, 0, 0, 0.35), 0 0 24px var(--glow-blue);
}

.feature-card:hover::after {
    opacity: 1;
}

.feature-icon {
    font-size: 36px;
    margin-bottom: 12px;
    display: block;
}

.feature-title {
    font-family: 'Playfair Display', serif;
    font-size: 17px;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 8px;
}

.feature-desc {
    font-size: 12px;
    color: var(--text-secondary);
    line-height: 1.5;
}

/* ==============================
   ANALYSIS PANEL
============================== */
.analysis-panel {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 20px;
    padding: 32px;
    backdrop-filter: var(--backdrop-blur);
    -webkit-backdrop-filter: var(--backdrop-blur);
    box-shadow: var(--shadow-card);
    margin-bottom: 32px;
    position: relative;
    z-index: 10;
}

.panel-title {
    font-family: 'Playfair Display', serif;
    font-size: 20px;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 10px;
}

/* ==============================
   RESULT CARDS
============================== */
.result-strip {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
    margin-top: 24px;
    position: relative;
    z-index: 10;
}

.result-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 14px;
    padding: 20px;
    backdrop-filter: var(--backdrop-blur);
    box-shadow: var(--shadow-card);
    text-align: center;
}

.result-card .rc-icon { font-size: 28px; margin-bottom: 8px; }
.result-card .rc-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 6px;
}
.result-card .rc-value {
    font-family: 'Playfair Display', serif;
    font-size: 22px;
    font-weight: 700;
    color: var(--text-primary);
}
.result-card .rc-score {
    font-size: 12px;
    color: var(--text-secondary);
    margin-top: 4px;
}

/* ==============================
   SUMMARY COLUMNS
============================== */
.summary-cols {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    margin-top: 24px;
    position: relative;
    z-index: 10;
}

.summary-box {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 14px;
    padding: 20px;
    backdrop-filter: var(--backdrop-blur);
    box-shadow: var(--shadow-card);
}

.summary-box h4 {
    font-family: 'Playfair Display', serif;
    font-size: 15px;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0 0 12px 0;
    padding-bottom: 10px;
    border-bottom: 1px solid var(--border-color);
    display: flex;
    align-items: center;
    gap: 8px;
}

.summary-box p {
    font-size: 14px;
    color: var(--text-secondary);
    line-height: 1.7;
    margin: 0;
}

/* ==============================
   WORD CLOUD CHART
============================== */
.wordcloud-display {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 16px;
    padding: 24px;
    backdrop-filter: var(--backdrop-blur);
    box-shadow: var(--shadow-card);
    margin-top: 24px;
    position: relative;
    z-index: 10;
}

.wordcloud-display h4 {
    font-family: 'Playfair Display', serif;
    font-size: 16px;
    color: var(--text-primary);
    margin: 0 0 20px 0;
}

.wc-word-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    justify-content: center;
    align-items: center;
}

.wc-word {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 6px;
    background: rgba(59, 130, 246, 0.08);
    border: 1px solid rgba(59, 130, 246, 0.15);
    color: var(--text-secondary);
    transition: all 0.2s ease;
    cursor: default;
    white-space: nowrap;
}

.wc-word:hover {
    background: rgba(59, 130, 246, 0.18);
    color: var(--text-primary);
    transform: scale(1.05);
}

/* ==============================
   HISTORY SIDEBAR
============================== */
.sidebar-title {
    font-family: 'Playfair Display', serif;
    font-size: 18px;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 16px;
    padding-bottom: 10px;
    border-bottom: 1px solid var(--border-color);
}

/* ==============================
   BUTTON OVERRIDES
============================== */
div.stButton > button {
    width: 100% !important;
    height: 48px !important;
    border-radius: 10px !important;
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan)) !important;
    color: #fff !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    border: none !important;
    box-shadow: 0 4px 16px var(--glow-blue) !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}

div.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px var(--glow-blue) !important;
    opacity: 0.92 !important;
}

div.stButton > button:disabled {
    opacity: 0.45 !important;
    cursor: not-allowed !important;
    transform: none !important;
}

/* ==============================
   TEXTAREA / SELECT OVERRIDES
============================== */
.stTextArea textarea {
    border-radius: 12px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    line-height: 1.6 !important;
    border: 1px solid var(--border-color) !important;
    background: var(--bg-secondary) !important;
    color: var(--text-primary) !important;
    resize: vertical !important;
}

.stTextArea textarea:focus {
    border-color: var(--accent-blue) !important;
    box-shadow: 0 0 0 2px var(--glow-blue) !important;
}

.stMultiSelect > div > div {
    border-radius: 10px !important;
    border-color: var(--border-color) !important;
    background: var(--bg-secondary) !important;
}

/* ==============================
   LABELS & TEXT
============================== */
.stTextArea label, .stMultiSelect label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    color: var(--text-secondary) !important;
    text-transform: uppercase !important;
}

/* ==============================
   ALERT/STATUS OVERRIDES
============================== */
.stAlert {
    border-radius: 12px !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ==============================
   PROGRESS BAR
============================== */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan)) !important;
}

/* ==============================
   SCROLLBAR
============================== */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: rgba(59, 130, 246, 0.3);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover { background: rgba(59, 130, 246, 0.5); }

/* ==============================
   HIDE DEFAULT STREAMLIT UI
============================== */
#MainMenu, footer, .stDeployButton { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent !important; }

/* ==============================
   FAKE NEWS BANNER
============================== */
.fake-banner {
    padding: 16px 20px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 8px;
    font-family: 'DM Sans', sans-serif;
    font-size: 15px;
    font-weight: 600;
}

.fake-banner.fake {
    background: rgba(239, 68, 68, 0.12);
    border: 1px solid rgba(239, 68, 68, 0.3);
    color: #ef4444;
}

.fake-banner.real {
    background: rgba(34, 197, 94, 0.1);
    border: 1px solid rgba(34, 197, 94, 0.25);
    color: #22c55e;
}

.fake-banner.suspicious {
    background: rgba(245, 158, 11, 0.1);
    border: 1px solid rgba(245, 158, 11, 0.3);
    color: #f59e0b;
}

/* ==============================
   CONFIDENCE BAR
============================== */
.confidence-bar-wrapper {
    background: rgba(148, 163, 184, 0.1);
    border-radius: 6px;
    height: 6px;
    overflow: hidden;
    margin-top: 8px;
}
.confidence-bar-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 0.6s ease;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Word Cloud Background
# =========================
wc_words = [
    ("BREAKING", 5, 8, 900, 42), ("POLITICS", 20, 65, 700, 28),
    ("AI", 50, 35, 900, 56), ("ECONOMY", 75, 15, 600, 22),
    ("WORLD", 10, 80, 800, 34), ("CLIMATE", 60, 70, 600, 20),
    ("TECH", 85, 50, 900, 38), ("ELECTION", 35, 25, 700, 24),
    ("MARKETS", 5, 45, 600, 18), ("CRISIS", 70, 85, 800, 30),
    ("ANALYSIS", 45, 5, 500, 16), ("BREAKING", 88, 30, 700, 22),
    ("GLOBAL", 25, 55, 600, 20), ("SCIENCE", 55, 92, 700, 26),
    ("HEALTH", 80, 75, 600, 19), ("ENERGY", 15, 20, 800, 32),
    ("FINANCE", 65, 40, 600, 18), ("SECURITY", 40, 75, 700, 24),
    ("INNOVATION", 3, 60, 500, 15), ("CONFLICT", 78, 5, 700, 22),
]

wc_spans = " ".join([
    f'<span style="left:{x}%;top:{y}%;font-weight:{fw};font-size:{fs}px;">{word}</span>'
    for word, x, y, fw, fs in wc_words
])

st.markdown(f'<div class="wordcloud-bg">{wc_spans}</div>', unsafe_allow_html=True)

# =========================
# NEWS TICKER
# =========================
ticker_content = (
    'BREAKING NEWS <span class="dot">•</span> '
    'AI surpasses human benchmarks in reasoning tests '
    '<span class="dot">•</span> Global markets rise amid positive trade signals '
    '<span class="dot">•</span> Climate summit reaches historic agreement '
    '<span class="dot">•</span> Tech giants face new regulatory scrutiny '
    '<span class="dot">•</span> Scientists discover potential breakthrough in clean energy '
    '<span class="dot">•</span> Central banks signal interest rate pivot '
    '<span class="dot">•</span> Space agency announces next-generation mission '
    '<span class="dot">•</span> BREAKING NEWS <span class="dot">•</span> '
    'AI surpasses human benchmarks in reasoning tests '
    '<span class="dot">•</span> Global markets rise amid positive trade signals '
    '<span class="dot">•</span> Climate summit reaches historic agreement '
    '<span class="dot">•</span> Tech giants face new regulatory scrutiny '
    '<span class="dot">•</span> Scientists discover potential breakthrough in clean energy '
    '<span class="dot">•</span> Central banks signal interest rate pivot '
    '<span class="dot">•</span> Space agency announces next-generation mission '
)

st.markdown(f"""
<div class="ticker-wrapper">
    <div class="ticker-inner">
        <span class="ticker-label">Breaking</span>
        <div class="ticker-track">
            <div class="ticker-text">{ticker_content}</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# MASTHEAD
# =========================
st.markdown("""
<div class="masthead">
    <div class="masthead-eyebrow">AI-Powered Intelligence Platform</div>
    <h1 class="masthead-title">News<span>Intel</span> AI</h1>
    <p class="masthead-subtitle">
        Advanced machine learning for fake news detection, content classification,
        intelligent summarization &amp; keyword analysis.
    </p>
    <div class="masthead-divider"></div>
</div>
""", unsafe_allow_html=True)

# =========================
# STATS STRIP
# =========================
top_cat = st.session_state.top_categories[-1] if st.session_state.top_categories else "—"

st.markdown(f"""
<div class="stats-strip">
    <div class="stat-card blue">
        <span class="stat-icon">📊</span>
        <div class="stat-value">{st.session_state.total_analyzed}</div>
        <div class="stat-label">Total Analyzed</div>
        <div class="stat-desc">Articles processed since session start</div>
    </div>
    <div class="stat-card red">
        <span class="stat-icon">🚨</span>
        <div class="stat-value">{st.session_state.fake_count}</div>
        <div class="stat-label">Fake Detected</div>
        <div class="stat-desc">Potentially misleading articles flagged</div>
    </div>
    <div class="stat-card green">
        <span class="stat-icon">✅</span>
        <div class="stat-value">{st.session_state.real_count}</div>
        <div class="stat-label">Real News</div>
        <div class="stat-desc">Verified credible articles found</div>
    </div>
    <div class="stat-card amber">
        <span class="stat-icon">🏷️</span>
        <div class="stat-value" style="font-size:18px;line-height:1.3;">{top_cat}</div>
        <div class="stat-label">Top Category</div>
        <div class="stat-desc">Most frequently classified topic</div>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# FEATURE CARDS
# =========================
st.markdown("""
<div class="section-header">
    <h2>Core Intelligence Modules</h2>
    <div class="section-line"></div>
</div>

<div class="feature-grid">
    <div class="feature-card fc-red">
        <span class="feature-icon">🚨</span>
        <div class="feature-title">Fake News Detection</div>
        <div class="feature-desc">BERT-powered model identifies misinformation with confidence scoring</div>
    </div>
    <div class="feature-card fc-blue">
        <span class="feature-icon">🏷️</span>
        <div class="feature-title">News Classification</div>
        <div class="feature-desc">Categorize articles across 15 topic domains automatically</div>
    </div>
    <div class="feature-card fc-cyan">
        <span class="feature-icon">🧠</span>
        <div class="feature-title">Smart Summarization</div>
        <div class="feature-desc">Distil long articles into concise, readable summaries instantly</div>
    </div>
    <div class="feature-card fc-green">
        <span class="feature-icon">☁️</span>
        <div class="feature-title">Word Cloud</div>
        <div class="feature-desc">Visualize key terms and topic frequency from your text</div>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown('<div class="sidebar-title">📜 Analysis History</div>', unsafe_allow_html=True)
    if st.session_state.history:
        for i, text in enumerate(st.session_state.history[::-1]):
            preview = text[:35] + "..." if len(text) > 35 else text
            if st.button(f"📄 {preview}", key=f"hist_{i}_{hash(text)}"):
                st.session_state.text_input = text
                st.rerun()
    else:
        st.markdown('<p style="color:var(--text-muted);font-size:13px;">No history yet — analyze some articles to get started.</p>', unsafe_allow_html=True)

# =========================
# INPUT PANEL
# =========================
st.markdown("""
<div class="section-header">
    <h2>Analyze Content</h2>
    <div class="section-line"></div>
</div>
""", unsafe_allow_html=True)

MAX_CHARS = 2000

with st.container():
    st.markdown('<div class="analysis-panel">', unsafe_allow_html=True)

    user_input = st.text_area(
        "Paste article or news text",
        key="text_input",
        height=180,
        placeholder="Paste your article, news headline, or any text here for AI-powered analysis...",
        help=f"Maximum {MAX_CHARS} characters. Longer text is truncated automatically."
    )

    if len(user_input) > MAX_CHARS:
        st.warning(f"⚠️ Text truncated to {MAX_CHARS} characters for optimal model performance.")
        user_input = user_input[:MAX_CHARS]

    all_tasks = ["All", "Fake Detection", "Classification", "Summarization", "Word Cloud"]

    selected_tasks = st.multiselect(
        "Select Analysis Modules",
        all_tasks,
        default=st.session_state.tasks
    )

    if "All" in selected_tasks:
        tasks = ["Fake Detection", "Classification", "Summarization", "Word Cloud"]
    else:
        tasks = selected_tasks

    st.session_state.tasks = selected_tasks

    col1, col2 = st.columns([3, 1])
    is_disabled = not user_input.strip()

    with col1:
        analyze_btn = st.button("🔍  Run Analysis", disabled=is_disabled)
    with col2:
        if st.button("✕  Clear"):
            st.session_state.text_input = ""
            st.session_state.history = []
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# ANALYSIS EXECUTION
# =========================
if analyze_btn:
    if not user_input.strip():
        st.warning("Please enter text before running analysis.")
    else:
        results = {}

        with st.spinner(""):
            if "Fake Detection" in tasks:
                fake_label, fake_score = detect_fake(user_input)
                results["fake"] = fake_label
                results["fake_score"] = fake_score

            if "Classification" in tasks:
                category_label, category_score = classify_text(user_input)
                results["category"] = category_label
                results["category_score"] = category_score

            if "Summarization" in tasks:
                results["summary"] = summarize_text(user_input)

            if "Word Cloud" in tasks:
                results["word_freq"] = generate_word_freq(user_input)

        # Update session stats
        if user_input not in st.session_state.history:
            st.session_state.history.append(user_input)
        st.session_state.total_analyzed += 1

        if "fake" in results:
            if "Fake" in results["fake"]:
                st.session_state.fake_count += 1
            elif "Real" in results["fake"]:
                st.session_state.real_count += 1
            # Suspicious doesn't increment either counter

        if "category" in results:
            cat_name = decode_label(results["category"])
            st.session_state.top_categories.append(cat_name)

        # ---- RESULTS HEADER ----
        st.markdown("""
        <div class="section-header" style="margin-top:40px;">
            <h2>Analysis Results</h2>
            <div class="section-line"></div>
        </div>
        """, unsafe_allow_html=True)

        # ---- FAKE NEWS BANNER ----
        if "fake" in results:
            lbl = results["fake"]
            is_fake = "Fake" in lbl
            is_suspicious = "Suspicious" in lbl
            if is_fake:
                banner_class, banner_icon = "fake", "🚨"
                banner_text = f"MISINFORMATION ALERT — This article shows {round(results['fake_score']*100, 1)}% probability of being fake."
            elif is_suspicious:
                banner_class, banner_icon = "suspicious", "⚠️"
                banner_text = f"UNVERIFIED CONTENT — Credibility is uncertain ({round(results['fake_score']*100, 1)}% confidence). Treat with caution."
            else:
                banner_class, banner_icon = "real", "✅"
                banner_text = f"CREDIBILITY VERIFIED — This article appears authentic with {round(results['fake_score']*100, 1)}% confidence."
            st.markdown(f"""
            <div class="fake-banner {banner_class}">
                <span style="font-size:20px;">{banner_icon}</span>
                <span>{banner_text}</span>
            </div>
            """, unsafe_allow_html=True)

        # ---- METRICS ROW ----
        # Build list of (html_string) for each active card, then render each
        # in its own st.columns() cell — avoids Streamlit truncating joined HTML.
        active_cards = []

        if "fake" in results:
            lbl = results["fake"]
            if "Fake" in lbl:
                f_color, f_icon = "#ef4444", "🚨"
            elif "Suspicious" in lbl:
                f_color, f_icon = "#f59e0b", "⚠️"
            else:
                f_color, f_icon = "#22c55e", "✅"
            f_pct = round(results["fake_score"] * 100, 1)
            active_cards.append(f"""
<div class="result-card">
    <div class="rc-icon">{f_icon}</div>
    <div class="rc-label">Fake / Real</div>
    <div class="rc-value" style="color:{f_color};">{results['fake']}</div>
    <div class="rc-score">Confidence: {f_pct}%</div>
    <div class="confidence-bar-wrapper" style="margin-top:10px;">
        <div class="confidence-bar-fill" style="width:{f_pct}%;background:{f_color};"></div>
    </div>
</div>""")

        if "category" in results:
            cat_name = decode_label(results["category"])
            cat_icon = category_icons.get(cat_name, "🏷️")
            c_pct = round(results["category_score"] * 100, 1)
            active_cards.append(f"""
<div class="result-card">
    <div class="rc-icon">{cat_icon}</div>
    <div class="rc-label">Category</div>
    <div class="rc-value" style="font-size:16px;">{cat_name}</div>
    <div class="rc-score">Confidence: {c_pct}%</div>
    <div class="confidence-bar-wrapper" style="margin-top:10px;">
        <div class="confidence-bar-fill" style="width:{c_pct}%;background:var(--accent-blue);"></div>
    </div>
</div>""")

        if "summary" in results:
            wc = len(results["summary"].split())
            orig_wc = len(user_input.split())
            ratio = round((wc / orig_wc) * 100) if orig_wc > 0 else 0
            active_cards.append(f"""
<div class="result-card">
    <div class="rc-icon">📝</div>
    <div class="rc-label">Summary</div>
    <div class="rc-value">{wc} words</div>
    <div class="rc-score">Compressed to {ratio}% of original</div>
    <div class="confidence-bar-wrapper" style="margin-top:10px;">
        <div class="confidence-bar-fill" style="width:{ratio}%;background:var(--accent-cyan);"></div>
    </div>
</div>""")

        # Render each card in its own native Streamlit column cell.
        # This is the safe pattern: one st.markdown per column, no shared wrapper div.
        if active_cards:
            cols = st.columns(len(active_cards))
            for col, card_html in zip(cols, active_cards):
                with col:
                    st.markdown(card_html, unsafe_allow_html=True)

        # ---- SUMMARY COLUMNS ----
        # Use native st.columns + st.markdown per box so user text (which may
        # contain HTML special chars) never breaks the surrounding HTML structure.
        if "summary" in results:
            import html as html_lib
            orig_preview = html_lib.escape(user_input[:600]) + ("..." if len(user_input) > 600 else "")
            summary_escaped = html_lib.escape(results["summary"])

            col_orig, col_summ = st.columns(2)
            with col_orig:
                st.markdown(f"""
<div class="summary-box">
    <h4>📄 Original Text</h4>
    <p>{orig_preview}</p>
</div>""", unsafe_allow_html=True)
            with col_summ:
                st.markdown(f"""
<div class="summary-box">
    <h4>🧠 AI Summary</h4>
    <p>{summary_escaped}</p>
</div>""", unsafe_allow_html=True)

        # ---- WORD CLOUD ----
        if "word_freq" in results and results["word_freq"]:
            freq_data = results["word_freq"]
            max_freq = freq_data[0][1] if freq_data else 1

            words_html = ""
            for word, count in freq_data:
                size_ratio = count / max_freq
                font_size = int(12 + size_ratio * 28)
                opacity = 0.5 + size_ratio * 0.5
                words_html += f'<span class="wc-word" style="font-size:{font_size}px;opacity:{opacity};" title="{count} occurrences">{word}</span>'

            st.markdown(f"""
            <div class="wordcloud-display">
                <h4>☁️ Key Terms Visualization</h4>
                <div class="wc-word-grid">{words_html}</div>
            </div>
            """, unsafe_allow_html=True)

# =========================
# FOOTER
# =========================
st.markdown("""
<div style="text-align:center;padding:48px 0 24px;position:relative;z-index:10;">
    <div style="width:40px;height:1px;background:var(--border-color);margin:0 auto 16px;"></div>
    <p style="font-family:'DM Sans',sans-serif;font-size:12px;color:var(--text-muted);letter-spacing:1px;text-transform:uppercase;margin:0;">
        NewsIntel AI &nbsp;·&nbsp; Powered by BERT &amp; DistilBART &nbsp;·&nbsp; Built with Streamlit
    </p>
</div>
""", unsafe_allow_html=True)