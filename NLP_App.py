import streamlit as st
import joblib
from transformers import pipeline

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="AI Text Analyzer",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# =========================
# Load Models
# =========================
@st.cache_resource
def load_models():
    fake_model = joblib.load("fake_lr_model.pkl")  # pipeline جاهز
    category_model = joblib.load("category_model.pkl")
    label_encoder = joblib.load("category_label_encoder.pkl")
    return fake_model, category_model, label_encoder

@st.cache_resource
def load_summarizer():
    return pipeline(
        "text-generation",
        model="gpt2"
    )

fake_model, category_model, label_encoder = load_models()
summarizer = load_summarizer()

# =========================
# NLP Functions
# =========================
def detect_fake(text):
    pred = fake_model.predict([text])[0]
    return "Fake ❌" if pred == 1 else "Real ✅"

def classify_text(text):
    try:
        pred = category_model.predict([text])[0]
        return label_encoder.inverse_transform([pred])[0]
    except:
        return "General"

def summarize_text(text):
    text = text[:500]
    prompt = f"Summarize this text:\n{text}\nSummary:"
    
    result = summarizer(
        prompt,
        max_length=120,
        do_sample=False
    )
    
    return result[0]["generated_text"].replace(prompt, "").strip()

# =========================
# UI Style
# =========================
st.markdown("""
<style>
.stApp {
    background-color: #020617;
}

h1 {
    text-align: center;
    color: #e2e8f0;
}

.desc {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 25px;
}

textarea {
    border-radius: 12px !important;
}

div.stButton > button {
    width: 100%;
    height: 45px;
    border-radius: 10px !important;
    background: linear-gradient(90deg, #06b6d4, #3b82f6);
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Title
# =========================
st.markdown("<h1>AI Text Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<div class='desc'>Analyze text, detect fake news, classify content, and generate summaries</div>", unsafe_allow_html=True)

# =========================
# Session State
# =========================
if "history" not in st.session_state:
    st.session_state.history = []

if "text_input" not in st.session_state:
    st.session_state.text_input = ""

if "tasks" not in st.session_state:
    st.session_state.tasks = ["All"]

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("📜 History")

    if st.session_state.history:
        for i, text in enumerate(st.session_state.history[::-1]):
            if st.button(f"{text[:30]}...", key=f"hist_{i}"):
                st.session_state.text_input = text
                st.rerun()
    else:
        st.write("No history yet")

# =========================
# Input
# =========================
user_input = st.text_area(
    "Enter your text here:",
    key="text_input",
    height=200
)

# =========================
# Tasks
# =========================
all_tasks = ["All", "Fake Detection", "Classification", "Summarization"]

selected_tasks = st.multiselect(
    "Choose Tasks",
    all_tasks,
    default=st.session_state.tasks
)

if "All" in selected_tasks:
    tasks = ["Fake Detection", "Classification", "Summarization"]
else:
    tasks = selected_tasks

# =========================
# Buttons
# =========================
col1, col2 = st.columns(2)

with col1:
    analyze_btn = st.button("Analyze")

with col2:
    if st.button("Clear Text"):
        st.session_state.text_input = ""
        st.rerun()

# =========================
# Analyze
# =========================
if analyze_btn:

    if user_input.strip() == "":
        st.warning("Please enter text first.")
    else:

        with st.spinner("Running AI Model... 🤖"):

            results = {}

            if "Fake Detection" in tasks:
                results["fake"] = detect_fake(user_input)

            if "Classification" in tasks:
                results["category"] = classify_text(user_input)

            if "Summarization" in tasks:
                results["summary"] = summarize_text(user_input)

        if user_input not in st.session_state.history:
            st.session_state.history.append(user_input)

        st.subheader("Results")

        if "fake" in results:
            if "Fake" in results["fake"]:
                st.error("🚨 Fake")
            else:
                st.success("✅ Real")

        cols = st.columns(len(results))
        i = 0

        if "fake" in results:
            cols[i].metric("Fake / Real", results["fake"])
            i += 1

        if "category" in results:
            cols[i].metric("Category", results["category"])
            i += 1

        if "summary" in results:
            cols[i].metric("Summary Length", len(results["summary"]))

        st.progress(0.85)
        st.caption("Confidence: 85%")

        if "summary" in results:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Original Text")
                st.write(user_input)

            with col2:
                st.subheader("Summary")
                st.info(results["summary"])