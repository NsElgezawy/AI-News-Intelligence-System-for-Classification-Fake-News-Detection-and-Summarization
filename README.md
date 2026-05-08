# 📰 NewsIntel AI

> **An AI-powered news intelligence platform** for fake news detection, content classification, smart summarization, and keyword visualization — built with Streamlit and Hugging Face Transformers.

---

## ✨ Overview

NewsIntel AI is a premium, production-grade web application that lets you paste any news article or text and instantly analyze it using state-of-the-art NLP models. The UI is designed to feel like a real newsroom intelligence tool — clean, fast, and professional.

---

## 🖼️ Features

### 🚨 Fake News Detection
Uses **DeBERTa-v3** (NLI-based) to classify text credibility:

| Verdict | Color | Meaning |
|---|---|---|
| Real ✅ | 🟢 Green | ENTAILMENT — article is credible |
| Fake ❌ | 🔴 Red | CONTRADICTION — likely misinformation |
| Suspicious ⚠️ | 🟡 Amber | NEUTRAL — unverifiable, treat with caution |

### 🏷️ News Classification
Classifies articles into **15 topic categories** using a fine-tuned DistilBERT model:

`CRIME` · `DIVORCE` · `ENTERTAINMENT` · `FOOD & DRINK` · `HOME & LIVING` · `MONEY` · `PARENTING` · `POLITICS` · `QUEER VOICES` · `SPORTS` · `STYLE & BEAUTY` · `TRAVEL` · `WEDDINGS` · `WELLNESS` · `WORLD NEWS`

### 🧠 Smart Summarization
Condenses long articles into concise summaries using **DistilBART-CNN**, showing compression ratio and word count.

### ☁️ Word Cloud Visualization
Extracts the top 30 keywords from the input text, sized by frequency, rendered as an interactive tag cloud — directly in the browser.

### 📊 Live Session Stats
A stats strip tracks across the session:
- **Total Analyzed** — articles processed
- **Fake Detected** — flagged articles
- **Real News** — verified articles
- **Top Category** — most recent classification

### 📰 Breaking News Ticker
A smooth-scrolling horizontal ticker at the top of the page with live-style headlines.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **UI Framework** | [Streamlit](https://streamlit.io/) |
| **NLP Models** | [Hugging Face Transformers](https://huggingface.co/transformers/) |
| **Fake Detection** | `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` |
| **Classification** | `nselgezawy/news-classifier-distilbert` |
| **Summarization** | `sshleifer/distilbart-cnn-12-6` |
| **Inference Backend** | PyTorch (CPU or CUDA) |
| **Styling** | Custom CSS — glassmorphism, CSS variables, dark/light theme |
| **Fonts** | Playfair Display · DM Sans (Google Fonts) |

---

## 📁 Project Structure

```
newsintel-ai/
│
├── app.py              # Main Streamlit application
└── README.md           # This file
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/newsintel-ai.git
cd newsintel-ai
```

### 2. Create a Virtual Environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install streamlit transformers torch
```

> **GPU support (optional):** If you have a CUDA-compatible GPU, install the appropriate PyTorch build from [pytorch.org](https://pytorch.org/get-started/locally/) for faster inference. The app auto-detects CUDA availability.

### 4. Run the App

```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`.

---

## 📦 Dependencies

```txt
streamlit
transformers
torch
```

> Models are downloaded automatically from Hugging Face Hub on first run and cached locally. Expect a short delay on the first launch (~1–2 minutes depending on your connection).

---

## 🎨 UI Design

### Dark / Light Mode
The app fully supports **Streamlit's built-in theme toggle** (System / Light / Dark) via the Deploy menu → Settings. No custom toggle is needed — all colors, shadows, and backgrounds adapt automatically through CSS variables.

| Mode | Background | Cards |
|---|---|---|
| Dark | `#020617` deep navy | Glassmorphism with backdrop blur |
| Light | `#f8fafc` soft white | Clean white surfaces |

### Color Palette

| Token | Dark Mode | Usage |
|---|---|---|
| Primary | `#3b82f6` | Buttons, accents, bars |
| Accent | `#06b6d4` | Gradient highlights, ticker |
| Success | `#22c55e` | Real news verdict |
| Error | `#ef4444` | Fake news verdict |
| Warning | `#f59e0b` | Suspicious verdict |

---

## ⚙️ Configuration

All key settings live at the top of `app.py`:

```python
# Maximum characters accepted per input
MAX_CHARS = 2000

# Model input truncation (per model limits)
# Fake detection & classification: 512 tokens
# Summarization: 1000 characters

# GPU device — auto-detected
DEVICE = 0 if torch.cuda.is_available() else -1
```

---

## 🔍 How the Models Work

### Fake News Detection — DeBERTa NLI
`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` is a **Natural Language Inference** model trained on MNLI, FEVER, and ANLI datasets. It outputs one of three labels:

- `ENTAILMENT` → the claim is supported → **Real ✅**
- `CONTRADICTION` → the claim contradicts known facts → **Fake ❌**
- `NEUTRAL` → insufficient evidence → **Suspicious ⚠️**

### News Classification — DistilBERT
`nselgezawy/news-classifier-distilbert` is fine-tuned on a multi-class news corpus to assign articles to one of 15 topic categories.

### Summarization — DistilBART-CNN
`sshleifer/distilbart-cnn-12-6` is a distilled version of BART fine-tuned on CNN/DailyMail for abstractive summarization. Summary length is dynamically scaled to 30–60% of the input word count.

---

## 🧩 Analysis Modules

You can run any combination of modules per analysis:

| Module | Toggle in UI | Description |
|---|---|---|
| Fake Detection | ✅ | NLI-based credibility scoring |
| Classification | ✅ | Topic category assignment |
| Summarization | ✅ | Abstractive summary generation |
| Word Cloud | ✅ | Top-30 keyword frequency map |
| All | ✅ | Runs all four simultaneously |

---

## 🐛 Known Limitations

- **Input length:** Text is truncated at 512 tokens for classification/detection and 1000 chars for summarization due to model context limits.
- **Language:** All models are English-only.
- **Cold start:** First run downloads ~1–2 GB of model weights from Hugging Face Hub.
- **Fake detection nuance:** The NLI model compares text against its training knowledge — it does not browse the web or fact-check against live sources.

---

## 🗺️ Roadmap

- [ ] Add URL input to fetch and analyze articles directly from a link
- [ ] Export results as PDF report
- [ ] Multi-language support
- [ ] Sentiment analysis module
- [ ] Named Entity Recognition (NER) highlighting
- [ ] Batch analysis for multiple articles

---

## 🤝 Contributing

Contributions are welcome! Please open an issue first to discuss what you'd like to change, then submit a pull request.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [Hugging Face](https://huggingface.co/) for the open-source model ecosystem
- [Streamlit](https://streamlit.io/) for the rapid web app framework
- [MoritzLaurer](https://huggingface.co/MoritzLaurer) for the DeBERTa NLI model
- [sshleifer](https://huggingface.co/sshleifer) for DistilBART-CNN

---

<p align="center">
  Built with ❤️ using Streamlit &amp; Hugging Face Transformers
</p>