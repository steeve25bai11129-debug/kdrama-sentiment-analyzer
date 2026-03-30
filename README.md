# 🎬 K-Drama Sentiment Analyzer

A Python tool that analyzes the sentiment of K-drama and C-drama reviews — classifying them as **Positive**, **Negative**, or **Mixed/Neutral**, with a confidence score and tone description.

Built as a capstone project for the **Fundamentals of AI and ML** course.

---

## 📌 What It Does

Paste any K-drama review and the tool will:

- Classify the overall sentiment (Positive / Negative / Mixed)
- Show a normalized sentiment score
- Show a confidence percentage
- Describe the tone (e.g., "highly emotional", "mostly factual")
- Highlight which specific words drove the classification

**Example output:**

```
==============================================================
  K-DRAMA SENTIMENT ANALYZER
==============================================================

  Review    : "Crash Landing on You was absolutely breathtaking! The chemistry..."
  Verdict   : Positive 😊
  Score     : +1.287
  Confidence: [████████████████████] 100.0%
  Tone      : highly emotional / opinionated
  Positive  : breathtaking, best, chemistry, cried
==============================================================
```

---

## 🧠 How It Works

The analyzer uses a rule-based NLP pipeline:

1. **Tokenization** — lowercases the review and splits it into word tokens
2. **Lexicon Matching** — checks tokens against a curated positive and negative word list
3. **Negation Handling** — detects negators like "not" or "never" and flips the sentiment sign
4. **Intensifier Weighting** — detects words like "absolutely" and multiplies the score by 1.5
5. **Log Normalization** — divides the raw score by `log(word_count + 2)` to normalize for review length
6. **Confidence Scoring** — combines score magnitude with sentiment vocabulary diversity

No external ML libraries or internet connection required.

---

## 🚀 Getting Started

### Requirements

- Python 3.8 or higher
- No external packages needed (uses only Python standard library)

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/kdrama-sentiment-analyzer.git
cd kdrama-sentiment-analyzer
```

### Run in Interactive Mode

```bash
python sentiment_analyzer.py
```

You'll be prompted to type a review. Type `quit` to exit.

### Run the Demo

```bash
python sentiment_analyzer.py --demo
```

Runs analysis on three pre-written sample reviews.

---

## 📁 Project Structure

```
kdrama-sentiment-analyzer/
│
├── sentiment_analyzer.py   # Main analyzer — run this
└── README.md               # This file
```

---

## 🔬 Concepts Applied

| Concept | Where Used |
|---|---|
| Tokenization | `tokenize()` function |
| Feature Engineering | Sentiment lexicons, negation/intensifier detection |
| Score Normalization | Log-scale normalization by review length |
| Classification | Threshold-based label assignment |
| Confidence Estimation | Weighted formula combining magnitude + vocabulary diversity |

---

## 🔮 Future Improvements

- Train a supervised classifier (Naive Bayes or Logistic Regression) on a labelled dataset
- Add a Flask web interface for browser-based use
- Implement aspect-based sentiment (separate scores for acting, plot, OST)
- Connect to a drama database API for bulk review analysis

---

## 📄 License

MIT License — free to use and modify.

