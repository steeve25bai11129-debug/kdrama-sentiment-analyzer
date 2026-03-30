"""
K-Drama Sentiment Analyzer
Analyzes K-drama reviews using rule-based NLP + TF-IDF keyword scoring.

Course: Fundamentals of AI and ML
"""

import re
import math

POSITIVE_WORDS = {
    "amazing", "awesome", "brilliant", "beautiful", "breathtaking", "captivating",
    "charming", "compelling", "cute", "delightful", "emotional", "enchanting",
    "entertaining", "excellent", "fantastic", "fascinating", "flawless", "great",
    "heartwarming", "incredible", "inspiring", "love", "loved", "lovely",
    "magnificent", "masterpiece", "outstanding", "perfect", "phenomenal",
    "powerful", "recommended", "romantic", "stunning", "superb", "touching",
    "unforgettable", "wonderful", "best", "good", "epic", "gem", "iconic",
    "cried", "tears", "addictive", "binge", "obsessed", "excited", "happy",
    "enjoyable", "fun", "sweet", "chemistry"
}

NEGATIVE_WORDS = {
    "awful", "bad", "bland", "boring", "cliche", "confusing", "depressing",
    "disappointing", "disappoints", "dull", "flat", "frustrating", "hate",
    "horrible", "lazy", "mediocre", "messy", "nonsense", "overrated",
    "painful", "pointless", "predictable", "ridiculous", "slow", "stupid",
    "terrible", "tired", "toxic", "trash", "underwhelming", "unfair",
    "unrealistic", "waste", "weak", "worst", "dropped", "dragged", "dragging",
    "forgettable", "rushed", "underdeveloped", "unlikable", "laughable"
}

INTENSIFIERS = {"very", "really", "so", "extremely", "absolutely", "totally", "incredibly"}
NEGATORS     = {"not", "never", "no", "hardly", "barely", "dont", "doesnt", "didnt", "wasnt", "isnt"}


def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def compute_sentiment_score(tokens):
    score = 0.0
    for i, token in enumerate(tokens):
        base = 0.0
        if token in POSITIVE_WORDS:
            base = 1.0
        elif token in NEGATIVE_WORDS:
            base = -1.0
        else:
            continue
        window = tokens[max(0, i-2):i]
        if any(w in NEGATORS for w in window):
            base *= -1
        if any(w in INTENSIFIERS for w in window):
            base *= 1.5
        score += base
    return score


def analyze_sentiment(review):
    tokens = tokenize(review)
    if not tokens:
        return {"label": "No Input", "score": 0.0, "confidence": 0.0,
                "tone": "n/a", "color": "gray", "pos_words": [], "neg_words": []}

    raw_score  = compute_sentiment_score(tokens)
    word_count = len(tokens)
    norm_score = raw_score / math.log(word_count + 2)

    if norm_score > 0.4:
        label, color = "Positive", "green"
    elif norm_score < -0.4:
        label, color = "Negative", "red"
    else:
        label, color = "Mixed / Neutral", "orange"

    pos_hits = [t for t in tokens if t in POSITIVE_WORDS]
    neg_hits = [t for t in tokens if t in NEGATIVE_WORDS]
    sentiment_density = (len(pos_hits) + len(neg_hits)) / word_count

    unique_ratio = len(set(pos_hits + neg_hits)) / max(len(pos_hits + neg_hits), 1)
    confidence = min(round(abs(norm_score) * 60 + unique_ratio * 40, 1), 100.0)

    if sentiment_density > 0.12:
        tone = "highly emotional / opinionated"
    elif sentiment_density > 0.06:
        tone = "moderately expressive"
    else:
        tone = "mostly factual / calm"

    return {
        "label":      label,
        "score":      round(norm_score, 3),
        "raw_score":  round(raw_score, 3),
        "confidence": confidence,
        "tone":       tone,
        "color":      color,
        "word_count": word_count,
        "pos_words":  list(set(pos_hits)),
        "neg_words":  list(set(neg_hits)),
    }


def print_result(review, result):
    preview = review[:100] + ("..." if len(review) > 100 else "")
    bar_len  = int(result["confidence"] / 5)
    bar      = "█" * bar_len + "░" * (20 - bar_len)
    emoji    = {"Positive": "😊", "Negative": "😞", "Mixed / Neutral": "😐"}.get(result["label"], "")

    print("\n" + "="*62)
    print("  K-DRAMA SENTIMENT ANALYZER")
    print("="*62)
    print(f"\n  Review    : \"{preview}\"")
    print(f"  Verdict   : {result['label']} {emoji}")
    print(f"  Score     : {result['score']:+.3f}")
    print(f"  Confidence: [{bar}] {result['confidence']}%")
    print(f"  Tone      : {result['tone']}")
    if result["pos_words"]:
        print(f"  Positive  : {', '.join(result['pos_words'])}")
    if result["neg_words"]:
        print(f"  Negative  : {', '.join(result['neg_words'])}")
    print("="*62 + "\n")


DEMO_REVIEWS = [
    "Crash Landing on You was absolutely breathtaking! The chemistry between the leads was electric and I cried in every single episode. Easily one of the best dramas I have ever watched.",
    "Honestly, the plot was all over the place. The first half was great but it dragged badly toward the end. The villain was so underdeveloped it was almost laughable. Very disappointing.",
    "It was okay. Some episodes were good, some were slow. The OST was nice. Nothing groundbreaking but watchable if you have time."
]


def run_demo():
    print("\n  Running K-Drama Sentiment Analyzer — Demo Mode\n")
    for review in DEMO_REVIEWS:
        result = analyze_sentiment(review)
        print_result(review, result)


def interactive_mode():
    print("\n  K-Drama Sentiment Analyzer")
    print("  Type a review and press Enter. Type 'quit' to exit.\n")
    while True:
        review = input("Review: ").strip()
        if review.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not review:
            continue
        result = analyze_sentiment(review)
        print_result(review, result)


if __name__ == "__main__":
    import sys
    if "--demo" in sys.argv:
        run_demo()
    else:
        interactive_mode()
