from transformers import pipeline

# Load sentiment analysis pipeline (DistilBERT fine-tuned on SST-2)
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def analyze_sentiment(text: str):
    """
    Analyze sentiment of input text.

    Returns a dict with label (POSITIVE/NEGATIVE), score, and emoji.
    """
    if not text.strip():
        return {"label": "NEUTRAL", "score": 0.0, "emoji": "😐"}

    result = classifier(text)[0]
    label = result['label']
    score = result['score']

    emoji_map = {
        "POSITIVE": "😄",
        "NEGATIVE": "😢"
    }
    emoji = emoji_map.get(label, "😐")

    return {
        "label": label,
        "score": score,
        "emoji": emoji
    }
