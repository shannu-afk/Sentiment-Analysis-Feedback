import asyncio
try:
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
except AttributeError:
    pass

import streamlit as st
from transformers import pipeline
from sentiment import analyze_sentiment

st.set_page_config(
    page_title="🧠 Sentiment Feedback Analyzer",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Animated gradient background CSS
st.markdown(
    """
    <style>
    /* Fullscreen animated gradient background */
    body, .stApp {
        margin: 0;
        padding: 0;
        height: 100vh;
        background: linear-gradient(-45deg, #ee7752, #e73c7e, #23d5ab, #23a6d5);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: white;
    }

    @keyframes gradientBG {
        0% {
            background-position: 0% 50%;
        }
        50% {
            background-position: 100% 50%;
        }
        100% {
            background-position: 0% 50%;
        }
    }

    /* Container styling */
    .main-container {
        background: rgba(0, 0, 0, 0.6);
        border-radius: 20px;
        padding: 30px 40px;
        max-width: 700px;
        margin: 50px auto;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.7);
    }

    /* Title style */
    h1 {
        text-align: center;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 10px;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.7);
    }

    /* Text area style */
    textarea {
        background-color: #222;
        color: #eee;
        border-radius: 12px;
        border: 2px solid #444;
        padding: 15px;
        font-size: 1.1rem;
        resize: vertical;
    }

    /* Sentiment result box */
    .result-box {
        background-color: rgba(255, 255, 255, 0.1);
        border-left: 8px solid;
        padding: 20px;
        border-radius: 12px;
        margin-top: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
        font-size: 1.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# App container
st.markdown('<div class="main-container">', unsafe_allow_html=True)

st.markdown("### 🧠 Real-Time Sentiment Feedback App")
st.markdown(
    '<p style="text-align:center; margin-bottom:30px;">Type or paste your feedback below and get instant sentiment analysis.</p>',
    unsafe_allow_html=True,
)

# Load zero-shot classifier once
zero_shot_classifier = pipeline("zero-shot-classification")

def is_feedback(text: str) -> bool:
    candidate_labels = ["feedback", "review", "comment", "complaint", "suggestion", "opinion", "praise", "rating"]
    text_lower = text.lower()
    feedback_keywords = ["like", "love", "hate", "dislike", "suggest", "problem", "issue", "complain", "feedback", "review"]
    if any(word in text_lower for word in feedback_keywords):
        return True
    result = zero_shot_classifier(text, candidate_labels)
    for label, score in zip(result['labels'], result['scores']):
        if label in candidate_labels and score > 0.4:
            return True
    return False

user_input = st.text_area("Enter your feedback:", height=180)

if user_input.strip():
    if is_feedback(user_input):
        with st.spinner("Analyzing sentiment..."):
            result = analyze_sentiment(user_input)

        label = result["label"]
        score = result["score"]
        emoji = result["emoji"]

        color_map = {
            "POSITIVE": "#4CAF50",
            "NEGATIVE": "#F44336",
            "NEUTRAL": "#FFC107"
        }
        box_color = color_map.get(label, "#FFC107")

        st.markdown(
            f"""
            <div class="result-box" style="border-left-color: {box_color};">
                <strong>Sentiment:</strong> {label} {emoji}<br>
                <strong>Confidence:</strong> {score:.2%}
            </div>
            """,
            unsafe_allow_html=True,
        )

        if label == "POSITIVE":
            st.success("🎉 Glad to see positive vibes! Keep it up!")
        elif label == "NEGATIVE":
            st.error("💪 Looks like there’s some negativity. Hope things improve soon!")
        else:
            st.info("😐 Neutral sentiment detected.")
    else:
        st.warning("⚠️ Not a proper feedback — please enter a relevant comment.")
else:
    st.info("📝 Please enter some text to analyze.")

st.markdown('</div>', unsafe_allow_html=True)
