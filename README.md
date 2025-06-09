# 🧠 Real-Time Sentiment Feedback App

A Streamlit web application that performs **real-time sentiment analysis** on user input feedback.  
It detects if the input text is a valid feedback/comment using zero-shot classification, then analyzes the sentiment (positive, negative, neutral) with a powerful transformer model and gives instant, professional feedback.

---

## Features

- **Real-time sentiment analysis** using Hugging Face transformers.
- Detects if input text is a proper feedback or irrelevant comment.
- Color-coded sentiment results with emojis and confidence scores.
- Friendly and professional user messages based on sentiment.
- Smooth, animated gradient background for a modern, engaging UI.
- Simple, clean interface optimized for readability.

---

## Technologies Used

- [Python 3.10+](https://www.python.org/)
- [Streamlit](https://streamlit.io/) for UI
- [Transformers](https://huggingface.co/transformers/) library from Hugging Face
- [PyTorch](https://pytorch.org/) as backend for model inference
- Pretrained transformer models:
  - Zero-shot classification: `facebook/bart-large-mnli`
  - Sentiment analysis: `distilbert-base-uncased-finetuned-sst-2-english`

---

## Installation

1. **Clone the repository or copy the files**

git clone https://github.com/yourusername/sentiment-feedback-app.git

cd sentiment-feedback-app

Install dependencies

pip install -r requirements.txt

Running the App
Run the Streamlit app using:

streamlit run app.py
