# =========================================
# AI-Based Mental Health Sentiment Monitoring System
# =========================================

# IMPORT LIBRARIES

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import joblib
import string
import nltk
import gdown
import os

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =========================================
# DOWNLOAD NLTK RESOURCES
# =========================================

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# =========================================
# PAGE CONFIGURATION
# =========================================

st.set_page_config(
    page_title="Mental Health Monitoring",
    page_icon="🧠",
    layout="wide"
)

# =========================================
# DOWNLOAD MODEL FROM GOOGLE DRIVE
# =========================================

model_url = "https://drive.google.com/uc?id=11SduMJijJMfGF1hXxNaDojuymxgzd5Fe"

model_path = "mental_health_rnn_model.h5"

if not os.path.exists(model_path):

    with st.spinner("Downloading AI model... Please wait"):

        gdown.download(
            model_url,
            model_path,
            quiet=False
        )

# =========================================
# LOAD SAVED FILES
# =========================================

model = load_model(
    model_path
)

tokenizer = joblib.load(
    "tokenizer.pkl"
)

label_encoder = joblib.load(
    "label_encoder.pkl"
)

max_length = joblib.load(
    "max_length.pkl"
)

# =========================================
# STOPWORDS
# =========================================

stop_words = set(
    stopwords.words('english')
)

# =========================================
# SECTION 1 — HEADER
# =========================================

st.title(
    "🧠 AI-Based Mental Health Sentiment Monitoring System"
)

st.subheader(
    "Emotion Detection using Simple Recurrent Neural Networks"
)

# =========================================
# SECTION 2 — ABOUT PROJECT
# =========================================

st.markdown("---")

st.header("📘 About the Project")

st.write("""
This AI-powered Mental Health Sentiment Monitoring System
uses Natural Language Processing (NLP) and Simple Recurrent
Neural Networks (RNN) to detect emotional sentiment from text.

Importance of Emotional AI:
- Helps monitor emotional well-being
- Detects negative sentiment trends
- Supports early emotional awareness

NLP Applications:
- Sentiment Analysis
- Chatbots
- Emotion Detection
- Language Translation

Role of RNN:
RNN learns sequential text patterns by remembering
previous words using hidden states.
""")

# =========================================
# TEXT PREPROCESSING FUNCTION
# =========================================

def preprocess_text(text):

    # Lowercase conversion
    text = text.lower()

    # Remove punctuation
    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )

    # Tokenization
    words = word_tokenize(text)

    # Stopword removal
    filtered_words = [
        word for word in words
        if word not in stop_words
    ]

    # Join cleaned words
    text = " ".join(filtered_words)

    # Convert text into sequence
    sequence = tokenizer.texts_to_sequences(
        [text]
    )

    # Padding
    padded = pad_sequences(
        sequence,
        maxlen=max_length,
        padding='post',
        truncating='post'
    )

    return padded

# =========================================
# SECTION 3 — USER INPUT AREA
# =========================================

st.markdown("---")

st.header("✍ User Text Input Area")

st.write("Sample Sentence Suggestions:")

st.code("""
I feel happy and confident today
I feel lonely and depressed
Nobody understands my pain
My anxiety is increasing every day
""")

user_input = st.text_area(
    "Enter your thoughts or feelings here...",
    height=200
)

# =========================================
# SECTION 4 — PREDICTION BUTTON
# =========================================

analyze = st.button(
    "🔍 Analyze Emotion"
)

# =========================================
# PREDICTION SECTION
# =========================================

if analyze:

    if user_input.strip() == "":

        st.warning(
            "Please enter some text."
        )

    else:

        # Preprocess user input
        processed_text = preprocess_text(
            user_input
        )

        # Predict emotion
        prediction = model.predict(
            processed_text,
            verbose=0
        )

        # Predicted class
        predicted_class = np.argmax(
            prediction
        )

        # Confidence score
        confidence_score = np.max(
            prediction
        ) * 100

        # Convert class into emotion label
        predicted_emotion = (
            label_encoder.inverse_transform(
                [predicted_class]
            )[0]
        )

        # =========================================
        # SECTION 5 — PREDICTION OUTPUT
        # =========================================

        st.markdown("---")

        st.header("📊 Prediction Output")

        st.success(
            f"Emotion Detected: {predicted_emotion}"
        )

        st.info(
            f"Confidence Score: {confidence_score:.2f}%"
        )

        # Emotional Status

        if confidence_score >= 80:

            st.success(
                "Emotional Status: Strong Prediction"
            )

        elif confidence_score >= 50:

            st.warning(
                "Emotional Status: Moderate Prediction"
            )

        else:

            st.error(
                "Emotional Status: Low Confidence Prediction"
            )

        # =========================================
        # SECTION 6 — VISUALIZATION AREA
        # =========================================

        st.markdown("---")

        st.header("📈 Sentiment Confidence Graph")

        emotions = label_encoder.classes_

        probabilities = prediction[0]

        fig, ax = plt.subplots(
            figsize=(10,5)
        )

        ax.bar(
            emotions,
            probabilities
        )

        ax.set_xlabel(
            "Emotion Categories"
        )

        ax.set_ylabel(
            "Probability"
        )

        ax.set_title(
            "Emotion Probability Chart"
        )

        st.pyplot(fig)

        # =========================================
        # SECTION 7 — EMOTIONAL GUIDANCE
        # =========================================

        st.markdown("---")

        st.header("💡 Emotional Guidance")

        if predicted_emotion.lower() in [
            "depression",
            "anxiety",
            "stress",
            "suicidal"
        ]:

            st.warning("""
Take a short break and talk with someone you trust.

Suggested Activities:
- Deep breathing
- Listening to calm music
- Walking outdoors
- Talking with friends

Emotional Wellness Tip:
You are not alone. Seeking support is important.
""")

        elif predicted_emotion.lower() in [
            "normal",
            "happy"
        ]:

            st.success("""
Great to see positive emotions.

Suggested Activities:
- Continue healthy habits
- Exercise regularly
- Spend time with loved ones

Emotional Wellness Tip:
Keep maintaining positivity and balance.
""")

        else:

            st.info("""
Stay mindful of your emotional health.

Suggested Activities:
- Meditation
- Journaling
- Relaxation exercises

Emotional Wellness Tip:
Small self-care steps can improve emotional well-being.
""")

# =========================================
# FOOTER
# =========================================

st.markdown("---")

st.caption(
    "Developed using Streamlit, TensorFlow, NLP, and SimpleRNN"
)
