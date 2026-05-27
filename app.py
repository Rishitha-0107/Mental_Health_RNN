# Enhanced app.py with Images, Emojis, and Modern UI

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
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================
# CUSTOM CSS
# =========================================

st.markdown("""
<style>

body {
    background-color: #f5f7fa;
}

.main-title {
    font-size: 48px;
    font-weight: bold;
    text-align: center;
    color: #2E86C1;
}

.sub-title {
    font-size: 22px;
    text-align: center;
    color: #5D6D7E;
    margin-bottom: 30px;
}

.section-card {
    background-color: white;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
    margin-bottom: 25px;
}

.stButton>button {
    width: 100%;
    background: linear-gradient(to right, #2E86C1, #5DADE2);
    color: white;
    font-size: 20px;
    border-radius: 12px;
    padding: 14px;
    border: none;
    font-weight: bold;
}

.stButton>button:hover {
    background: linear-gradient(to right, #1B4F72, #2E86C1);
    color: white;
}

.tip-box {
    background-color: #EBF5FB;
    padding: 20px;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

# =========================================
# DOWNLOAD MODEL
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

model = load_model(model_path)

tokenizer = joblib.load("tokenizer.pkl")

label_encoder = joblib.load("label_encoder.pkl")

max_length = joblib.load("max_length.pkl")

# =========================================
# STOPWORDS
# =========================================

stop_words = set(stopwords.words('english'))

# =========================================
# SIDEBAR
# =========================================

st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/3771/3771417.png",
    width=120
)

st.sidebar.title("🧠 Mental Health AI")

st.sidebar.info("""
This AI application predicts emotional sentiment
using Simple Recurrent Neural Networks.
""")

st.sidebar.markdown("### 🌈 Supported Emotions")

for emotion in label_encoder.classes_:
    st.sidebar.write(f"✅ {emotion}")

st.sidebar.markdown("---")

st.sidebar.success("💡 Stay Positive & Take Care of Your Mind")

# =========================================
# HEADER SECTION
# =========================================

col1, col2, col3 = st.columns([1,2,1])

with col2:
    st.image(
        "https://cdn-icons-png.flaticon.com/512/4320/4320337.png",
        width=180
    )

st.markdown(
    '<p class="main-title">🧠 AI-Based Mental Health Sentiment Monitoring System</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="sub-title">Emotion Detection using Simple Recurrent Neural Networks 💬</p>',
    unsafe_allow_html=True
)

# =========================================
# ABOUT PROJECT SECTION
# =========================================

st.markdown('<div class="section-card">', unsafe_allow_html=True)

st.header("📘 About the Project")

col1, col2 = st.columns([2,1])

with col1:

    st.write("""
This AI-powered Mental Health Sentiment Monitoring System
uses Natural Language Processing (NLP) and Simple Recurrent
Neural Networks (RNN) to detect emotional sentiment from text.

### 🌟 Importance of Emotional AI
- Helps monitor emotional well-being
- Detects negative sentiment trends
- Supports early emotional awareness

### 🤖 NLP Applications
- Sentiment Analysis
- Chatbots
- Emotion Detection
- Language Translation

### 🔄 Role of RNN
RNN models remember previous words using hidden states,
making them suitable for sequential text learning.
""")

with col2:

    st.image(
        "https://cdn-icons-png.flaticon.com/512/4140/4140048.png"
    )

st.markdown('</div>', unsafe_allow_html=True)

# =========================================
# TEXT PREPROCESSING FUNCTION
# =========================================

def preprocess_text(text):

    text = text.lower()

    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )

    words = word_tokenize(text)

    filtered_words = [
        word for word in words
        if word not in stop_words
    ]

    text = " ".join(filtered_words)

    sequence = tokenizer.texts_to_sequences([text])

    padded = pad_sequences(
        sequence,
        maxlen=max_length,
        padding='post',
        truncating='post'
    )

    return padded

# =========================================
# USER INPUT SECTION
# =========================================

st.markdown('<div class="section-card">', unsafe_allow_html=True)

st.header("✍ Express Your Feelings")

st.image(
    "https://cdn-icons-png.flaticon.com/512/4712/4712109.png",
    width=120
)

st.write("### 💬 Sample Sentences")

col1, col2 = st.columns(2)

with col1:
    st.success("😊 I feel happy and confident today")
    st.success("🌈 Life feels beautiful and peaceful")

with col2:
    st.error("😔 I feel lonely and depressed")
    st.error("😟 My anxiety is increasing every day")

user_input = st.text_area(
    "📝 Enter your thoughts or feelings here...",
    height=220,
    placeholder="Type your emotions, thoughts, or feelings here..."
)

analyze = st.button("🔍 Analyze Emotion")

st.markdown('</div>', unsafe_allow_html=True)

# =========================================
# PREDICTION SECTION
# =========================================

if analyze:

    if user_input.strip() == "":

        st.warning("⚠ Please enter some text.")

    else:

        processed_text = preprocess_text(user_input)

        prediction = model.predict(
            processed_text,
            verbose=0
        )

        predicted_class = np.argmax(prediction)

        confidence_score = np.max(prediction) * 100

        predicted_emotion = (
            label_encoder.inverse_transform(
                [predicted_class]
            )[0]
        )

        # =========================================
        # OUTPUT SECTION
        # =========================================

        st.markdown('<div class="section-card">', unsafe_allow_html=True)

        st.header("📊 Prediction Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "🧠 Emotion",
                predicted_emotion
            )

        with col2:
            st.metric(
                "🎯 Confidence",
                f"{confidence_score:.2f}%"
            )

        with col3:

            if confidence_score >= 80:
                status = "Strong"

            elif confidence_score >= 50:
                status = "Moderate"

            else:
                status = "Low"

            st.metric(
                "📌 Prediction Strength",
                status
            )

        st.markdown('</div>', unsafe_allow_html=True)

        # =========================================
        # VISUALIZATION SECTION
        # =========================================

        st.markdown('<div class="section-card">', unsafe_allow_html=True)

        st.header("📈 Emotion Confidence Graph")

        emotions = label_encoder.classes_

        probabilities = prediction[0]

        fig, ax = plt.subplots(figsize=(10,5))

        bars = ax.bar(
            emotions,
            probabilities
        )

        ax.set_xlabel("Emotion Categories")

        ax.set_ylabel("Probability")

        ax.set_title("Sentiment Confidence Levels")

        plt.xticks(rotation=15)

        st.pyplot(fig)

        st.markdown('</div>', unsafe_allow_html=True)

        # =========================================
        # EMOTIONAL GUIDANCE
        # =========================================

        st.markdown('<div class="section-card">', unsafe_allow_html=True)

        st.header("💡 Emotional Wellness Guidance")

        if predicted_emotion.lower() in [
            "depression",
            "anxiety",
            "stress",
            "suicidal"
        ]:

            st.image(
                "https://cdn-icons-png.flaticon.com/512/3774/3774299.png",
                width=120
            )

            st.error("😔 Take a short break and talk with someone you trust.")

            st.markdown("""
### 🌿 Suggested Activities
- Deep breathing exercises
- Listening to calming music
- Taking a short walk
- Talking with friends or family

### ❤️ Wellness Tip
You are not alone. Small positive steps can help improve emotional well-being.
""")

        elif predicted_emotion.lower() in [
            "normal",
            "happy"
        ]:

            st.image(
                "https://cdn-icons-png.flaticon.com/512/742/742751.png",
                width=120
            )

            st.success("😊 Great to see positive emotions today.")

            st.markdown("""
### 🌞 Suggested Activities
- Exercise regularly
- Continue productive habits
- Spend time outdoors
- Enjoy time with loved ones

### 💚 Wellness Tip
Maintaining positive routines supports long-term mental wellness.
""")

        else:

            st.image(
                "https://cdn-icons-png.flaticon.com/512/4228/4228703.png",
                width=120
            )

            st.info("🧘 Stay mindful of your emotional balance.")

            st.markdown("""
### 🌸 Suggested Activities
- Meditation
- Journaling
- Creative hobbies
- Proper sleep routine

### ✨ Wellness Tip
Self-care and emotional awareness are important for mental wellness.
""")

        st.markdown('</div>', unsafe_allow_html=True)

# =========================================
# FOOTER
# =========================================

st.markdown("---")

st.markdown(
    """
<div style='text-align:center'>
<h4>🧠 Developed using Streamlit, TensorFlow, NLP, and SimpleRNN</h4>
<p>💙 Mental Health Matters • Stay Positive • Stay Strong 💙</p>
</div>
""",
    unsafe_allow_html=True
)

