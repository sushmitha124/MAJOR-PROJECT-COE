import streamlit as st
import torch
from transformers import pipeline
import urllib.parse
import requests
import time

# PAGE CONFIG
st.set_page_config(
    page_title="Emotion AI Story Generator",
    page_icon="🎭",
    layout="centered"
)

# -------------------------------
# 🎨 UI STYLE (same as yours)
# -------------------------------
st.markdown("""
<style>
.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    margin-top: 40px;
    margin-bottom: 10px;
    background: linear-gradient(90deg, #ff758c, #ff7eb3, #ffd86f);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.stApp {
    background: linear-gradient(270deg, #ff6ec4, #7873f5, #42e695, #f9ca24);
    background-size: 800% 800%;
    animation: gradientMove 10s ease infinite;
}

@keyframes gradientMove {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

.block-container {
    background: rgba(0,0,0,0.65);
    padding: 35px;
    border-radius: 18px;
}

.stTextArea textarea {
    background: #ffffff !important;
    color: #333 !important;
    border-radius: 12px !important;
}

.stButton > button {
    background: linear-gradient(135deg, #ff758c, #ff7eb3);
    color: white !important;
    border-radius: 25px !important;
}

.emotion-badge {
    padding: 10px 25px;
    border-radius: 30px;
    display: inline-block;
    margin-top: 15px;
    font-weight: bold;
}

.story-box {
    background: #ffffff;
    color: #333;
    border-radius: 12px;
    padding: 20px;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# EMOTION STYLE
# -------------------------------
EMOTION_STYLES = {
    "sadness": {"bg": "#1e2a3a", "color": "#7ab3e0", "emoji": "💙"},
    "joy": {"bg": "#2a2a1a", "color": "#f0d060", "emoji": "✨"},
    "fear": {"bg": "#1f1a2e", "color": "#b388e8", "emoji": "🌑"},
    "anger": {"bg": "#2e1a1a", "color": "#e87070", "emoji": "🔥"},
    "neutral": {"bg": "#1e1e1e", "color": "#aaaaaa", "emoji": "🌫️"},
}

# -------------------------------
# STORY TEMPLATES
# -------------------------------
STORY_TEMPLATES = {
    "sadness": "{prompt} She felt heavy inside, yet slowly hope returned.",
    "joy": "{prompt} Happiness filled her heart with warmth and light.",
    "fear": "{prompt} Fear surrounded her, but courage pushed her forward.",
    "anger": "{prompt} Anger burned, but she chose calmness instead.",
    "neutral": "{prompt} She moved forward with quiet balance.",
}

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_emotion_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=1,
        device=0 if torch.cuda.is_available() else -1
    )

# -------------------------------
# FUNCTIONS
# -------------------------------
def detect_emotion(clf, text):
    return clf(text)[0][0]["label"].lower()

def generate_story(prompt, emotion):
    return STORY_TEMPLATES.get(emotion, STORY_TEMPLATES["neutral"]).format(prompt=prompt)

# -------------------------------
# IMAGE GENERATION (Pollinations)
# -------------------------------
def generate_image(prompt, emotion):
    image_prompt = f"{emotion} cinematic realistic scene, ultra detailed, 4k, {prompt}"
    encoded = urllib.parse.quote(image_prompt)

    url = f"https://image.pollinations.ai/prompt/{encoded}"

    for _ in range(3):
        try:
            res = requests.get(url, timeout=5)
            if res.status_code == 200:
                return url
        except:
            pass
        time.sleep(2)

    return None

# -------------------------------
# UI
# -------------------------------
st.markdown("<div class='main-title'>🎭 Emotion AI Story Generator ✨</div>", unsafe_allow_html=True)

user_prompt = st.text_area("", placeholder="Type your feeling or story idea...")

if st.button("✨ Generate"):

    if not user_prompt.strip():
        st.warning("Please enter a prompt")
        st.stop()

    clf = load_emotion_model()
    emotion = detect_emotion(clf, user_prompt)

    style = EMOTION_STYLES.get(emotion, EMOTION_STYLES["neutral"])

    # Emotion badge
    st.markdown(
        f"<div class='emotion-badge' style='background:{style['bg']}; color:{style['color']}'>"
        f"{style['emoji']} {emotion.upper()}</div>",
        unsafe_allow_html=True
    )

    # Story
    story = generate_story(user_prompt, emotion)
    st.markdown(f"<div class='story-box'>{story}</div>", unsafe_allow_html=True)

    # Image
    with st.spinner("Generating image..."):
        image_url = generate_image(user_prompt, emotion)

    if image_url:
        st.image(image_url, use_column_width=True)
    else:
        st.warning("⚠️ Image generation failed. Try again.")
