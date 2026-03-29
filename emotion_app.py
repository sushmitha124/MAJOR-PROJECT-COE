import streamlit as st
import torch
from transformers import pipeline
from PIL import Image
import requests
import io

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Emotion AI Story Generator",
    page_icon="🎭",
    layout="centered"
)

# ---------------- UI STYLE ----------------
st.markdown("""
<style>
.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    margin-top: 30px;
    background: linear-gradient(90deg, #ff758c, #ff7eb3, #ffd86f);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.stApp {
    background: linear-gradient(270deg, #42e695, #3bb2b8);
}
.block-container {
    background: rgba(0,0,0,0.65);
    padding: 30px;
    border-radius: 15px;
}
.story-box {
    background: #ffffff;
    color: #333;
    border-radius: 12px;
    padding: 20px;
    margin-top: 20px;
}
.quote-box {
    background: #ffffff;
    color: #333;
    border-radius: 12px;
    padding: 10px;
    margin-top: 10px;
    text-align: center;
    font-style: italic;
}
</style>
""", unsafe_allow_html=True)

# ---------------- EMOTION DATA ----------------
EMOTION_STYLES = {
    "sadness": {"color": "#7ab3e0", "emoji": "💙"},
    "joy": {"color": "#f0d060", "emoji": "✨"},
    "fear": {"color": "#b388e8", "emoji": "🌑"},
    "anger": {"color": "#e87070", "emoji": "🔥"},
    "disgust": {"color": "#80c980", "emoji": "🌿"},
    "neutral": {"color": "#aaaaaa", "emoji": "🌫️"},
    "surprise": {"color": "#f0a0e0", "emoji": "⚡"},
}

EMOTION_QUOTES = {
    "sadness": "💙 It's okay to feel lost — even dark nights end.",
    "anger": "🔥 Calm mind leads to better decisions.",
    "joy": "🎉 Celebrate small wins.",
    "fear": "🌑 Courage grows when you move forward.",
    "neutral": "🌫️ Peace lies in balance.",
    "surprise": "⚡ Life is full of unexpected beauty.",
    "disgust": "🌿 Distance brings clarity."
}

STORY_TEMPLATES = {
    "sadness": "{prompt} She felt a deep heaviness but slowly found peace.",
    "joy": "{prompt} Happiness filled the moment and everything felt bright.",
    "fear": "{prompt} Fear crept in, but she chose courage.",
    "anger": "{prompt} Anger burned, but she regained control.",
    "disgust": "{prompt} She stepped away and found clarity.",
    "surprise": "{prompt} The moment filled her with wonder.",
    "neutral": "{prompt} She moved forward calmly and steadily."
}

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        device=-1
    )

# ---------------- FUNCTIONS ----------------
def detect_emotion(model, text):
    return model(text)[0]["label"].lower()

def generate_story(prompt, emotion):
    return STORY_TEMPLATES.get(emotion, STORY_TEMPLATES["neutral"]).format(prompt=prompt)

# ---------------- IMAGE GENERATION (FINAL FIX) ----------------
def generate_image(prompt):
    try:
        response = requests.post(
            "https://hf.space/embed/stabilityai/stable-diffusion/+/api/predict/",
            json={"data": [prompt]},
            timeout=60
        )

        result = response.json()

        if "data" in result:
            image_url = result["data"][0]

            image_response = requests.get(image_url)
            return Image.open(io.BytesIO(image_response.content))

        else:
            st.error("Image generation failed")
            return None

    except Exception as e:
        st.error(f"Image error: {e}")
        return None

# ---------------- UI ----------------
st.markdown("<div class='main-title'>🎭 Emotion AI Story Generator ✨</div>", unsafe_allow_html=True)

user_prompt = st.text_area("", placeholder="Type your feeling or story idea...")

if st.button("✨ Generate"):

    if not user_prompt.strip():
        st.warning("Please enter a prompt")
        st.stop()

    model = load_model()
    emotion = detect_emotion(model, user_prompt)

    style = EMOTION_STYLES.get(emotion, EMOTION_STYLES["neutral"])

    st.markdown(
        f"<h3 style='color:{style['color']}'>{style['emoji']} {emotion.upper()}</h3>",
        unsafe_allow_html=True
    )

    # Quote
    st.markdown(
        f"<div class='quote-box'>{EMOTION_QUOTES.get(emotion, '')}</div>",
        unsafe_allow_html=True
    )

    # Story
    story = generate_story(user_prompt, emotion)
    st.markdown(f"<div class='story-box'>{story}</div>", unsafe_allow_html=True)

    # Image
    with st.spinner("Generating image..."):
        image = generate_image(user_prompt)

    if image:
        st.image(image, use_column_width=True)
    else:
        st.warning("⚠️ Image generation failed.")
