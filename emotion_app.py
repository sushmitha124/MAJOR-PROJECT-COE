import streamlit as st
import torch
from transformers import pipeline
from PIL import Image
import google.generativeai as genai
import io

# 🔐 CONFIG GEMINI
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# PAGE CONFIG
st.set_page_config(
    page_title="Emotion AI Story Generator",
    page_icon="🎭",
    layout="centered"
)

# 🎨 UI STYLE
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

# 🎭 EMOTION DATA
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
    "sadness": "{prompt} She felt a deep heaviness in her heart but slowly found peace.",
    "joy": "{prompt} Happiness filled the moment and everything felt bright.",
    "fear": "{prompt} Fear crept in, but she chose courage and moved forward.",
    "anger": "{prompt} Anger burned, but she regained calm and control.",
    "disgust": "{prompt} She stepped away to regain balance and clarity.",
    "surprise": "{prompt} The unexpected moment filled her with wonder.",
    "neutral": "{prompt} She moved forward calmly, grounded and aware."
}

# 🚀 LOAD MODELS
@st.cache_resource
def load_emotion_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=1,
        device=-1
    )

# 🔍 FUNCTIONS
def detect_emotion(model, text):
    return model(text)[0][0]["label"].lower()

def generate_story(prompt, emotion):
    return STORY_TEMPLATES.get(emotion, STORY_TEMPLATES["neutral"]).format(prompt=prompt)

def generate_image(prompt):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")

        response = model.generate_content(
            f"Create a realistic cinematic image showing emotion '{prompt}'.",
            generation_config={"response_modalities": ["TEXT", "IMAGE"]}
        )

        for part in response.parts:
            if hasattr(part, "inline_data"):
                image_bytes = part.inline_data.data
                return Image.open(io.BytesIO(image_bytes))

    except Exception as e:
        st.error(f"Image error: {e}")
        return None

# 🎭 UI
st.markdown("<div class='main-title'>🎭 Emotion AI Story Generator ✨</div>", unsafe_allow_html=True)

user_prompt = st.text_area("", placeholder="Type your feeling or story idea...")

if st.button("✨ Generate"):

    if not user_prompt.strip():
        st.warning("Please enter a prompt")
        st.stop()

    # Emotion Detection
    clf = load_emotion_model()
    emotion = detect_emotion(clf, user_prompt)

    style = EMOTION_STYLES.get(emotion, EMOTION_STYLES["neutral"])

    st.markdown(
        f"<h3 style='color:{style['color']}'>{style['emoji']} {emotion.upper()}</h3>",
        unsafe_allow_html=True
    )

    # Quote
    quote = EMOTION_QUOTES.get(emotion, "")
    st.markdown(f"<div class='quote-box'>{quote}</div>", unsafe_allow_html=True)

    # Story
    story = generate_story(user_prompt, emotion)
    st.markdown(f"<div class='story-box'>{story}</div>", unsafe_allow_html=True)

    # 🖼️ IMAGE GENERATION (GEMINI)
    with st.spinner("Generating image..."):
        image = generate_image(user_prompt)

    if image:
        st.image(image, use_column_width=True)
    else:
        st.warning("⚠️ Image generation failed.")
