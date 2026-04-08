import streamlit as st
import torch
from transformers import pipeline
import urllib.parse
import requests
import time

# PAGE CONFIG
st.set_page_config(
    page_title="Emotion AI Story Engine",
    page_icon="🎭",
    layout="centered"
)

# -------------------------------
# 🎨 UI STYLE
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
    background: rgba(0,0,0,0.70);
    padding: 35px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.1);
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
    border: none;
    font-weight: bold;
    padding: 0.5rem 2rem;
}

.emotion-badge {
    padding: 8px 20px;
    border-radius: 30px;
    display: inline-block;
    margin-top: 15px;
    margin-bottom: 10px;
    font-weight: bold;
    text-transform: uppercase;
    font-size: 0.9rem;
    color: white;
}

.story-box {
    background: #ffffff;
    color: #333;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
    line-height: 1.6;
    border-left: 5px solid #ff758c;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# EMOTION UI THEMES
# -------------------------------
EMOTION_UI = {
    "sadness": {"bg": "#1e2a3a", "emoji": "💙"},
    "joy": {"bg": "#d4af37", "emoji": "✨"},
    "fear": {"bg": "#5a4a78", "emoji": "🌑"},
    "anger": {"bg": "#a83232", "emoji": "🔥"},
    "neutral": {"bg": "#555555", "emoji": "🌫️"},
    "surprise": {"bg": "#2e7d32", "emoji": "😲"},
}

# -------------------------------
# LOAD MODELS
# -------------------------------
@st.cache_resource
def load_emotion_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=1,
        device=-1 # Set to 0 if you have a GPU
    )

# -------------------------------
# CORE LOGIC FUNCTIONS
# -------------------------------
def generate_dynamic_story(prompt, emotion):
    # This uses the Pollinations Text API to generate a unique story paragraph
    instruction = f"Write one meaningful, realistic story paragraph about: {prompt}. The mood is {emotion}."
    encoded_text = urllib.parse.quote(instruction)
    url = f"https://text.pollinations.ai/{encoded_text}?model=openai"
    try:
        res = requests.get(url, timeout=10)
        return res.text.strip()
    except:
        return f"{prompt}. The moment felt deeply connected to a sense of {emotion}."

def generate_image(prompt, emotion):
    # Uses high-quality Flux model for realistic visuals
    image_prompt = f"A high-quality realistic photo, {emotion} atmosphere, {prompt}, cinematic lighting, 8k, detailed human features"
    encoded_img = urllib.parse.quote(image_prompt)
    return f"https://image.pollinations.ai/prompt/{encoded_img}?width=1024&height=768&model=flux&nologo=true"

# -------------------------------
# UI EXECUTION
# -------------------------------
st.markdown("<div class='main-title'>🎭 Emotion AI Story Engine ✨</div>", unsafe_allow_html=True)

user_prompt = st.text_area("", placeholder="What's on your mind? (e.g., walking home in the rain...)")

if st.button("✨ Generate Experience"):
    if not user_prompt.strip():
        st.warning("Please enter a prompt first.")
        st.stop()

    with st.spinner("Analyzing emotions and weaving your story..."):
        # 1. Detect Emotion
        clf = load_emotion_model()
        emo_label = clf(user_prompt)[0][0]["label"].lower()
        
        # 2. Generate Story Content
        story_text = generate_dynamic_story(user_prompt, emo_label)
        
        # 3. Get Image URL
        image_url = generate_image(user_prompt, emo_label)
        
        # --- DISPLAY RESULTS ---
        theme = EMOTION_UI.get(emo_label, EMOTION_UI["neutral"])

        # Badge
        st.markdown(
            f"<div class='emotion-badge' style='background:{theme['bg']}'>"
            f"{theme['emoji']} {emo_label}</div>",
            unsafe_allow_html=True
        )

        # Story Box
        st.markdown(f"""
            <div class='story-box'>
                <strong>The Scene:</strong><br>
                {story_text}
            </div>
        """, unsafe_allow_html=True)

        # Image
        st.image(image_url, use_container_width=True)
