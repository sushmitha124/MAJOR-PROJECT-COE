import streamlit as st
import torch
from transformers import pipeline
import requests
import random
from urllib.parse import quote
import time

# --- 1. PAGE CONFIG ---
st.set_page_config(
    page_title="Emotion AI Story Engine",
    page_icon="🎭",
    layout="centered"
)

# --- 2. ADVANCED UI STYLE ---
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
    background: rgba(0,0,0,0.75);
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
    width: 100%;
    font-weight: bold;
    border: none;
    padding: 10px;
}

.emotion-badge {
    padding: 10px 25px;
    border-radius: 30px;
    display: inline-block;
    margin-top: 15px;
    font-weight: bold;
    color: white;
}

.story-box {
    background: #ffffff;
    color: #333;
    border-radius: 12px;
    padding: 20px;
    margin-top: 20px;
    line-height: 1.6;
    border-left: 6px solid #ff758c;
}
</style>
""", unsafe_allow_html=True)

# --- 3. LOGIC SETTINGS ---
EMOTION_UI_THEMES = {
    "joy": {"bg": "#f0d060", "emoji": "✨"},
    "sadness": {"bg": "#7ab3e0", "emoji": "💙"},
    "fear": {"bg": "#b388e8", "emoji": "🌑"},
    "anger": {"bg": "#e87070", "emoji": "🔥"},
    "surprise": {"bg": "#42e695", "emoji": "😲"},
    "neutral": {"bg": "#aaaaaa", "emoji": "🌫️"},
}

STYLE_MAP = {
    "joy": "clear happy face, authentic smile, natural sunlight, candid photo, symmetrical, realistic skin",
    "sadness": "clear face, quiet expression, soft window light, natural skin texture, sharp focus on eyes",
    "fear": "clear focused face, wide eyes, sharp facial features, realistic human anatomy",
    "anger": "sharp clear face, intense gaze, realistic skin, indoor lighting, candid photo",
    "surprise": "clear shocked face, sharp focus, eyebrows raised, authentic human reaction",
    "neutral": "clear normal face, unposed headshot, natural daylight, ordinary person, sharp details"
}

# --- 4. BACKEND FUNCTIONS ---
@st.cache_resource
def load_emotion_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=1,
        device=0 if torch.cuda.is_available() else -1
    )

def generate_visual_story(prompt, emotion):
    # Text Generation via Pollinations OpenAI model
    instruction = (
        f"Write a short, meaningful story paragraph. "
        f"Continue this prompt: '{prompt}'. Mood: {emotion}. "
        f"Use simple words and stay grounded in daily life."
    )
    text_url = f"https://text.pollinations.ai/{quote(instruction)}?model=openai"
    try:
        story = requests.get(text_url, timeout=10).text.strip()
    except:
        story = f"{prompt}. It was a moment that defined the day, carrying the weight of {emotion}."
    
    # Image Generation via Pollinations Flux model
    tech_style = STYLE_MAP.get(emotion, "realistic photography, clear face")
    image_prompt = (
        f"High-quality realistic photo of a human: {prompt}. "
        f"{tech_style}, natural lighting, no distortions, 8k."
    )
    image_url = f"https://image.pollinations.ai/prompt/{quote(image_prompt)}?width=1024&height=768&model=flux&nologo=true"
    
    return story, image_url

# --- 5. MAIN UI ---
st.markdown("<div class='main-title'>🎭 Emotion AI Story Engine ✨</div>", unsafe_allow_html=True)

clf = load_emotion_model()

user_prompt = st.text_area("", placeholder="Type a feeling or a moment (e.g., Walking home in the rain...)")

if st.button("✨ Generate Experience"):
    if not user_prompt.strip():
        st.warning("Please enter a prompt first!")
        st.stop()

    with st.spinner("Analyzing emotions and crafting your world..."):
        # Detect Emotion
        raw_emotion = clf(user_prompt)[0][0]["label"].lower()
        
        # Get Story and Image
        story_text, img_url = generate_visual_story(user_prompt, raw_emotion)
        
        # UI Styling
        theme = EMOTION_UI_THEMES.get(raw_emotion, EMOTION_UI_THEMES["neutral"])

        # Display Emotion Badge
        st.markdown(
            f"<div class='emotion-badge' style='background:{theme['bg']}'>"
            f"{theme['emoji']} {raw_emotion.upper()}</div>",
            unsafe_allow_html=True
        )

        # Display Story
        st.markdown(f"<div class='story-box'><b>The Scene:</b><br>{story_text}</div>", unsafe_allow_html=True)

        # Display Image
        st.image(img_url, use_container_width=True)
        
        st.success("Scene Generated Successfully!")

# --- 6. HISTORY (Optional) ---
if "history" not in st.session_state:
    st.session_state.history = []

# If you want to see previous results, they appear here
if st.session_state.history:
    st.markdown("---")
    st.subheader("Previous Explorations")
    for h in reversed(st.session_state.history):
        st.text(f"{h['mood'].upper()}: {h['input'][:50]}...")
