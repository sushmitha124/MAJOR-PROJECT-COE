import streamlit as st
import torch
from transformers import pipeline
import requests
import io
from PIL import Image
import os

# PAGE CONFIG
st.set_page_config(page_title="Emotion AI 2026", page_icon="🎭", layout="centered")

# -------------------------------
# 🔐 API SETUP (Hugging Face)
# -------------------------------
# On Render: Add HF_TOKEN to 'Environment Variables'
HF_TOKEN = os.getenv("HF_TOKEN") 
API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# -------------------------------
# 🎨 UI STYLE
# -------------------------------
st.markdown("""
<style>
    .main-title { text-align: center; font-size: 40px; font-weight: bold; background: linear-gradient(90deg, #6a11cb, #2575fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .stApp { background-color: #0e1117; }
    .block-container { background: rgba(255,255,255,0.05); padding: 30px; border-radius: 20px; border: 1px solid rgba(255,255,255,0.1); }
    .story-box { background: #161b22; color: #c9d1d9; border-radius: 10px; padding: 20px; border-left: 5px solid #2575fc; margin: 20px 0; font-size: 18px; }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# 🧠 LOGIC FUNCTIONS
# -------------------------------
@st.cache_resource
def load_emotion_model():
    # Force CPU usage for Render stability
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", device=-1)

def query_image_api(prompt):
    if not HF_TOKEN:
        return "error_no_key"
    try:
        response = requests.post(API_URL, headers=headers, json={"inputs": prompt}, timeout=60)
        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content))
        return None
    except:
        return None

# -------------------------------
# 🎭 UI LAYOUT
# -------------------------------
st.markdown("<div class='main-title'>🎭 Emotion Story Engine</div>", unsafe_allow_html=True)

user_input = st.text_area("How are you feeling?", placeholder="e.g. I saw the sun rise over the mountains today...")

if st.button("✨ Create Magic"):
    if not user_input.strip():
        st.error("Please write something first!")
    elif not HF_TOKEN:
        st.warning("⚠️ HF_TOKEN missing. Image generation disabled. Add it to Render Environment Variables.")
    else:
        with st.spinner("Analyzing emotions..."):
            classifier = load_emotion_model()
            result = classifier(user_input)[0]
            emotion = result['label']
            
        # 1. Display Emotion
        st.subheader(f"Detected Emotion: {emotion.upper()}")
        
        # 2. Generate Story snippet
        story_templates = {
            "joy": f"The world seemed to hum in harmony with her. {user_input} A golden light followed every step.",
            "sadness": f"The rain mirrored the quiet weight in her chest. {user_input} Yet, in the silence, she found peace.",
            "anger": f"The air crackled with unspoken tension. {user_input} But she breathed, turning fire into focus.",
            "fear": f"The shadows grew long and cold. {user_input} However, she realized the light was always within.",
            "neutral": f"The day unfolded with a steady, calm rhythm. {user_input} Balance was her greatest strength."
        }
        story = story_templates.get(emotion, f"A strange feeling took hold. {user_input}")
        st.markdown(f"<div class='story-box'>{story}</div>", unsafe_allow_html=True)
        
        # 3. Generate Image
        with st.spinner("Painting your mood..."):
            img_prompt = f"Cinematic digital art, {emotion} atmosphere, {user_input}, hyper-detailed, 8k"
            img = query_image_api(img_prompt)
            
            if img:
                st.image(img, caption=f"Visualized: {emotion.capitalize()}", use_column_width=True)
            else:
                st.info("The Image AI is warming up. Please try again in a few seconds.")
