import streamlit as st
import requests
from transformers import pipeline
from urllib.parse import quote
from PIL import Image
from io import BytesIO
import os

# Fix HF cache
os.environ["HF_HOME"] = "/tmp/huggingface"

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="Emotion Story Engine", page_icon="📖", layout="wide")

# ─────────────────────────────────────────────
# LOAD EMOTION MODEL (LIGHTWEIGHT)
# ─────────────────────────────────────────────
@st.cache_resource
def load_emotion_model():
    return pipeline(
        "text-classification",
        model="bhadresh-savani/distilbert-base-uncased-emotion",
        top_k=1,
        device=-1
    )

emotion_classifier = load_emotion_model()

# ─────────────────────────────────────────────
# STYLE MAP
# ─────────────────────────────────────────────
STYLE_MAP = {
    "joy": "happy face, natural smile, sunlight, DSLR photo",
    "sadness": "sad face, soft light, emotional expression",
    "fear": "wide eyes, tense face, dark lighting",
    "anger": "angry expression, sharp features",
    "surprise": "shocked face, raised eyebrows",
    "neutral": "normal face, natural daylight"
}

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ─────────────────────────────────────────────
# FUNCTIONS
# ─────────────────────────────────────────────
def generate_image(prompt, emotion):
    API_URL = "https://api-inference.huggingface.co/models/stabilityai/sdxl-turbo"

    headers = {
        "Authorization": f"Bearer {st.secrets['HF_TOKEN']}"
    }

    payload = {
        "inputs": f"{prompt}, {emotion}, realistic human photo",
        "options": {"wait_for_model": True}
    }

    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code != 200:
        return None

    return Image.open(BytesIO(response.content))


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.title("Emotion-Aware AI Story & Visual Generator")

prompt = st.text_area("Enter your prompt")

if st.button("Generate Story and Visual"):

    if not prompt.strip():
        st.warning("Please enter a prompt")
        st.stop()

    # Emotion
    with st.spinner("Detecting emotion..."):
        emo_res = emotion_classifier(prompt)[0][0]
        emotion = emo_res['label']
        confidence = round(emo_res['score'], 2)

    st.success(f"Emotion: {emotion.upper()} ({confidence})")

    # Story (simple)
    story = f"{prompt} The moment carried a strong sense of {emotion}, shaping everything around it."

    st.subheader("📖 Story")
    st.write(story)

    # Image
    with st.spinner("Generating image (10–20 sec)..."):
        image = generate_image(prompt, emotion)

    if image:
        st.subheader("🖼️ Image")
        st.image(image, use_column_width=True)
    else:
        st.error("Image generation failed. Try again.")

    # Save history
    st.session_state.history.append((prompt, emotion))

# ─────────────────────────────────────────────
# HISTORY
# ─────────────────────────────────────────────
if st.session_state.history:
    st.subheader("History")
    for p, e in reversed(st.session_state.history):
        st.write(f"{p} → {e}")
