import streamlit as st
import requests
from transformers import pipeline
from urllib.parse import quote
import os

# Fix HF cache (important for cloud)
os.environ["HF_HOME"] = "/tmp/huggingface"

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="Emotion Story Engine", page_icon="📖", layout="wide")

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_emotion_model():
    return pipeline(
        "text-classification",
        model="bhadresh-savani/distilbert-base-uncased-emotion",  # lighter model
        top_k=1,
        device=-1
    )

emotion_classifier = load_emotion_model()

# ─────────────────────────────────────────────
# STYLE MAP
# ─────────────────────────────────────────────
STYLE_MAP = {
    "joy": "happy face, natural smile, sunlight, realistic skin, DSLR photo",
    "sadness": "sad face, soft light, emotional expression, realistic human",
    "fear": "wide eyes, tense face, low light, realistic human",
    "anger": "angry expression, sharp features, realistic skin",
    "surprise": "shocked face, raised eyebrows, realistic reaction",
    "neutral": "normal face, natural light, realistic portrait"
}

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

if "current_prompt" not in st.session_state:
    st.session_state.current_prompt = ""

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.title("📖 Controls")

if st.sidebar.button("🗑️ Reset Story"):
    st.session_state.history = []
    st.session_state.current_prompt = ""
    st.rerun()

# ─────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────
st.title("Emotion-Aware AI Story & Visual Generator")

prompt = st.text_area("Enter your prompt", value=st.session_state.current_prompt, height=100)

# ─────────────────────────────────────────────
# GENERATION
# ─────────────────────────────────────────────
if st.button("Generate Story and Visual"):

    if not prompt.strip():
        st.warning("Please enter a prompt!")
        st.stop()

    with st.spinner("Analyzing emotion..."):
        try:
            emo_res = emotion_classifier(prompt)[0][0]
            emotion = emo_res['label']
            confidence = round(emo_res['score'], 2)
        except:
            st.error("Emotion detection failed")
            st.stop()

    st.success(f"Detected Emotion: {emotion.upper()} (confidence: {confidence})")

    # ── STORY GENERATION ──
    with st.spinner("Generating story..."):
        try:
            instruction = (
                f"Write a simple, realistic short paragraph. Mood: {emotion}. "
                f"Prompt: {prompt}"
            )

            text_api_url = f"https://text.pollinations.ai/{quote(instruction)}?model=openai"
            story_ext = requests.get(text_api_url, timeout=20).text.strip()
        except:
            story_ext = "Story generation failed. Please try again."

    # ── IMAGE GENERATION ──
    with st.spinner("Generating image... (10–20 sec)"):
        try:
            style = STYLE_MAP.get(emotion, "realistic photo")

            image_prompt = (
                f"realistic human photo, {prompt}, {style}, "
                f"natural lighting, sharp face, no distortion, DSLR quality"
            )

            image_url = (
                f"https://image.pollinations.ai/prompt/{quote(image_prompt)}"
                f"?width=1024&height=768&model=flux&nologo=true"
            )
        except:
            image_url = None

    # SAVE
    st.session_state.history.append({
        "mood": emotion,
        "input": prompt,
        "story": story_ext,
        "image": image_url
    })

# ─────────────────────────────────────────────
# DISPLAY RESULTS
# ─────────────────────────────────────────────
for item in reversed(st.session_state.history):
    with st.container(border=True):
        st.subheader(f"🎭 Mood: {item['mood'].upper()}")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.write(f"**Prompt:** {item['input']}")
            st.write(item['story'])

        with col2:
            if item["image"]:
                st.image(item["image"], use_container_width=True)
            else:
                st.warning("Image not available")
