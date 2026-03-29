import streamlit as st
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

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        device=-1
    )

# ---------------- DATA ----------------
EMOTION_QUOTES = {
    "sadness": "💙 Even the darkest nights end.",
    "joy": "✨ Celebrate small wins.",
    "fear": "🌑 Courage grows when you move forward.",
    "anger": "🔥 Stay calm and think clearly.",
    "neutral": "🌫️ Peace lies in balance.",
    "surprise": "⚡ Life is full of beauty.",
    "disgust": "🌿 Distance brings clarity."
}

# ---------------- FUNCTIONS ----------------
def detect_emotion(model, text):
    return model(text)[0]["label"].lower()

def generate_story(prompt, emotion):
    return f"{prompt} — A moment shaped by {emotion} emotions."

# ✅ FINAL IMAGE FUNCTION (WORKING)
def generate_image(prompt):
    try:
        url = f"https://image.pollinations.ai/prompt/{prompt}"
        response = requests.get(url, timeout=20)

        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content))
        else:
            return None
    except:
        return None

# ---------------- UI ----------------
st.title("🎭 Emotion AI Story Generator")

user_prompt = st.text_area("Enter your idea:")

if st.button("✨ Generate"):

    if not user_prompt.strip():
        st.warning("Please enter text")
        st.stop()

    model = load_model()
    emotion = detect_emotion(model, user_prompt)

    st.subheader(f"{emotion.upper()}")

    st.write(EMOTION_QUOTES.get(emotion, ""))

    story = generate_story(user_prompt, emotion)
    st.write(story)

    with st.spinner("Generating image..."):
        image = generate_image(user_prompt)

    if image:
        st.image(image, use_column_width=True)
    else:
        st.warning("⚠️ Image generation failed. Try again.")
