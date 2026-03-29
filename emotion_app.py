import streamlit as st
from transformers import pipeline
from diffusers import StableDiffusionPipeline
import torch

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Emotion AI Story Generator", page_icon="🎭")

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_emotion_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        device=-1
    )

@st.cache_resource
def load_image_model():
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5"
    )
    pipe = pipe.to("cpu")  # CPU safe
    return pipe

# ---------------- DATA ----------------
EMOTION_QUOTES = {
    "joy": "Celebrate small wins 🎉",
    "sadness": "Even dark nights end 💙",
    "anger": "Stay calm 🔥",
    "fear": "Be brave 🌑",
    "neutral": "Peace lies in balance 🌫️",
    "surprise": "Life is beautiful ⚡",
    "disgust": "Clarity comes with distance 🌿"
}

# ---------------- FUNCTIONS ----------------
def detect_emotion(model, text):
    return model(text)[0]["label"].lower()

def generate_story(prompt, emotion):
    return f"{prompt} — A story shaped by {emotion} emotions."

def generate_image(prompt):
    pipe = load_image_model()
    image = pipe(prompt).images[0]
    return image

# ---------------- UI ----------------
st.title("🎭 Emotion AI Story Generator")

prompt = st.text_area("Enter your text:")

if st.button("Generate"):

    if not prompt:
        st.warning("Enter text")
        st.stop()

    model = load_emotion_model()
    emotion = detect_emotion(model, prompt)

    st.subheader(f"Emotion: {emotion.upper()}")
    st.write(EMOTION_QUOTES.get(emotion, ""))

    story = generate_story(prompt, emotion)
    st.write(story)

    with st.spinner("Generating image..."):
        image = generate_image(prompt)

    st.image(image)
