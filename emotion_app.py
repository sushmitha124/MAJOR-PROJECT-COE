import streamlit as st
import urllib.parse
import requests
import time
from transformers import pipeline

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Emotion AI", layout="centered")

st.title("🎭 Emotion-Aware AI Story Generator")
st.write("Detect emotion → Generate story → Generate image")

# -------------------------------
# LOAD MODELS (CACHED)
# -------------------------------
@st.cache_resource
def load_models():
    emotion_model = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=1
    )

    story_model = pipeline(
        "text-generation",
        model="google/flan-t5-base"
    )

    return emotion_model, story_model

emotion_model, story_model = load_models()

# -------------------------------
# IMAGE GENERATION (SAFE)
# -------------------------------
import urllib.parse
import requests
import time

def generate_image(prompt, emotion):
    image_prompt = f"{emotion} cinematic realistic scene, 4k, {prompt}"
    encoded = urllib.parse.quote(image_prompt)

    url = f"https://image.pollinations.ai/prompt/{encoded}"

    # 🔁 Retry 3 times
    for _ in range(3):
        try:
            res = requests.get(url, timeout=5)

            if res.status_code == 200:
                return url

        except:
            pass

        time.sleep(2)

    return None  # ❌ no fake URL

# -------------------------------
# UI INPUT
# -------------------------------
user_input = st.text_area("Enter your thoughts:")

# -------------------------------
# GENERATE BUTTON
# -------------------------------
if st.button("Generate"):

    if not user_input.strip():
        st.warning("Please enter some text")
        st.stop()

    # -------------------------------
    # EMOTION DETECTION
    # -------------------------------
    with st.spinner("Detecting emotion..."):
        emo = emotion_model(user_input)[0][0]
        emotion = emo["label"]
        score = round(emo["score"], 3)

    st.success(f"🎭 Emotion: {emotion} ({score})")

    # -------------------------------
    # STORY GENERATION
    # -------------------------------
    with st.spinner("Generating story..."):
        story_prompt = f"Write a short emotional story. Mood: {emotion}. Context: {user_input}"
        story = story_model(
            story_prompt,
            max_length=150
        )[0]["generated_text"]

    st.subheader("📖 Story")
    st.write(story)

    # -------------------------------
    # IMAGE GENERATION
    # -------------------------------
    st.subheader("🖼️ Image")

    image_url = generate_image(user_input, emotion)

    if image_url:
        st.image(image_url)
    else:
        st.warning("⚠️ Image generation failed. Try again.")
