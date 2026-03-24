import streamlit as st
import torch
from transformers import pipeline
import requests
from PIL import Image
from io import BytesIO
import time
import os

# Fix HuggingFace cache (important for Streamlit Cloud)
os.environ["HF_HOME"] = "/tmp/huggingface"

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Emotion Story Generator",
    page_icon="🎭",
    layout="centered"
)

st.title("🎭 Emotion Story Generator")
st.write("Type anything — get emotion, story, and image ✨")

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

# ─────────────────────────────────────────────
# STORY TEMPLATES
# ─────────────────────────────────────────────
STORY_TEMPLATES = {
    "sadness": "{prompt} She felt a deep heaviness in her heart as memories returned slowly.",
    "joy": "{prompt} A warm smile spread across her face as happiness filled the moment.",
    "fear": "{prompt} Her heart raced as fear quietly crept into her thoughts.",
    "anger": "{prompt} Her emotions burned intensely as frustration took control.",
    "disgust": "{prompt} She felt uncomfortable, stepping away from the situation.",
    "surprise": "{prompt} She froze, caught completely off guard by what happened.",
    "neutral": "{prompt} She quietly observed everything with a calm and steady mind."
}

# ─────────────────────────────────────────────
# FUNCTIONS
# ─────────────────────────────────────────────
def detect_emotion(classifier, text):
    result = classifier(text)
    return result[0][0]["label"].lower()

def generate_story(prompt, emotion):
    template = STORY_TEMPLATES.get(emotion, STORY_TEMPLATES["neutral"])
    return template.format(prompt=prompt)

def generate_image(prompt, emotion):
    API_URL = "https://api-inference.huggingface.co/models/stabilityai/sdxl-turbo"

    headers = {
        "Authorization": f"Bearer {st.secrets['HF_TOKEN']}"
    }

    full_prompt = f"{prompt}, {emotion} mood, photorealistic, high quality, realistic human"

    response = requests.post(
        API_URL,
        headers=headers,
        json={"inputs": full_prompt}
    )

    if response.status_code != 200:
        return None

    image = Image.open(BytesIO(response.content))
    return image

# ─────────────────────────────────────────────
# UI INPUT
# ─────────────────────────────────────────────
user_prompt = st.text_area("Enter your prompt")

if st.button("Generate"):

    if not user_prompt.strip():
        st.warning("Please enter something!")
        st.stop()

    # Load model
    with st.spinner("Loading emotion model..."):
        clf = load_emotion_model()

    # Detect emotion
    with st.spinner("Detecting emotion..."):
        emotion = detect_emotion(clf, user_prompt)

    st.success(f"Detected Emotion: {emotion.upper()}")

    # Generate story
    with st.spinner("Generating story..."):
        story = generate_story(user_prompt, emotion)
        time.sleep(0.5)

    st.subheader("📖 Story")
    st.write(story)

    # Generate image
    with st.spinner("Generating image..."):
        image = generate_image(user_prompt, emotion)

    if image:
        st.subheader("🖼️ Image")
        st.image(image, use_column_width=True)

        # Download
        image.save("output.jpg")
        with open("output.jpg", "rb") as f:
            st.download_button(
                label="⬇ Download Image",
                data=f,
                file_name="emotion_image.jpg",
                mime="image/jpeg"
            )
    else:
        st.error("Image generation failed. Try again.")

    st.success("Done! Try another prompt ✨")
