import streamlit as st
from PIL import Image
from emotion_detector import detect_emotion
from story_generator import continue_story
from image_prompt_generator import generate_image_prompt
from image_generator import generate_emotion_image  

st.set_page_config(page_title="Emotion-Aware Story Generator", layout="centered")
st.title("🎭 Emotion-Aware AI Story Assistant")

scene_input = st.text_area("Enter the opening scene:", height=150)

if st.button("Generate"):
    if not scene_input.strip():
        st.error("Please enter some text to analyze.")
    else:
        with st.spinner("Detecting emotion..."):
            emotion = detect_emotion(scene_input)
            st.success(f"Detected Emotion: **{emotion.capitalize()}**")

        with st.spinner("Generating continuation..."):
            continuation = continue_story(scene_input, emotion)
            st.subheader("📘 Continued Story")
            st.write(continuation)

        with st.spinner("Generating visual prompt..."):
            img_prompt = generate_image_prompt(scene_input, emotion)
            st.text("Prompt used for image generation:")
            st.write(img_prompt)

        with st.spinner("Generating image..."):
            image_path = generate_emotion_image(scene_input,emotion)
            st.image(image_path, caption=f"Generated image for '{emotion}'", use_container_width=True)