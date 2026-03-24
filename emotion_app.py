# import streamlit as st
# import torch
# from transformers import pipeline
# from diffusers import StableDiffusionPipeline
# from PIL import Image
# import time

# # ─────────────────────────────────────────────
# # PAGE CONFIG
# # ─────────────────────────────────────────────
# st.set_page_config(
#     page_title="Emotion Story Generator",
#     page_icon="🎭",
#     layout="centered"
# )

# # ─────────────────────────────────────────────
# # CUSTOM CSS
# # ─────────────────────────────────────────────
# st.markdown("""
# <style>
#     @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=DM+Sans:wght@300;400;500&display=swap');

#     html, body, [class*="css"] {
#         font-family: 'DM Sans', sans-serif;
#         background-color: #0f0f0f;
#         color: #f0ece3;
#     }

#     .main {
#         background-color: #0f0f0f;
#     }

#     h1, h2, h3 {
#         font-family: 'Playfair Display', serif !important;
#         color: #f0ece3 !important;
#     }

#     .stTextArea textarea {
#         background-color: #1a1a1a !important;
#         color: #f0ece3 !important;
#         border: 1px solid #333 !important;
#         border-radius: 8px !important;
#         font-family: 'DM Sans', sans-serif !important;
#         font-size: 15px !important;
#     }

#     .stButton > button {
#         background: linear-gradient(135deg, #c9a96e, #a0784a) !important;
#         color: #0f0f0f !important;
#         font-family: 'DM Sans', sans-serif !important;
#         font-weight: 600 !important;
#         font-size: 15px !important;
#         border: none !important;
#         border-radius: 8px !important;
#         padding: 0.6rem 2rem !important;
#         width: 100% !important;
#         transition: opacity 0.2s ease !important;
#     }

#     .stButton > button:hover {
#         opacity: 0.85 !important;
#     }

#     .emotion-badge {
#         display: inline-block;
#         padding: 6px 20px;
#         border-radius: 30px;
#         font-size: 14px;
#         font-weight: 600;
#         letter-spacing: 1px;
#         text-transform: uppercase;
#         margin-bottom: 1rem;
#     }

#     .story-box {
#         background: #1a1a1a;
#         border-left: 3px solid #c9a96e;
#         border-radius: 8px;
#         padding: 1.5rem 1.8rem;
#         font-family: 'Playfair Display', serif;
#         font-size: 17px;
#         line-height: 1.9;
#         color: #e8e0d0;
#         margin: 1rem 0;
#     }

#     .section-label {
#         font-size: 11px;
#         letter-spacing: 2px;
#         text-transform: uppercase;
#         color: #c9a96e;
#         margin-bottom: 0.4rem;
#         font-weight: 500;
#     }

#     .divider {
#         border: none;
#         border-top: 1px solid #2a2a2a;
#         margin: 2rem 0;
#     }

#     /* hide streamlit branding */
#     #MainMenu {visibility: hidden;}
#     footer {visibility: hidden;}
#     header {visibility: hidden;}
# </style>
# """, unsafe_allow_html=True)


# # ─────────────────────────────────────────────
# # EMOTION COLOR MAP
# # ─────────────────────────────────────────────
# EMOTION_STYLES = {
#     "sadness":  {"bg": "#1e2a3a", "color": "#7ab3e0", "emoji": "💙"},
#     "joy":      {"bg": "#2a2a1a", "color": "#f0d060", "emoji": "✨"},
#     "fear":     {"bg": "#1f1a2e", "color": "#b388e8", "emoji": "🌑"},
#     "anger":    {"bg": "#2e1a1a", "color": "#e87070", "emoji": "🔥"},
#     "disgust":  {"bg": "#1a2e1a", "color": "#80c980", "emoji": "🌿"},
#     "neutral":  {"bg": "#1e1e1e", "color": "#aaaaaa", "emoji": "🌫️"},
#     "surprise": {"bg": "#2a1e2e", "color": "#f0a0e0", "emoji": "⚡"},
# }


# # ─────────────────────────────────────────────
# # STORY TEMPLATES
# # ─────────────────────────────────────────────
# STORY_TEMPLATES = {
#     "sadness": (
#         "{prompt} She felt a deep heaviness in her heart as memories slowly filled her mind. "
#         "The silence around her made everything more painful, like a fog that refused to lift. "
#         "Taking a long, slow breath, she closed her eyes and tried to accept the moment — "
#         "not to fight it, but to let it pass through her like rain."
#     ),
#     "joy": (
#         "{prompt} A warm smile spread across her face as happiness filled every corner of the moment. "
#         "Everything around her felt brighter, lighter, as if the world had been turned up just a little. "
#         "She held onto that feeling with both hands, knowing it was rare and worth savoring fully."
#     ),
#     "fear": (
#         "{prompt} Her heart started beating faster as a cold sense of fear crept slowly up her spine. "
#         "She looked around, unsure of what lay ahead, every shadow feeling like a threat. "
#         "Then, gathering every last piece of courage she had, she took one small step forward."
#     ),
#     "anger": (
#         "{prompt} Her emotions burned with a fierce intensity as frustration took over her thoughts entirely. "
#         "She struggled to stay calm, trying hard not to let the fire inside her spill over. "
#         "Slowly, steadily, she pulled herself back — choosing clarity over the heat of the moment."
#     ),
#     "disgust": (
#         "{prompt} She felt deeply uncomfortable, unsettled by what she had just witnessed or heard. "
#         "It was difficult even to process — her mind recoiling from the weight of the moment. "
#         "She stepped away, quietly, quickly, letting clean air slowly replace what had been there."
#     ),
#     "surprise": (
#         "{prompt} She stopped completely — the moment catching her entirely off guard and breathless. "
#         "Her mind raced to catch up with what her eyes had just seen unfold before her. "
#         "After a long pause, a slow smile of disbelief crept across her face. She hadn't expected this."
#     ),
#     "neutral": (
#         "{prompt} She paused for a moment, quietly observing everything that surrounded her in the space. "
#         "The situation felt ordinary on the surface, yet held a quiet meaning she couldn't quite name. "
#         "She continued forward with a calm and settled mind, neither rushed nor reluctant."
#     ),
# }


# # ─────────────────────────────────────────────
# # CACHED MODEL LOADERS
# # ─────────────────────────────────────────────
# @st.cache_resource(show_spinner=False)
# def load_emotion_model():
#     return pipeline(
#         "text-classification",
#         model="j-hartmann/emotion-english-distilroberta-base",
#         top_k=1,
#         device=0 if torch.cuda.is_available() else -1
#     )

# @st.cache_resource(show_spinner=False)
# def load_sd_pipeline():
#     model_id = "runwayml/stable-diffusion-v1-5"
#     dtype = torch.float16 if torch.cuda.is_available() else torch.float32
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
#     pipe = pipe.to(device)
#     pipe.safety_checker = None  # optional: disable for speed
#     return pipe


# # ─────────────────────────────────────────────
# # CORE FUNCTIONS
# # ─────────────────────────────────────────────
# def detect_emotion(classifier, text: str) -> str:
#     result = classifier(text)
#     return result[0][0]["label"].lower()

# def generate_story(prompt: str, emotion: str) -> str:
#     template = STORY_TEMPLATES.get(emotion, STORY_TEMPLATES["neutral"])
#     return template.format(prompt=prompt.strip())

# def generate_image(sd_pipe, prompt: str, emotion: str) -> Image.Image:
#     image_prompt = (
#         f"{prompt}, {emotion} mood, "
#         "shot on Canon EOS R5, 85mm lens, natural lighting, "
#         "real photograph, DSLR, shallow depth of field, "
#         "candid moment, true to life, documentary style, "
#         "high resolution photography, no filters"
#     )
#     negative_prompt = (
#         "cartoon, anime, illustration, painting, digital art, "
#         "CGI, render, 3D, fantasy, unrealistic, plastic skin, "
#         "overly smooth, airbrushed, studio backdrop, fake lighting, "
#         "low quality, blurry, bad anatomy, distorted face, "
#         "watermark, text, duplicate, cropped, deformed, "
#         "oversaturated, dramatic fx, cinematic grade, lens flare"
#     )
#     result = sd_pipe(
#         image_prompt,
#         negative_prompt=negative_prompt,
#         guidance_scale=6.5,      # lower = more natural, less "forced"
#         num_inference_steps=40,  # more steps = more detail
#         height=512,
#         width=512,
#     )
#     return result.images[0]


# # ─────────────────────────────────────────────
# # UI — HEADER
# # ─────────────────────────────────────────────
# st.markdown("<br>", unsafe_allow_html=True)
# st.markdown("## 🎭 Emotion Story Generator")
# st.markdown(
#     "<p style='color:#888; font-size:15px; margin-top:-10px;'>"
#     "Type anything — a feeling, a moment, a thought. We'll detect the emotion, "
#     "write a story, and generate an image for it."
#     "</p>",
#     unsafe_allow_html=True
# )
# st.markdown("<hr class='divider'>", unsafe_allow_html=True)


# # ─────────────────────────────────────────────
# # UI — INPUT
# # ─────────────────────────────────────────────
# user_prompt = st.text_area(
#     label="Your prompt",
#     placeholder="e.g. She stood at the edge of the cliff, looking down at the waves...",
#     height=120,
#     label_visibility="collapsed"
# )

# generate_btn = st.button("✦ Generate")


# # ─────────────────────────────────────────────
# # UI — GENERATION
# # ─────────────────────────────────────────────
# if generate_btn:

#     if not user_prompt.strip():
#         st.warning("Please enter a prompt before generating.")
#         st.stop()

#     # Load models
#     with st.spinner("Loading emotion model..."):
#         emotion_clf = load_emotion_model()

#     with st.spinner("Loading image model (first run may take a few minutes)..."):
#         sd_pipe = load_sd_pipeline()

#     st.markdown("<hr class='divider'>", unsafe_allow_html=True)

#     # ── Step 1: Emotion
#     st.markdown("<p class='section-label'>Step 1 — Emotion Detected</p>", unsafe_allow_html=True)
#     with st.spinner("Analyzing emotion..."):
#         emotion = detect_emotion(emotion_clf, user_prompt)

#     style = EMOTION_STYLES.get(emotion, EMOTION_STYLES["neutral"])
#     st.markdown(
#         f"<div class='emotion-badge' style='background:{style['bg']}; color:{style['color']};'>"
#         f"{style['emoji']} &nbsp; {emotion.upper()}"
#         f"</div>",
#         unsafe_allow_html=True
#     )

#     # ── Step 2: Story
#     st.markdown("<p class='section-label'>Step 2 — Generated Story</p>", unsafe_allow_html=True)
#     with st.spinner("Writing story..."):
#         story = generate_story(user_prompt, emotion)
#         time.sleep(0.5)  # small delay for UX feel

#     st.markdown(f"<div class='story-box'>{story}</div>", unsafe_allow_html=True)

#     # ── Step 3: Image
#     st.markdown("<p class='section-label'>Step 3 — Generated Image</p>", unsafe_allow_html=True)
#     with st.spinner("Generating image (30 steps)... this takes ~30–60 seconds"):
#         image = generate_image(sd_pipe, user_prompt, emotion)

#     st.image(image, use_column_width=True, caption=f"{style['emoji']} {emotion.capitalize()} — AI Generated")

#     # Save image
#     image.save("output.jpg")

#     # Download button
#     with open("output.jpg", "rb") as f:
#         st.download_button(
#             label="⬇ Download Image",
#             data=f,
#             file_name="emotion_image.jpg",
#             mime="image/jpeg"
#         )

#     st.markdown("<hr class='divider'>", unsafe_allow_html=True)
#     st.success("✦ Done! Try a new prompt above.")


import streamlit as st
import requests
from PIL import Image
import io
import time

# CONFIG
st.set_page_config(page_title="Emotion Story Generator", page_icon="🎭")

HF_TOKEN = st.secrets["HF_TOKEN"]

# EMOTION STYLES
EMOTION_STYLES = {
    "sadness":  {"emoji": "💙"},
    "joy":      {"emoji": "✨"},
    "fear":     {"emoji": "🌑"},
    "anger":    {"emoji": "🔥"},
    "disgust":  {"emoji": "🌿"},
    "neutral":  {"emoji": "🌫️"},
    "surprise": {"emoji": "⚡"},
}

# STORY
STORY_TEMPLATES = {
    "sadness": "{prompt} She felt a deep heaviness in her heart...",
    "joy": "{prompt} A warm smile spread across her face...",
    "fear": "{prompt} Her heart started beating faster...",
    "anger": "{prompt} Her emotions burned with intensity...",
    "disgust": "{prompt} She felt uncomfortable...",
    "surprise": "{prompt} She stopped completely...",
    "neutral": "{prompt} She paused quietly..."
}

# ✅ EMOTION API (ONLY ONE FUNCTION)
def detect_emotion(text):
    API_URL = "https://api-inference.huggingface.co/models/j-hartmann/emotion-english-distilroberta-base"

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}"
    }

    response = requests.post(API_URL, headers=headers, json={"inputs": text})

    if response.status_code != 200:
        st.warning("Emotion API failed, using neutral")
        return "neutral"

    result = response.json()

    try:
        return result[0][0]["label"].lower()
    except:
        return "neutral"

# IMAGE API
def generate_image(prompt, emotion):
    API_URL = "https://api-inference.huggingface.co/models/stabilityai/sdxl-turbo"

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}"
    }

    full_prompt = f"{prompt}, {emotion} emotion, ultra realistic, DSLR photo"

    try:
        response = requests.post(API_URL, headers=headers, json={"inputs": full_prompt})

        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content))

        elif response.status_code == 503:
            st.warning("Model loading... try again in few seconds")
            return None

        else:
            st.error(f"Image API Error: {response.status_code}")
            return None

    except:
        st.error("Image generation failed")
        return None

# STORY FUNCTION
def generate_story(prompt, emotion):
    return STORY_TEMPLATES.get(emotion, STORY_TEMPLATES["neutral"]).format(prompt=prompt)

# UI
st.title("🎭 Emotion Story Generator")

user_prompt = st.text_area("Enter your prompt")

if st.button("Generate"):

    if not user_prompt.strip():
        st.warning("Enter something")
        st.stop()

    # Emotion
    with st.spinner("Analyzing emotion..."):
        emotion = detect_emotion(user_prompt)

    emoji = EMOTION_STYLES.get(emotion, {}).get("emoji", "🌫️")
    st.subheader(f"{emoji} Emotion: {emotion.upper()}")

    # Story
    with st.spinner("Generating story..."):
        story = generate_story(user_prompt, emotion)
        time.sleep(0.5)

    st.write(story)

    # Image
    with st.spinner("Generating image..."):
        image = generate_image(user_prompt, emotion)

    if image:
        st.image(image, caption=f"{emotion} mood")
    else:
        st.warning("Image not generated. Try again.")

    st.success("Done!")
