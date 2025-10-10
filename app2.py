import os
import textwrap
from typing import List, Dict
from contextlib import nullcontext

import streamlit as st
from PIL import Image
from transformers import pipeline

# ✅ Optional: Only use token for Stable Diffusion image generation
HF_TOKEN = "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"  # replace with your real Hugging Face token

# Optional image generation support
try:
    from diffusers import StableDiffusionPipeline
    import torch
    DIFFUSERS_AVAILABLE = True
except Exception:
    DIFFUSERS_AVAILABLE = False


# -----------------------------
# Model Loading (cached)
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_emotion_pipeline():
    """Load emotion classification model safely without sending HF_TOKEN."""
    model_name = "j-hartmann/emotion-english-distilroberta-base"
    try:
        # Temporarily remove HF_TOKEN to avoid 401 errors for public models
        old_token = os.environ.pop("HF_TOKEN", None)
        emo_pipe = pipeline("text-classification", model=model_name, top_k=None)
        if old_token:
            os.environ["HF_TOKEN"] = old_token
    except Exception as e:
        st.error(f"Failed to load emotion model: {e}")
        raise
    return emo_pipe



@st.cache_resource(show_spinner=False)
def load_textgen_pipeline():
    """Load lightweight text generation model."""
    gen_model = "distilgpt2"
    try:
        gen_pipe = pipeline("text-generation", model=gen_model, device_map="auto")
    except Exception:
        gen_pipe = pipeline("text-generation", model=gen_model)
    return gen_pipe


@st.cache_resource(show_spinner=False)
def load_sd_pipeline(hf_token: str = None):
    """Load Stable Diffusion pipeline (requires token)."""
    if not DIFFUSERS_AVAILABLE:
        return None
    model_id = "runwayml/stable-diffusion-v1-5"
    try:
        if torch.cuda.is_available():
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                use_safetensors=True,
                torch_dtype=torch.float16,
                revision="fp16",
                use_auth_token=hf_token,
            ).to("cuda")
        else:
            pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=hf_token)
    except Exception as e:
        st.warning(f"Stable Diffusion pipeline failed to load: {e}")
        return None
    return pipe


# -----------------------------
# Core Functions
# -----------------------------
def detect_emotion(emo_pipe, text: str) -> List[Dict]:
    results = emo_pipe(text)
    if isinstance(results, list) and len(results) and isinstance(results[0], list):
        return sorted(results[0], key=lambda x: x["score"], reverse=True)
    return results


def build_continuation_prompt(original_text: str, top_emotion: str) -> str:
    return textwrap.dedent(f"""
    Continue the following story in the same voice and maintain the emotional tone of '{top_emotion}'.

    Story start:
    {original_text}

    Continue:
    """)


def generate_continuation(gen_pipe, prompt: str, max_new_tokens: int = 120) -> str:
    try:
        out = gen_pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.9,
            top_k=50,
            top_p=0.95,
        )
        return out[0]["generated_text"][len(prompt):].strip()
    except TypeError:
        out = gen_pipe(prompt, max_length=len(prompt.split()) + max_new_tokens, do_sample=True, temperature=0.9)
        return out[0]["generated_text"][len(prompt):].strip()


def generate_image(sd_pipe, prompt: str, guidance_scale: float = 7.5, num_inference_steps: int = 30):
    if sd_pipe is None:
        raise RuntimeError("Stable Diffusion pipeline not available")
    with torch.autocast("cuda") if torch.cuda.is_available() else nullcontext():
        image = sd_pipe(prompt, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps).images[0]
    return image


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Emotion-Aware Story Assistant", layout="wide")
st.title("🎭 Emotion-Aware AI Writing Assistant — Dynamic Storytelling")

st.sidebar.header("Settings")
show_images = st.sidebar.checkbox("Enable image generation (requires HF token)", value=False)

# ✅ Only use HF_TOKEN for Stable Diffusion, not text models
hf_token = HF_TOKEN if show_images else None

if show_images and not hf_token:
    st.sidebar.warning("Please set your valid Hugging Face token to enable image generation.")

with st.spinner("Loading models..."):
    emo_pipe = load_emotion_pipeline()
    gen_pipe = load_textgen_pipeline()
    sd_pipe = load_sd_pipeline(hf_token) if show_images else None

st.sidebar.markdown("---")
max_new_tokens = st.sidebar.slider("Max new tokens for continuation", 50, 400, 140)

st.subheader("Enter your story or scene snippet")
user_text = st.text_area("Story input", height=200)

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("Analyze Emotion & Generate Continuation"):
        if not user_text.strip():
            st.warning("Please type a short story fragment first.")
        else:
            with st.spinner("Detecting emotion..."):
                emo_results = detect_emotion(emo_pipe, user_text)
                top = emo_results[0]
                st.markdown(f"**Top emotion:** {top['label']} — score: {top['score']:.2f}")
                st.write(emo_results)

            with st.spinner("Generating continuation..."):
                prompt = build_continuation_prompt(user_text, top['label'])
                continuation = generate_continuation(gen_pipe, prompt, max_new_tokens=max_new_tokens)
                st.markdown("### Story Continuation")
                st.write(continuation)

            if show_images and sd_pipe is not None:
                img_prompt = (
                    f"Cinematic, highly detailed illustration reflecting the emotion '{top['label']}' "
                    f"for this scene: {user_text} Continue: {continuation} --ar 16:9"
                )
                st.markdown("### Generated Image")
                try:
                    with st.spinner("Generating image..."):
                        image = generate_image(sd_pipe, img_prompt)
                        st.image(image, use_container_width=True)
                except Exception as e:
                    st.error(f"Image generation failed: {e}")

with col2:
    st.markdown("### Quick Prompts & Controls")
    st.markdown("- Tip: keep input short (1–5 paragraphs) for best results.")
    st.markdown("---")
    st.markdown("### Example Inputs")
    if st.button("Load example: melancholic swing"):
        st.session_state['example'] = (
            "She stood under the oak, watching the empty swing sway in the autumn breeze, "
            "thinking of laughter that would never return."
        )
        st.experimental_rerun()
    if 'example' in st.session_state:
        st.text_area("Story input", value=st.session_state['example'], key='story_example')

st.markdown("---")
st.markdown("Built with ❤️ — Customize models & UI freely!")

