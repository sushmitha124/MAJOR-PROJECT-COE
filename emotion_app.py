import gradio as gr
import google.generativeai as genai
import os
import urllib.parse
import requests

# -------------------------------
# Gemini Config (STABLE MODEL)
# -------------------------------
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("❌ GEMINI_API_KEY not set")

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-pro")  # ✅ stable working model

# -------------------------------
# Hugging Face Emotion Model
# -------------------------------
HF_API_URL = "https://api-inference.huggingface.co/models/j-hartmann/emotion-english-distilroberta-base"
HF_TOKEN = os.getenv("HF_API_KEY")

headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

def detect_emotion(text):
    try:
        payload = {"inputs": text}
        response = requests.post(HF_API_URL, headers=headers, json=payload)

        if response.status_code != 200:
            return "Neutral"

        result = response.json()
        emotions = result[0]

        top_emotion = max(emotions, key=lambda x: x['score'])
        return top_emotion['label']

    except:
        return "Neutral"

# -------------------------------
# Story + Image Generation
# -------------------------------
def generate_story(user_input):
    try:
        emotion = detect_emotion(user_input)

        prompt = f"""
        The user is feeling {emotion}.
        Continue a short, engaging story based on this emotion:
        {user_input}
        """

        response = model.generate_content(prompt)
        story = response.text

        # Image generation (Pollinations)
        image_prompt = f"{emotion} cinematic scene, realistic, {user_input}"
        encoded_prompt = urllib.parse.quote(image_prompt)
        image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}"

        return emotion, story, image_url

    except Exception as e:
        return "Error", str(e), None

# -------------------------------
# UI
# -------------------------------
iface = gr.Interface(
    fn=generate_story,
    inputs=gr.Textbox(lines=4, placeholder="Enter your thoughts..."),
    outputs=[
        gr.Textbox(label="Detected Emotion"),
        gr.Textbox(label="Generated Story"),
        gr.Image(label="Generated Image")
    ],
    title="Emotion-Aware AI Assistant with Visual Storytelling",
    description="AI detects emotion, generates story and image."
)

# -------------------------------
# Render Entry Point
# -------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    iface.launch(server_name="0.0.0.0", server_port=port)
