import gradio as gr
import os
import urllib.parse
import requests

# -------------------------------
# Hugging Face APIs
# -------------------------------
HF_TEXT_API = "https://api-inference.huggingface.co/models/google/flan-t5-large"
HF_EMOTION_API = "https://api-inference.huggingface.co/models/j-hartmann/emotion-english-distilroberta-base"

HF_TOKEN = os.getenv("HF_API_KEY")
headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

# -------------------------------
# Emotion Detection
# -------------------------------
def detect_emotion(text):
    try:
        payload = {"inputs": text}
        response = requests.post(HF_EMOTION_API, headers=headers, json=payload)

        if response.status_code != 200:
            return "Neutral"

        result = response.json()
        emotions = result[0]

        top_emotion = max(emotions, key=lambda x: x['score'])
        return top_emotion['label']

    except:
        return "Neutral"

# -------------------------------
# Story Generation (NO GEMINI)
# -------------------------------
def generate_story_text(prompt):
    try:
        payload = {
            "inputs": prompt,
            "parameters": {"max_length": 200}
        }

        response = requests.post(HF_TEXT_API, headers=headers, json=payload)

        if response.status_code != 200:
            return "Error generating story"

        result = response.json()

        return result[0]["generated_text"]

    except:
        return "Error generating story"

# -------------------------------
# Main Function
# -------------------------------
def generate_story(user_input):
    try:
        emotion = detect_emotion(user_input)

        prompt = f"""
        The user is feeling {emotion}.
        Continue a short, engaging story based on this emotion:
        {user_input}
        """

        story = generate_story_text(prompt)

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
    description="AI detects emotion, generates story and image (fully free, no Gemini)."
)

# -------------------------------
# Render Entry Point
# -------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    iface.launch(server_name="0.0.0.0", server_port=port)
