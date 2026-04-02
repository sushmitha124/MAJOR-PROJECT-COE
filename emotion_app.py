import gradio as gr
import google.generativeai as genai
import os
import urllib.parse

# -------------------------------
# Gemini Config
# -------------------------------
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("❌ GEMINI_API_KEY not set")

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

# -------------------------------
# Emotion Detection
# -------------------------------
def detect_emotion(text):
    text = text.lower()

    if any(word in text for word in ["sad", "depressed", "cry", "lonely"]):
        return "Sad"
    elif any(word in text for word in ["happy", "excited", "joy", "love"]):
        return "Happy"
    elif any(word in text for word in ["angry", "mad", "furious"]):
        return "Angry"
    elif any(word in text for word in ["fear", "scared", "afraid"]):
        return "Fear"
    else:
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

        # 🎨 Create image prompt
        image_prompt = f"{emotion} cinematic scene, realistic, {user_input}"

        # Encode URL safely
        encoded_prompt = urllib.parse.quote(image_prompt)

        image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}"

        return f"Emotion: {emotion}\n\n{story}", image_url

    except Exception as e:
        return f"❌ Error: {str(e)}", None

# -------------------------------
# UI
# -------------------------------
iface = gr.Interface(
    fn=generate_story,
    inputs=gr.Textbox(lines=4, placeholder="Enter your thoughts..."),
    outputs=[
        gr.Textbox(label="Generated Story"),
        gr.Image(label="Generated Image")
    ],
    title="Emotion-Aware AI Assistant with Visual Storytelling",
    description="AI generates both story and image based on your emotions."
)

# -------------------------------
# Render Entry Point
# -------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    iface.launch(server_name="0.0.0.0", server_port=port)
