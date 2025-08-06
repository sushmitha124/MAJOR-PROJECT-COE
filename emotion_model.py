import google.generativeai as genai
import os

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash-lite")

def detect_emotion(text):
    prompt = f"Detect the dominant emotion from this text: '{text}'. Respond with one word like 'fear', 'joy', 'sadness', 'anger', or 'surprise'."
    response = model.generate_content(prompt)
    return response.text.strip().lower()