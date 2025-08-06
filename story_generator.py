import google.generativeai as genai
import os

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash-lite")

def continue_story(scene, emotion):
    prompt = (
        f"Continue this story in the tone of {emotion}. Make it emotionally rich and realistic.\n\n"
        f"Scene: {scene}\n\n"
        f"Continuation:"
    )
    response = model.generate_content(prompt)
    return response.text.strip()