import google.generativeai as genai
import os

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash-lite")

def generate_image_prompt(scene, emotion):
    prompt = (
        f"Generate a detailed visual prompt for an image showing the emotion '{emotion}' in the following scene:\n\n"
        f"{scene}\n\n"
        f"Make sure the facial expressions and environment reflect the emotion clearly."
    )
    response = model.generate_content(prompt)
    return response.text.strip()