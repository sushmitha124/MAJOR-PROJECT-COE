from diffusers import StableDiffusionXLPipeline
import torch

# Load the model
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    variant="fp16" if torch.cuda.is_available() else None,
    use_safetensors=True
).to("cuda" if torch.cuda.is_available() else "cpu")

def generate_emotion_image(prompt, emotion):
    styled_prompt = (
        f"Highly detailed, cinematic, ultra-realistic of a real human showing strong {emotion.lower()} emotion. "
        f"{prompt}, expressive facial features matching the emotion, dramatic but natural lighting, realistic background, "
        f"35mm photography, shallow depth of field, photorealism, 8k resolution"
    )
    result = pipe(prompt=styled_prompt, guidance_scale=1.5, num_inference_steps=4)
    image = result.images[0]
    filename = f"{emotion.lower()}_image.png"
    image.save(filename)
    return filename