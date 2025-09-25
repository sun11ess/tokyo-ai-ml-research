import os
import torch
import clip
from PIL import Image
import pandas as pd
import statistics

PROMPT_FILE = "sample_eval_prompts.txt"
IMAGE_DIR = "juggernautXL_rundiffusion/"
RESULTS_FILE = "results/clipscore_results.csv"

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

prompts = {}
with open(PROMPT_FILE, "r") as f:
    for line in f:
        if ":" in line:
            idx, text = line.strip().split(":", 1)
            prompts[int(idx)] = text.strip()

results = []

# Loop through images
for idx, prompt in prompts.items():
    image_path = os.path.join(IMAGE_DIR, f"{idx}.png")
    if not os.path.exists(image_path):
        print(f" Missing image: {image_path}")
        continue

    # Load image
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text = clip.tokenize([prompt]).to(device)

    # Compute CLIPScore
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        similarity = torch.cosine_similarity(image_features, text_features).item()

    # Save result
    results.append({"Index": idx, "Prompt": prompt, "CLIPScore": similarity})
    print(f"{idx}.png - CLIPScore: {similarity:.4f}")


df = pd.DataFrame(results).sort_values("Index")
os.makedirs("results", exist_ok=True)
df.to_csv(RESULTS_FILE, index=False)

print(f"\n Results saved to {RESULTS_FILE}")
