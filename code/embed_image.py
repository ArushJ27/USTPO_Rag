import pandas as pd
import pickle
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

# ---------- Configuration ----------
RAW_ROOT        = Path("data/raw/Patent Gazettes/2025")
CSV_PATH        = Path("data/processed/patents_2025.csv")
EMBEDDINGS_DIR  = Path("embeddings")
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_EMB_PATH  = EMBEDDINGS_DIR / "image_embeddings_2025.npy"
IMAGE_META_PATH = EMBEDDINGS_DIR / "image_metadata_2025.pkl"

# 1) Load CSV with image paths as strings
print(f"Loading CSV from {CSV_PATH}...")
df = pd.read_csv(CSV_PATH, dtype={"gazette": str, "image_path": str})
df = df.dropna(subset=["image_path"])
print(f"Found {len(df)} patents with images.")

# 2) Preload images and build metadata list
images = []
metadata = []
for _, row in df.iterrows():
    gazette = row["gazette"].strip()
    img_file = row["image_path"].strip()
    patent_number = row["patent_number"]
    html_file = row.get("html_file", "")
    img_path = RAW_ROOT / gazette / "OG" / "html" / gazette / img_file
    if not img_path.exists():
        print(f"Warning: {img_path} not found, skipping.")
        continue
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"Error opening {img_path}: {e}, skipping.")
        continue
    images.append(img)
    metadata.append({
        "patent_number": patent_number,
        "title":       row.get("title", ""),
        "inventor":    row.get("inventor", ""),
        "assignee":    row.get("assignee", ""),
        "classification": row.get("classification", ""),
        "description": row.get("description", ""),
        "gazette":     gazette,
        "image_path":  img_file,
        "html_file":   html_file
    })

# Save metadata
with open(IMAGE_META_PATH, "wb") as f:
    pickle.dump(metadata, f)
print(f"Saved image metadata ({len(metadata)} records) to {IMAGE_META_PATH}")

# 3) Embed images on MPS (if available)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

# 4) Embed in batches with progress bar
batch_size = 16
emb_list = []
num_batches = (len(images) + batch_size - 1) // batch_size
print("Starting image embedding in batches...")
for i in tqdm(range(0, len(images), batch_size), desc="Embedding batches", total=num_batches):
    batch = images[i : i + batch_size]
    inputs = processor(images=batch, return_tensors="pt").to(device)
    with torch.no_grad():
        feats = clip_model.get_image_features(**inputs)
    emb_list.append(feats.cpu().numpy())

# 5) Concatenate and save embeddings
embeddings = np.concatenate(emb_list, axis=0).astype("float32")
np.save(IMAGE_EMB_PATH, embeddings)
print(f"Saved image embeddings to {IMAGE_EMB_PATH}")
