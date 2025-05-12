import pandas as pd
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path

# ---------- Configuration ----------
CSV_PATH = Path("data/processed/patents_2025.csv")
EMB_DIR = Path("embeddings")
EMB_DIR.mkdir(parents=True, exist_ok=True)
METADATA_PATH = EMB_DIR / "metadata_text_2025.pkl"
TEXT_INDEX_PATH = EMB_DIR / "faiss_index_text_2025.bin"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# ---------- 1. Load CSV ----------
# Read all columns as strings to avoid dtype issues
df = pd.read_csv(CSV_PATH, dtype=str).fillna("")
print(f"Loaded {len(df)} records from {CSV_PATH}")

# ---------- 2. Build Metadata ----------
# Include html_file plus other relevant fields
metadata = df[[
    "patent_number", "title", "inventor", "assignee", "classification",
    "image_path", "description", "gazette", "html_file"
]].to_dict(orient="records")

with open(METADATA_PATH, "wb") as f:
    pickle.dump(metadata, f)
print(f"Saved text metadata ({len(metadata)} records) to {METADATA_PATH}")

# ---------- 3. Prepare Texts ----------
# Safely concatenate title and description (both strings)
titles = df["title"].astype(str)
descs  = df["description"].astype(str)
texts  = (titles + ". " + descs).tolist()

# ---------- 4. Embed Texts ----------
model = SentenceTransformer(EMBED_MODEL_NAME)
print(f"Embedding texts with {EMBED_MODEL_NAME}...")
embs = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True
).astype("float32")
print(f"Generated embeddings shape: {embs.shape}")

# ---------- 5. Build & Save FAISS Index ----------
dim   = embs.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embs)
faiss.write_index(index, str(TEXT_INDEX_PATH))
print(f"Built & saved text FAISS index ({index.ntotal} vectors) to {TEXT_INDEX_PATH}")