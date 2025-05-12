# build_image_faiss_2025.py

import numpy as np
import faiss
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────
EMB_DIR           = Path("embeddings")
IMAGE_NPY_PATH    = EMB_DIR / "image_embeddings_2025.npy"
IMAGE_INDEX_PATH  = EMB_DIR / "faiss_index_image_2025.bin"

# ── 1. Load the embeddings ────────────────────────────────────────────────
print(f"Loading image embeddings from {IMAGE_NPY_PATH} …")
embs = np.load(IMAGE_NPY_PATH).astype("float32")
print(f"Loaded embeddings array of shape {embs.shape}")

# ── 2. Build an L2 FAISS index ───────────────────────────────────────────
dim   = embs.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embs)
print(f"Built FAISS index with {index.ntotal} vectors (dim={dim})")

# ── 3. Save the index ────────────────────────────────────────────────────
print(f"Saving FAISS index to {IMAGE_INDEX_PATH} …")
faiss.write_index(index, str(IMAGE_INDEX_PATH))
print("Done.")