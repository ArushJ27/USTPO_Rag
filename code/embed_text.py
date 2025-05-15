import pandas as pd
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path
import re

# ---------- Configuration ----------
CSV_PATH          = Path("data/processed/patents_2025.csv")
EMB_DIR           = Path("embeddings")
EMB_DIR.mkdir(parents=True, exist_ok=True)
METADATA_PATH     = EMB_DIR / "metadata_text_2025.pkl"
TEXT_INDEX_PATH   = EMB_DIR / "faiss_index_text_2025.bin"
EMBED_MODEL_NAME  = "all-MiniLM-L6-v2"

# ---------- 1. Load CSV ----------
df = pd.read_csv(CSV_PATH, dtype=str).fillna("")
print(f"Loaded {len(df)} records from {CSV_PATH}")

# ---------- 2. Build Metadata ----------
metadata = df[
    [
        "patent_number", "title", "inventor", "assignee", "classification",
        "image_path", "description", "gazette", "html_file"
    ]
].to_dict(orient="records")
with open(METADATA_PATH, "wb") as f:
    pickle.dump(metadata, f)
print(f"Saved text metadata ({len(metadata)} records) to {METADATA_PATH}")

# ---------- 3. Helpful tokens (company & US‑state) ----------
# We embed:
#   • Company  (full assignee string – already captured below)
#   • State    (two‑letter US state taken first from assignee, else from inventor)

state_re = r"\b(AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|" \
           r"MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|" \
           r"TN|TX|UT|VT|VA|WA|WV|WI|WY)\b"

# Try assignee first
states = (
    df["assignee"]
      .str.extract(state_re, expand=False)
      .fillna("")
      .str.upper()
)

# Fallback: look inside inventor string where assignee had no state
mask = states.eq("")
if mask.any():
    states[mask] = (
        df.loc[mask, "inventor"]
          .str.extract(state_re, expand=False)
          .fillna("")
          .str.upper()
    )

# ---------- 4. Prepare Texts (extended) ----------
titles    = df["title"].astype(str)
descs     = df["description"].astype(str)
assignees = df["assignee"].astype(str)
inventors = df["inventor"].astype(str)
classes   = df["classification"].astype(str)
gazettes  = df["gazette"].astype(str)
htmlfiles = df["html_file"].astype(str)

# Extract CPC codes and parent section (e.g. "H01L" → section "H")
cpc_full   = classes.str.extract(r"([A-Z]\d{2}[A-Z]?\s*\d+\/\d+)", expand=False).fillna("")
cpc_parent = cpc_full.str.slice(stop=3).fillna("")

# Extract filing / publication date (YYYYMMDD) from html_file, fallback gazette‑derived year
dates = htmlfiles.str.extract(r"(\d{8})", expand=False)
dates = dates.fillna("")  # blank if not found

texts = (
    "Patent "   + df["patent_number"] + ". " +
    "Title: "   + titles     + ". " +
    "Abstract: "+ descs      + ". " +
    "Assignee: "+ assignees  + ". " +
    "Inventor: "+ inventors  + ". " +
    "Class: "   + classes    + ". " +
    "CPC: "     + cpc_full   + ". " +
    "CPCsec: "  + cpc_parent + ". " +
    "Company: " + assignees  + ". " +
    "State: "   + states     + ". " +
    "Filed: "   + dates
).tolist()

# ---------- 5. Embed Texts ----------
model = SentenceTransformer(EMBED_MODEL_NAME)
print(f"Embedding texts with {EMBED_MODEL_NAME}…")
embs = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True
).astype("float32")
print(f"Generated embeddings shape: {embs.shape}")

# ---------- 6. Build & Save FAISS Index ----------
dim   = embs.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embs)
faiss.write_index(index, str(TEXT_INDEX_PATH))
print(f"Built & saved text FAISS index ({index.ntotal} vectors) to {TEXT_INDEX_PATH}")