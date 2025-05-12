"""
app_gradio.py – Patent RAG demo with inline HTML preview (CPU‑only image search)
───────────────────────────────────────────────────────────────────────────────
• Search (text or image) → top‑5 patents table (inline GIF + 250‑char desc)
• Enter a Patent # from the table → full patent front page rendered in Gradio
"""

# ────────────────────────────────────────────────────────
# 0. GLOBAL SINGLE‑THREAD SWITCH (avoid BLAS segfaults)
# ────────────────────────────────────────────────────────
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
          "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"

# ────────────────────────────────────────────────────────
# 1. Imports
# ────────────────────────────────────────────────────────
import base64
import pickle
import re
import numpy as np
import faiss                     # now sees OMP_NUM_THREADS=1
import torch
import gradio as gr
from pathlib import Path
from dotenv import load_dotenv
import litellm
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel

# ────────────────────────────────────────────────────────
# 2. Paths & environment
# ────────────────────────────────────────────────────────
RAW_ROOT = Path("data/raw/Patent Gazettes/2025")
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

# ────────────────────────────────────────────────────────
# 3. Load FAISS & models
# ────────────────────────────────────────────────────────
TEXT_IDX = faiss.read_index("embeddings/faiss_index_text_2025.bin")
with open("embeddings/metadata_text_2025.pkl", "rb") as f:
    TEXT_META = pickle.load(f)
ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

IMG_IDX = faiss.read_index("embeddings/faiss_index_image_2025.bin")
with open("embeddings/image_metadata_2025.pkl", "rb") as f:
    IMG_META = pickle.load(f)

# Force CPU for CLIP to avoid MPS/thread issues
device = torch.device("cpu")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
CLIP      = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

# ────────────────────────────────────────────────────────
# 4. Retrieval helpers
# ────────────────────────────────────────────────────────
def search_text(q: str, k: int = 5):
    vec = ST_MODEL.encode([q]).astype("float32")
    _, I = TEXT_IDX.search(vec, k)
    return [TEXT_META[i] for i in I[0]]

def search_image(img: Image.Image, k: int = 5):
    try:
        img = img.convert("RGB")
        if max(img.size) > 1024:         # down‑size gigantic uploads
            img.thumbnail((1024, 1024))
        inputs = processor(images=img, return_tensors="pt")
        with torch.no_grad():
            feats = CLIP.get_image_features(**inputs).cpu().numpy().astype("float32")
        _, I = IMG_IDX.search(feats, k)
        return [IMG_META[i] for i in I[0]]
    except Exception as e:
        print("[ERROR] search_image:", e, flush=True)
        raise

# ────────────────────────────────────────────────────────
# 5. Table builder
# ────────────────────────────────────────────────────────
def build_table(hits: list[dict]) -> str:
    lines = ["| Patent # | Title | Description | Image |",
             "|---|---|---|---|"]
    for h in hits:
        pn    = h.get("patent_number", "")
        title = h.get("title", "").replace("|", "\\|")
        desc  = (h.get("description", "")[:250] + "…").replace("|", "\\|")
        img_md = ""
        m = next((x for x in IMG_META if x.get("patent_number") == pn), None)
        if m:
            gif = RAW_ROOT / m["gazette"] / "OG" / "html" / m["image_path"]
            if gif.exists():
                b64 = base64.b64encode(gif.read_bytes()).decode()
                img_md = f"![{title}](data:image/gif;base64,{b64})"
        lines.append(f"| {pn} | {title} | {desc} | {img_md} |")
    return "\n".join(lines)

# ────────────────────────────────────────────────────────
# 6. LLM summary
# ────────────────────────────────────────────────────────
def llm_summary(query: str, hits: list[dict]) -> str:
    bullets = "\n".join(
        f"- **{h['title']}** (#{h['patent_number']}): {h['description'][:120]}…"
        for h in hits
    )
    prompt = ("Summarize in ≤100 words why these patents are relevant.\n\n"
              f"Query: {query}\n\n{bullets}")
    resp = litellm.completion(
        model="gpt-4.1-nano",
        api_key=API_KEY,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=160,
        temperature=0.3
    )
    return resp.choices[0].message.content.strip()

# ────────────────────────────────────────────────────────
# 7. Unified search endpoint
# ────────────────────────────────────────────────────────
def run_search(text_query: str, image_input: Image.Image) -> str:
    if not text_query.strip() and image_input is None:
        return "⚠️ Please enter a query or upload an image."
    if image_input and not text_query.strip():
        hits  = search_image(image_input)
        query = "<image query>"
    else:
        hits  = search_text(text_query)
        query = text_query.strip() or "<empty>"
    if not hits:
        return "No results found."
    table   = build_table(hits)
    summary = llm_summary(query, hits)
    return table + "\n\n**LLM Summary**\n" + summary

# ────────────────────────────────────────────────────────
# 8. Patent HTML preview helper
# ────────────────────────────────────────────────────────
def load_patent_html(pat_num: str):
    rec = next((r for r in TEXT_META if r.get("patent_number") == pat_num), None)
    if not rec:
        return gr.update(value=f"<p style='color:red'>Patent {pat_num} not found.</p>", visible=True)
    gaz = rec.get("gazette"); html_f = rec.get("html_file")
    if not gaz or not html_f:
        return gr.update(value=f"<p style='color:red'>No HTML for {pat_num}.</p>", visible=True)
    path = RAW_ROOT / gaz / "OG" / "html" / html_f
    if not path.exists():
        return gr.update(value=f"<p style='color:red'>Missing: {path}</p>", visible=True)
    html_str = path.read_text(errors="ignore")
    def _inline(m):
        src = m.group(1); gif_path = path.parent / src
        if gif_path.exists():
            data = base64.b64encode(gif_path.read_bytes()).decode()
            return f'src="data:image/gif;base64,{data}"'
        return f'src="{src}"'
    html_str = re.sub(r'src="([^"]+\.gif)"', _inline, html_str, flags=re.I)
    return gr.update(value=html_str, visible=True)

# ────────────────────────────────────────────────────────
# 9. Gradio UI
# ────────────────────────────────────────────────────────
with gr.Blocks(title="Patent RAG Demo") as demo:
    gr.Markdown("## 🔍 USPTO Patent Retrieval‑Augmented Demo")
    with gr.Row():
        txt = gr.Textbox(label="Text Query", placeholder="e.g. semiconductor packaging")
        img = gr.Image(label="Or upload a drawing (leave text blank for image search)",
                       type="pil")
    btn = gr.Button("Search")
    out = gr.Markdown()

    with gr.Row():
        pat_in   = gr.Textbox(label="Patent # to preview (copy from table)", lines=1)
        prev_btn = gr.Button("Preview")
    preview = gr.HTML(label="Patent front page", visible=False)

    btn.click(run_search, inputs=[txt, img], outputs=out)
    txt.submit(run_search, inputs=[txt, img], outputs=out)

    prev_btn.click(load_patent_html, inputs=pat_in, outputs=preview)
    pat_in.submit(load_patent_html, inputs=pat_in, outputs=preview)

# synchronous launch (no extra queue args)
if __name__ == "__main__":
    demo.launch()