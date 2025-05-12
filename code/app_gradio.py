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
def build_table(hits: list[dict], summaries: list[str]) -> str:
    """Return an HTML table with clickable patent numbers and summary descriptions."""
    rows = [
        "<table>",
        "<thead><tr><th>Patent #</th><th>Title</th><th>Description</th><th>Image</th></tr></thead>",
        "<tbody>",
    ]
    for i, h in enumerate(hits):
        pn    = h["patent_number"]
        title = h.get("title", "")
        desc  = summaries[i].replace("|", "\\|")

        # clickable link fills input, triggers preview, and scrolls
        link = (
            "<a href='#' "
            "onclick=\""
            "const box=document.getElementById('pnInput').querySelector('input,textarea');"
            f"box.value='{pn}';"
            "box.dispatchEvent(new Event('input',{bubbles:true}));"
            "document.getElementById('previewBtn').click();"
            "setTimeout(()=>window.scrollTo({top:document.body.scrollHeight,behavior:'smooth'}),200);"
            "return false;\">"
            f"{pn}</a>"
        )

        # inline first GIF if available
        img_td = ""
        m = next((x for x in IMG_META if x["patent_number"] == pn), None)
        if m:
            gif = RAW_ROOT / m["gazette"] / "OG" / "html" / m["image_path"]
            if gif.exists():
                b64 = base64.b64encode(gif.read_bytes()).decode()
                img_td = f"<img src='data:image/gif;base64,{b64}' height='80'/>"

        rows.append(f"<tr><td>{link}</td><td>{title}</td><td>{desc}</td><td>{img_td}</td></tr>")
    rows.append("</tbody></table>")
    return "\n".join(rows)

# ────────────────────────────────────────────────────────
# 5b. Short description summaries
# ────────────────────────────────────────────────────────
def get_description_summaries(hits: list[dict]) -> list[str]:
    """Return a 1‑2 sentence LLM summary for each patent description."""
    summaries: list[str] = []
    for h in hits:
        desc = (h.get("description") or "").strip()
        if not desc:
            summaries.append("(No description available)")
            continue
        prompt = (
            "Provide a brief, technical 2-sentence summary of the following patent description:\n\n"
            f"{desc}"
        )
        try:
            resp = litellm.completion(
                model="gpt-4.1-nano",
                api_key=API_KEY,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=60,
                temperature=0.0,
            )
            summaries.append(resp.choices[0].message.content.strip())
        except Exception as e:
            print("[WARN] summary failed:", e, flush=True)
            summaries.append(desc[:160] + ("…" if len(desc) > 160 else ""))
    return summaries

# ────────────────────────────────────────────────────────
# 6. LLM summary
# ────────────────────────────────────────────────────────
def llm_summary(query: str, hits: list[dict]) -> str:
    bullets = "\n".join(
        f"- **{h['title']}** (#{h['patent_number']}): {h['description'][:120]}…"
        for h in hits
    )
    prompt = ("Provide a technical summary (about 100 words) of these patents based on their descriptions:\n\n"
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
        # If text query provided, ask LLM to rewrite it for better retrieval
        if text_query.strip():
            rewrite_prompt = (
                "Rewrite the following search query to be concise and technical for patent retrieval:\n"
                f"```\n{text_query.strip()}\n```"
            )
            rewrite_resp = litellm.completion(
                model="gpt-4.1-nano",
                api_key=API_KEY,
                messages=[{"role":"user","content":rewrite_prompt}],
                max_tokens=32,
                temperature=0.0
            )
            embed_query = rewrite_resp.choices[0].message.content.strip()
        else:
            embed_query = ""
        hits  = search_text(embed_query)
        query = text_query.strip() or "<empty>"
    if not hits:
        return "No results found."
    summaries = get_description_summaries(hits)
    table = build_table(hits, summaries)
    summary = llm_summary(query, hits)
    n = len(hits)
    if image_input and not text_query.strip():
        intro = f"Here are the top {n} results related to your image:\n\n"
    else:
        intro = f"Here are the top {n} results related to your search **{query}**:\n\n"
    return intro + table + "\n\n**Summary of Results**\n" + summary

# ────────────────────────────────────────────────────────
# 8. Patent HTML preview helper
# ────────────────────────────────────────────────────────
def load_patent_html(pat_num: str):
    print ('load_patet_html called for', {pat_num})
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
    # LLM call for a detailed summary of the front page
    text_only = re.sub(r'<[^>]+>', '', html_str).strip()
    summary_prompt = (
        "Provide a detailed, technical 2-3 sentence summary of this patent front page content:\n\n"
        + text_only
    )
    summary_resp = litellm.completion(
        model="gpt-4.1-nano",
        api_key=API_KEY,
        messages=[{"role":"user","content":summary_prompt}],
        max_tokens=100,
        temperature=0.0
    )
    html_summary = summary_resp.choices[0].message.content.strip()
    full_content = html_str + f"<hr><h3>Front Page Summary</h3><p>{html_summary}</p>"
    return gr.update(value=full_content, visible=True)

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
    out = gr.HTML()

    with gr.Row():
        pat_in   = gr.Textbox(label="Patent # to preview (copy from table)", lines=1, elem_id="pnInput")
        prev_btn = gr.Button("Preview", elem_id="previewBtn")
    preview = gr.HTML(label="Patent front page", visible=False)

    btn.click(run_search, inputs=[txt, img], outputs=out)
    txt.submit(run_search, inputs=[txt, img], outputs=out)

    prev_btn.click(load_patent_html, inputs=pat_in, outputs=preview)
    pat_in.submit(load_patent_html, inputs=pat_in, outputs=preview)

# synchronous launch (no extra queue args)
if __name__ == "__main__":
    demo.launch()