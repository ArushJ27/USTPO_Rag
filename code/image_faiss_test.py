import faiss, numpy as np
# Load metadata length
import pickle
meta = pickle.load(open("embeddings/image_metadata_2025.pkl","rb"))
print("Metadata entries:", len(meta))

# Load a small slice of embeddings
embs = np.load("embeddings/image_embeddings_2025.npy").astype("float32")
idx = faiss.read_index("embeddings/faiss_index_image_2025.bin")
print("Index ntotal:", idx.ntotal, "Emb array length:", embs.shape[0])

# Try a simple self-search (should retrieve itself at distance 0)
D, I = idx.search(embs[:1], k=1)
print("Nearest neighbor distance:", D[0][0], "Index returned:", I[0][0])