from fastapi import FastAPI, Query
import faiss
import torch
import clip
from PIL import Image
import numpy as np
import json
import os

app = FastAPI()

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Paths
index_path = "C:/Users/mash/Desktop/kaggle/faiss_index.bin"
image_paths_file = "C:/Users/mash/Desktop/kaggle/image_paths.json"  # Change this!

# Load FAISS index
if not os.path.exists(index_path):
    raise FileNotFoundError(f"FAISS index file not found at {index_path}")

index = faiss.read_index(index_path)

# Load image paths from JSON
if not os.path.exists(image_paths_file):
    raise FileNotFoundError(f"Image paths file not found at {image_paths_file}")

with open(image_paths_file, "r") as f:
    image_paths = json.load(f)

@app.get("/")
def read_root():
    return {"message": "FastAPI Multi-Modal Retrieval is running!"}

@app.get("/search")
def search_images(query: str, top_k: int = Query(10, gt=0)):
    # Convert query text to CLIP embedding
    with torch.no_grad():
        text_tokens = clip.tokenize([query]).to(device)
        text_embedding = model.encode_text(text_tokens).cpu().numpy()

    # Normalize text embedding
    text_embedding /= np.linalg.norm(text_embedding, axis=1, keepdims=True)
    text_embedding = np.ascontiguousarray(text_embedding.reshape(1, -1))

    # Search FAISS index
    distances, indices = index.search(text_embedding, top_k)

 

    # Retrieve matching images
    results = [image_paths[i] for i in indices[0] if i < len(image_paths)]

    if not results:
        print("⚠️ No matching images found in FastAPI!")
    else:
        print(f"✅ Retrieved images in FastAPI: {results}")

    return {"query": query, "results": results}
