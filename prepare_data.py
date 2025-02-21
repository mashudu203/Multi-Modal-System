import os
import json
import faiss
import torch
import clip
import numpy as np
from PIL import Image
from tqdm import tqdm

# Paths (User must replace with actual dataset path after downloading)
DATASET_PATH = "path/to/test2_data_v2"  # Update this!
IMAGE_PATHS_FILE = "image_paths.json"
FAISS_INDEX_FILE = "faiss_index.bin"

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Get all image file paths
image_files = [f for f in os.listdir(DATASET_PATH) if f.endswith((".jpg", ".png", ".jpeg"))]
image_paths = [os.path.join(DATASET_PATH, img) for img in image_files]

# Save image paths to JSON
with open(IMAGE_PATHS_FILE, "w") as f:
    json.dump(image_paths, f)

print(f"{IMAGE_PATHS_FILE} created with {len(image_paths)} images.")

# Compute image embeddings
image_embeddings = []
for img_path in tqdm(image_paths, desc="Processing images"):
    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(image).cpu().numpy()
    image_embeddings.append(embedding)

# Convert to NumPy array and normalize
image_embeddings = np.vstack(image_embeddings)
image_embeddings /= np.linalg.norm(image_embeddings, axis=1, keepdims=True)

# Build FAISS index
index = faiss.IndexFlatL2(image_embeddings.shape[1])
index.add(image_embeddings)
faiss.write_index(index, FAISS_INDEX_FILE)

print(f"{FAISS_INDEX_FILE} created with {len(image_embeddings)} embeddings.")
