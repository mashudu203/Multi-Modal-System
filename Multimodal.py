import streamlit as V
import requests

# Set page configuration
V.set_page_config(page_title="Multi-Modal Image Retrieval", layout="wide")

# Read the CSS file
with open("style.css") as f:
    V.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Read the HTML file
with open("index.html") as f:
    html_content = f.read()

# Back-end API URL
API_URL = "http://127.0.0.1:8000"

# Add an icon
V.image("icon.png", width=80)

# Display the HTML content
V.markdown(html_content, unsafe_allow_html=True)

V.write("Enter a text description of an image, and we’ll find similar images")

query = V.text_input("Enter your search query:")

if V.button("Search"):
    if query:
        # Call the FastAPI backend (Fixed API URL)
        response = requests.get(f"{API_URL}/search", params={"query": query, "top_k": 5})

        if response.status_code == 200:
            results = response.json().get("results", [])

            if results:
                V.subheader("🔹 Results:")
                cols = V.columns(5)  # Display images in 5 columns

                for i, img_path in enumerate(results):
                    with cols[i % 5]:  # Arrange images in a row
                        V.image(img_path, caption=f"Result {i+1}")
            else:
                V.warning("⚠️ No matching images found!")
        else:
            V.error("❌ Failed to fetch results. Please check the API.")
    else:
        V.error("❌ Please enter a query.")
