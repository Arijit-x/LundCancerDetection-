import streamlit as st
import cv2
import numpy as np

def analyze_image(image):
    """Show actual feature values"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    edges = cv2.Canny(gray, 30, 100)
    edge_density = np.sum(edges > 0) / edges.size
    
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    texture_variance = np.var(laplacian)
    
    return edge_density, texture_variance

st.title("🫁 Feature Value Analyzer")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    st.image(image, caption="Image", channels="BGR")
    
    img = cv2.resize(image, (224, 224))
    edge_density, texture_variance = analyze_image(img)
    
    st.write(f"**Edge Density:** {edge_density:.6f}")
    st.write(f"**Texture Variance:** {texture_variance:.2f}")
    
    # Show ranges
    st.divider()
    st.write("**Your data ranges (from earlier analysis):**")
    st.write("Cancer: edge ~0.068, texture ~173")
    st.write("Normal: edge ~0.190, texture ~4388")
    st.write("")
    st.write(f"This image edge is closer to: {'CANCER' if edge_density < 0.13 else 'NORMAL'}")
    st.write(f"This image texture is closer to: {'CANCER' if texture_variance < 2000 else 'NORMAL'}")
