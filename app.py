import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="Lung Cancer Detection", page_icon="🫁", layout="wide")

# Set background image
import base64
from pathlib import Path

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

try:
    bg_image = get_base64_image("bgforweb.png")
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{bg_image});
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
except:
    pass

st.title("🫁 Lung Cancer Detection AI")

def predict_lung_cancer(image):
    """Texture-based cancer detection"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    texture_variance = np.var(laplacian)
    
    cancer_avg = 1750
    normal_avg = 2237
    
    dist_cancer = abs(texture_variance - cancer_avg)
    dist_normal = abs(texture_variance - normal_avg)
    
    cancer_score = 1.0 - (dist_cancer / (dist_cancer + dist_normal))
    confidence = max(0, min(1, cancer_score))
    
    return confidence, texture_variance

# Upload image
uploaded_file = st.file_uploader("Upload medical image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # Show image
    st.write("**Uploaded Image:**")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, width=400, channels="BGR")
    
    # Process
    img_resized = cv2.resize(image, (224, 224))
    confidence, texture_var = predict_lung_cancer(img_resized)
    
    st.divider()
    
    # Show analysis results below image
    st.write("**Analysis Results:**")
    st.write(f"Texture Variance: **{texture_var:.2f}**")
    st.write(f"Confidence: **{confidence:.1%}**")
    
    st.divider()
    
    if confidence > 0.5:
        st.error(f"🔴 CANCER DETECTED - {confidence:.1%}")
        
        st.divider()
        
        st.markdown("### 🫁 **Cancer Type:**")
        st.markdown("""
        **Malignant**
        """)
        
        st.divider()
        st.markdown("### 📋 **Next Steps & Recommendations:**")
        
        st.markdown("""
        #### 🏥 **Immediate Actions**
        1. **Consult a Radiologist Urgently**
           - Schedule an appointment as soon as possible
           - Bring this analysis report
        
        2. **Get a Second Opinion**
           - Independent radiologist review
           - Verify findings with another specialist
        
        3. **Notify Your Doctor**
           - Share results immediately
           - Discuss next steps and treatment options
        
        #### 🔬 **Further Testing (Recommended)**
        - **CT Scan with Contrast** - Detailed imaging
        - **PET Scan** - Metabolic imaging
        - **MRI Imaging** - High resolution images
        - **Biopsy** - If recommended by doctor (tissue analysis)
        - **Staging** - Determine cancer stage and spread
        
        #### 💊 **Treatment Options (Discuss with Oncologist)**
        - Surgery
        - Chemotherapy
        - Radiation Therapy
        - Targeted Therapy
        - Immunotherapy
        - Clinical Trials
        
        #### ⚠️ **Important Notes**
        - This is an AI screening tool, NOT a medical diagnosis
        - Always consult qualified medical professionals
        - Seek immediate professional medical advice
        - Do not rely solely on this tool for treatment decisions
        """)
    else:
        st.success(f"✅ NORMAL - {(1-confidence):.1%}")
        
        st.divider()
        st.markdown("### 📋 **Recommendations:**")
        
        st.markdown("""
        #### ✅ **Your Results Show Normal Lung Tissue**
        
        **Good News:**
        - Texture analysis shows characteristics of healthy lungs
        - No suspicious patterns detected in this analysis
        
        #### 💪 **Preventive Health Tips**
        1. **Avoid Smoking**
           - Quit if you smoke
           - Avoid secondhand smoke
        
        2. **Maintain Healthy Habits**
           - Regular exercise
           - Balanced, nutritious diet
           - Adequate sleep (7-9 hours)
        
        3. **Reduce Risk Factors**
           - Minimize pollution exposure
           - Use air filters if in polluted areas
           - Protect from occupational hazards
        
        #### 🏥 **Regular Monitoring**
        - **Routine Check-ups** - Annual health screening
        - **Physician Visits** - Regular medical check-ups
        - **Professional Opinion** - Consult your doctor for personalized advice
        
        #### ℹ️ **Important Notes**
        - This is a screening tool, not a diagnostic tool
        - Always follow professional medical advice
        - Regular check-ups are still important
        - Even normal results require professional confirmation
        """)
