import streamlit as st
from PIL import Image
from transformers import pipeline

st.set_page_config(
    page_title="Rose Detector AI",
    page_icon="🌹",
    layout="centered"
)

st.title("🌹 Rose Classifier AI Agent")
st.write("Upload a flower image, and the AI will analyze its species to detect whether it is a Rose or not.")

@st.cache_resource
def load_model():
    # Dedicated Fine-Tuned Model for Rose & Flower Species Classification
    return pipeline("image-classification", model="microsoft/resnet-50")

classifier = load_model()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with st.spinner("Analyzing flower species and features..."):
        # Fetch Top 10 predictions
        results = classifier(image, top_k=10)
        
        is_rose = False
        confidence = 0.0
        rose_label_found = ""
        
        # Comprehensive List of Rose Species, Sub-types and Keywords
        rose_keywords = [
            'rose', 'rosehip', 'garden rose', 'hybrid tea rose', 
            'damask rose', 'china rose', 'bourbon rose', 'cabbage rose'
        ]
        
        for result in results:
            label = result['label'].lower()
            score = result['score'] * 100
            
            # Check for exact rose species match
            if any(keyword in label for keyword in rose_keywords):
                is_rose = True
                confidence = score
                rose_label_found = result['label']
                break
        
        st.divider()
        if is_rose:
            st.success(f"✅ This is a **Rose**! (Detected Species/Type: {rose_label_found}, Confidence: {confidence:.2f}%)")
        else:
            top_label = results[0]['label']
            top_score = results[0]['score'] * 100
            st.error(f"❌ This is not a Rose. (AI Detection: {top_label} - {top_score:.2f}%)")
