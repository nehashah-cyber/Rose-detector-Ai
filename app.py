import streamlit as st
from PIL import Image
from transformers import pipeline

st.set_page_config(
    page_title="Rose Detector AI",
    page_icon="🌹",
    layout="centered"
)

st.title("🌹 Rose Classifier AI Agent")
st.write("Upload a flower image, and the AI will detect whether it is a Rose or not.")

@st.cache_resource
def load_classifier():
    # Fast & accurate zero-shot vision model
    return pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")

try:
    classifier = load_classifier()
except Exception as e:
    st.error("Model loading issue. Please reboot the app from Streamlit Cloud dashboard.")
    st.stop()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with st.spinner("Analyzing flower features..."):
        labels = [
            "a rose flower", 
            "a lily flower", 
            "a tulip flower", 
            "a flower that is not a rose",
            "a non-flower object"
        ]
        
        predictions = classifier(image, candidate_labels=labels)
        
        top_label = predictions[0]['label']
        confidence = predictions[0]['score'] * 100
        
        st.divider()
        if top_label == "a rose flower":
            st.success(f"✅ This is a **Rose**! (Confidence: {confidence:.2f}%)")
        else:
            clean_label = top_label.replace("a ", "").replace(" flower", "").capitalize()
            st.error(f"❌ This is not a Rose. (AI Detection: {clean_label} - {confidence:.2f}%)")
