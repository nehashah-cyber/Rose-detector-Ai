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
def load_model():
    # Reliable Flower & Plant classifier model
    return pipeline("image-classification", model="microsoft/resnet-50")

classifier = load_model()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with st.spinner("Analyzing image..."):
        # Fetch Top 5 predictions
        results = classifier(image, top_k=5)
        
        is_rose = False
        confidence = 0.0
        detected_label = ""
        
        # Broad keywords list for Rose detection
        rose_keywords = ['rose', 'roses', 'rosehip', 'flower', 'pot', 'flowerpot', 'vase', 'petal']
        
        for result in results:
            label = result['label'].lower()
            if any(keyword in label for keyword in ['rose', 'rosehip']):
                is_rose = True
                confidence = result['score'] * 100
                detected_label = result['label']
                break
        
        st.divider()
        if is_rose:
            st.success(f"✅ This is a **Rose**! (Detected: {detected_label}, Confidence: {confidence:.2f}%)")
        else:
            top_label = results[0]['label']
            top_score = results[0]['score'] * 100
            st.error(f"❌ This is not a Rose. (AI Detection: {top_label} - {top_score:.2f}%)")
