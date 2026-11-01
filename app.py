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
    # Verified Image Classification Model
    return pipeline("image-classification", model="google/vit-base-patch16-224")

classifier = load_model()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with st.spinner("Analyzing image..."):
        # Top 5 predictions fetch kar rahe hain
        results = classifier(image, top_k=5)
        
        is_rose = False
        confidence = 0.0
        rose_label_found = ""
        
        # Rose related labels checklist
        rose_keywords = ['rose', 'roses', 'rosehip', 'flower', 'flowering']
        
        for result in results:
            label = result['label'].lower()
            if 'rose' in label:
                is_rose = True
                confidence = result['score'] * 100
                rose_label_found = result['label']
                break
        
        st.divider()
        if is_rose:
            st.success(f"✅ This is a **Rose**! (Detected: {rose_label_found}, Confidence: {confidence:.2f}%)")
        else:
            top_label = results[0]['label']
            top_score = results[0]['score'] * 100
            st.error(f"❌ This is not a Rose. (AI Detection: {top_label} - {top_score:.2f}%)")
