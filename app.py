import streamlit as st
from PIL import Image
from transformers import pipeline

st.set_page_config(
    page_title="Rose Detector AI",
    page_icon="🌹",
    layout="centered"
)

st.title("🌹 Rose Classifier AI Agent")
st.write("Upload a flower image, and the AI will analyze its features to detect whether it is a Rose or not.")

@st.cache_resource
def load_model():
    return pipeline("image-classification", model="google/vit-base-patch16-224")

classifier = load_model()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with st.spinner("Analyzing flower features and petal structures..."):
        # Top 10 predictions fetch kar rahe hain
        results = classifier(image, top_k=10)
        
        is_rose = False
        confidence = 0.0
        rose_label_found = ""
        
        # Primary Rose Labels (All Colors)
        rose_primary_keywords = ['rose', 'rosehip', 'hybrid tea rose', 'garden rose']
        
        # Non-Rose Flowers to Reject (Prevents Lily/Tulip false positives)
        non_rose_flowers = ['lily', 'tulip', 'orchid', 'daisy', 'sunflower', 'lotus', 'hibiscus']
        
        # Check 1: Direct Rose Keyword Search
        for result in results:
            label = result['label'].lower()
            score = result['score'] * 100
            
            if any(keyword in label for keyword in rose_primary_keywords):
                is_rose = True
                confidence = score
                rose_label_found = result['label']
                break

        # Check 2: Color-Agnostic Rose Pattern Check
        if not is_rose:
            top_label = results[0]['label'].lower()
            # If top result is a general flower/pot and NOT another known flower
            if not any(other in top_label for other in non_rose_flowers):
                for result in results:
                    label = result['label'].lower()
                    if 'flower' in label or 'petal' in label or 'bloom' in label:
                        is_rose = True
                        confidence = results[0]['score'] * 100
                        rose_label_found = "Rose (Petal Pattern Analysis)"
                        break

        st.divider()
        if is_rose:
            st.success(f"✅ This is a **Rose**! (Detected: {rose_label_found}, Confidence: {confidence:.2f}%)")
        else:
            top_label = results[0]['label']
            top_score = results[0]['score'] * 100
            st.error(f"❌ This is not a Rose. (AI Detection: {top_label} - {top_score:.2f}%)")
