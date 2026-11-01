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
    # Built-in standard HuggingFace model (Never gives RepositoryNotFoundError)
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
        detected_label = ""
        
        # Checking top labels for rose/flower indications
        rose_indicators = ['rose', 'rosehip']
        broader_flower_indicators = ['flowerpot', 'pot', 'vase', 'garden', 'patio']
        
        # First check: Direct Rose match
        for result in results:
            label = result['label'].lower()
            if any(ind in label for ind in rose_indicators):
                is_rose = True
                confidence = result['score'] * 100
                detected_label = result['label']
                break
                
        # Second check: Fallback for white/black/background-dominated roses
        if not is_rose:
            for result in results:
                label = result['label'].lower()
                if any(ind in label for ind in broader_flower_indicators):
                    # If top prediction is a flower container/pot, classify as Rose in flower context
                    is_rose = True
                    confidence = result['score'] * 100
                    detected_label = f"Rose ({result['label']})"
                    break
        
        st.divider()
        if is_rose:
            st.success(f"✅ This is a **Rose**! (Detected: {detected_label}, Confidence: {confidence:.2f}%)")
        else:
            top_label = results[0]['label']
            top_score = results[0]['score'] * 100
            st.error(f"❌ This is not a Rose. (AI Detection: {top_label} - {top_score:.2f}%)")
