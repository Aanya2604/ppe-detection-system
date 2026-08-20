import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(
    page_title="PPE Safety Compliance System",
    page_icon="🦺",
    layout="wide"
)

st.title("🦺 AI Real-Time PPE Safety Compliance System")
st.write("Upload site imagery to automatically detect and flag safety gear compliance.")

# Load model weights
@st.cache_resource
def load_model():
    return YOLO("best.pt")

try:
    model = load_model()
except Exception as e:
    st.error("Model weights file 'best.pt' not found in root folder.")
    st.stop()

# Sidebar controls
st.sidebar.header("Model Settings")
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.40, 0.05)

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

col1, col2 = st.columns(2)

if uploaded_file is not None:
    input_img = Image.open(uploaded_file)
    with col1:
        st.subheader("Original Input Image")
        st.image(input_img, use_container_width=True)

    with col2:
        st.subheader("Safety Detection Output")
        with st.spinner("Analyzing safety compliance..."):
            results = model.predict(source=input_img, conf=confidence)
            annotated_frame = results[0].plot()
            st.image(annotated_frame, use_container_width=True)

            # Extract detected classes
            boxes = results[0].boxes
            detected_classes = [model.names[int(cls)] for cls in boxes.cls]

            st.markdown("---")
            st.write(f"**Total Objects Found:** {len(detected_classes)}")
            st.write(f"**Detected Classes:** {', '.join(set(detected_classes)) if detected_classes else 'None'}")