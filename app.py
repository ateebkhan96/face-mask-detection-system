# -------------------------------
# 😷 Face Mask Detection System
# Developed with Streamlit, TFLite, OpenCV, and MediaPipe
# -------------------------------

import streamlit as st
import numpy as np
import tensorflow.lite as tflite
import mediapipe as mp
import cv2
from PIL import Image, ImageOps
import time
from datetime import datetime
import os

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Face Mask Detection System",
    page_icon="😷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Custom CSS Styling
# -------------------------------
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        font-weight: bold;
    }
    .success-text { color: #28a745; font-weight: bold; }
    .warning-text { color: #000000; font-weight: bold; }
    .status-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .info-text {
        font-size: 0.9em;
        color: #6c757d;
        font-style: italic;
        margin-top: -15px;
        padding-bottom: 10px;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        color: #212529;
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# Initialize Session State Variables
# -------------------------------
if 'detection_history' not in st.session_state:
    st.session_state['detection_history'] = []
if 'processing_time' not in st.session_state:
    st.session_state['processing_time'] = []
if 'camera_on' not in st.session_state:
    st.session_state['camera_on'] = False

# -------------------------------
# Load MediaPipe Face Detection
# -------------------------------
mp_face_detection = mp.solutions.face_detection

# -------------------------------
# Load TensorFlow Lite Model
# -------------------------------
@st.cache_resource
def load_model():
    try:
        interpreter = tflite.Interpreter(model_path="model/face_mask_model.tflite")
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# -------------------------------
# Draw Bounding Box with Styling
# -------------------------------
def draw_fancy_bbox(image, bbox, label, confidence, confidence_level):
    """
    Draw a modern-looking bounding box with label and confidence.
    """
    xmin, ymin, xmax, ymax = bbox
    h, w = image.shape[:2]

    # Dynamically adjust visuals based on image size
    thickness = max(1, int(min(h, w) * 0.003))
    corner_radius = max(3, int(min(h, w) * 0.02))
    font_scale = min(0.7, max(0.4, min(h, w) * 0.001))
    padding = max(2, int(min(h, w) * 0.01))

    # Color logic based on mask status and confidence
    if "With Mask" in label:
        color = (0, 200, 0) if confidence > 0.9 else (0, 170, 0)
    else:
        color = (200, 0, 0) if confidence > 0.9 else (170, 0, 0)

    # Helper to draw rounded corners
    def draw_corner(x1, y1, x2, y2, corner_type):
        corners = {
            'top_left': ((x1 + corner_radius, y1 + corner_radius), 180),
            'top_right': ((x2 - corner_radius, y1 + corner_radius), 270),
            'bottom_left': ((x1 + corner_radius, y2 - corner_radius), 90),
            'bottom_right': ((x2 - corner_radius, y2 - corner_radius), 0)
        }
        center, angle = corners[corner_type]
        cv2.ellipse(image, center, (corner_radius, corner_radius), angle, 0, 90, color, thickness)

    # Draw box lines and corners
    cv2.line(image, (xmin + corner_radius, ymin), (xmax - corner_radius, ymin), color, thickness)
    cv2.line(image, (xmin + corner_radius, ymax), (xmax - corner_radius, ymax), color, thickness)
    cv2.line(image, (xmin, ymin + corner_radius), (xmin, ymax - corner_radius), color, thickness)
    cv2.line(image, (xmax, ymin + corner_radius), (xmax, ymax - corner_radius), color, thickness)
    for corner in ['top_left', 'top_right', 'bottom_left', 'bottom_right']:
        draw_corner(xmin, ymin, xmax, ymax, corner)

    # Add label
    label_text = f"{label} ({confidence:.2f})"
    if confidence_level == "Low Confidence":
        label_text += " ⚠️"
    (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

    # Draw gradient label background
    gradient_height = text_h + 2 * padding
    for i in range(gradient_height):
        alpha = 1 - (i / gradient_height) * 0.3
        current_color = tuple(int(c * alpha) for c in color)
        cv2.line(image, (xmin, ymin - gradient_height + i), (xmin + text_w + 2 * padding, ymin - gradient_height + i), current_color, 1)

    # Draw text
    cv2.putText(image, label_text, (xmin + padding, ymin - padding),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return image

# -------------------------------
# Preprocess Face Image
# -------------------------------
def preprocess_face(face):
    """
    Resize, normalize and prepare face image for model input.
    """
    try:
        if face is None or face.size == 0:
            raise ValueError("Invalid face image")
        face = cv2.resize(face, (128, 128))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = face.astype(np.float32) / 255.0
        return np.expand_dims(face, axis=0)
    except Exception as e:
        st.warning(f"Error in preprocessing: {str(e)}")
        return None

# -------------------------------
# Predict Using TFLite Model
# -------------------------------
def predict(face, interpreter, confidence_threshold=0.7):
    """
    Perform prediction using the loaded TFLite model.
    """
    try:
        interpreter.set_tensor(input_details[0]['index'], face)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        confidence = np.max(output_data)
        return output_data, "High Confidence" if confidence >= confidence_threshold else "Low Confidence"
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None

# -------------------------------
# Face Detection and Prediction
# -------------------------------
def detect_and_predict(image, interpreter):
    """
    Detect faces using MediaPipe and predict mask status.
    """
    start_time = time.time()
    image_np = np.array(image)
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    faces_detected = 0

    with mp_face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=1) as face_detection:
        results = face_detection.process(image_rgb)

        if results.detections:
            for detection in results.detections:
                faces_detected += 1
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = image_np.shape
                xmin, ymin = int(bbox.xmin * w), int(bbox.ymin * h)
                xmax, ymax = xmin + int(bbox.width * w), ymin + int(bbox.height * h)

                xmin, ymin = max(0, xmin), max(0, ymin)
                xmax, ymax = min(w, xmax), min(h, ymax)
                face_crop = image_np[ymin:ymax, xmin:xmax]

                if face_crop.size > 0:
                    processed_face = preprocess_face(face_crop)
                    if processed_face is not None:
                        prediction, confidence_level = predict(processed_face, interpreter)
                        if prediction is not None:
                            class_labels = ["With Mask", "Without Mask"]
                            predicted_label = class_labels[np.argmax(prediction)]
                            confidence = np.max(prediction)
                            image_np = draw_fancy_bbox(image_np, (xmin, ymin, xmax, ymax), predicted_label, confidence, confidence_level)

                            st.session_state['detection_history'].append({
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'prediction': predicted_label,
                                'confidence': confidence,
                                'confidence_level': confidence_level
                            })

    processing_time = time.time() - start_time
    st.session_state['processing_time'].append(processing_time)
    return Image.fromarray(image_np), faces_detected, processing_time

# -------------------------------
# Process Uploaded Image
# -------------------------------
def process_image(image):
    """Process a single image and display original vs detected results"""
    try:
        image = image.convert("RGB")
        image = ImageOps.exif_transpose(image)
        processed_image, faces, proc_time = detect_and_predict(image, interpreter)
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_container_width=True)
        with col2:
            st.image(processed_image, caption=f"Detected Image ({faces} faces, {proc_time:.3f}s)", use_container_width=True)
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")

# -------------------------------
# Load Sample Images Dynamically
# -------------------------------
def get_sample_images():
    """Load available sample images from a folder"""
    sample_dir = "sample_images"
    image_extensions = (".jpg", ".jpeg", ".png")
    samples = {}
    if os.path.exists(sample_dir):
        for filename in os.listdir(sample_dir):
            if filename.lower().endswith(image_extensions):
                label = os.path.splitext(filename)[0].replace("_", " ").title()
                samples[label] = os.path.join(sample_dir, filename)
    return samples

# -------------------------------
# Display Mask Detection Warning
# -------------------------------
def display_detection_warning():
    """Show limitations of the detection system"""
    st.markdown("""
    <div class="warning-box">
        <strong>⚠️ Detection Limitations:</strong><br>
        • Results may be less accurate in low lighting<br>
        • Beards, unusual masks, occlusion can reduce accuracy<br>
        • Ensure good lighting and frontal view for best results
    </div>
    """, unsafe_allow_html=True)

# -------------------------------
# Main UI Logic
# -------------------------------
def main():
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings & Statistics")
        confidence_threshold = st.slider("Confidence Threshold", 0.5, 1.0, 0.7, 0.1)
        st.markdown("""<div class="info-text">
            Adjust how confident the system must be to make a decision.
        </div>""", unsafe_allow_html=True)

        if st.session_state['processing_time']:
            st.subheader("📊 Performance Metrics")
            avg_time = np.mean(st.session_state['processing_time'])
            st.metric("Average Processing Time", f"{avg_time:.3f}s")

        if st.session_state['detection_history']:
            st.subheader("🔍 Detection History")
            detections = st.session_state['detection_history']
            st.write(f"Total Detections: {len(detections)}")
            st.write(f"With Mask: {sum(d['prediction'] == 'With Mask' for d in detections)}")
            st.write(f"Without Mask: {sum(d['prediction'] == 'Without Mask' for d in detections)}")

    # Main Interface
    st.title("😷 Face Mask Detection System")
    st.markdown("Use this tool to check for face masks in images using AI.")

    tab1, tab2, tab3 = st.tabs(["📷 Camera Input", "📁 File Upload", "🖼️ Sample Images"])

    with tab1:
        st.subheader("Camera Input")
        display_detection_warning()
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🎥 Start Camera", key="start_camera", disabled=st.session_state['camera_on']):
                st.session_state['camera_on'] = True
                st.rerun()
        with col2:
            if st.button("⏹️ Stop Camera", key="stop_camera", disabled=not st.session_state['camera_on']):
                st.session_state['camera_on'] = False
                st.rerun()

        if st.session_state['camera_on']:
            camera_image = st.camera_input("Take a picture")
            if camera_image:
                image = Image.open(camera_image)
                process_image(image)
        else:
            st.info("Click 'Start Camera' to begin using the webcam.")

    with tab2:
        st.subheader("Image Upload")
        display_detection_warning()
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            process_image(image)

    with tab3:
        st.subheader("Sample Images")
        st.markdown("Try the model on built-in examples.")
        sample_images = get_sample_images()
        if sample_images:
            selected_sample = st.selectbox("Choose a sample image:", list(sample_images.keys()))
            if st.button("📊 View Detection Results", key="view_sample"):
                image = Image.open(sample_images[selected_sample])
                process_image(image)
        else:
            st.warning("No sample images found in `sample_images` directory.")

# -------------------------------
# App Entry Point
# -------------------------------
interpreter = load_model()
if interpreter:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    main()
else:
    st.error("Failed to load the model. Please verify the model file.")
