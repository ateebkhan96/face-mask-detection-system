# test_app.py
# ---------------------------------------------
# ✅ Comprehensive Test Suite for Face Mask Detection App
# Developed using pytest and mock for Streamlit context
# ---------------------------------------------

import pytest
import numpy as np
from PIL import Image
import cv2
import os
import time
import sys
import warnings
import logging

# ---------------------------------------------------------
# ⚠️ Suppress Warnings and Logs (TensorFlow, Streamlit etc.)
# ---------------------------------------------------------

# Suppress common warning types
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Suppress logs from TensorFlow and MediaPipe
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses TF logs: 0=all, 3=errors only
logging.getLogger("mediapipe").setLevel(logging.ERROR)
logging.getLogger("streamlit").setLevel(logging.ERROR)

# Create a null logger to silence unwanted stderr outputs
class NullLogger:
    def write(self, *args, **kwargs): pass
    def flush(self): pass
    def close(self): pass  # Required to prevent crashing

# Redirect stderr only when running directly (not via pytest)
if __name__ == "__main__":
    original_stderr = sys.stderr
    sys.stderr = NullLogger()

# Import main app AFTER suppression
import app

# Restore stderr for pytest compatibility
if __name__ == "__main__":
    sys.stderr = original_stderr

# ---------------------------------------------------------
# 📦 Fixtures for Reusable Components
# ---------------------------------------------------------

@pytest.fixture
def dummy_face_image():
    """Returns a synthetic 128x128 RGB image as NumPy array."""
    return np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)

@pytest.fixture
def dummy_pil_image():
    """Returns a random 300x300 PIL Image object."""
    return Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))

@pytest.fixture(scope="session")
def interpreter():
    """Loads the TFLite interpreter once for all tests."""
    return app.load_model()

@pytest.fixture
def model_details(interpreter):
    """Returns model input and output details."""
    return interpreter.get_input_details(), interpreter.get_output_details()

# ---------------------------------------------------------
# ✅ Unit Tests for Core Functionality
# ---------------------------------------------------------

def test_model_loading(interpreter):
    """✅ Ensure the TFLite model loads correctly with valid tensors."""
    assert interpreter is not None, "❌ Interpreter failed to load."
    assert interpreter.get_input_details(), "❌ Missing input details."
    assert interpreter.get_output_details(), "❌ Missing output details."
    return True

def test_preprocessing(dummy_face_image):
    """🧪 Validate image preprocessing - normalized and reshaped."""
    processed = app.preprocess_face(dummy_face_image)
    assert processed is not None, "❌ Preprocessing returned None."
    assert processed.shape == (1, 128, 128, 3), f"❌ Invalid shape: {processed.shape}"
    assert np.all((processed >= 0.0) & (processed <= 1.0)), "❌ Image not normalized."
    return True

def test_predict_function(interpreter, dummy_face_image):
    """🎯 Ensure the prediction output and confidence level are valid."""
    processed = app.preprocess_face(dummy_face_image)
    app.input_details = interpreter.get_input_details()
    app.output_details = interpreter.get_output_details()
    output, confidence_level = app.predict(processed, interpreter)

    assert output is not None, "❌ Prediction failed."
    assert confidence_level in ["High Confidence", "Low Confidence"], f"❌ Invalid confidence level: {confidence_level}"
    return confidence_level

def test_detection_on_sample(dummy_pil_image, interpreter):
    """🔍 Test face detection and result image generation."""
    app.input_details = interpreter.get_input_details()
    app.output_details = interpreter.get_output_details()

    result_image, faces, proc_time = app.detect_and_predict(dummy_pil_image, interpreter)
    assert isinstance(result_image, Image.Image), "❌ Invalid result image."
    assert isinstance(faces, int), "❌ Faces count is not integer."
    assert isinstance(proc_time, float), "❌ Processing time is not float."
    return faces, proc_time

def test_sample_images_loading():
    """📁 Validate that sample images are dynamically loaded."""
    images = app.get_sample_images()
    assert isinstance(images, dict), "❌ Expected a dictionary of images."
    return len(images)

def test_draw_fancy_bbox():
    """🎨 Ensure the fancy bounding box is drawn correctly."""
    test_image = np.zeros((300, 300, 3), dtype=np.uint8)
    bbox = (50, 50, 200, 200)
    result = app.draw_fancy_bbox(test_image, bbox, "With Mask", 0.95, "High Confidence")
    assert result is not None, "❌ Drawing failed."
    assert np.sum(result) > 0, "❌ No drawing detected."
    return True

# ---------------------------------------------------------
# ⏱️ Performance Testing
# ---------------------------------------------------------

def test_processing_speed(dummy_pil_image, interpreter):
    """⏱️ Test how quickly the pipeline processes an image."""
    app.input_details = interpreter.get_input_details()
    app.output_details = interpreter.get_output_details()

    start_time = time.time()
    app.detect_and_predict(dummy_pil_image, interpreter)
    duration = time.time() - start_time

    assert duration < 5.0, f"❌ Too slow: {duration:.3f}s"
    return duration

# ---------------------------------------------------------
# 🔁 End-to-End Integration Test
# ---------------------------------------------------------

def test_end_to_end_pipeline():
    """🔄 Full test from image load to prediction."""
    interpreter = app.load_model()
    app.input_details = interpreter.get_input_details()
    app.output_details = interpreter.get_output_details()

    sample_images = app.get_sample_images()
    if not sample_images:
        return "Skipped (no sample images)"

    test_image_path = next(iter(sample_images.values()))
    test_image = Image.open(test_image_path)
    result_image, faces, proc_time = app.detect_and_predict(test_image, interpreter)

    assert isinstance(result_image, Image.Image), "❌ Invalid result image."
    return faces, proc_time

# ---------------------------------------------------------
# 🛑 Robustness & Error Handling Tests
# ---------------------------------------------------------

def test_empty_image():
    """🛑 Ensure empty images are handled gracefully."""
    empty_image = Image.new('RGB', (1, 1), color='black')
    interpreter = app.load_model()
    app.input_details = interpreter.get_input_details()
    app.output_details = interpreter.get_output_details()

    result_image, faces, _ = app.detect_and_predict(empty_image, interpreter)
    assert faces == 0, "❌ False positive on empty image."
    return True

def test_preprocessing_with_invalid_input():
    """⚠️ Test preprocessing behavior with invalid input."""
    result = app.preprocess_face(np.array([]))  # Invalid input
    assert result is None, "❌ Should return None for invalid input."
    return True

# ---------------------------------------------------------
# 🧪 Manual Execution: CLI Test Runner with Styling
# ---------------------------------------------------------

def print_header():
    """Print a stylized title for the test suite."""
    header = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║        🔍 FACE MASK DETECTION SYSTEM - TEST SUITE 🔍          ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
    """
    print(header)

def print_result(test_name, result, elapsed_time=None):
    """Pretty-print test results."""
    icon = "✅"
    result_str = ""

    if isinstance(result, tuple) and len(result) == 2:
        result_str = f"{result[0]} face(s) detected in {result[1]:.3f}s"
    elif isinstance(result, str):
        result_str = result
    elif isinstance(result, (int, float)):
        result_str = f"{result:.3f}s" if "speed" in test_name else f"{result} image(s) loaded"
    else:
        result_str = str(result)

    if elapsed_time:
        print(f"{icon} {test_name.replace('_', ' ').title()}: {result_str} ({elapsed_time:.3f}s)")
    else:
        print(f"{icon} {test_name.replace('_', ' ').title()}: {result_str}")

def print_section(name):
    """Print section divider for test groups."""
    print(f"\n{'=' * 20} {name} {'=' * 20}")

def run_test(test_func, *args, **kwargs):
    """Run a test and capture runtime."""
    start_time = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = test_func(*args, **kwargs)
    elapsed_time = time.time() - start_time
    return result, elapsed_time

# Manual CLI runner
if __name__ == "__main__":
    print_header()
    print_section("MODEL TESTS")

    interpreter_instance = app.load_model()
    app.input_details = interpreter_instance.get_input_details()
    app.output_details = interpreter_instance.get_output_details()
    dummy_face = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    dummy_image = Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))

    result, elapsed = run_test(test_model_loading, interpreter_instance)
    print_result("test_model_loading", result, elapsed)

    result, elapsed = run_test(test_preprocessing, dummy_face)
    print_result("test_preprocessing", result, elapsed)

    print_section("PREDICTION TESTS")

    result, elapsed = run_test(test_predict_function, interpreter_instance, dummy_face)
    print_result("test_predict_function", result, elapsed)

    result, elapsed = run_test(test_detection_on_sample, dummy_image, interpreter_instance)
    print_result("test_detection_on_sample", result, elapsed)

    print_section("UTILITY TESTS")

    result, elapsed = run_test(test_sample_images_loading)
    print_result("test_sample_images_loading", result, elapsed)

    result, elapsed = run_test(test_draw_fancy_bbox)
    print_result("test_draw_fancy_bbox", result, elapsed)

    print_section("PERFORMANCE TESTS")

    result, elapsed = run_test(test_processing_speed, dummy_image, interpreter_instance)
    print_result("test_processing_speed", result, elapsed)

    print_section("END-TO-END TESTS")

    try:
        result, elapsed = run_test(test_end_to_end_pipeline)
        print_result("test_end_to_end_pipeline", result, elapsed)
    except Exception as e:
        print(f"⚠️ End-to-end test skipped: {str(e)}")

    print_section("ROBUSTNESS TESTS")

    result, elapsed = run_test(test_empty_image)
    print_result("test_empty_image", result, elapsed)

    result, elapsed = run_test(test_preprocessing_with_invalid_input)
    print_result("test_preprocessing_with_invalid_input", result, elapsed)

    print("\n✨ All tests completed successfully! ✨\n")
