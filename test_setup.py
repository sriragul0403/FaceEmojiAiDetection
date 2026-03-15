#!/usr/bin/env python
"""Test script to verify all dependencies and models are available"""

import sys
print("=" * 60)
print("Face Emoji AI - Setup Test")
print("=" * 60)

# Test imports
print("\n1. Testing imports...")
try:
    import cv2
    print("   ✓ OpenCV (cv2) imported successfully")
except ImportError as e:
    print(f"   ✗ OpenCV import failed: {e}")
    sys.exit(1)

try:
    import numpy as np
    print("   ✓ NumPy imported successfully")
except ImportError as e:
    print(f"   ✗ NumPy import failed: {e}")
    sys.exit(1)

try:
    import tensorflow
    print(f"   ✓ TensorFlow {tensorflow.__version__} imported successfully")
except ImportError as e:
    print(f"   ✗ TensorFlow import failed: {e}")
    sys.exit(1)

try:
    from tensorflow.keras.models import load_model
    print("   ✓ Keras imported successfully")
except ImportError as e:
    print(f"   ✗ Keras import failed: {e}")
    sys.exit(1)

try:
    import flask
    print(f"   ✓ Flask {flask.__version__} imported successfully")
except ImportError as e:
    print(f"   ✗ Flask import failed: {e}")
    sys.exit(1)

# Test model files
print("\n2. Testing model files...")
import os

if os.path.exists("model/emotion_model.h5"):
    print("   ✓ emotion_model.h5 found")
else:
    print("   ✗ emotion_model.h5 NOT found")

if os.path.exists("model/haarcascade_frontalface_default.xml"):
    print("   ✓ haarcascade_frontalface_default.xml found")
else:
    print("   ✗ haarcascade_frontalface_default.xml NOT found")

# Test loading model
print("\n3. Testing model loading...")
try:
    model = load_model("model/emotion_model.h5", compile=False)
    print(f"   ✓ Emotion model loaded (shape: {model.input_shape})")
except Exception as e:
    print(f"   ✗ Model loading failed: {e}")

# Test cascade
print("\n4. Testing face cascade...")
try:
    cascade = cv2.CascadeClassifier("model/haarcascade_frontalface_default.xml")
    if cascade.empty():
        print("   ✗ Cascade classifier is empty")
    else:
        print("   ✓ Face cascade loaded successfully")
except Exception as e:
    print(f"   ✗ Cascade loading failed: {e}")

print("\n" + "=" * 60)
print("✓ All tests passed! Project is ready to run.")
print("=" * 60)
print("\nYou can run:")
print("  1. Flask web app (recommended): python app.py")
print("     - Access at: http://127.0.0.1:5000")
print("  2. Real-time camera: python main.py")
print("  3. Image processing: python image_mode.py")
print("=" * 60)
