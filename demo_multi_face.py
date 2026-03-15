import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# Load face detector
face_cascade = cv2.CascadeClassifier(
    "model/haarcascade_frontalface_default.xml"
)

# Load emotion model
emotion_model = load_model("model/emotion_model.h5", compile=False)

emotion_labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# Emoji mapping
emoji_map = {
    "happy": "emojis/happy.png",
    "kiss": "emojis/kiss.png",
    "sad": "emojis/sad.png",
    "angry": "emojis/angry.png",
    "neutral": "emojis/neutral.png",
    "surprise": "emojis/surprise.png",
    "fear": "emojis/fear.png" if os.path.exists("emojis/fear.png") else None,
    "disgust": "emojis/disgust.png" if os.path.exists("emojis/disgust.png") else None,
}

# Remove None values
emoji_map = {k: v for k, v in emoji_map.items() if v}

# Emoji overlay function
def overlay_emoji(frame, emoji, x, y):
    h, w = emoji.shape[:2]
    
    # Prevent overflow
    if y + h > frame.shape[0] or x + w > frame.shape[1]:
        return frame
    
    if len(emoji.shape) == 3 and emoji.shape[2] == 4:  # Has alpha
        alpha = emoji[:, :, 3] / 255.0
        for c in range(3):
            frame[y:y+h, x:x+w, c] = (
                frame[y:y+h, x:x+w, c] * (1 - alpha)
                + emoji[:, :, c] * alpha
            )
    else:
        frame[y:y+h, x:x+w] = emoji[:, :, :3]
    
    return frame


def process_image_multi_face(image_path):
    """Process image and detect emotions for multiple faces"""
    frame = cv2.imread(image_path)
    
    if frame is None:
        print(f"❌ Could not load image: {image_path}")
        return
    
    frame = cv2.resize(frame, (1280, 720))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # Limit to 4 people
    faces = faces[:4]
    
    print(f"\n✓ Image loaded: {image_path}")
    print(f"✓ Detected {len(faces)} face(s) - Processing emotions...")
    
    # Add counter
    cv2.putText(frame, f"Detected: {len(faces)}/4 people", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    for idx, (x, y, w, h) in enumerate(faces, 1):
        # Preprocess face
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (64, 64))
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=-1)
        face = np.expand_dims(face, axis=0)
        
        # Predict emotion
        preds = emotion_model.predict(face, verbose=0)
        emotion_index = np.argmax(preds)
        emotion = emotion_labels[emotion_index]
        confidence = preds[0][emotion_index]
        
        print(f"  Person {idx}: {emotion} ({confidence:.2%})")
        
        # Smart kiss logic
        if emotion == "happy" and confidence > 0.85:
            emotion = "kiss"
        
        # Load emoji
        if emotion in emoji_map:
            emoji = cv2.imread(emoji_map[emotion], cv2.IMREAD_UNCHANGED)
            if emoji is not None:
                emoji = cv2.resize(emoji, (100, 100))
                frame = overlay_emoji(frame, emoji, x, max(0, y-110))
        
        # Draw rectangle and text
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
        label = f"Person {idx} | {emotion} ({confidence:.2f})"
        cv2.putText(frame, label, (x, y-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        
        # Confidence bar
        bar_width = w
        bar_height = 5
        bar_y = y + h + 10
        cv2.rectangle(frame, (x, bar_y), (x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        cv2.rectangle(frame, (x, bar_y), (x + int(bar_width * confidence), bar_y + bar_height), 
                      (0, 255, 0), -1)
    
    # Display result
    cv2.imshow("Multi-Face Emotion Detection - Demo Mode", frame)
    print("✓ Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return frame


def process_webcam():
    """Process live webcam feed"""
    print("\n🎥 Attempting to access WebCam...")
    
    # Try multiple camera backends
    cap = None
    for camera_index in range(3):
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            print(f"✓ Camera found at index {camera_index}")
            break
    
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("❌ No camera found")
            return False
    
    print("Press Q to quit\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (1280, 720))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Limit to 4 people
        faces = faces[:4]
        
        # Counter display
        cv2.putText(frame, f"Detected: {len(faces)}/4 people", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        for idx, (x, y, w, h) in enumerate(faces, 1):
            # Preprocess
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (64, 64))
            face = face.astype("float32") / 255.0
            face = np.expand_dims(face, axis=-1)
            face = np.expand_dims(face, axis=0)
            
            # Predict
            preds = emotion_model.predict(face, verbose=0)
            emotion_index = np.argmax(preds)
            emotion = emotion_labels[emotion_index]
            confidence = preds[0][emotion_index]
            
            # Kiss logic
            if emotion == "happy" and confidence > 0.85:
                emotion = "kiss"
            
            # Load emoji
            if emotion in emoji_map:
                emoji = cv2.imread(emoji_map[emotion], cv2.IMREAD_UNCHANGED)
                if emoji is not None:
                    emoji = cv2.resize(emoji, (100, 100))
                    frame = overlay_emoji(frame, emoji, x, max(0, y-110))
            
            # Draw
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
            label = f"Person {idx} | {emotion} ({confidence:.2f})"
            cv2.putText(frame, label, (x, y-15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            
            # Bar
            bar_width = w
            bar_height = 5
            bar_y = y + h + 10
            cv2.rectangle(frame, (x, bar_y), (x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
            cv2.rectangle(frame, (x, bar_y), (x + int(bar_width * confidence), bar_y + bar_height), 
                          (0, 255, 0), -1)
        
        cv2.imshow("Multi-Face Emotion Detection - WebCam Mode (Up to 4 People)", frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Multi-Face Emotion Detection System")
    print("=" * 60)
    print("\nModes:")
    print("1. WebCam (Live - Press Q to quit)")
    print("2. Demo Image (Static)")
    print("3. Exit")
    print()
    
    choice = input("Select mode (1-3): ").strip()
    
    if choice == "1":
        print("\n" + "=" * 60)
        process_webcam()
    elif choice == "2":
        print("\n" + "=" * 60)
        print("Demo Mode - Testing with synthetic frames")
        print("=" * 60)
        
        # Create test image with multiple faces for demo
        # (You can replace with actual image path)
        test_image = "uploads/test.jpg" if os.path.exists("uploads/test.jpg") else None
        
        if test_image:
            process_image_multi_face(test_image)
        else:
            print("No test image found. Please upload an image to /uploads/test.jpg")
            print("Or use the Flask web app to upload images.")
    elif choice == "3":
        print("Goodbye!")
    else:
        print("Invalid choice")
