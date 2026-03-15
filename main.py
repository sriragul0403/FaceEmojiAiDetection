import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load face detector
face_cascade = cv2.CascadeClassifier(
    "model/haarcascade_frontalface_default.xml"
)

# Load emotion model (no compile needed)
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
}

# Emoji overlay function
def overlay_emoji(frame, emoji, x, y):
    h, w = emoji.shape[:2]

    # Prevent overflow outside frame
    if y + h > frame.shape[0] or x + w > frame.shape[1]:
        return frame

    if emoji.shape[2] == 4:  # Has alpha channel
        alpha = emoji[:, :, 3] / 255.0
        for c in range(3):
            frame[y:y+h, x:x+w, c] = (
                frame[y:y+h, x:x+w, c] * (1 - alpha)
                + emoji[:, :, c] * alpha
            )
    else:
        frame[y:y+h, x:x+w] = emoji

    return frame


# Try multiple camera backends for better compatibility
cap = None
for camera_index in range(3):
    cap = cv2.VideoCapture(camera_index)
    if cap.isOpened():
        print(f"✓ Camera found at index {camera_index}")
        break

if cap is None or not cap.isOpened():
    # Try DSHOW as fallback
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ No camera found")
        exit()

print("Press Q to quit")
print("Multi-face detection mode - Detects up to 4 people\n")

person_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # Limit to 4 people max
    faces = faces[:4]
    person_count = len(faces)
    
    # Add counter display at top
    cv2.putText(frame, f"Detected: {person_count}/4 people", (10, 30),
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

        # Smart kiss logic (strong happy = kiss)
        if emotion == "happy" and confidence > 0.85:
            emotion = "kiss"

        # Load emoji safely
        if emotion in emoji_map:
            emoji = cv2.imread(emoji_map[emotion], cv2.IMREAD_UNCHANGED)
            if emoji is not None:
                emoji = cv2.resize(emoji, (100, 100))
                frame = overlay_emoji(frame, emoji, x, max(0, y-110))

        # Draw rectangle and text with person label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
        label = f"Person {idx} | {emotion} ({confidence:.2f})"
        cv2.putText(frame, label, (x, y-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        
        # Draw confidence bar
        bar_width = w
        bar_height = 5
        bar_y = y + h + 10
        cv2.rectangle(frame, (x, bar_y), (x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        cv2.rectangle(frame, (x, bar_y), (x + int(bar_width * confidence), bar_y + bar_height), 
                      (0, 255, 0), -1)

    cv2.imshow("Face Expression Emoji Detector - Multi-Person Mode", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()