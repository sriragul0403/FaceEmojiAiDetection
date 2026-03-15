import cv2
import numpy as np
from tensorflow.keras.models import load_model

emotion_model = load_model("model/emotion_model.h5", compile=False)

face_cascade = cv2.CascadeClassifier(
    "model/haarcascade_frontalface_default.xml"
)

emotion_labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

emoji_map = {
    "angry": "😠",
    "disgust": "🤢",
    "fear": "😨",
    "happy": "😊",
    "sad": "😢",
    "surprise": "😲",
    "neutral": "😐",
    "kiss": "💋"   # added kiss emoji
}


def detect_emotion(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None

    for (x, y, w, h) in faces:

        face = gray[y:y+h, x:x+w]

        face = cv2.resize(face, (64, 64))

        face = face / 255.0

        face = np.reshape(face, (1, 64, 64, 1))

        preds = emotion_model.predict(face, verbose=0)[0]

        emotion_index = np.argmax(preds)

        emotion = emotion_labels[emotion_index]

        percentage = int(preds[emotion_index] * 100)

        # ---- KISS DETECTION (extra rule) ----
        if emotion == "happy" and percentage > 90:
            emotion = "kiss"
        # ------------------------------------

        emoji = emoji_map[emotion]

        return {
            "emotion": emotion,
            "emoji": emoji,
            "percentage": percentage
        }