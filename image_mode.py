import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model and cascade
emotion_model = load_model("model/emotion_model.h5", compile=False)
face_cascade = cv2.CascadeClassifier(
    "model/haarcascade_frontalface_default.xml"
)

emotion_labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# Emoji image paths
emoji_map = {
    "happy": "emojis/happy.png",
    "sad": "emojis/sad.png",
    "angry": "emojis/angry.png",
    "neutral": "emojis/neutral.png",
    "surprise": "emojis/surprise.png",
    "fear": "emojis/fear.png",
    "disgust": "emojis/disgust.png"
}

# Emoji symbols for terminal output
emoji_symbol_map = {
    "happy": "😊",
    "sad": "😢",
    "angry": "😠",
    "neutral": "😐",
    "surprise": "😲",
    "fear": "😨",
    "disgust": "🤢"
}

def overlay_emoji(frame, emoji, x, y):
    h, w = emoji.shape[:2]

    if y < 0 or x < 0:
        return frame

    if y + h > frame.shape[0] or x + w > frame.shape[1]:
        return frame

    if emoji.shape[2] == 4:
        alpha = emoji[:, :, 3] / 255.0
        for c in range(3):
            frame[y:y+h, x:x+w, c] = (
                frame[y:y+h, x:x+w, c] * (1 - alpha)
                + emoji[:, :, c] * alpha
            )
    else:
        frame[y:y+h, x:x+w] = emoji

    return frame


# Get image path
image_path = input("Enter image path: ").strip('"')

image = cv2.imread(image_path)

if image is None:
    print("❌ Image not found!")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=4,
    minSize=(30, 30)
)

if len(faces) == 0:
    print("❌ No face detected!")
    cv2.putText(image, "No Face Detected!",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3)
else:
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (64, 64))
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=-1)
        face = np.expand_dims(face, axis=0)

        preds = emotion_model.predict(face, verbose=0)
        emotion_index = np.argmax(preds)
        emotion = emotion_labels[emotion_index]
        confidence = preds[0][emotion_index]

        # 🔥 TERMINAL OUTPUT
        print("\n📊 Emotion Probabilities:")
        for i, label in enumerate(emotion_labels):
            percentage = preds[0][i] * 100
            print(f"{label:<8}: {percentage:.2f}%")

        emoji_symbol = emoji_symbol_map.get(emotion, "")
        final_percentage = confidence * 100
        print(f"\n🎯 Final Emotion: {emotion.upper()} {emoji_symbol} ({final_percentage:.2f}%)")
        print("-" * 40)

        # Overlay emoji image
        if emotion in emoji_map:
            emoji = cv2.imread(emoji_map[emotion], cv2.IMREAD_UNCHANGED)
            if emoji is not None:
                emoji = cv2.resize(emoji, (120, 120))
                image = overlay_emoji(image, emoji, x, max(0, y-130))

        # Draw face rectangle
        cv2.rectangle(image, (x, y), (x+w, y+h), (0,255,0), 2)

        # Show emotion on image
        result_text = f"{emotion.upper()} ({confidence:.2f})"
        cv2.putText(image,
                    result_text,
                    (30, image.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    3)

# Show final output image
cv2.imshow("Final Emotion Result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
