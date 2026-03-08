import os
import cv2
import numpy as np
from fer import FER

# ---------------- PATH SETUP (VERY IMPORTANT) ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ---------------- LOAD DNN MODELS ----------------
face_net = cv2.dnn.readNetFromCaffe(
    os.path.join(MODEL_DIR, "deploy.prototxt"),
    os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
)

age_net = cv2.dnn.readNetFromCaffe(
    os.path.join(MODEL_DIR, "age_deploy.prototxt"),
    os.path.join(MODEL_DIR, "age_net.caffemodel")
)

gender_net = cv2.dnn.readNetFromCaffe(
    os.path.join(MODEL_DIR, "gender_deploy.prototxt"),
    os.path.join(MODEL_DIR, "gender_net.caffemodel")
)

# ---------------- LABELS ----------------
AGE_LIST = ['(0-2)','(4-6)','(8-12)','(15-20)',('21-25'),
            '(25-32)','(38-43)','(48-53)','(60-100)']
GENDER_LIST = ['Male','Female']

# ---------------- INITIALIZE ----------------
emotion_detector = FER(mtcnn=True)m
cap = cv2.VideoCapture(0)

print("Starting Camera... Press 'q' to quit")

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        frame, 1.0, (300, 300),
        (104.0, 177.0, 123.0)
    )

    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)

            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            # -------- Emotion --------
            emotion, _ = emotion_detector.top_emotion(face)
            if emotion is None:
                emotion = "Unknown"

            # -------- Age & Gender --------
            face_blob = cv2.dnn.blobFromImage(
                face, 1.0, (227, 227),
                (78.426, 87.768, 114.895)
            )

            gender_net.setInput(face_blob)
            gender = GENDER_LIST[gender_net.forward()[0].argmax()]

            age_net.setInput(face_blob)
            age = AGE_LIST[age_net.forward()[0].argmax()]

            label = f"{gender}, {age}, {emotion}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 255), 2
            )

    cv2.imshow("Emotion + Age + Gender", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------------- CLEANUP ----------------
cap.release()
cv2.destroyAllWindows()
