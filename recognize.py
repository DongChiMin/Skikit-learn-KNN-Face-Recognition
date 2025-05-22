import cv2
import face_recognition
import pickle
import os
from tkinter import messagebox
from utils import ENCODINGS_FILE

def recognize_faces():
    if not os.path.exists(ENCODINGS_FILE):
        messagebox.showerror("Error", "Model not found.")
        return

    with open(ENCODINGS_FILE, 'rb') as f:
        knn_clf = pickle.load(f)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Cannot access camera.")
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    process_frame = True #xử lý xen kẽ từng khung hình để tăng tốc độ
    face_names = [] #danh sách tên người nhận diện được trong mỗi khung hình.
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        if process_frame:
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []

            for encoding in face_encodings:
                distances, indices = knn_clf.kneighbors([encoding], n_neighbors=1)
                distance = distances[0][0]
                name = knn_clf.predict([encoding])[0] if distance < 0.4 else "Unknown"
                face_names.append(name)
        process_frame = not process_frame

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2

            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, top - 10), font, 0.8, color, 2)

        cv2.imshow('Face Recognition - Press q to quit', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
