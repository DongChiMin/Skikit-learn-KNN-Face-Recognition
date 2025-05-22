import cv2
import os
import time
import face_recognition
from tkinter import messagebox
from utils import create_person_dir

def capture_data(id_, name):
    try:
        id_ = int(id_)
        if not name.strip():
            raise ValueError("Name cannot be empty")
    except ValueError:
        messagebox.showerror("Error", "Invalid ID or empty name!")
        return
    
    person_dir = create_person_dir(id_, name)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        messagebox.showerror("Error", "Cannot access camera!")
        return

    count = 0   #Đếm số ảnh đã chụp
    max_images = 100    #Số ảnh tối đa sẽ chụp (100)
    start_time = time.time()    #Ghi lại thời điểm bắt đầu
    timeout = 30    #Giới hạn 30 giây
    font = cv2.FONT_HERSHEY_SIMPLEX
    messagebox.showinfo("Info", f"Capturing images for {name}. Please look at the camera.")
    while True:
        if time.time() - start_time > timeout:
            messagebox.showwarning("Timeout", "No face detected within 30 seconds.")
            break

        ret, frame = cap.read()
        if not ret:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)

        for (top, right, bottom, left) in face_locations:
            count += 1
            face_img = frame[top:bottom, left:right]
            cv2.imwrite(os.path.join(person_dir, f"{count}.jpg"), face_img)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, f"Captured: {count}/{max_images}", (10, 30), font, 1, (0, 255, 0), 2)

        cv2.imshow("Capture Data - Press q to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or count >= max_images:
            break

    cap.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Success", f"Captured {count} images for {name}.")
