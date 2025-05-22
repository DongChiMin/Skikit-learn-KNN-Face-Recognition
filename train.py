import os
import pickle
import random
import face_recognition
from sklearn import neighbors
from sklearn.metrics import accuracy_score
from tkinter import messagebox
from utils import DATA_DIR, ENCODINGS_FILE

def split_data(person_path, test_ratio=0.1):
    images = [img for img in os.listdir(person_path) if img.endswith('.jpg')]
    random.shuffle(images)
    split_index = int(len(images) * (1 - test_ratio))
    return images[:split_index], images[split_index:]

def train_model():
    if not os.path.exists(DATA_DIR):
        messagebox.showerror("Error", "No data found.")
        return

    train_encodings, train_names = [], []
    test_encodings, test_names = [], []

    for folder in os.listdir(DATA_DIR):
        person_path = os.path.join(DATA_DIR, folder)
        if not os.path.isdir(person_path):
            continue

        name = folder.split('_', 1)[1] if '_' in folder else folder
        print(f"Processing person: {name}")

        train_images, test_images = split_data(person_path, test_ratio = 0.1)

        # Process training images
        for idx, image_name in enumerate(train_images, 1):
            print(f"   > Training image {idx}/{len(train_images)}: {image_name}")
            image_path = os.path.join(person_path, image_name)
            image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(image)
            if len(face_locations) != 1:
                continue
            encodings = face_recognition.face_encodings(image, face_locations)
            if encodings:
                train_encodings.append(encodings[0])
                train_names.append(name)

        # Process test images
        for image_name in test_images:
            image_path = os.path.join(person_path, image_name)
            image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(image)
            if len(face_locations) != 1:
                continue
            encodings = face_recognition.face_encodings(image, face_locations)
            if encodings:
                test_encodings.append(encodings[0])
                test_names.append(name)

    if not train_encodings:
        messagebox.showerror("Error", "No valid training data found.")
        return

    best_accuracy = 0
    best_model = None
    best_k = None

    for k in [1, 3, 5, 7]:
        knn_clf = neighbors.KNeighborsClassifier(n_neighbors=k, algorithm='ball_tree', weights='distance')
        knn_clf.fit(train_encodings, train_names)

        if test_encodings:
            predictions = knn_clf.predict(test_encodings)
            accuracy = accuracy_score(test_names, predictions)
            print(f"[Validation] Accuracy with k={k}: {accuracy:.2%}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = knn_clf
                best_k = k
        else:
            print("No test data available for validation.")
            break

    if best_model:
        with open(ENCODINGS_FILE, 'wb') as f:
            pickle.dump(best_model, f)
        messagebox.showinfo("Success", f"Training complete! Best k = {best_k} with accuracy = {best_accuracy:.2%}")
    else:
        messagebox.showerror("Error", "Training failed - no model saved.")
