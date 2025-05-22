# SKIKITLEARN KNN Face Recognition
A Group Project made by OpenCV, face_recognition, Skikit-learn (K-nearest neighbors) and Tkinter, Numpy.

A simple face recognition trainer built with Python, using face_recognition, scikit-learn (KNN), and a Tkinter GUI. The project extracts face embeddings from images, trains a K-Nearest Neighbors model, and validates it to select the best hyperparameter k.

## 🚀 Features

- 🧠 Automatically trains a face recognition model using the KNN algorithm  
- 📁 Supports structured datasets organized per person (one folder per person)  
- 🎯 Validates different values of `k` (1, 3, 5, 7) and selects the best one  
- 💾 Saves the trained model to a file using `pickle`  
- 📊 Shows training progress and validation accuracy  
- ❌ Skips images with no face or multiple faces

## 🛠️ Requirements

- Python 3.7+
- face_recognition
- scikit-learn
- opencv-python
- tkinter

## ▶️ How to Run

1. **Install dependencies** (if not already installed):

    ```bash
    pip install face_recognition scikit-learn opencv-python tkinter
    ```

2. **Run the main program**:

    ```bash
    python main.py
    ```

3. **Capture your face data**:
   - Click the **"Capture Data"** button.
   - The webcam will open and automatically capture **100 photos** of your face.
   - Make sure only one face appears clearly in the frame.

4. **Train the model**:
   - After capturing data, click the **"Train"** button.
   - The system will process the collected images, extract facial encodings, and train a KNN model.
   - It will automatically validate different values of `k` and save the best model.

5. **Start face recognition**:
   - Click the **"Recognize"** button.
   - The webcam will open and perform real-time face recognition using the trained model.
   - Recognized names will appear on the screen.

