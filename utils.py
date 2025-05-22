import os

DATA_DIR = "face_data"
ENCODINGS_FILE = "face_encodings.pkl"

def create_person_dir(id_, name):
    name = name.replace(" ", "_")
    person_dir = os.path.join(DATA_DIR, f"{id_}_{name}")
    os.makedirs(person_dir, exist_ok=True)
    return person_dir
