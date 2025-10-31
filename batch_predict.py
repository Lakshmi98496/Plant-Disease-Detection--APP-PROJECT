import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load your saved model
model = load_model('plant_disease_model.keras')

# Folder containing images for prediction
folder_path = 'test_images'  # Edit as needed

# Class indices mapping (update this with ALL your class labels)
class_indices = {
    'Apple_scab': 0,
    'Black_rot': 1,
    'Cedar_apple_rust': 2,
    'healthy': 3,
    # ...
    'Pepper_bell__Bacterial_spot': 18,
    'Some_other_class': 19,
}
indices_class = {v: k for k, v in class_indices.items()}

# Output predictions to a CSV file for easy GUI integration
with open("results.csv", "w") as f:
    f.write("filename,predicted_class\n")
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(folder_path, filename)
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0
            prediction = model.predict(np.expand_dims(img_array, axis=0))
            class_idx = int(np.argmax(prediction))
            predicted_class = indices_class.get(class_idx, f"Unknown_class_{class_idx}")
            f.write(f"{filename},{predicted_class}\n")

print("Predictions written to results.csv")