from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np

# Load your saved model
model = keras.models.load_model('plant_disease_model.keras')

# Load and preprocess the image to predict (replace 'path/to/image.jpg')
img_path ='test_images/sample1.JPG'

img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Make prediction
prediction = model.predict(img_array)
predicted_class = np.argmax(prediction, axis=1)
print(f"Predicted class index: {predicted_class}")

# Optionally, print the class names mapping if you have it saved or available
# print(train_gen.class_indices)  # If train_gen is accessible
# Define the class indices mapping dictionary (put your full classes here)
class_indices = {
    'Apple_scab': 0,
    'Black_rot': 1,
    'Cedar_apple_rust': 2,
    'healthy': 3,
    # ...
    'Pepper__bell___Bacterial_spot': 18,
    # Add rest of your classes with their indices
}

# Reverse mapping index -> class label
indices_class = {v: k for k, v in class_indices.items()}

# Get predicted class label
predicted_label = indices_class[predicted_class[0]]
print("Predicted class:", predicted_label)

