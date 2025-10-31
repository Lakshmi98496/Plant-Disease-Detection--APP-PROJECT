import os
import io
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# --- CONFIGURATION ---
MODEL_PATH = 'plant_disease_model.keras'
# Define Max File Size (4MB) for server-side validation
MAX_FILE_SIZE = 4 * 1024 * 1024 

# Load the trained model once when the app starts
try:
    # Ensure this model file exists in the same directory
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    # Application needs to handle the case where the model fails to load

# --- CLASS MAPPING (38 Classes) ---
CLASS_INDICES = {
    'Apple_Apple_scab': 0, 'Apple_Black_rot': 1, 'Apple_Cedar_apple_rust': 2, 'Apple_healthy': 3, 
    'Blueberry__healthy': 4, 'Cherry(including_sour)Powdery_mildew': 5, 'Cherry(including_sour)_healthy': 6, 
    'Corn_(maize)Cercospora_leaf_spot_Gray_leaf_spot': 7, 'Corn(maize)Common_rust': 8, 
    'Corn_(maize)Northern_Leaf_Blight': 9, 'Corn(maize)healthy': 10, 'Grape__Black_rot': 11, 
    'Grape_Esca(Black_Measles)': 12, 'GrapeLeaf_blight(Isariopsis_Leaf_Spot)': 13, 'Grape__healthy': 14, 
    'Orange_Haunglongbing(Citrus_greening)': 15, 'Peach_Bacterial_spot': 16, 'Peach__healthy': 17, 
    'Pepper,bell_Bacterial_spot': 18, 'Pepper,bellhealthy': 19, 'Potato__Early_blight': 20, 
    'Potato_Late_blight': 21, 'Potato_healthy': 22, 'Raspberry_healthy': 23, 'Soybean_healthy': 24, 
    'Squash_Powdery_mildew': 25, 'Strawberry_Leaf_scorch': 26, 'Strawberry_healthy': 27, 
    'Tomato_Bacterial_spot': 28, 'Tomato_Early_blight': 29, 'Tomato_Late_blight': 30, 
    'Tomato_Leaf_Mold': 31, 'Tomato_Septoria_leaf_spot': 32, 
    'Tomato_Spider_mites_Two-spotted_spider_mite': 33, 'Tomato_Target_Spot': 34, 
    'Tomato_Tomato_Yellow_Leaf_Curl_Virus': 35, 'Tomato_Tomato_mosaic_virus': 36, 'Tomato_healthy': 37
}

# Reverse map index to class name
INDICES_CLASS = {v: k for k, v in CLASS_INDICES.items()}


# --- DIAGNOSIS DATA (Actionable Treatment Advice) ---
DIAGNOSIS_INFO = {
    'Apple___Apple_scab': {
        'Diagnosis': 'Fungal disease causing scabby spots. Requires intervention to prevent spread.',
        'Treatment': 'Apply copper-based fungicide and remove fallen leaves.'
    },
    'Apple_Black_rot': { # Corrected key formatting to match CLASS_INDICES
        'Diagnosis': 'Advanced fungal rot that often leads to internal decay and can spread to other fruit.',
        'Treatment': 'Prune out dead wood immediately and use protective sprays.'
    },
    'Apple_Cedar_apple_rust': { # Corrected key formatting
        'Diagnosis': 'Rust disease characterized by orange/yellow spots, requiring local control measures.',
        'Treatment': 'Use preventative fungicide early in the spring.'
    },
    'Apple__healthy': { # Corrected key formatting
        'Diagnosis': 'The plant shows no signs of disease and appears perfectly healthy.',
        'Treatment': 'Continue routine care and monitoring.'
    },
    'Blueberry__healthy': {
        'Diagnosis': 'The plant is healthy.',
        'Treatment': 'Maintain standard care practices.'
    },
    'Cherry(including_sour)Powdery_mildew': {
        'Diagnosis': 'White, powdery fungal growth on leaves and twigs. Can stunt growth.',
        'Treatment': 'Apply sulfur fungicide and ensure good air circulation.'
    },
    'Cherry(including_sour)_healthy': {
        'Diagnosis': 'The plant is healthy.',
        'Treatment': 'Maintain standard care practices.'
    },
    'Corn_(maize)Cercospora_leaf_spot_Gray_leaf_spot': {
        'Diagnosis': 'Severe fungal leaf spot causing gray, rectangular lesions. Can significantly reduce yield.',
        'Treatment': 'Rotate crops and apply foliar fungicides.'
    },
    'Corn(maize)Common_rust': {
        'Diagnosis': 'Pustules containing reddish-brown spores on leaves. Common in cool, moist conditions.',
        'Treatment': 'Use resistant hybrids or apply fungicide at the first sign of rust.'
    },
    'Corn_(maize)Northern_Leaf_Blight': {
        'Diagnosis': 'Long, cigar-shaped gray-green lesions on leaves, leading to leaf death.',
        'Treatment': 'Plant resistant varieties and use timely fungicide application.'
    },
    'Corn(maize)healthy': {
        'Diagnosis': 'The plant is healthy.',
        'Treatment': 'Maintain standard care practices.'
    },
    'Grape__Black_rot': {
        'Diagnosis': 'Destructive fungal disease causing shriveled, mummified berries.',
        'Treatment': 'Practice good sanitation (remove mummies) and use protective fungicide schedule.'
    },
    'Grape_Esca(Black_Measles)': {
        'Diagnosis': 'A wood-canker disease causing leaf discoloration and eventual vine collapse.',
        'Treatment': 'Prune infected wood back to healthy tissue and seal cuts.'
    },
    'GrapeLeaf_blight(Isariopsis_Leaf_Spot)': {
        'Diagnosis': 'Causes small, dark leaf spots. Generally less severe than other grape diseases.',
        'Treatment': 'Maintain general fungicide program; ensure good air flow.'
    },
    'Grape__healthy': {
        'Diagnosis': 'The plant is healthy.',
        'Treatment': 'Maintain standard care practices.'
    },
    'Orange_Haunglongbing(Citrus_greening)': {
        'Diagnosis': 'A serious bacterial disease spread by psyllids, causing blotchy mottling on leaves and distorted fruit.',
        'Treatment': 'No cure; management involves removing infected trees and controlling psyllid vectors.'
    },
    'Peach_Bacterial_spot': {
        'Diagnosis': 'Bacterial infection causing small, dark, angular spots on leaves and fruit.',
        'Treatment': 'Use resistant varieties and apply copper bactericides.'
    },
    'Peach__healthy': {
        'Diagnosis': 'The plant is healthy.',
        'Treatment': 'Maintain standard care practices.'
    },
    'Pepper,bell_Bacterial_spot': {
        'Diagnosis': 'Bacterial infection causing irregular, dark spots on leaves and scabs on fruit.',
        'Treatment': 'Use disease-free seeds/transplants and apply copper-based sprays.'
    },
    'Pepper,bellhealthy': {
        'Diagnosis': 'The plant is healthy.',
        'Treatment': 'Maintain standard care practices.'
    },
    'Potato__Early_blight': {
        'Diagnosis': 'Fungal disease causing dark, concentric rings on older leaves (target spots).',
        'Treatment': 'Rotate crops, use resistant cultivars, and apply fungicides.'
    },
    'Potato__Late_blight': {
        'Diagnosis': 'Highly aggressive water mold causing rapid tissue death. Appears as dark, water-soaked spots.',
        'Treatment': 'Immediate and aggressive use of fungicides is required.'
    },
    'Potato_healthy': {
        'Diagnosis': 'The plant is healthy.',
        'Treatment': 'Maintain standard care practices.'
    },
    'Raspberry_healthy': {
        'Diagnosis': 'The plant is healthy.',
        'Treatment': 'Maintain standard care practices.'
    },
    'Soybean__healthy': {
        'Diagnosis': 'The plant is healthy.',
        'Treatment': 'Maintain standard care practices.'
    },
    'Squash__Powdery_mildew': {
        'Diagnosis': 'White, powdery coating on leaves, reducing photosynthesis and yield.',
        'Treatment': 'Apply horticultural oil or biological control agents.'
    },
    'Strawberry_Leaf_scorch': {
        'Diagnosis': 'Fungal infection causing purple to brown spots, leading to leaves appearing scorched.',
        'Treatment': 'Mow/remove old leaves after harvest and use an appropriate fungicide.'
    },
    'Strawberry__healthy': {
        'Diagnosis': 'The plant is healthy.',
        'Treatment': 'Maintain standard care practices.'
    },
    'Tomato__Bacterial_spot': {
        'Diagnosis': 'Bacterial disease causing dark, angular spots on leaves and raised scabs on fruit.',
        'Treatment': 'Use copper sprays and avoid working with plants when wet.'
    },
    'Tomato_Early_blight': {
        'Diagnosis': 'Fungal disease causing dark spots with concentric rings on older leaves.',
        'Treatment': 'Fungicide treatment (chlorothalonil) and good sanitation.'
    },
    'Tomato__Late_blight': {
        'Diagnosis': 'Highly aggressive disease causing large, dark, water-soaked lesions. Requires immediate action.',
        'Treatment': 'Aggressive application of fungicides.'
    },
    'Tomato__Leaf_Mold': {
        'Diagnosis': 'Fungal disease causing velvety, olive-green/brown spots on the underside of leaves, common in greenhouses.',
        'Treatment': 'Increase ventilation and reduce humidity.'
    },
    'Tomato__Septoria_leaf_spot': {
        'Diagnosis': 'Fungal infection causing small, circular spots with gray centers. Causes premature leaf drop.',
        'Treatment': 'Apply fungicides and practice crop rotation.'
    },
    'Tomato__Spider_mites_Two-spotted_spider_mite': {
        'Diagnosis': 'Pest infestation causing yellow stippling and fine webbing on leaves.',
        'Treatment': 'Use miticides or horticultural oils/soaps.'
    },
    'Tomato__Target_Spot': {
        'Diagnosis': 'Fungal disease causing dark spots with light centers, resembling a bullseye.',
        'Treatment': 'Fungicide applications and proper watering (avoid overhead watering).'
    },
    'Tomato__Tomato_Yellow_Leaf_Curl_Virus': {
        'Diagnosis': 'Viral disease transmitted by whiteflies, causing severe stunting and leaf curling.',
        'Treatment': 'No cure; control whitefly population and remove infected plants.'
    },
    'Tomato_Tomato_mosaic_virus': {
        'Diagnosis': 'Viral disease causing a mosaic pattern (light and dark green areas) and distorted leaves.',
        'Treatment': 'No cure; remove infected plants and avoid using tobacco products near plants.'
    },
    'Tomato__healthy': {
        'Diagnosis': 'The plant is healthy.',
        'Treatment': 'Maintain standard care practices.'
    },
    'Unknown Class': {
        'Diagnosis': 'Prediction failed or class not recognized by the model. Check image clarity.',
        'Treatment': 'Consult a local agricultural expert for tailored advice.'
    }
}
# -------------------------------------------------------------


@app.route('/')
def home():
    """Renders the main dashboard page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles image upload, prediction, and returns results with diagnosis and severity."""
    # Ensure a file was sent
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request."}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file."}), 400
    
    file_path = None
    if file:
        try:
            # --- SERVER-SIDE FILE SIZE CHECK (Robustness) ---
            # Save the file temporarily to check its size
            temp_dir = 'temp'
            os.makedirs(temp_dir, exist_ok=True)
            file_path = os.path.join(temp_dir, file.filename)
            
            # Use a stream for size check instead of saving twice, if possible, 
            # but for simplicity and compatibility with model loading, we save here.
            file.save(file_path)
            
            if os.path.getsize(file_path) > MAX_FILE_SIZE:
                 os.remove(file_path)
                 return jsonify({"error": f"File size exceeds {MAX_FILE_SIZE / (1024 * 1024):.0f}MB limit. Please upload a smaller image."}), 400
            # -----------------------------------------------

            # Preprocess the image
            # Note: The model's target size should match your training setup (e.g., 224x224 or 256x256)
            img = image.load_img(file_path, target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0
            
            # --- MODEL INFERENCE ---
            prediction = model.predict(np.expand_dims(img_array, axis=0))
            
            # 1. Get Prediction and Confidence Score
            predicted_class_index = np.argmax(prediction)
            predicted_class_name = INDICES_CLASS.get(predicted_class_index, "Unknown Class")
            
            confidence_score = float(prediction[0][predicted_class_index]) * 100
            confidence_string = f"{confidence_score:.2f}%"

            # 2. Calculate Severity Score (FIXED LOGIC)
            if 'healthy' in predicted_class_name.lower():
                # CRITICAL FIX: Healthy predictions MUST be LOW severity
                severity = "LOW ✅" 
            elif confidence_score > 90.0:
                severity = "HIGH 🚨"
            elif confidence_score > 75.0:
                severity = "MODERATE ⚠"
            else:
                severity = "LOW / UNCERTAIN ⚪"

            # 3. Look up Diagnosis and Treatment
            diagnosis_details = DIAGNOSIS_INFO.get(predicted_class_name, DIAGNOSIS_INFO['Unknown Class'])

            # Clean up the temporary file
            os.remove(file_path)
            
            # --- FINAL JSON RESPONSE ---
            return jsonify({
                "filename": file.filename,
                "predicted_class": predicted_class_name,
                "confidence": confidence_string,     
                "severity": severity,               
                "diagnosis": diagnosis_details['Diagnosis'],    
                "treatment": diagnosis_details['Treatment']      
            })
        
        except Exception as e:
            # Clean up the file if it was saved before the error
            if file_path and os.path.exists(file_path):
                 os.remove(file_path)
            
            # Return a detailed error message (Improved Error Handling)
            print(f"Prediction Error: {e}")
            return jsonify({"error": f"An internal error occurred during prediction: {e}"}), 500

if __name__ == '__main__':
    # Ensure temp directory exists at startup
    os.makedirs('temp', exist_ok=True)
    app.run(debug=True)