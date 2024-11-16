from flask import Flask, render_template, request, jsonify
import os
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model (make sure the model file is in the same directory or specify the correct path)
model = tf.keras.models.load_model('plant_health_model.keras')  # Change this path if needed

# Define the labels for the diseases
LABELS = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", 
    "Apple___healthy", "Blueberry___healthy", "Cherry___healthy", "Cherry___Powdery_mildew",
    "Corn___Cercospora_leaf_spot Gray_leaf_spot", "Corn___Common_rust", "Corn___healthy", 
    "Corn___Northern_Leaf_Blight", "Grape___Black_rot", "Grape___Esca_(Black_Measles)", 
    "Grape___healthy", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot", "Peach___healthy", "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", 
    "Potato___Early_blight", "Potato___healthy", "Potato___Late_blight", "Raspberry___healthy", 
    "Soybean___healthy", "Squash___Powdery_mildew", "Strawberry___healthy", "Strawberry___Leaf_scorch", 
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___healthy", "Tomato___Late_blight", 
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite", 
    "Tomato___Target_Spot", "Tomato___Tomato_mosaic_virus", "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
]

# Function to classify the plant image and get the solution
def classify_and_get_solution(image_path):
    # Load and process the image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)  # Expand dims to add batch size dimension
    img = img / 255.0  # Normalize image to range [0, 1]

    # Predict the disease
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=1)[0]
    disease = LABELS[predicted_class]

    # Simulate a solution (you can integrate an API to get actual solutions)
    solution = f"To treat {disease}, consult your local plant care guide or visit a specialist."

    return disease, solution

# Route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the uploaded image file
        file = request.files['image']
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)

        # Classify the image and get the solution
        disease, solution = classify_and_get_solution(file_path)

        return jsonify({
            'disease': disease,
            'solution': solution
        })

if __name__ == '__main__':
    app.run(debug=True)
