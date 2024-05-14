from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "http://localhost:5173"}})

# Load the pre-trained model
model = keras.models.load_model('./model/model.keras')
categories = ["NonDemented", "MildDemented", "ModerateDemented", "VeryMildDemented"]

# Define a function to process the uploaded image
def process_image(image_file):
    SIZE = 120
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (SIZE, SIZE))
    image = image / 255.0
    prediction = model.predict(np.array([image]).reshape(-1, SIZE, SIZE, 1))
    pclass = np.argmax(prediction)
    predicted_category = categories[int(pclass)]
    return predicted_category

# Route for handling image upload and prediction
@app.route('/predict', methods=['POST'])
def predict_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        predicted_category = process_image(file)
        return jsonify({'prediction': predicted_category}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
