from flask import Flask, request, jsonify
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image
from flask_cors import CORS
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

# Load models
COTTON_CHECK_MODEL_PATH = 'cotton_model2.h5'  # Binary classifier
DISEASE_MODEL_PATH = 'model_resnet152V2.h5'   # Multi-class classifier

cotton_model = load_model(COTTON_CHECK_MODEL_PATH)
disease_model = load_model(DISEASE_MODEL_PATH)

IMG_SIZE = 224

# ✅ Endpoint 1: Check if image is cotton plant or leaf
@app.route('/api/is-cotton', methods=['POST'])
def check_cotton():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        img = Image.open(file).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = cotton_model.predict(img_array)[0][0]

        if prediction > 0.5:
            result = "Not a Cotton Image.."
        else:
            result = "Yes Cotton Image "

        return jsonify({
            'prediction_score': float(prediction),
            'result': result
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ✅ Helper function for disease classification
def classify_disease(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    preds = disease_model.predict(x)
    preds = np.argmax(preds, axis=1)

    if preds == 0:
        return "The leaf is diseased cotton leaf"
    elif preds == 1:
        return "The leaf is diseased cotton plant"
    elif preds == 2:
        return "The leaf is fresh cotton leaf"
    else:
        return "The leaf is fresh cotton plant"


# ✅ Endpoint 2: Classify cotton disease
@app.route('/api/predict', methods=['POST'])
def classify_disease_api():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        basepath = os.path.dirname(__file__)
        upload_dir = os.path.join(basepath, 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, secure_filename(file.filename))
        file.save(file_path)

        result = classify_disease(file_path)
        os.remove(file_path)

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ✅ Run the app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
