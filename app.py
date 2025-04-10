from flask import Flask, request, jsonify
from keras.models import load_model
from PIL import Image
from flask_cors import CORS
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load binary classification model (cotton vs non-cotton)
COTTON_CHECK_MODEL_PATH = 'cotton_model_v2.keras'
cotton_model = load_model(COTTON_CHECK_MODEL_PATH)

IMG_SIZE = 224

@app.route('/api/is-cotton', methods=['POST'])
def check_cotton():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Load and preprocess image in memory
        img = Image.open(file).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        prediction = cotton_model.predict(img_array)[0][0]
        result = "Yes, this is a cotton image." if prediction <= 0.5 else "No, this is not a cotton image."

        return jsonify({
            'prediction_score': float(prediction),
            'result': result
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ✅ Run the app with dynamic PORT for Render
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
