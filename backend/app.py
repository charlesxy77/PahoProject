from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
import logging

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = None
model_loaded_successfully = False

def load_model():
    global model, model_loaded_successfully
    model_path = os.path.join(os.path.dirname(__file__), 'Livestock_Random_Forest_Model (1).pkl')
    logger.info(f"Attempting to load model from: {model_path}")
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded successfully. Type: {type(model)}")
        model_loaded_successfully = True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        model_loaded_successfully = False

# Load the model when the module is imported
load_model()

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 204
    
    if not model_loaded_successfully:
        logger.error("Model not loaded successfully")
        return jsonify({"error": "Model not loaded successfully"}), 500

    try:
        data = request.json
        logger.info(f"Received data: {data}")
        input_data = pd.DataFrame([data])
        
        # Version mismatch handling
        if hasattr(model, 'monotonic_cst_'):
            model.monotonic_cst = model.monotonic_cst_
        
        predictions = model.predict(input_data)
        
        # Ensure predictions is a 2D array
        if predictions.ndim == 1:
            predictions = predictions.reshape(1, -1)
        
        result = {
            "DMD": round(predictions[0][0], 2),
            "OMD": round(predictions[0][1], 2),
            "ME": round(predictions[0][2], 2),
            "CH4": round(predictions[0][3], 2)
        }
        
        logger.info(f"Prediction result: {result}")
        return jsonify(result), 200
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({"error": str(e)}), 400

@app.route('/test', methods=['GET'])
def test():
    return jsonify({"message": "Backend is working!", "model_loaded": model_loaded_successfully}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)