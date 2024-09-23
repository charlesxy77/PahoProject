from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import pickle
import os
#from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import logging

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


model = None

def load_model():
    global model
    model_path = os.path.join(os.path.dirname(__file__), 'Livestock_Random_Forest_Model (1).pkl')
    print(f"Attempting to load model from: {model_path}")
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded successfully. Type: {type(model)}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 204
    
    if model is None:
        logger.error("Model not loaded")
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.json
        print(f"Received data: {data}")
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
        
        print(f"Prediction result: {result}")
        logger.info(f"Prediction result: {result}")
        return jsonify(result), 200
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({"error": str(e)}), 400

@app.route('/test', methods=['GET'])
def test():
    return jsonify({"message": "Backend is working!", "model_loaded": model is not None}), 200


# Serve React App
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')


if __name__ == '__main__':
    load_model()
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)