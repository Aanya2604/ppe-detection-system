from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import numpy as np
import onnxruntime as ort
import os

app = Flask(__name__)

# Paths
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "best.onnx")
public_dir = os.path.join(base_dir, "..", "public")

# Load ONNX model
session = None
if os.path.exists(model_path):
    session = ort.InferenceSession(model_path)

CLASSES = ["helmet", "mask", "person", "vest"]

# Root route: serve frontend UI
@app.route('/')
def home():
    return send_from_directory(public_dir, 'index.html')

# Catch-all static route
@app.route('/<path:path>')
def static_files(path):
    return send_from_directory(public_dir, path)

# Inference API endpoint
@app.route('/api/detect', methods=['POST'])
def detect():
    if session is None:
        return jsonify({"error": "Model not found on server"}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    img = Image.open(file.stream).convert('RGB').resize((640, 640))
    
    # Preprocess
    img_data = np.array(img).astype('float32') / 255.0
    img_data = np.transpose(img_data, (2, 0, 1))
    img_data = np.expand_dims(img_data, axis=0)

    # Inference
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img_data})[0]

    detections = []
    for pred in outputs[0].T:
        scores = pred[4:]
        class_id = int(np.argmax(scores))
        conf = float(scores[class_id])
        if conf >= 0.40:
            label = CLASSES[class_id] if class_id < len(CLASSES) else f"Class_{class_id}"
            detections.append({
                "label": label,
                "confidence": f"{round(conf * 100, 1)}%"
            })

    return jsonify({
        "status": "success",
        "total_detected": len(detections),
        "detections": detections
    })

if __name__ == '__main__':
    app.run(debug=True)