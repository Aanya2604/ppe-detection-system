from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import onnxruntime as ort
import os

app = Flask(__name__)

# Load ONNX model
model_path = os.path.join(os.path.dirname(__file__), "..", "best.onnx")
session = ort.InferenceSession(model_path)

CLASSES = ["helmet", "mask", "person", "vest"]

@app.route('/api/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    img = Image.open(file.stream).convert('RGB').resize((640, 640))
    
    # Normalize and transform to NCHW
    img_data = np.array(img).astype('float32') / 255.0
    img_data = np.transpose(img_data, (2, 0, 1))
    img_data = np.expand_dims(img_data, axis=0)

    # Run inference
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img_data})[0]

    # Filter detections
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