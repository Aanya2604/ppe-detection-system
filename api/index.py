from flask import Flask, request, jsonify, render_template_string
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Full Roboflow PPE 10/11-Class list
CLASSES = [
    "Hardhat",
    "Mask",
    "NO-Hardhat",
    "NO-Mask",
    "NO-Safety Vest",
    "Person",
    "Safety Cone",
    "Safety Vest",
    "Machinery",
    "Vehicle",
    "Worker"
]

_session = None

def get_model():
    global _session
    if _session is None:
        import onnxruntime as ort
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "best.onnx")
        if not os.path.exists(model_path):
            model_path = os.path.join(os.getcwd(), "best.onnx")
        _session = ort.InferenceSession(model_path)
    return _session

def nms(boxes, scores, iou_threshold=0.45):
    """Filter overlapping bounding boxes."""
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]

    return keep

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>PPE Safety Detector</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #f8fafc; text-align: center; padding: 40px 20px; }
    .card { background: #1e293b; max-width: 500px; margin: auto; padding: 30px; border-radius: 12px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }
    h2 { margin-top: 0; color: #38bdf8; }
    input[type=file] { margin: 20px 0; color: #94a3b8; font-size: 14px; }
    button { background: #38bdf8; color: #0f172a; border: none; font-weight: bold; padding: 12px 24px; border-radius: 6px; cursor: pointer; transition: 0.2s; }
    button:hover { background: #0284c7; color: white; }
    #preview { max-width: 100%; border-radius: 8px; margin-top: 15px; display: none; }
    #result { margin-top: 20px; text-align: left; background: #0f172a; padding: 15px; border-radius: 8px; font-size: 14px; line-height: 1.6; display: none; }
    .tag { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; margin-right: 6px; }
    .safe { background: #15803d; color: #fff; }
    .danger { background: #b91c1c; color: #fff; }
  </style>
</head>
<body>
  <div class="card">
    <h2>🦺 PPE Safety Compliance System</h2>
    <p>Upload a site image to check for safety gear compliance.</p>
    <input type="file" id="imageInput" accept="image/*" onchange="previewImg(event)">
    <br>
    <img id="preview" alt="Input Preview" />
    <br>
    <button onclick="runDetection()">Analyze PPE Safety</button>
    <div id="result"></div>
  </div>

  <script>
    function previewImg(e) {
      const img = document.getElementById('preview');
      img.src = URL.createObjectURL(e.target.files[0]);
      img.style.display = 'block';
    }

    async function runDetection() {
      const fileInput = document.getElementById('imageInput');
      const resultBox = document.getElementById('result');

      if (!fileInput.files[0]) {
        alert("Please select an image first!");
        return;
      }

      resultBox.style.display = 'block';
      resultBox.innerHTML = "<em>Analyzing image with YOLO model...</em>";

      const formData = new FormData();
      formData.append('file', fileInput.files[0]);

      try {
        const res = await fetch('/api/detect', { method: 'POST', body: formData });
        const data = await res.json();

        if (data.detections && data.detections.length > 0) {
          resultBox.innerHTML = `<strong>Total Objects Found:</strong> ${data.total_detected}<br><br>` +
            data.detections.map(d => {
              const isDanger = d.label.toLowerCase().includes('no-');
              const tagClass = isDanger ? 'danger' : 'safe';
              return `<span class="tag ${tagClass}">${d.label.toUpperCase()}</span> ${d.confidence}`;
            }).join("<br>");
        } else if (data.error) {
          resultBox.innerHTML = `<span style='color:#ef4444;'>${data.error}</span>`;
        } else {
          resultBox.innerHTML = "No PPE items detected above the confidence threshold.";
        }
      } catch (err) {
        resultBox.innerHTML = "<span style='color:#ef4444;'>Error connecting to detection API.</span>";
      }
    }
  </script>
</body>
</html>
"""

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def home(path):
    return render_template_string(HTML_PAGE)

@app.route('/api/detect', methods=['POST'])
def detect():
    try:
        session = get_model()
    except Exception as e:
        return jsonify({"error": f"Model loading error: {str(e)}"}), 500

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

    # Parse Bounding Boxes & Confidence Scores
    boxes = []
    scores = []
    class_ids = []

    for pred in outputs[0].T:
        class_scores = pred[4:]
        c_id = int(np.argmax(class_scores))
        conf = float(class_scores[c_id])
        if conf >= 0.50:
            # Convert center-x, center-y, width, height -> x1, y1, x2, y2
            cx, cy, w, h = pred[0], pred[1], pred[2], pred[3]
            boxes.append([cx - w/2, cy - h/2, cx + w/2, cy + h/2])
            scores.append(conf)
            class_ids.append(c_id)

    # Apply NMS
    detections = []
    if len(boxes) > 0:
        keep_indices = nms(np.array(boxes), np.array(scores), iou_threshold=0.45)
        for idx in keep_indices:
            cid = class_ids[idx]
            label = CLASSES[cid] if cid < len(CLASSES) else f"Object_{cid}"
            detections.append({
                "label": label,
                "confidence": f"{round(scores[idx] * 100, 1)}%"
            })

    return jsonify({
        "status": "success",
        "total_detected": len(detections),
        "detections": detections
    })

app_handler = app