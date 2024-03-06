from flask import Flask, render_template, Response
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLO model
yolo_model = YOLO("../../Library/Application Support/JetBrains/PyCharm2023.3/scratches/yolo-Weights/yolov8n.pt")

# Define object classes and colors
object_classes = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat',
                  'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
                  'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
                  'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                  'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                  'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
                  'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed',
                  'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                  'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                  'teddy bear', 'hair drier', 'toothbrush']
colors = np.random.uniform(0, 255, size=(len(object_classes), 3))

def detect_objects(frame):
    """Detects objects in the frame and returns the frame with bounding boxes."""
    detection_results = yolo_model(frame, stream=True)
    for idx, result in enumerate(detection_results):
        for bbox in result.boxes:
            # Unpack bounding box coordinates and class ID
            left, top, right, bottom = map(int, bbox.xyxy[0])
            class_id = int(bbox.cls[0])
            # Draw the bounding box and label
            color = colors[class_id]
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            label = f"{object_classes[class_id]}: {bbox.conf[0]:.2f}"
            cv2.putText(frame, label, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
    return frame

def generate_frames():
    video_feed = cv2.VideoCapture(0)  # Open webcam feed
    while True:
        success, frame = video_feed.read()
        if not success:
            break
        else:
            frame_with_objects = detect_objects(frame)
            ret, buffer = cv2.imencode('.jpg', frame_with_objects)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # Concat frame one by one and show result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
