from flask import Flask, render_template, Response, request, jsonify
import cv2
import os
import numpy as np
import pandas as pd
from ultralytics import YOLO
from tracker import Tracker

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load YOLO model
model = YOLO("yolov8s.pt")

# Global Variables
tracker = Tracker()
selected_areas = []
people_entering = set()
people_exiting = set()
cap = None
video_path = None

def get_first_frame():
    """ Extracts and returns the first frame for ROI selection. """
    global cap, video_path
    if not video_path:
        return None
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if ret:
        _, buffer = cv2.imencode(".jpg", frame)
        return buffer.tobytes()
    return None

def process_video():
    """ Process video and track people based on selected ROIs. """
    global people_entering, people_exiting, cap, selected_areas, video_path

    if not video_path or len(selected_areas) != 2:
        return None

    people_entering.clear()
    people_exiting.clear()

    cap = cv2.VideoCapture(video_path)
    
    # Convert selected areas to NumPy arrays
    area1 = np.array(selected_areas[0], dtype=np.int32)
    area2 = np.array(selected_areas[1], dtype=np.int32)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1020, 500))
        results = model.predict(frame)
        detections = results[0].boxes.data
        px = pd.DataFrame(detections).astype("float")

        person_list = []
        for _, row in px.iterrows():
            x1, y1, x2, y2, d = map(int, row[:5])
            
            # Validate COCO class index
            try:
                with open("coco.txt", "r") as f:
                    coco_classes = f.read().splitlines()
                if d < len(coco_classes) and coco_classes[d] == "person":
                    person_list.append([x1, y1, x2, y2])
            except Exception as e:
                print(f"Error reading coco.txt: {e}")
                continue

        bbox_id = tracker.update(person_list)

        for bbox in bbox_id:
            x3, y3, x4, y4, id = bbox

            # Check if the person enters area 2 first
            if cv2.pointPolygonTest(area2, (x4, y4), False) >= 0:
                people_entering.add(id)

            # Check if the person moves into area 1 after area 2
            if id in people_entering and cv2.pointPolygonTest(area1, (x4, y4), False) >= 0:
                people_exiting.add(id)

        # Draw ROIs
        cv2.polylines(frame, [area1], isClosed=True, color=(255, 0, 0), thickness=2)
        cv2.polylines(frame, [area2], isClosed=True, color=(0, 255, 0), thickness=2)

        # Display count
        cv2.putText(frame, f"Entering: {len(people_entering)}", (60, 80), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Exiting: {len(people_exiting)}", (60, 140), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 255), 2)

        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

    cap.release()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_video():
    global video_path
    if "file" not in request.files:
        return "No file uploaded", 400
    file = request.files["file"]
    if file.filename == "":
        return "No selected file", 400

    video_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(video_path)
    return jsonify({"message": "File uploaded successfully"})


@app.route("/first_frame")
def first_frame():
    global cap, video_path
    if not video_path:
        return jsonify({"error": "No video uploaded"}), 400

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if ret:
        original_height, original_width = frame.shape[:2]  # Get actual video size
        _, buffer = cv2.imencode(".jpg", frame)
        return jsonify({
            "frame": buffer.tobytes().decode("latin1"), 
            "width": original_width,
            "height": original_height
        })

    return jsonify({"error": "Failed to capture frame"}), 400


@app.route("/set_roi", methods=["POST"])
def set_roi():
    global selected_areas
    data = request.json
    selected_areas = data.get("areas", [])

    if len(selected_areas) != 2 or any(len(area) != 4 for area in selected_areas):
        return jsonify({"error": "Invalid ROI data"}), 400

    return jsonify({"message": "ROI set successfully"})

@app.route("/video_feed")
def video_feed():
    return Response(process_video(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/count")
def get_count():
    return {"entered": len(people_entering), "exited": len(people_exiting)}

if __name__ == "__main__":
    app.run(debug=True)
