import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import *

model = YOLO('yolov8s.pt')

# Global variables for ROI selection
roi_points = []
selected_areas = []
selecting = True

def select_roi(event, x, y, flags, param):
    global roi_points, selecting

    if event == cv2.EVENT_LBUTTONDOWN and selecting:
        roi_points.append((x, y))
        if len(roi_points) == 4:
            selected_areas.append(roi_points.copy())
            roi_points.clear()
            print(f"Area {len(selected_areas)} selected: {selected_areas[-1]}")
            if len(selected_areas) == 2:
                selecting = False  # Stop selection after two areas

cap = cv2.VideoCapture('video_testing.mp4')
cv2.namedWindow("Select ROI")
cv2.setMouseCallback("Select ROI", select_roi)

# ROI Selection
while selecting:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (1020, 500))
    
    # Draw selected points
    for point in roi_points:
        cv2.circle(frame, point, 5, (0, 0, 255), -1)
    
    cv2.imshow("Select ROI", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyWindow("Select ROI")

if len(selected_areas) < 2:
    print("Not enough areas selected. Exiting.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

area1, area2 = selected_areas

# Object tracking
tracker = Tracker()
people_entering = {}
entering = set()
people_exiting = {}
exiting = set()
count = 0

my_file = open("coco.txt", "r")
class_list = my_file.read().split("\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 2 != 0:
        continue
    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    
    list = []
    for index, row in px.iterrows():
        x1, y1, x2, y2, d = map(int, row[:5])
        c = class_list[d]
        if 'person' in c:
            list.append([x1, y1, x2, y2])
    
    bbox_id = tracker.update(list)
    
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        results = cv2.pointPolygonTest(np.array(area2, np.int32), (x4, y4), False)
        if results >= 0:
            people_entering[id] = (x4, y4)
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)

        if id in people_entering:
            results1 = cv2.pointPolygonTest(np.array(area1, np.int32), (x4, y4), False)
            if results1 >= 0:
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
                cv2.circle(frame, (x4, y4), 5, (255, 0, 255), -1)
                cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                entering.add(id)

        results2 = cv2.pointPolygonTest(np.array(area1, np.int32), (x4, y4), False)
        if results2 >= 0:
            people_exiting[id] = (x4, y4)
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)

        if id in people_exiting:
            results3 = cv2.pointPolygonTest(np.array(area2, np.int32), (x4, y4), False)
            if results3 >= 0:
                cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 255), 2)
                cv2.circle(frame, (x4, y4), 5, (255, 0, 255), -1)
                cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                exiting.add(id)

    cv2.polylines(frame, [np.array(area1, np.int32)], True, (255, 0, 0), 2)
    cv2.putText(frame, str('1'), area1[0], cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)

    cv2.polylines(frame, [np.array(area2, np.int32)], True, (255, 0, 0), 2)
    cv2.putText(frame, str('2'), area2[0], cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)

    i, o = len(entering), len(exiting)
    cv2.putText(frame, f"Entering: {i}", (60, 80), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f"Exiting: {o}", (60, 140), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 255), 2)

    cv2.imshow("People Counting", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
