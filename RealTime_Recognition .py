import cv2
import numpy as np
import torch
from threading import Thread
import time
import pandas as pd
from shapely.geometry import Point, Polygon
import pyttsx3

# Initializing auditory system
engine = pyttsx3.init()
engine.setProperty('rate', 150) 
engine.setProperty('volume', 5)  

# Set english
voices = engine.getProperty('voices')
for voice in voices:
    if voice.id == 'HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_EN-US_DAVID_11.0':
        engine.setProperty('voice', voice.id)
        break

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=r'C:\Python_projects\Real_time_Apparatus\Trained_models\all_add_655.pt')

def balance_white(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

class VideoStream:
    def __init__(self, src):
        self.stream = cv2.VideoCapture(src)
        self.stopped = False
        self.grabbed, self.frame = self.stream.read()
        self.thread = Thread(target=self.update, args=())
        self.thread.start()

    def update(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                self.grabbed, self.frame = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.thread.join()

# RTSP connection
rtsp_url = f"rtsp://admin:oshimalab2024@192.168.11.3:64580/ipcam_profile_1.sdp"
cap = VideoStream(rtsp_url)
current_zones = {}
zone_polygons = {}
zone_counter = 1
zone_timeout = {}
fixed_zones = False
fixed_coordinates = {}
fixed_polygons = {}
last_update_time = time.time()
update_interval = 3  # 3s
last_detections = []
stop_speech = False
engine_running = False

def nothing(x):
    pass

cv2.namedWindow('Mask')
cv2.createTrackbar('H Lower', 'Mask', 0, 179, nothing)
cv2.createTrackbar('S Lower', 'Mask', 0, 255, nothing)
cv2.createTrackbar('V Lower', 'Mask', 0, 255, nothing)
cv2.createTrackbar('H Upper', 'Mask', 0, 179, nothing)
cv2.createTrackbar('S Upper', 'Mask', 0, 255, nothing)
cv2.createTrackbar('V Upper', 'Mask', 0, 255, nothing)

# HSV thresholds in 425
cv2.setTrackbarPos('H Lower', 'Mask', 33)
cv2.setTrackbarPos('S Lower', 'Mask', 20)
cv2.setTrackbarPos('V Lower', 'Mask', 150)
cv2.setTrackbarPos('H Upper', 'Mask', 70)
cv2.setTrackbarPos('S Upper', 'Mask', 150)
cv2.setTrackbarPos('V Upper', 'Mask', 255)

def point_in_polygon(point, polygon):
    return polygon.contains(Point(point))

def assign_fixed_zone_ids():
    global fixed_coordinates, fixed_polygons
    if len(fixed_coordinates) != 6:
        return

    # Obtain center points in pictures
    zone_centers = list(fixed_coordinates.values())
    
    # sort to 2 lines
    zone_centers.sort(key=lambda x: x[1])
    row1 = sorted(zone_centers[:3], key=lambda x: x[0], reverse=True)  # 第一排，从右到左
    row2 = sorted(zone_centers[3:], key=lambda x: x[0], reverse=True)  # 第二排，从右到左

    # Renumber zones
    new_fixed_coordinates = {}
    new_polygons = {}
    new_fixed_coordinates[1] = row1[0]
    new_polygons[1] = fixed_polygons[next(k for k, v in fixed_coordinates.items() if v == row1[0])]
    new_fixed_coordinates[6] = row1[1]
    new_polygons[6] = fixed_polygons[next(k for k, v in fixed_coordinates.items() if v == row1[1])]
    new_fixed_coordinates[5] = row1[2]
    new_polygons[5] = fixed_polygons[next(k for k, v in fixed_coordinates.items() if v == row1[2])]
    new_fixed_coordinates[2] = row2[0]
    new_polygons[2] = fixed_polygons[next(k for k, v in fixed_coordinates.items() if v == row2[0])]
    new_fixed_coordinates[3] = row2[1]
    new_polygons[3] = fixed_polygons[next(k for k, v in fixed_coordinates.items() if v == row2[1])]
    new_fixed_coordinates[4] = row2[2]
    new_polygons[4] = fixed_polygons[next(k for k, v in fixed_coordinates.items() if v == row2[2])]

    fixed_coordinates = new_fixed_coordinates
    fixed_polygons = new_polygons

# NMS
def nms(boxes, scores, threshold):
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.5, nms_threshold=threshold)
    return indices

# Exception
excluded_classes = ['glassrod']

while True:
    frame = cap.read()
    if frame is None:
        break

    original_frame = frame.copy()  
    processed_frame = balance_white(frame.copy())  

    # 获取HSV阈值
    h_lower = cv2.getTrackbarPos('H Lower', 'Mask')
    s_lower = cv2.getTrackbarPos('S Lower', 'Mask')
    v_lower = cv2.getTrackbarPos('V Lower', 'Mask')
    h_upper = cv2.getTrackbarPos('H Upper', 'Mask')
    s_upper = cv2.getTrackbarPos('S Upper', 'Mask')
    v_upper = cv2.getTrackbarPos('V Upper', 'Mask')

    lower_hsv = np.array([h_lower, s_lower, v_lower])
    upper_hsv = np.array([h_upper, s_upper, v_upper])

    # Object detection by YOLOv5 model
    results = model(original_frame)
    detections = results.pandas().xyxy[0]

    selected_detections = []
    for index, detection in detections.iterrows():
        if detection['name'] in excluded_classes:
            next_best = detections[(detections['xmin'] == detection['xmin']) &
                                   (detections['ymin'] == detection['ymin']) &
                                   (detections['xmax'] == detection['xmax']) &
                                   (detections['ymax'] == detection['ymax']) &
                                   (~detections['name'].isin(excluded_classes))]
            if not next_best.empty:
                next_best = next_best.sort_values(by='confidence', ascending=False).iloc[0]
                selected_detections.append(next_best)
        else:
            selected_detections.append(detection)

    # NMS application
    if selected_detections:
        filtered_detections = pd.DataFrame(selected_detections)
        boxes = filtered_detections[['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()
        scores = filtered_detections['confidence'].values.tolist()
        indices = nms(boxes, scores, 0.4)

        if len(indices) > 0:
            detections_for_broadcast = []
            for i in indices.flatten():
                x1, y1, x2, y2, conf, cls, name = filtered_detections.iloc[i]
                cx = (int(x1) + int(x2)) // 2
                cy = (int(y1) + int(y2)) // 2
                cv2.rectangle(original_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(original_frame, name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Judging the zones of apparatus
                for zone_id, polygon in fixed_polygons.items():
                    if point_in_polygon((cx, cy), polygon):
                        zone_text = f'{name} is in Zone {zone_id}'
                        cv2.putText(original_frame, zone_text, (int(x1), int(y1) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        detections_for_broadcast.append(zone_text)
                        break
            last_detections = detections_for_broadcast

    if not fixed_zones:
        hsv = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(original_frame, contours, -1, (255, 200, 100), 2)

        new_zones = {}
        new_polygons = {}
        for contour in contours:
            if 5000 < cv2.contourArea(contour) < 400000:
                approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
                M = cv2.moments(approx)
                if M["m00"] != 0:
                    rx = int(M["m10"] / M["m00"])
                    ry = int(M["m01"] / M["m00"])
                    matched_id = None
                    for zone_id, (px, py) in current_zones.items():
                        if np.sqrt((rx - px)**2 + (ry - py)**2) < 50:
                            matched_id = zone_id
                            zone_timeout[zone_id] = 0
                            break
                    if matched_id is None:
                        matched_id = zone_counter
                        zone_counter += 1
                        zone_timeout[matched_id] = 0
                    if matched_id is not None:
                        new_zones[matched_id] = (rx, ry)
                        new_polygons[matched_id] = Polygon([tuple(pt[0]) for pt in approx])
                        cv2.drawContours(original_frame, [approx], -1, (0, 255, 0), 2)
                        cv2.circle(original_frame, (rx, ry), 10, (0, 0, 255), -1)
                        cv2.putText(original_frame, f'Zone {matched_id}', (rx, ry - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        for zone_id in list(current_zones.keys()):
            if zone_id not in new_zones:
                zone_timeout[zone_id] += 1
                if zone_timeout[zone_id] > 99999:
                    zone_timeout.pop(zone_id)
                else:
                    new_zones[zone_id] = current_zones[zone_id]
                    new_polygons[zone_id] = zone_polygons[zone_id]

        current_zones = new_zones
        zone_polygons = new_polygons

        if len(current_zones) == 6:
            fixed_coordinates = current_zones.copy()
            fixed_polygons = zone_polygons.copy()
            assign_fixed_zone_ids()

    else:
        for zone_id, (rx, ry) in fixed_coordinates.items():
            cv2.putText(original_frame, f'Zone {zone_id}', (rx, ry - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.circle(original_frame, (rx, ry), 10, (0, 0, 255), -1)

    # Show the numbers
    y_offset = 20
    for zone_id, (rx, ry) in current_zones.items():
        text = f'Zone {zone_id}: ({rx}, {ry})'
        cv2.putText(original_frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 20

    # 每隔update_interval秒重新分配区域编号 Keep on updating zone numbers
    if time.time() - last_update_time > update_interval and len(fixed_coordinates) == 6:
        assign_fixed_zone_ids()
        last_update_time = time.time()

    cv2.imshow('Combined Detections', original_frame)  # show results in same time
    cv2.imshow('Mask', mask)  # show the mask for HSV thresholds

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('f'):
        fixed_zones = not fixed_zones
        if fixed_zones:
            assign_fixed_zone_ids()
    elif key == ord('s'):
        # broadcast the Information
        engine_running = True
        for detection in last_detections:
            if stop_speech:
                stop_speech = False
                break
            engine.say(detection)
            engine.runAndWait()
        engine_running = False
    elif key == ord('c'):
        # Shut down the broadcasting
        stop_speech = True
        if engine_running:
            engine.stop()

cap.stop()
cv2.destroyAllWindows()
