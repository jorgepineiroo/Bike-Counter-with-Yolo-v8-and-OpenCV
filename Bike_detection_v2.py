# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 20:17:47 2024

@author: jorge
"""

from ultralytics import YOLO
import cv2

# Video path
path = "video4.mp4"

# Save the video in a variable
cap = cv2.VideoCapture(path)

# Verify if the video opened correctly
if not cap.isOpened():
    print("Error: Cannot open video.")

model = YOLO("pesos_dataset_jorge.pt")

# Classes
classNames = ["bicycle"]

# Confidence threshold
confidence_threshold = 0.7

# Bicycle counter
bike_count = 0

# Position history for smoother tracking
history = []
history_length = 10  # number of frames to keep in history
iou_threshold = 0.15  # IoU threshold to consider same bicycle

# Function to calculate Intersection over Union
def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1) # Intersection area


    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1) # Area of first bounding box
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1) # Area of second bounding box

    iou = interArea / float(boxAArea + boxBArea - interArea) # Intersection over Union ratio
    return iou

while cap.isOpened():
    try:
        success, img = cap.read() 
        if not success:
            break               # If video doesn't open, break loop
        
        img = cv2.convertScaleAbs(img, alpha=1.15) # Slightly increase exposure to improve results
        
        results = model(img, stream=True)  # Pass each frame through bicycle detection model

        current_positions = []

        for r in results:           # Iterate over detection results collection for each video frame
            boxes = r.boxes         # Extract information from each detected object in the video
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]      # Extract coordinates of top-left and bottom-right corners of bounding box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)   # Convert to integers

                confidence = box.conf[0]             # Extract detection confidence level
                if confidence < confidence_threshold:   # Check if detection confidence is below threshold
                    continue

                cls = int(box.cls[0])      # Extract detected object class and convert to integer
                current_positions.append((x1, y1, x2, y2)) # Add detected bicycle bounding box coordinates to list

                # Draw box in video
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # Object details
                org = (x1, y1)
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)  # Draw text on video over box

        # Compare current positions with history
        for current_pos in current_positions:    # Iterate over each bicycle detected in the frame
            new_bike = True
            
            # Compare current bicycle position (current_pos) with each previous position (prev_pos) in history
            for history_frame in history:
                for prev_pos in history_frame:
                    iou = calculate_iou(current_pos, prev_pos)   # Calculate intersection of bounding boxes
                    if iou > iou_threshold:                      # If greater than threshold, won't detect a new bicycle
                        new_bike = False
                        break
                    if not new_bike:
                        break
            
            # If new_bike is still True after comparing with history, increment bicycle counter
            if new_bike:
                bike_count += 1

        # Update position history
        history.append(current_positions)
        if len(history) > history_length:
            history.pop(0)

        # Show bicycle counter
        cv2.putText(img, f"Number of bikes: {bike_count}", (15, 460), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255,0), 2)

        cv2.imshow('Video', img)

        if cv2.waitKey(1) == ord('q'):
            break

    except Exception as e:
        print(f"Error during frame processing: {e}")
        break

cap.release()
cv2.destroyAllWindows()