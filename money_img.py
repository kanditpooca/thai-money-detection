import cv2
import torch
from ultralytics import YOLO
import numpy as np

model = YOLO("best.pt")

width = 640
height = 640

sum = 0

class_list = ['1', '1', '10', '100', '1000', '2', '20', '5', '5', '50', '500']

detection_colors = [(255,255,0),(255,255,0),(0,0,191),(0,255,0),(0,255,127),(255,0,127),(0,0,255),(255,0,0),(255,0,0),(0,204,255),(195, 235,52)]

img = cv2.imread("coins.jpg")

# resize the frame | small frame optimise the run
frame = cv2.resize(img, (width, height))

# Run YOLOv8 inference on the frame
detect_params = model.predict(source=frame,save=False,conf=0.5,imgsz=640)
print(type(detect_params[0]))

# Convert tensor array to numpy
DP = detect_params[0].cuda()
DP = DP.cpu()
DP = DP.to('cpu')
DP = DP.numpy()
print(len(DP))

if len(DP) != 0:
    for i in range(len(detect_params[0])):
        print(i)

        boxes = detect_params[0].boxes
        box = boxes[i]  # returns one box
        clsID = box.cls[0].cuda()
        clsID = clsID.cpu()
        clsID = clsID.to('cpu')
        clsID = clsID.numpy()
        
        conf = box.conf[0].cuda()
        conf = conf.cpu()
        conf = conf.to('cpu')
        conf = conf.numpy()
        
        bb = box.xyxy[0].cuda()
        bb = bb.cpu()
        bb = bb.to('cpu')
        bb = bb.numpy()

        cv2.rectangle(
            frame,
            (int(bb[0]), int(bb[1])),
            (int(bb[2]), int(bb[3])),
            detection_colors[int(clsID)],
            3,
        )

        # Display class name and confidence
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(
            frame,
            class_list[int(clsID)] + " " + str(np.round(conf*100, 1)) + "%",
            (int(bb[0]), int(bb[1]) - 10),
            font,
            1,
            detection_colors[int(clsID)],
            2,
        )

        if class_list[int(clsID)] == "1":
            sum += 1
        if class_list[int(clsID)] == "2":
            sum += 2
        if class_list[int(clsID)] == "5":
            sum += 5
        if class_list[int(clsID)] == "10":
            sum += 10
        if class_list[int(clsID)] == "20":
            sum += 20
        if class_list[int(clsID)] == "50":
            sum += 50
        if class_list[int(clsID)] == "100":
            sum += 100
        if class_list[int(clsID)] == "500":
            sum += 500
        if class_list[int(clsID)] == "1000":
            sum += 1000

cv2.putText(frame, f'Total Sum: {sum}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Display the resulting frame
cv2.imshow("What brand", frame)

print()

# Break the loop if 'q' is pressed
if cv2.waitKey(0) & 0xFF == ord("q"):
    cv2.destroyAllWindows()