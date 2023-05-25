import cv2
import torch
from ultralytics import YOLO
import numpy as np

model = YOLO("best.pt")

width = 640
height = 640

sum = 0
font = cv2.FONT_HERSHEY_DUPLEX

class_list = ['1','1','10', '100', '1000', '2', '20', '5','5','50', '500']

detection_colors = [(255,255,0),(255,255,0),(0,0,191),(0,255,0),(0,255,127),(255,0,127),(0,0,255),(255,0,0),(255,0,0),(0,204,255),(195, 235,52),]

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera or file")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # resize the frame | small frame optimise the run
    frame = cv2.resize(frame, (width, height))

    # Run YOLOv8 inference on the frame
    detect_params = model.predict(source=frame,save=False,conf=0.7)

    # Convert tensor array to numpy
    DP = detect_params[0].cuda()
    DP = DP.cpu()
    DP = DP.to('cpu')
    DP = DP.numpy()

    if (len(DP) != 0):
        for i in range(len(detect_params[0])):

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

            all_cls = boxes.cls.cuda()
            all_cls = all_cls.cpu()
            all_cls = all_cls.to('cpu')
            all_cls = all_cls.numpy()

            cv2.rectangle(
                frame,
                (int(bb[0]), int(bb[1])),
                (int(bb[2]), int(bb[3])),
                detection_colors[int(clsID)],
                3,
            )

            # Display class name and confidence
            cv2.putText(
                frame,
                class_list[int(clsID)] + " " + str(np.round(conf*100, 1)) + "%",
                (int(bb[0]), int(bb[1]) - 10),
                font,
                1,
                detection_colors[int(clsID)],
                2,
            )

        for i in range(len(all_cls)):
            if all_cls[i] == 0:
                sum += 1
            elif all_cls[i] == 1:
                sum += 1
            elif all_cls[i] == 2:
                sum += 10
            elif all_cls[i] == 3:
                sum += 100
            elif all_cls[i] == 4:
                sum += 1000
            elif all_cls[i] == 5:
                sum += 2
            elif all_cls[i] == 6:
                sum += 20
            elif all_cls[i] == 7:
                sum += 5
            elif all_cls[i] == 8:
                sum += 5
            elif all_cls[i] == 9:
                sum += 50
            elif all_cls[i] == 10:
                sum += 500
    
    cv2.putText(frame, f'Total Sum: {sum}', (10, 30), font, 0.9, (0, 255, 0), 2)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(5) & 0xFF == 27:
        break

    # Display the resulting frame
    sum = 0
    cv2.imshow("1", frame)

cap.release()
cv2.destroyAllWindows()