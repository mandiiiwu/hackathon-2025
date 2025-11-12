import cv2
import numpy as np
from Servo2 import *

def roi(frame):
    center_x, center_y = 368, 288
    w, h = 60, 60           

    x1 = int(center_x - w / 2)
    y1 = int(center_y - h / 2)
    x2 = int(center_x + w / 2)
    y2 = int(center_y + h / 2)

    roi = frame[y1:y2, x1:x2]

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                  
    return roi


def circle_detect(frame):
    circledetect = False
    #loading image
    (h, w) = frame.shape[:2]
    new_width = 800
    aspect_ratio = h / w
    new_height = int(new_width * aspect_ratio)
    frame = cv2.resize(frame, (new_width, new_height))
    img = frame.copy()

    img = cv2.medianBlur(img, 11) 
    #only green 
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([30, 40, 40])    # adjust as needed
    upper_green = np.array([100, 255, 255])
    
    mask = cv2.inRange(hsv, lower_green, upper_green) 

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)  
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)  
    mask = cv2.medianBlur(mask, 5)
    #circle detection
    circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 1.2, 200, param1=500, param2=15, minRadius=10, maxRadius=70)
    if circles is not None:
    #convert x y coords into ints
        circles = np.round(circles[0, :]).astype("int")
    #looping over coords n radius
        for (x, y, r) in circles:
            #drawing circle
            cv2.circle(frame, (x, y), r, (200, 50, 500), 2)
            #drawing center
            cv2.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1) 
        circledetect = True
    return frame, circledetect
    

cap = cv2.VideoCapture(0)
# checking if camera is opened
if (cap.isOpened() == False):
    print("error")

while (cap.isOpened()):
    # capture frame
    ret, frame = cap.read()
    if ret == True:
    
        h, w = frame.shape[:2]
        aspect_ratio = w / h
        width = 780
        new_height = int(width / aspect_ratio)
        frame = cv2.resize(frame, (width, new_height), interpolation=cv2.INTER_AREA)

        #overlay
        roiim = roi(frame)
        circleim, _ = circle_detect(frame)
        _, circle = circle_detect(roiim)
        if circle:
            time.sleep(0.5)
            print("SHOOT")
            set_servo_angle(180)
            time.sleep(2)

        # exiting
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    else:
        print("vid finished")
        break

cap.release()
cv2.destroyAllWindows()
