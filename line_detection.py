import cv2
import numpy as np
from datetime import datetime
import math  

def time_diff(time1, time2):
    # format: minute:second:microsecond
    m1, s1, ms1 = map(int, time1.split(':'))
    m2, s2, ms2 = map(int, time2.split(':'))

    return (m2-m1)*60 + (s2-s1) + (ms2-ms1)/1000000 # returns time in seconds 

def calc_dist(p1, p2): 
    x1, y1 = p1[0], p1[1] 
    x2, y2 = p2[0], p2[1]
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)
    
def get_points(line1, line2):
    p1, p2 = line1[0], line1[1]
    p3, p4 = line2[0], line2[1]  
    
    distances = [
        (calc_dist(p1, p3), p2, p4),  # if p1-p3 shortest, return p2 + p4
        (calc_dist(p1, p4), p2, p3),  # if p1-p4 shortest, return p2 + p3
        (calc_dist(p2, p3), p1, p4),  # if p2-p3 shortest, return p1 + p4
        (calc_dist(p2, p4), p1, p3),  # if p2-p4 shortest, return p1 + p3
    ]
    
    min_dist, point1, point2 = min(distances, key=lambda x: x[0])
    return point1, point2

def get_angle(len1, len2):
    # len1 is the length of the isoceles sides
    # len2 is the length of the dist btwn the two endpoints 
    return np.arccos(1-(len2**2)/(2*len1**2))

def hsv_mask(frame):
    # only look for bright red
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    lower1 = np.array([0, 120, 100])
    upper1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower1, upper1)

    lower2 = np.array([170, 120, 100])
    upper2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower2, upper2)

    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=3)  

    # masked = cv2.bitwise_and(frame, frame, mask=mask)

    return mask

def line_detection(frame):
    thinned = cv2.ximgproc.thinning(frame)
    lines = cv2.HoughLines(thinned, 1, np.pi / 180, 150)
    result = cv2.cvtColor(thinned, cv2.COLOR_GRAY2BGR)
    if lines is not None:
        for r_theta in lines:
            arr = np.array(r_theta[0], dtype=np.float64)
            r, theta = arr
            a = np.cos(theta)
            b = np.sin(theta)

            x0 = a*r
            y0 = b*r 

            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))

            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*a)

            cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 3)
    return result 

def circle_detect(frame):
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
    return frame

camera = cv2.VideoCapture('video.mp4')

if not camera.isOpened():
    print("failed to connect to camera")
    exit()

while True:
    ret, frame = camera.read()

    if not ret:
        print("failed to get frame")
        break

    #cv2.imshow('vid stream', frame)

    circle = circle_detect(frame)
    cv2.imshow('Video Circle Detection Frame!', circle)

    hsv = hsv_mask(frame)
    cv2.imshow('hsv', hsv)

    thinned = cv2.ximgproc.thinning(hsv)
    cv2.imshow('thinned', thinned)

    overlay = line_detection(hsv)
    cv2.imshow('overlay', overlay)


    if cv2.waitKey(1) != -1:
        break

camera.release()
cv2.destroyAllWindows()
