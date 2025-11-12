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

    lower1 = np.array([0, 30, 80])
    upper1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower1, upper1)

    lower2 = np.array([170, 50, 100])
    upper2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower2, upper2)

    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=3)  

    # masked = cv2.bitwise_and(frame, frame, mask=mask)

    return mask

def skeletonizing(frame):
    thinned = cv2.ximgproc.thinning(frame)
    dilated = cv2.dilate(thinned, np.ones((2, 2), np.uint8), iterations=1)
    return dilated

def line_detection(frame):
    lines = cv2.HoughLinesP(frame, 1, np.pi/180, threshold=200, minLineLength=220, maxLineGap=250)
    result = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    count = 0
    if lines is not None:
        for point in lines:
            if count == 1: break
            x1, y1, x2, y2 = point[0]
            cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2) 
            count += 1
    # else:
        
    return result, (x1, y1), (x2, y2)

def roi(frame):
    copy = frame.copy() 
    cropped = copy[100:450, 780:1220]
    return cropped

camera = cv2.VideoCapture('video2.mov')

if not camera.isOpened():
    print("failed to connect to camera")
    exit()

frame_count = 0
while True:
    ret, frame = camera.read()

    if not ret:
        print("failed to get frame")
        break

    # cv2.imshow('vid stream', frame)
    # print(frame.shape)

    cropped = roi(frame)
    cv2.imshow('cropped', cropped)

    hsv = hsv_mask(cropped)
    cv2.imshow('hsv', hsv)

    lines = skeletonizing(hsv)
    cv2.imshow('lines', lines)

    overlay, p1, p2 = line_detection(lines)
    cv2.imshow('overlay', overlay)

    frame_count += 1
    print(f'{frame_count} frames')

    if cv2.waitKey(1) != -1:
        break

camera.release()
cv2.destroyAllWindows()
