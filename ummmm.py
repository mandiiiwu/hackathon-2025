import cv2
import numpy as np
from Servo2 import *
import time
import sys

def roi(frame):
    center_x, center_y = 368, 288
    w, h = 60, 60           

    x1 = int(center_x - w / 2)
    y1 = int(center_y - h / 2)
    x2 = int(center_x + w / 2)
    y2 = int(center_y + h / 2)

    roi = frame[y1:y2, x1:x2]
                  
    return roi


def circle_detect(frame):
    circledetect = False
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
        circledetect = True
    return circledetect


def initialize_camera():
    """Initialize camera with reduced resolution for better performance"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        sys.exit(1)
    
    # Set lower resolution for better performance on Pi
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)
    
    return cap


def main():
    # Initialize servo and camera
    print("Initializing camera...")
    cap = initialize_camera()
    
    # Allow camera to warm up
    time.sleep(2)
    
    last_shot_time = 0
    shot_cooldown = 4.0  # seconds between shots (1.5 + 2 + buffer)
    detection_start_time = 0
    target_detected = False
    shooting_delay = 1.5  # seconds to wait before shooting after detection
    
    frame_count = 0
    start_time = time.time()
    
    print("Starting target detection loop...")
    
    try:
        while True:
            current_time = time.time()
            
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Frame capture error, retrying...")
                time.sleep(0.1)
                continue
            
            frame_count += 1
            
            # Calculate FPS every 100 frames
            if frame_count % 100 == 0:
                fps = frame_count / (current_time - start_time)
                print(f"Running at {fps:.1f} FPS")
                frame_count = 0
                start_time = current_time
            
            # Process ROI
            roi_frame = roi(frame)
            circle_detected = circle_detect(roi_frame)
            
            can_shoot = (current_time - last_shot_time) > shot_cooldown
            
            if circle_detected and can_shoot and not target_detected:
                target_detected = True
                detection_start_time = current_time
                print("Target detected! Preparing to shoot...")
            
            if target_detected:
                if not circle_detected:
                    target_detected = False
                    print("Target lost - resetting")
                else:
                    time_since_detection = current_time - detection_start_time
                    
                    countdown = max(0, shooting_delay - time_since_detection)
                    if countdown > 0:
                        print(f"Shooting in {countdown:.1f}s...")
                    
                    if time_since_detection >= shooting_delay:
                        print("SHOOT!")
                        try:
                            set_servo_angle(180)
                            last_shot_time = current_time
                            target_detected = False
                            print("Shot completed. Cooldown active.")
                        except Exception as e:
                            print(f"Servo error: {e}")
            
            # Small delay to prevent excessive CPU usage
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        # Cleanup
        cap.release()
        print("Camera released")
        cv2.destroyAllWindows()  # This is safe even without display


if __name__ == "__main__":
    main()
