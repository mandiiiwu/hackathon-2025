import cv2
import numpy as np
from datetime import datetime
import math
from typing import Tuple, List, Optional, Deque
from collections import deque

class LineAnalyzer:
    def __init__(self, hist_size=5):
        self.pa_line = None  
        self.cr_line = None  
        self.prev_time = None
        self.curr_time = None
        self.frame_count = 0
        
        # store angular vel n acceleration history
        self.ang_vel_hist = deque(maxlen=hist_size)
        self.acc_hist = deque(maxlen=hist_size)
        self.timestamps = deque(maxlen=hist_size)
        
    @staticmethod
    def time_diff(time1, time2):
        m1, s1, ms1 = map(int, time1.split(':'))
        m2, s2, ms2 = map(int, time2.split(':'))
        return (m2 - m1) * 60 + (s2 - s1) + (ms2 - ms1) / 1000000
    
    @staticmethod
    def calc_dist(p1, p2):
        x1, y1 = p1[0], p1[1]
        x2, y2 = p2[0], p2[1]
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
    @staticmethod
    def get_points(line1, line2):
        p1, p2 = line1[0], line1[1]
        p3, p4 = line2[0], line2[1]
        
        distances = [
            (LineAnalyzer.calc_dist(p1, p3), p2, p4),
            (LineAnalyzer.calc_dist(p1, p4), p2, p3),
            (LineAnalyzer.calc_dist(p2, p3), p1, p4),
            (LineAnalyzer.calc_dist(p2, p4), p1, p3),
        ]
        
        min_dist, point1, point2 = min(distances, key=lambda x: x[0])
        return point1, point2
    
    @staticmethod
    def get_angle(len1, len2):
        return np.arccos(1 - (len2 ** 2) / (2 * len1 ** 2))
    
    def calc_ang_vel(self):
        if self.pa_line is None or self.cr_line is None:
            return None
        
        pa_len = self.calc_dist(self.pa_line[0], self.pa_line[1])
        cr_len = self.calc_dist(self.cr_line[0], self.cr_line[1])
        avg_len = (pa_len + cr_len) / 2
        
        p1, p2 = self.get_points(self.pa_line, self.cr_line)
        base = self.calc_dist(p1, p2)
        
        d_theta = self.get_angle(avg_len, base)
        
        if self.prev_time and self.curr_time:
            delta_time = self.time_diff(self.prev_time, self.curr_time)
            if delta_time > 0:  
                angular_velocity = d_theta / delta_time
                return angular_velocity

        return None
    
    def calc_acc(self):
        if len(self.ang_vel_hist) < 2 or len(self.timestamps) < 2:
            return None
        
        cur_vel = self.ang_vel_hist[-1]
        prev_vel = self.ang_vel_hist[-2]
        cur_time = self.timestamps[-1]
        prev_time = self.timestamps[-2]
        
        d_vel = cur_vel - prev_vel
        d_time = self.time_diff(prev_time, cur_time)
        
        if d_time > 0:
            acc = d_vel / d_time
            return acc
        
        return None
    
    def get_avgs(self, window= 3):
        if len(self.ang_vel_hist) < window:
            return None, None
        
        rec_vels = list(self.ang_vel_hist)[-window:]
        avg_vel = sum(rec_vels) / len(rec_vels)
        
        avg_acc = None
        if len(self.acc_hist) >= window:
            rec_accs = list(self.acc_hist)[-window:]
            avg_acc = sum(rec_accs) / len(rec_accs)
        
        return avg_vel, avg_acc
    
    def update_frame_data(self, line_pts, timestamp):
        self.pa_line = self.cr_line
        self.prev_time = self.curr_time
        
        self.cr_line = line_pts
        self.curr_time = timestamp
        self.frame_count += 1
        
        ang_vel = self.calc_ang_vel()
        if ang_vel is not None:
            self.ang_vel_hist.append(ang_vel)
            self.timestamps.append(timestamp)
            
            acc = self.calc_acc()
            if acc is not None:
                self.acc_hist.append(acc)


class VideoProcessor:
    def __init__(self, video_path):
        self.camera = cv2.VideoCapture(video_path)
        self.line_analyzer = LineAnalyzer()
        
        if not self.camera.isOpened():
            raise ValueError("failed connection")
    
    @staticmethod
    def hsv_mask(frame):
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        lower1 = np.array([0, 30, 80])
        upper1 = np.array([10, 255, 255])
        lower2 = np.array([170, 50, 100])
        upper2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=3)
        
        return mask
    
    @staticmethod
    def skeletonizing(frame):
        thinned = cv2.ximgproc.thinning(frame)
        dilated = cv2.dilate(thinned, np.ones((2, 2), np.uint8), iterations=1)
        return dilated
    
    @staticmethod
    def line_detection(frame):
        lines = cv2.HoughLinesP(frame, 1, np.pi/180, threshold=200, 
                               minLineLength=220, maxLineGap=250)
        
        result = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        p1, p2 = None, None
        
        if lines is not None and len(lines) > 0:
            x1, y1, x2, y2 = lines[0][0]
            cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)
            p1, p2 = (x1, y1), (x2, y2)
        
        return result, p1, p2
    
    @staticmethod
    def roi(frame):
        return frame[100:450, 780:1220]
    
    @staticmethod
    def get_timestamp():
        now = datetime.now()
        return f'{now.minute}:{now.second}:{now.microsecond}'
    
    def add_text(self, frame, vel, acc):
        overlay = frame.copy()
        
        cv2.putText(overlay, f"frame #{self.line_analyzer.frame_count}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if vel is not None:
            cv2.putText(overlay, f"ang vel: {vel:.4f} rad/s", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if acc is not None:
            color = (0, 255, 0) if abs(acc) < 15 else (0, 0, 255)  
            cv2.putText(overlay, f"ang acc: {acc:.4f} rad/s^s", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return overlay
    
    def process_frame(self, frame):
        cropped = self.roi(frame)
        
        hsv = self.hsv_mask(cropped)
        skeleton = self.skeletonizing(hsv)
        overlay, p1, p2 = self.line_detection(skeleton)
        
        results = {
            'cropped': cropped,
            'hsv_mask': hsv,
            'skeleton': skeleton,
            'overlay': overlay,
            'line_points': [p1, p2] if p1 and p2 else None
        }
        
        return results
    
    def run(self):
        while True:
            ret, frame = self.camera.read()
            if not ret:
                print("Failed to get frame or end of video")
                break
            
            results = self.process_frame(frame)
            
            if results['line_points']:
                timestamp = self.get_timestamp()
                self.line_analyzer.update_frame_data(results['line_points'], timestamp)
                
                vel, acc = self.line_analyzer.get_avgs()
                
                if vel is not None:
                    accel_text = f', acc = {acc:.4f} rad/s^2' if acc is not None else ''
                    print(f'frame #{self.line_analyzer.frame_count}: '
                          f'ang vel = {vel:.4f} rad/s{accel_text}')
                
                results['overlay'] = self.add_text(
                    results['overlay'], vel, acc
                )
            
            # Display processing steps
            cv2.imshow('cropped', results['cropped'])
            cv2.imshow('hsv', results['hsv_mask'])
            cv2.imshow('lines', results['skeleton'])
            cv2.imshow('overlay', results['overlay'])
            
            if cv2.waitKey(1) != -1:
                break
        
        self.camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        processor = VideoProcessor('video2.mov')
        processor.run()
    except ValueError as e:
        print(e)
