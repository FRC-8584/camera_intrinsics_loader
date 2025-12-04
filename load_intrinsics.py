import cv2
import numpy as np
import time
import sys
from tqdm import tqdm

CHECKERBOARD = (10, 7)
SQUARE_SIZE = 2.5 # centimeter
CAPTURE_PER_SECOND = 1.0
TARGET_CALIBRATION_IMAGES = 40

objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

obj_points = []
img_points = []

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open camera. Check device ID.")
    sys.exit()

print(f"STARTING CAMERA CALIBRATION DATA COLLECTION")
print(f"Goal: Collect {TARGET_CALIBRATION_IMAGES} good calibration images.")
print(f"Please move the checkerboard in front of the lens to capture from different angles.")

last_capture_time = time.time()
frame_size = None
pbar = tqdm(total=TARGET_CALIBRATION_IMAGES, desc="Collecting Data")

while len(obj_points) < TARGET_CALIBRATION_IMAGES:
    ret, frame = cap.read()
    
    if not ret:
        time.sleep(0.1)
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    if frame_size is None:
        frame_size = gray.shape[::-1]

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        current_time = time.time()
        
        if current_time - last_capture_time > 1.0 / CAPTURE_PER_SECOND:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), 
                                        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            
            obj_points.append(objp)
            img_points.append(corners2)
            last_capture_time = current_time
            
            pbar.update(1)
            pbar.write(f"Capture successful! Collected {len(obj_points)}/{TARGET_CALIBRATION_IMAGES} images")
            
    cv2.waitKey(1)

cap.release()
pbar.close()

if len(obj_points) > 0 and frame_size is not None:
    print("\nSTARTING CAMERA CALIBRATION")
    
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points,
        img_points,
        frame_size,
        None,
        None
    )

    if ret:
        print(f"\nCamera calibration complete! Based on {len(obj_points)} images.")
        print("Camera intrinsic matrix K =\n", K)
        print("Distortion coefficients dist =\n", dist)
        
        mean_error = 0
        for i in range(len(obj_points)):
            imgpoints2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], K, dist)
            error = cv2.norm(img_points[i], imgpoints2, cv2.NORM_L2) / len(img_points[i])
            mean_error += error
        print(f"Average re-projection error: {mean_error/len(obj_points)} pixels")
        
    else:
        print("ERROR: Calibration failed. Insufficient or poor quality images collected.")

else:
    print("\nERROR: Not enough good images collected to perform calibration.")