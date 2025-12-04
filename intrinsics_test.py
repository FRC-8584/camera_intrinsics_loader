import cv2 as cv
import numpy as np
import pupil_apriltags as aptags
import time

import json

# select your camera profile in "camera_intrinsics.json"
TARGET_CAMERA = "PW_313"

with open("camera_intrinsics.json", mode="rb") as f:
    _json = json.load(f)
    data = _json[TARGET_CAMERA]

    K_LIST = data["K"]
    DISTORTION_LIST = data["distortion"]

K = np.array(K_LIST)
DISTORTION = np.array(DISTORTION_LIST)

detector = aptags.Detector(
    families='tag36h11',
    nthreads=1,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.5,
    debug=0
)

TAG_SIZE = 0.08

CAP_INDEX = 0

cap = cv.VideoCapture(CAP_INDEX)

if not cap.isOpened():
    print(f"Start Failed: Can't open camera. (Index: {CAP_INDEX})")
    exit()

print("\nStart!")
print("Press 'q' to exit")
print("Waiting for frames stream......")

prev_time = time.time()
frame_count = 0
fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't get frame from camera (Exit?)")
        break
        
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    results = detector.detect(
        gray,
        camera_params=[K[0, 0], K[1, 1], K[0, 2], K[1, 2]],
        tag_size=TAG_SIZE,
        estimate_tag_pose=True
    )

    for r in results:
        (ptA, ptB, ptC, ptD) = r.corners.astype("int")
        cv.line(frame, tuple(ptA), tuple(ptB), (0, 255, 0), 2)
        cv.line(frame, tuple(ptB), tuple(ptC), (0, 255, 0), 2)
        cv.line(frame, tuple(ptC), tuple(ptD), (0, 255, 0), 2)
        cv.line(frame, tuple(ptD), tuple(ptA), (0, 255, 0), 2)

        (cX, cY) = r.center.astype("int")
        cv.circle(frame, (cX, cY), 5, (0, 0, 255), -1)

        tag_id = f"ID: {r.tag_id}"
        cv.putText(frame, tag_id, (cX, cY - 15),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if r.pose_R is not None and r.pose_t is not None:
            t_x = r.pose_t[0, 0] * 100
            t_y = r.pose_t[1, 0] * 100
            t_z = r.pose_t[2, 0] * 100
    
            pose_text = f"X:{t_x:.2f}cm Y:{t_y:.2f}cm Z:{t_z:.2f}cm"
            cv.putText(frame, pose_text, (cX, cY + 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
            def rotationMatrixToEulerAngles(R):
                sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
                singular = sy < 1e-6

                if not singular:
                    roll  = np.arctan2(R[2,1], R[2,2])   # X
                    pitch = np.arctan2(-R[2,0], sy)      # Y
                    yaw   = np.arctan2(R[1,0], R[0,0])   # Z
                else:
                    roll  = np.arctan2(-R[1,2], R[1,1])
                    pitch = np.arctan2(-R[2,0], sy)
                    yaw   = 0

                return np.degrees([yaw, pitch, roll])

            yaw, pitch, roll = rotationMatrixToEulerAngles(r.pose_R)

            euler_text = f"YAW:{yaw:.1f} PITCH:{pitch:.1f} ROLL:{roll:.1f}"
            cv.putText(frame, euler_text, (cX, cY + 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            rvec, _ = cv.Rodrigues(r.pose_R)
            tvec = r.pose_t
    
            axis = np.float32([[TAG_SIZE/2, 0, 0], [0, TAG_SIZE/2, 0], [0, 0, TAG_SIZE/2], [0, 0, 0]]).reshape(-1, 3)
    
        imgpts, jac = cv.projectPoints(axis, rvec, tvec, K, DISTORTION)
        imgpts = imgpts.astype(int)

        center = tuple(imgpts[3].ravel())
    
        frame = cv.line(frame, center, tuple(imgpts[0].ravel()), (0, 0, 255), 3)
        frame = cv.line(frame, center, tuple(imgpts[1].ravel()), (0, 255, 0), 3)
        frame = cv.line(frame, center, tuple(imgpts[2].ravel()), (255, 0, 0), 3)

    frame_count += 1
    current_time = time.time()
    elapsed_time = current_time - prev_time
    
    if elapsed_time > 0.1:
        fps = frame_count / elapsed_time
        prev_time = current_time
        frame_count = 0
        
    cv.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv.imshow("AprilTag Detector", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
print("Stopped")