# Usage

## 1. Calibrate your camera using load_intrinsics.py  
Run:
`python load_intrinsics.py`

The script will:

- Capture checkerboard images
- Detect corners
- Compute intrinsic matrix K and distortion coefficients

Output:
```py=
Camera intrinsic matrix K =
[[fx,  0, cx],
 [0, fy, cy],
 [0,  0,  1]]
```

```
Distortion coefficients dist =
[[k1, k2, p1, p2, k3]]
```

## 2. Add your intrinsics to camera_intrinsics.json

Copy the printed K and distortion values into the JSON file:

```
{
    "MyCamera": {
        "K": [
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ],
        "distortion": [[k1, k2, p1, p2, k3]]
    }
}

```

You may store multiple camera profiles.

## 3. Set your camera in intrinsics_test.py

Find this line:

`TARGET_CAMERA = "PW_313"`

Replace with the profile name you added:

`TARGET_CAMERA = "MyCamera"`

## 4. Run the AprilTag detector to test
`python intrinsics_test.py`

You will see:

- AprilTag bounding box
- Center point
- Tag ID
- 3D translation (X, Y, Z in cm)
- Euler angles (Yaw, Pitch, Roll)
- 3D axis projection
