import cv2
from picamera2 import Picamera2
import numpy as np
from ultralytics import YOLO
import os
import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import edge_detections

def preprocess(image, mask=False, new_shape=(640,640)):
        # Resize image to fit into new_shape maintaining aspect ratio
    h, w = image.shape[:2]
    scale = min(new_shape[1] / w, new_shape[0] / h)
    nw, nh = int(w * scale), int(h * scale)

    # Resize image with the scaling factor
    resized_image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)

    if mask == True:
        new_image = np.full((new_shape[0], new_shape[1]), False, dtype=bool)
    else:
        # Create a new image with padding
        new_image = np.full((new_shape[0], new_shape[1], 3), 128, dtype=np.uint8)

    # Calculate padding
    top = (new_shape[0] - nh) // 2
    left = (new_shape[1] - nw) // 2

    # Place the resized image in the new image with padding
    new_image[top:top + nh, left:left + nw] = resized_image

    return new_image, scale, top, left
# Initialize Picamera2
picam2 = Picamera2()

# Get the sensor's full resolution
camera_info = picam2.sensor_resolution  # Full sensor resolution (e.g., 4056x3040)
print("Camera resolution:", camera_info)

# Configure the camera to use full resolution
config = picam2.create_preview_configuration(
    main={"size": camera_info}  # Full field of view
)
picam2.configure(config)

# Start the camera
picam2.start()

output_width = 1280
output_height = 960

path = "/home/pi/Bachelor_Thesis/trains/multimodel_control/weights/best.pt"
# model = YOLO(path)
count =1
while True:
    # Capture the full frame
    frame = picam2.capture_array()

    # Convert RGB to BGR for OpenCV
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Rescale the frame to the desired size
    resized_frame = cv2.resize(frame_bgr, (output_width, output_height), interpolation=cv2.INTER_AREA)

    # resized_frame, scale, top, left = preprocess(resized_frame, mask=False, new_shape=(640,640))
    # Display the rescaled frame
    cv2.imshow("Rescaled Video", resized_frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord(' '):
        print("Captured image, predicting...")
        # model.predict(resized_frame, save=True, project="predictions", name="control_predict", conf=0.7)
        cv2.imwrite(f"test_pics/{count}.jpg", resized_frame)
        count += 1
        

# Stop the camera and close the window
picam2.stop()
cv2.destroyAllWindows()
