import csv
import time
from ultralytics import YOLO
import cv2
import os
import numpy as np
import edge_detections
import sys

system_path = "/home/reddy"
size = "x"
dataset = "8object"
models = {
          f"{dataset}_{size}_control" : {"model" : "",  "preprocess" : None, "args" : None}, 
          f"{dataset}_{size}_canny" : {"model" : "", "preprocess" : edge_detections.canny_edge, "args" : {"image" : ""}}, 
          f"{dataset}_{size}_active_canny" : {"model" : "", "preprocess" : edge_detections.active_canny, "args" : {"image" : ""}},
          f"{dataset}_{size}_HED1" : {"model" : "", "preprocess" : edge_detections.hed_edge, "args" : {"image" : "", "layer" : 1}},
          f"{dataset}_{size}_HED2" : {"model" : "", "preprocess" : edge_detections.hed_edge, "args" : {"image" : "", "layer" : 2}},
          f"{dataset}_{size}_anime_style" : {"model" : "", "preprocess" : edge_detections.info_drawing, "args" : {"image" : "", "model_name" : "anime_style"}},
          f"{dataset}_{size}_contour_style" : {"model" : "", "preprocess" : edge_detections.info_drawing, "args" : {"image" : "", "model_name" : "contour_style"}},
          f"{dataset}_{size}_opensketch_style" : {"model" : "", "preprocess" : edge_detections.info_drawing, "args" : {"image" : "", "model_name" : "opensketch_style"}},
          }

def preprocess(model_name, model):
    if model["preprocess"]:
        img = model["preprocess"](**model["args"])
        if isinstance(img, list):
            img = img[int(model_name.split("HED")[-1])-1]
    return img

# Load the trained model
for model_name in models:
    models[model_name]["model"] = YOLO(f'{system_path}/Bachelor_Thesis/trains/{dataset}/{dataset}_{size}/{model_name}/weights/best.pt')
    print(f"Loaded model: {model_name}")

# Directory containing the images to test
image_dir = f'{system_path}/Bachelor_Thesis/Demonstrator/test_pics'
output_dir = f'{system_path}/Bachelor_Thesis/benchmarks/{dataset}/{dataset}_{size}'
os.makedirs(output_dir, exist_ok=True)

for model_name, model_data in models.items():
    csv_file_path = os.path.join(output_dir, f'{model_name}_benchmark.csv')
    with open(csv_file_path, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Image Name", "Inference Time (s)", "Average Confidence"])
        
        # Process images
        for filename in sorted(os.listdir(image_dir)):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                img_path = os.path.join(image_dir, filename)
                img = cv2.imread(img_path)
                # Measure inference time
                start_time = time.time()
                if model_data["args"] is not None:
                    model_data["args"]["image"] = img
                    img = preprocess(model_name, model_data)
                
                
                results = model_data["model"](img, conf=0.80)
                inference_time = time.time() - start_time
                
                
                if results[0].obb:
                    confidences = [box.conf.item() for box in results[0].obb]
                    avg_confidence = np.mean(confidences)
                else:
                    avg_confidence = 0
                
                # Write results to CSV
                csv_writer.writerow([filename, inference_time, avg_confidence])
                
    print(f"Benchmark results saved for {model_name} at: {csv_file_path}")
