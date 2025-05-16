import shutil
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from PIL import Image, ImageFont, ImageDraw
import edge_detections

def resize_pad_image(image, mask = False, new_shape=(640, 640)):
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

def read_label_file(label_path):
    gt_boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 9:  # class_id + 8 coordinates (xyxyxyxy)
                class_id = int(parts[0])
                coordinates = [float(x) for x in parts[1:9]]  # Extract 8 coordinates
                gt_boxes.append((class_id, coordinates))
    return gt_boxes

def adjust_coordinates(box, scale, top, left):
    """Adjust coordinates based on scale and padding offset"""
    adjusted_box = box.copy()
    for i in range(0, len(box), 2):
        # Apply scale to x and y coordinates then add padding offset
        adjusted_box[i] = box[i] * scale + left
        adjusted_box[i+1] = box[i+1] * scale + top
    return adjusted_box

paths = [f"/home/reddy/Bachelor_Thesis/test_files/labeled_data/{i}" for i in os.listdir("/home/reddy/Bachelor_Thesis/test_files/labeled_data/") if not i.endswith("parts")]
paths.sort()
dataset_path = "/data/reddy/Bachelor_Thesis/datasets/publish_dataset"

for path in paths:
    class_name = "".join(path.split("/")[-1].split("_")[0])
    count = 0
    images = os.listdir(path + "/train/images")
    images.sort()
    name = "_".join(path.split("/")[-1].split("_")[:2])
    
    name_map = {"bit_holder" : "0",
                "bottle_holder" : "1",
                "cup_holder" : "2",
                "cutter_holder" : "3",
                "scissor_holder" : "4",
                "tool_holder" : "5",}
    
    os.makedirs(path + "/val/images", exist_ok=True)
    os.makedirs(path + "/val/labels", exist_ok=True)

    for image_name in images:
        if(count < len(images)/2):
            count += 1
            image_path = os.path.join(path + "/train/images", image_name)
            label_path = os.path.join(image_path.replace("images", "labels").replace(".jpg", ".txt"))
            print(image_path)
            print(label_path)


            img = cv2.imread(image_path)
            h, w = img.shape[:2]

            resized_image, scale, pad_top, pad_left = resize_pad_image(img)

            box = read_label_file(label_path)
            box = box[0][1]
            denormalized_box = box.copy()
            for i in range(0, len(box), 2):
                denormalized_box[i] *= w
                denormalized_box[i+1] *= h
            
            #Get pixel coordinates
            adjusted_box_pixel = adjust_coordinates(denormalized_box, scale, pad_top, pad_left)
            adjusted_box_pixel = np.array(adjusted_box_pixel).reshape(-1, 2).astype(np.float32)
            
            # Get normalized coordinates
            adjusted_box_normalized = adjusted_box_pixel.copy()
            for i in range(0, len(adjusted_box_normalized), 2):
                adjusted_box_normalized[i] /= resized_image.shape[1]
                adjusted_box_normalized[i+1] /= resized_image.shape[0]

            contour = np.array(adjusted_box_pixel, dtype=np.int32).reshape(-1, 1, 2)
            annotated_image = resized_image.copy()
            annotated_image = cv2.drawContours(annotated_image, [contour], 0, (0, 0, 255), 1)

            cv2.imwrite(f"{dataset_path}/control/val/images/real_image_{class_name}_{count}.jpg", resized_image)
            cv2.imwrite(f"{dataset_path}/annotated_images/control/real_annotated_image_{class_name}_{count}.jpg", annotated_image)

            edge_detections.canny_edge(f"{dataset_path}/canny/val/images/real_image_{class_name}_{count}.jpg", **{"image": resized_image, "annotation" : [dataset_path, contour]})
            edge_detections.active_canny(f"{dataset_path}/active_canny/val/images/real_image_{class_name}_{count}.jpg", **{"image": resized_image, "annotation" : [dataset_path, contour]})
            edge_detections.hed_edge(f"{dataset_path}/HED/PlAcEhOlDeR/val/images/real_image_{class_name}_{count}.jpg", **{"image": resized_image, "annotation" : [dataset_path, contour]})
            edge_detections.info_drawing(f"{dataset_path}/anime_style/val/images/real_image_{class_name}_{count}.jpg", **{"image": resized_image, "annotation" : [dataset_path, contour], "model_name" : "anime_style"})
            edge_detections.info_drawing(f"{dataset_path}/contour_style/val/images/real_image_{class_name}_{count}.jpg", **{"image": resized_image, "annotation" : [dataset_path, contour], "model_name" : "contour_style"})
            edge_detections.info_drawing(f"{dataset_path}/opensketch_style/val/images/real_image_{class_name}_{count}.jpg", **{"image": resized_image, "annotation" : [dataset_path, contour], "model_name" : "opensketch_style"})
            edge_detections.adaptive_threshold(f"{dataset_path}/adaptive_threshold/val/images/real_image_{class_name}_{count}.jpg", **{"image": resized_image, "annotation" : [dataset_path, contour]})
            
            class_id = name_map[name]
            annotation_text = f"{class_id} " + " ".join([f"{x:.6f}" for x in adjusted_box_normalized.flatten()])
            for style in ["control", "canny", "active_canny", "anime_style", "contour_style", "opensketch_style", "adaptive_threshold"]:
                with open(f"{dataset_path}/{style}/val/labels/real_image_{class_name}_{count}.txt", "w") as f:
                    f.write(annotation_text)
            for i in range(1, 6):
                with open(f"{dataset_path}/HED/{i}/val/labels/real_image_{class_name}_{count}.txt", "w") as f:
                    f.write(annotation_text)
            
            os.rename(image_path, path + "/val/images/" + image_name)
            os.rename(label_path, path + "/val/labels/" + image_name.replace(".jpg", ".txt"))

            print(f"Done image {class_name}_{count}")
            
