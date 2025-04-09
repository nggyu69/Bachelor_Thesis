from ultralytics import YOLO
import cv2
import os
import numpy as np
import edge_detections

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
    models[model_name]["model"] = YOLO(f'/home/reddy/Bachelor_Thesis/trains/{dataset}/{dataset}_{size}/{model_name}/weights/best.pt')
    print(f"Loaded model: {model_name}")

# model_name = "multimodel_HED1"


# Directory containing the images to test
image_dir = '/home/reddy/Bachelor_Thesis/test_files/all_black'
output_dir = f'/home/reddy/Bachelor_Thesis/predictions/{dataset}/{dataset}_{size}/{image_dir.split("/")[-1]}'
os.makedirs(output_dir, exist_ok=True)


grid_size = int(len(os.listdir(image_dir))**0.5) + 1


for model_name in models:
    image_results = []
    os.makedirs(os.path.join(output_dir, model_name), exist_ok=True)
    # Load images in their original size and predict
    for filename in sorted(os.listdir(image_dir)):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # Load image in its original size
            img_path = os.path.join(image_dir, filename)
            img = cv2.imread(img_path)
            
            if models[model_name]["args"] is not None:
                models[model_name]["args"]["image"] = img
            # Run prediction on the original-sized image, saving results in a single folder
                img = preprocess(model_name, models[model_name])

            results = models[model_name]["model"].predict(img, conf=0.9)
            
            # Save the visualized result and add it to the list for stitching
            result_image = results[0].plot()  # Visualize prediction
            result_image_path = os.path.join(output_dir, model_name, filename)
            cv2.imwrite(result_image_path, result_image)
            image_results.append(result_image)

    # Calculate the number of images required for the grid
    required_images = grid_size * grid_size

    # Add blank images if we have fewer than required images
    if len(image_results) < required_images:
        # Get the shape of the first image as a reference for creating blank images
        ref_height, ref_width = image_results[0].shape[:2]
        blank_image = np.zeros((ref_height, ref_width, 3), dtype=np.uint8)  # Black image placeholder
        image_results.extend([blank_image] * (required_images - len(image_results)))

    # Create the grid with filled blank images if necessary
    rows = [np.hstack(image_results[i*grid_size:(i+1)*grid_size]) for i in range(grid_size)]
    stitched_image = np.vstack(rows)  # Stack the rows vertically to form the grid

    # Save the stitched image
    stitched_image_path = os.path.join(output_dir, f'{model_name}/{model_name}_{image_dir.split("/")[-1]}_stitched_results_{grid_size}x{grid_size}.jpg')
    cv2.imwrite(stitched_image_path, stitched_image)

    print(f"Stitched {grid_size}x{grid_size} image saved at: {stitched_image_path}")