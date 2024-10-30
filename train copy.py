from ultralytics import YOLO
import torch

# Load a model
# model = YOLO("yolov11x-obb.yaml")  # build a new model from YAML
# model = YOLO("yolov11x-obb.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolo11x-obb.yaml")  # build from YAML and transfer weights
# model = YOLO("/home/reddy/runs/obb/train5/weights/best.pt")
# Train the model
model.load("/home/reddy/runs/obb/train5/weights/best.pt")

# Freeze all layers except for the detection head (last layer group)
for name, param in model.model.named_parameters():
    if 'head' not in name:  # Replace 'head' with the specific layer if necessary
        param.requires_grad = False  # Freeze all layers except for the head

results = model.train(data="/data/reddy/Bachelor_Thesis/data.yaml", epochs=100, imgsz=640, project="/home/reddy/Bachelor_Thesis/trains", pretrained=True , name="transferred_train")