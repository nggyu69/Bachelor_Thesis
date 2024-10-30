from ultralytics import YOLO

# Load a model
# model = YOLO("yolov11x-obb.yaml")  # build a new model from YAML
# model = YOLO("yolov11x-obb.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolo11x-obb.yaml").load("yolo11x.pt")  # build from YAML and transfer weights
# model = YOLO("/home/reddy/runs/obb/train5/weights/best.pt")
# Train the model
results = model.train(data="/data/reddy/Bachelor_Thesis/data.yaml", epochs=100, imgsz=640, project="/home/reddy/Bachelor_Thesis/trains", name="train_combined")