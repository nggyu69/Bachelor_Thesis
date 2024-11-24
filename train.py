from ultralytics import YOLO
import yaml

#data.yaml
yaml_data = {
    "path": "/data/reddy/Bachelor_Thesis/dataset_honeycomb_tool/control",
    "train": "train/images",
    "val": "val/images",
    "test": "test/images",
    "names": {
        0: "Honeycomb_Tool"
    }
}
with open ("/data/reddy/Bachelor_Thesis/data.yaml", "w") as file:
    yaml.dump(yaml_data, file)

for i in ["control", "active_canny", "HED/1", "HED/2"]:
    
    yaml_data["path"] = f"/data/reddy/Bachelor_Thesis/dataset_honeycomb_tool/{i}"
    model = YOLO("yolo11x-obb.yaml").load("yolo11x.pt")
    results = model.train(data="/data/reddy/Bachelor_Thesis/data.yaml", epochs=100, imgsz=640, project="/home/reddy/Bachelor_Thesis/trains", name=f"""train_honeycomb_tool_{"".join(i.split('/'))}""")
