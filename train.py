from ultralytics import YOLO
import yaml
import matplotlib

matplotlib.use('Agg')

#data.yaml
path = "/data/reddy/Bachelor_Thesis/multimodel"
yaml_data = {
    "path": f"{path}/control",
    "train": "train/images",
    "val": "val/images",
    "test": "test/images",
    "names": {
        0: "Honeycomb_Wall_Tool",
        1: "Honeycomb_Wall_Cup",
        2: "Honeycomb_Wall_Pliers"
    }
}
with open (f"{path}/data.yaml", "w") as file:
    yaml.dump(yaml_data, file)

for i in ["canny", "active_canny", "HED/1", "HED/2"]:
    yaml_data["path"] = f"{path}/{i}"
    with open (f"{path}/data.yaml", "w") as file:
        yaml.dump(yaml_data, file)

    model = YOLO("yolo11x-obb.yaml").load("yolo11x.pt")
    results = model.train(data=f"{path}/data.yaml", epochs=100, imgsz=640, project="/home/reddy/Bachelor_Thesis/trains", name=f"""multimodel_{"".join(i.split('/'))}""")
