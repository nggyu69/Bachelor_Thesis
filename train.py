# from ultralytics import YOLO
# import yaml
# import matplotlib

# matplotlib.use('Agg')

# #data.yaml
# path = "/data/reddy/Bachelor_Thesis/multimodel2"
# yaml_data = {
#     "path": f"{path}/control",
#     "train": "train/images",
#     "val": "val/images",
#     "test": "test/images",
#     "names": {
#         0: "Honeycomb_Wall_Tool",
#         1: "Honeycomb_Wall_Cup",
#         2: "Honeycomb_Wall_Pliers"
#     }
# }
# with open (f"{path}/data.yaml", "w") as file:
#     yaml.dump(yaml_data, file)

# models = ["opensketch_style", "contour_style"]
# for i in models:
#     yaml_data["path"] = f"{path}/{i}"
#     with open (f"{path}/data.yaml", "w") as file:
#         yaml.dump(yaml_data, file)

#     model = YOLO("yolo11x-obb.yaml").load("yolo11x.pt")
#     results = model.train(data=f"{path}/data.yaml", epochs=100, imgsz=640, project="/home/reddy/Bachelor_Thesis/trains", name=f"""multimodel_{"".join(i.split('/'))}""")

import argparse
from ultralytics import YOLO
import yaml
import os

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train YOLO model for different configurations.")
    parser.add_argument("--model", required=True, help="Model name (e.g., control, canny, etc.)")
    parser.add_argument("--dataset_path", required=True, help="Path to the dataset directory")
    args = parser.parse_args()

    model_name = args.model
    dataset_path = args.dataset_path

    # Update the YAML configuration
    yaml_data = {
        "path": f"{dataset_path}/{model_name}",
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": {
            0 : "Honeycomb_Wall_Scissor_Holder",
            1 : "Honeycomb_Cup_for_HC_wall",
            2 : "Honeycomb_Wall_Tool_holder",
            3 : "Honeycomb_Scraper",
            4 : "Honeycomb_Wall_Squirt_bottle",
            5 : "Honeycomb_wall_Allen_bit",
            6 : "Honeycomb_wall_pliers-cutter_holder",
            7 : "Honeycomb_NEW_DOUBLE_Wall_pliers-cutter",
        }
    }

    # Save the YAML configuration
    yaml_path = os.path.join(dataset_path, "data.yaml")
    with open(yaml_path, "w") as file:
        yaml.dump(yaml_data, file)
    
    model_size = "n"

    project_dir = dataset_path.replace("datasets", "trains").replace("data/", "home/")
    project_dir += f"_{model_size}"
    # project_dir = project_dir.replace("data", "home")
    save_name = f"8object_{model_size}_{''.join(model_name.split('/'))}"

    
    # Load and train the YOLO model
    model = YOLO(f"yolo11{model_size}-obb.yaml").load(f"yolo11{model_size}.pt")
    results = model.train(data=yaml_path, 
                          epochs=100, 
                          imgsz=640, 
                          project=project_dir, 
                          name=save_name)

if __name__ == "__main__":
    main()
