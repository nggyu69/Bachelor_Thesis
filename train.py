import argparse
from ultralytics import YOLO
import yaml
import os
import yaml
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stdout)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train YOLO model for different configurations.")
    parser.add_argument("--model", required=True, help="Model name (e.g., control, canny, etc.)")
    parser.add_argument("--dataset_path", required=True, help="Path to the dataset directory")
    parser.add_argument("--model_size", default="n", help="Model size (default: n)")
    args = parser.parse_args()

    model_name = args.model
    dataset_path = args.dataset_path
    model_size = args.model_size

    # Update the YAML configuration
    # yaml_data = {
    #     "path": f"{dataset_path}/{model_name}",
    #     "train": "train/images",
    #     "val": "val/images",
    #     "test": "test/images",
    #     "names": {
    #         0 : "Honeycomb_Wall_Scissor_Holder",
    #         1 : "Honeycomb_Cup_for_HC_wall",
    #         2 : "Honeycomb_Wall_Tool_holder",
    #         3 : "Honeycomb_Scraper",
    #         4 : "Honeycomb_Wall_Squirt_bottle",
    #         5 : "Honeycomb_wall_Allen_bit",
    #         6 : "Honeycomb_wall_pliers-cutter_holder",
    #         7 : "Honeycomb_NEW_DOUBLE_Wall_pliers-cutter",
    #     }
    # }

    # yaml_data = json.loads(f"{dataset_path}/data.yaml")
    yaml_data = yaml.load(open(f'{dataset_path}/data.yaml'), Loader=yaml.FullLoader)
    yaml_data["path"] = f"{dataset_path}/{model_name}"
    
    # Save the YAML configuration
    yaml_path = os.path.join(dataset_path, model_name, "data.yaml")
    with open(yaml_path, "w") as file:
        yaml.dump(yaml_data, file)
    
    dataset_name = dataset_path.split("/")[-1].split("_")[0]

    project_dir = dataset_path.replace("datasets", "trains").replace("data/", "home/").replace("_dataset", "")
    project_dir += f"/{dataset_name}_{model_size}_300epochs_75pat_realval"
    # project_dir = project_dir.replace("data", "home")
    save_name = f"{dataset_name}_{model_size}_{''.join(model_name.split('/'))}"

    
    # Load and train the YOLO model
    model = YOLO(f"yolo11{model_size}-obb.yaml").load(f"yolo11{model_size}.pt")
    results = model.train(data=yaml_path, 
                          epochs=300,
                          imgsz=640,
                          patience=75,
                          batch=32,
                          plots=True,
                          project=project_dir, 
                          name=save_name)

if __name__ == "__main__":
    main()
