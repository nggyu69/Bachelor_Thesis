from ultralytics import YOLO
import yaml
import os
import yaml
import sys
import logging
import json
import json

logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stdout)

def main():
    # Load config-only settings
    with open("/home/reddy/Bachelor_Thesis/config.json", "r") as f:
        config = json.load(f)

    training_cfg = config.get("training", {})

    model_name = training_cfg.get("model")
    dataset_path = training_cfg.get("dataset_path")
    model_size = training_cfg.get("model_size", "n")
    epochs = training_cfg.get("epochs", 300)
    imgsz = training_cfg.get("imgsz", 640)
    patience = training_cfg.get("patience", 75)
    batch = training_cfg.get("batch", 32)

    if not model_name or not dataset_path:
        raise SystemExit("model and dataset_path must be provided either via config or CLI.")

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

    project_suffix = training_cfg.get("project_suffix", "75pat_realval")
    project_dir = dataset_path.replace("datasets", "trains").replace("data/", "home/").replace("_dataset", "")
    project_dir += f"/{dataset_name}_{model_size}_{project_suffix}"
    # project_dir = project_dir.replace("data", "home")
    save_name = f"{dataset_name}_{model_size}_{''.join(model_name.split('/'))}"

    
    # Load and train the YOLO model
    yolo_cfg_pattern = training_cfg.get("yolo_config_pattern", "yolo11{size}-obb.yaml")
    yolo_wts_pattern = training_cfg.get("yolo_weights_pattern", "yolo11{size}.pt")
    model = YOLO(yolo_cfg_pattern.format(size=model_size)).load(yolo_wts_pattern.format(size=model_size))
    results = model.train(data=yaml_path, 
                          epochs=epochs,
                          imgsz=imgsz,
                          patience=patience,
                          batch=batch,
                          plots=True,
                          project=project_dir, 
                          name=save_name)

if __name__ == "__main__":
    main()
