from ultralytics import YOLO
import yaml
import os

system_path = "/home/reddy"
size = "n"
dataset = "8object"
models = {
          f"{dataset}_{size}_control" : {"model" : ""}, 
        #   f"{dataset}_{size}_canny" : {"model" : ""}, 
        #   f"{dataset}_{size}_active_canny" : {"model" : ""},
        #   f"{dataset}_{size}_HED1" : {"model" : ""},
        #   f"{dataset}_{size}_HED2" : {"model" : ""},
        #   f"{dataset}_{size}_anime_style" : {"model" : "", "model_name" : "anime_style"},
        #   f"{dataset}_{size}_contour_style" : {"model" : "", "model_name" : "contour_style"},
        #   f"{dataset}_{size}_opensketch_style" : {"model" : "", "model_name" : "opensketch_style"},
}

for model_name in models:
    models[model_name]["model"] = YOLO(f'{system_path}/Bachelor_Thesis/trains/{dataset}/{dataset}_{size}/{model_name}/weights/best.pt')
    print(f"Loaded model: {model_name}")

dataset_path = "/data/reddy/Bachelor_Thesis/datasets/8object_dataset"

model_name = "control"

yaml_data = yaml.load(open(f"{dataset_path}/data.yaml"), Loader=yaml.FullLoader)
yaml_data["path"] = f"{dataset_path}/{model_name}"
yaml_path = os.path.join(dataset_path, model_name, "data.yaml")
with open(yaml_path, "w") as file:
    yaml.dump(yaml_data, file)

model = models[f"{dataset}_{size}_control"]["model"]
results = model.val(data=yaml_path)

# Print specific metrics
print("Class indices with average precision:", results.ap_class_index)
print("Average precision for all classes:", results.box.all_ap)
print("Average precision:", results.box.ap)
print("Average precision at IoU=0.50:", results.box.ap50)
print("Class indices for average precision:", results.box.ap_class_index)
print("Class-specific results:", results.box.class_result)
print("F1 score:", results.box.f1)
print("F1 score curve:", results.box.f1_curve)
print("Overall fitness score:", results.box.fitness)
print("Mean average precision:", results.box.map)
print("Mean average precision at IoU=0.50:", results.box.map50)
print("Mean average precision at IoU=0.75:", results.box.map75)
print("Mean average precision for different IoU thresholds:", results.box.maps)
print("Mean results for different metrics:", results.box.mean_results)
print("Mean precision:", results.box.mp)
print("Mean recall:", results.box.mr)
print("Precision:", results.box.p)
print("Precision curve:", results.box.p_curve)
print("Precision values:", results.box.prec_values)
print("Specific precision metrics:", results.box.px)
print("Recall:", results.box.r)
print("Recall curve:", results.box.r_curve)