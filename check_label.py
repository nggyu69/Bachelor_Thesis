import os

images = os.listdir("/data/reddy/Bachelor_Thesis/datasets/8object/active_canny/val/images")

label_dict = {}
for image in images:
    label_name = image.split(".")[0] + ".txt"
    with open (f"/data/reddy/Bachelor_Thesis/datasets/8object/active_canny/val/labels/{label_name}", "r") as file:
        label_dict[file.read()[0]] = "_".join(image.split("_")[1:-1])

for i in range(8):
    print(str(i), label_dict[str(i)])