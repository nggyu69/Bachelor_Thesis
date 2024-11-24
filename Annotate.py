import json
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from PIL import Image, ImageFont, ImageDraw


def resize_pad_image(image, mask = False, new_shape=(640, 640)):
    # Resize image to fit into new_shape maintaining aspect ratio
    h, w = image.shape[:2]
    scale = min(new_shape[1] / w, new_shape[0] / h)
    nw, nh = int(w * scale), int(h * scale)

    # Resize image with the scaling factor
    resized_image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)

    if mask == True:
        new_image = np.full((new_shape[0], new_shape[1]), False, dtype=bool)
    else:
        # Create a new image with padding
        new_image = np.full((new_shape[0], new_shape[1], 3), 128, dtype=np.uint8)

    # Calculate padding
    top = (new_shape[0] - nh) // 2
    left = (new_shape[1] - nw) // 2

    # Place the resized image in the new image with padding
    new_image[top:top + nh, left:left + nw] = resized_image

    return new_image, scale, top, left

def resize_pad_mask(mask, new_shape=(640, 640)):
    # Get the original dimensions
    original_height, original_width = mask.shape
    
    # Calculate the scale to maintain aspect ratio
    scale = min(new_shape[1] / original_width, new_shape[0] / original_height)
    
    # Calculate the new width and height while maintaining aspect ratio
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    # Resize the mask
    resized_mask = cv2.resize(mask.astype(np.uint8), (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    
    # Create a new blank mask with the target dimensions
    new_mask = np.full(new_shape, False, dtype=bool)
    
    # Calculate padding
    pad_top = (new_shape[0] - new_height) // 2
    pad_left = (new_shape[1] - new_width) // 2
    
    # Place the resized mask into the new mask with padding
    new_mask[pad_top:pad_top + new_height, pad_left:pad_left + new_width] = resized_mask.astype(bool)
    
    return new_mask, scale, pad_top, pad_left

def rle_to_binary_mask(rle):
    """Converts a COCOs run-length encoding (RLE) to binary mask.
    :param rle: Mask in RLE format
    :return: a 2D binary numpy array where '1's represent the object
    """
    binary_array = np.zeros(np.prod(rle.get('size')), dtype=bool)
    counts = rle.get('counts')

    start = 0
    for i in range(len(counts) - 1):
        start += counts[i]
        end = start + counts[i + 1]
        binary_array[start:end] = (i + 1) % 2

    binary_mask = binary_array.reshape(*rle.get('size'), order='F')

    return binary_mask

def resize_bounding_box(bbox, scale, pad_top, pad_left):
    resized_bbox = []
    for x, y in bbox:
        new_x = x * scale + pad_left
        new_y = y * scale + pad_top
        resized_bbox.append((new_x, new_y))
    return resized_bbox

def canny_edge(image, path):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, threshold1=75, threshold2=75)

    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)  # Convert edges to BGR for stacking
    # combined_image = np.hstack((image, edges_color))

    cv2.imwrite(path, edges_color)

def active_canny(image, path):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute the median of the pixel intensities
    median_intensity = np.median(gray)

    # Set lower and upper thresholds for Canny edge detection based on median intensity
    # These constants can be adjusted for more or less sensitivity
    sigma = 0.2
    lower_threshold = int(max(0, (1.0 - sigma) * median_intensity))
    upper_threshold = int(min(255, (1.0 + sigma) * median_intensity))

    # Apply Canny edge detection with adaptive thresholds
    edges = cv2.Canny(gray, lower_threshold, upper_threshold)

    # Convert edges to BGR so it can be stacked with the original image
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # # Stack the original image and edge-detected image side by side
    # combined_image = np.hstack((image, edges_color))

    # Save the result
    cv2.imwrite(path, edges_color)

def hed_edge(image, path, net):
    # Prepare the image for HED
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(w, h), mean=(104.00698793, 116.66876762, 122.67891434), swapRB=False, crop=False)

    # Pass the image blob through the HED model
    net.setInput(blob)

    # Specify the layer names to capture intermediate outputs
    layer_names = ['sigmoid-dsn1', 'sigmoid-dsn2', 'sigmoid-dsn3', 'sigmoid-dsn4', 'sigmoid-dsn5', 'sigmoid-fuse']
    outputs = net.forward(layer_names)
    
    # Process and resize each output to match the original image dimensions
    output_images = [(255 * cv2.resize(out[0, 0], (w, h))).astype("uint8") for out in outputs]

    # Convert each edge map to BGR so it can be stacked with the original image
    output_images_bgr = [cv2.cvtColor(out_img, cv2.COLOR_GRAY2BGR) for out_img in output_images]
    # # Stack the original image and each of the intermediate outputs side by side
    # combined_image = np.hstack([image] + output_images_bgr)

    # Save the result
    for i, output_image in enumerate(output_images_bgr):
        #create path based on layer name

        path = path.replace("PlAcEhOlDeR", str(i+1))
        cv2.imwrite(path, output_image)
        path = path.replace(f"/{str(i+1)}/", "/PlAcEhOlDeR/")
    # cv2.imwrite(path, output_images_bgr[0])


path = "/data/reddy/Bachelor_Thesis/part_2/coco_data"
dataset_path = "/data/reddy/Bachelor_Thesis/dataset_honeycomb_tool"

existing_images = []
if os.path.exists(dataset_path + "/annotated_images"):
    existing_images = [int(i.split("_")[-1][0:-4]) for i in os.listdir(dataset_path + "/annotated_images")]


# path = "/data/reddy/coco_data"
with open(path + "/coco_annotations.json") as f:
    data = json.load(f)
    annotations = data["annotations"]
    images = data["images"]
    categories = data["categories"]
    categories = {category["id"]: category["name"] for category in categories}
    # classes = {"Honeycomb Cup for HC wall" : 0}



train_num = int(round(len(images) * 0.7))
test_num = int(round(len(images) * 0.2))
val_num = len(images) - train_num - test_num
split_list = ["train" for i in range(train_num)] + ["test" for i in range(test_num)] + ["val" for i in range(val_num)]
np.random.shuffle(split_list)

os.makedirs(dataset_path + "/annotated_images", exist_ok=True)
os.makedirs(dataset_path + "/masked_images", exist_ok=True)

for i in ["control", "canny", "active_canny"]:
    os.makedirs(dataset_path + f"/{i}/train/images", exist_ok=True)
    os.makedirs(dataset_path + f"/{i}/test/images", exist_ok=True)
    os.makedirs(dataset_path + f"/{i}/val/images", exist_ok=True)
    os.makedirs(dataset_path + f"/{i}/train/labels", exist_ok=True)
    os.makedirs(dataset_path + f"/{i}/test/labels", exist_ok=True)
    os.makedirs(dataset_path + f"/{i}/val/labels", exist_ok=True)

for i in range(1, 6):
    os.makedirs(dataset_path + "/HED/" + str(i) + "/train/images", exist_ok=True)
    os.makedirs(dataset_path + "/HED/" + str(i) + "/test/images", exist_ok=True)
    os.makedirs(dataset_path + "/HED/" + str(i) + "/val/images", exist_ok=True)
    os.makedirs(dataset_path + "/HED/" + str(i) + "/train/labels", exist_ok=True)
    os.makedirs(dataset_path + "/HED/" + str(i) + "/test/labels", exist_ok=True)
    os.makedirs(dataset_path + "/HED/" + str(i) + "/val/labels", exist_ok=True)


annotations_grouped = {}
for annotation in annotations:
    image_id = annotation["image_id"]
    if image_id not in annotations_grouped:
        annotations_grouped[image_id] = []
    annotations_grouped[image_id].append(annotation)

prototxt_path = 'Bachelor_Thesis/HED_Files/deploy.prototxt'
caffemodel_path = 'Bachelor_Thesis/HED_Files/hed_pretrained_bsds.caffemodel'
net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

for image in annotations_grouped:
    if image in existing_images:
        print(f"Image {image} already exists in the dataset. Skipping...")
        continue
    img = cv2.imread(f"{path}/{images[image-2000]['file_name']}")
    image_type = split_list.pop()
    resized_image, scale, pad_top, pad_left = resize_pad_image(img)
    
    annotated_image = np.copy(resized_image)
    mask_image = Image.fromarray(cv2.cvtColor(np.copy(resized_image), cv2.COLOR_BGR2RGB))

    annotations_text = ""
    for annotation in annotations_grouped[image]:
        mask = rle_to_binary_mask(annotation["segmentation"]).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)
        (x,y),(w,h), a = rect
        
        box = cv2.boxPoints(rect)
        box = np.intp(box) #turn into ints
        resized_box = resize_bounding_box(box, scale, pad_top, pad_left)
        resized_box = np.intp(resized_box)

        Class = annotation["category_id"]
        
        annotations_text += f"{Class} {' '.join([str((coord/640)) for coord in resized_box.flatten()])}\n"
        color = (0, 255, 0) if Class == 0 else (0, 0, 255)
        annotated_image = cv2.drawContours(annotated_image, [resized_box], 0, color, 1)

        mask = resize_pad_mask(mask)[0]
        mask = mask.astype(np.uint8) * 255
        mask_image.putalpha(255)
        mask = Image.fromarray(mask, mode="L")
        overlay = Image.new('RGBA', mask_image.size)
        draw_ov = ImageDraw.Draw(overlay)
        draw_ov.bitmap((0, 0), mask, fill=(color[2], color[1], color[0], 64))
        mask_image = Image.alpha_composite(mask_image, overlay)

    for i in ["control", "canny", "active_canny"]:
        with open(f"{dataset_path}/{i}/{image_type}/labels/image_{image}.txt", "w") as f:
            f.write(annotations_text)
    for i in range(1, 6):
        with open(f"{dataset_path}/HED/{i}/{image_type}/labels/image_{image}.txt", "w") as f:
            f.write(annotations_text)    
    
    mask_image.save(f"{dataset_path}/masked_images/masked_image_{image}.png")
    cv2.imwrite(f"{dataset_path}/annotated_images/annotated_image_{image}.jpg", annotated_image)

    cv2.imwrite(f"{dataset_path}/control/{image_type}/images/image_{image}.jpg", resized_image)

    canny_edge(resized_image, f"{dataset_path}/canny/{image_type}/images/image_{image}.jpg")
    active_canny(resized_image, f"{dataset_path}/active_canny/{image_type}/images/image_{image}.jpg")
    hed_edge(resized_image, f"{dataset_path}/HED/PlAcEhOlDeR/{image_type}/images/image_{image}.jpg",net)
    print("Done image", image)



#Converting new data to old dataset 

# import os
# import json
# import sys

# path = "/home/reddy/Bachelor_Thesis/examples/part_2/coco_data"
# dataset_path = "/data/reddy/Bachelor_Thesis/dataset2"

# existing_images = []
# if os.path.exists(dataset_path + "/annotated_images"):
#     existing_images = [int(i.split("_")[-1][0:-4]) for i in os.listdir(dataset_path + "/annotated_images")]

# # path = "/data/reddy/coco_data"
# with open(path + "/coco_annotations.json") as f:
#     data = json.load(f)

#     for image in data["images"]:
#         image["id"] = int(image["id"]+2000)
#         os.rename(path + "/" + image["file_name"], path + "/" + image["file_name"].split("/")[0] + "/" + f"{image['id']:06}.jpg")
#         image["file_name"] = image["file_name"].split("/")[0] + "/" + f"{image['id']:06}.jpg"

#     for annotation in data["annotations"]:
#         annotation["image_id"] = int(annotation["image_id"]+2000)
        

# with open(path + "/coco_annotations.json", "w") as f:
#     json.dump(data, f)

