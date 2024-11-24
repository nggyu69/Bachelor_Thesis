import cv2
import numpy as np
import os


def canny_edge(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, threshold1=75, threshold2=75)

    # Stack the original and edge-detected images side by side
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)  # Convert edges to BGR for stacking
    # combined_image = np.hstack((image, edges_color))

    # Save the result
    output_path = os.path.join(dataset_path, "canny", image_name)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, edges_color)

    print(f"Saved the combined image with edges at: {output_path}")

def active_canny(image):
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
    output_path = os.path.join(dataset_path, "active_canny", image_name)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, edges_color)

    print(f"Saved the combined image with adaptive Canny edges at: {output_path}")

def hed_edge(image, net):
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
        output_path = os.path.join(dataset_path, "HED", layer_names[i], image_name)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, output_image)

    print(f"Saved the combined image with HED edges at: {output_path}")

# path = "/home/reddy/Bachelor_Thesis/examples/coco_data/images"
dataset_path = "/data/reddy/Bachelor_Thesis/test_hed"

# images = os.listdir(path)
# images.sort()

prototxt_path = 'Bachelor_Thesis/HED_Files/deploy.prototxt'
caffemodel_path = 'Bachelor_Thesis/HED_Files/hed_pretrained_bsds.caffemodel'
net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

# for image_name in images:
#     img = cv2.imread(f"{path}/{image_name}")

#     canny_edge(img)
#     active_canny(img)
#     hed_edge(img, net)
image_name = "test_hed.jpg"
img = cv2.imread("/data/reddy/Bachelor_Thesis/cup_black/img2.jpg")
# canny_edge(img)
# active_canny(img)
hed_edge(img, net)