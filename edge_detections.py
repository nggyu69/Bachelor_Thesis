import cv2
import numpy as np
import os
import torch
from torchvision.transforms import transforms
from torchvision.utils import save_image
from PIL import Image
from Info_Drawing_Files.model import Generator

def canny_edge(path="", **kwargs):
    image = kwargs["image"]
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, threshold1=75, threshold2=75)

    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    if path:
        cv2.imwrite(path, edges_color)
    
    if "annotation" in kwargs:
        annotation = kwargs["annotation"]
        annotated_image = cv2.drawContours(edges_color, [annotation[1]], 0, (255, 255, 255), 1)
        #split tail and head
        path = f"{annotation[0]}/annotated_images/canny/" + os.path.split(path)[1]
        
        cv2.imwrite(f"{path}", annotated_image)
    return edges_color

def active_canny(path="", **kwargs):
    image = kwargs["image"]
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

    if path:
        cv2.imwrite(path, edges_color)

    if "annotation" in kwargs:
        annotation = kwargs["annotation"]
        annotated_image = cv2.drawContours(edges_color, [annotation[1]], 0, (255, 255, 255), 1)
        #split tail and head
        path = f"{annotation[0]}/annotated_images/active_canny/" + os.path.split(path)[1]
        
        cv2.imwrite(f"{path}", annotated_image)
    return edges_color
 
def hed_edge(path="", **kwargs):
    # Prepare the image for HED
    image = kwargs["image"]
    layer = None
    if "layer" in kwargs:
        layer = kwargs["layer"]
    

    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(640, 640), mean=(104.00698793, 116.66876762, 122.67891434), swapRB=False, crop=True)

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

    if path:
        for i, output_image in enumerate(output_images_bgr):
            path = path.replace("PlAcEhOlDeR", str(i+1))
            cv2.imwrite(path, output_image)
            

            if "annotation" in kwargs:
                annotation = kwargs["annotation"]
                temp_path = path
                annotated_image = cv2.drawContours(output_image, [annotation[1]], 0, (255, 255, 255), 1)
                #split tail and head
                temp_path = f"{annotation[0]}/annotated_images/HED/{i+1}/" + os.path.split(path)[1]
                cv2.imwrite(f"{temp_path}", annotated_image)
            
            path = path.replace(f"/{str(i+1)}/", "/PlAcEhOlDeR/")
    if layer:
        return output_images_bgr[layer-1]
    else:
        return output_images_bgr

def info_drawing(path="", **kwargs):

    image = kwargs["image"]
    model_name = kwargs["model_name"]

    checkpoints_dir = f"{system_path}/Bachelor_Thesis/Info_Drawing_Files/checkpoints"

    with torch.no_grad():
        
        net_G = 0
        net_G = Generator(3, 1, 3)
        net_G.cuda()

        net_G.load_state_dict(torch.load(os.path.join(checkpoints_dir, model_name, 'netG_A_latest.pth')))
        net_G.eval()

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(640, Image.BICUBIC),
            transforms.ToTensor()
        ])
        
        
        input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        input_tensor = input_tensor.cuda()

        output_tensor = net_G(input_tensor)
        # # Save the generated image
        # os.makedirs(path, exist_ok=True)
        # output_file = os.path.join(path, os.path.basename(image))
    if path:
        save_image(output_tensor.data, path)

        # Convert tensor to OpenCV format
    output_tensor = output_tensor.squeeze(0).cpu().detach()  # Remove batch dim, move to CPU
    output_tensor = output_tensor.numpy()  # Convert to NumPy
    output_tensor = np.transpose(output_tensor, (1, 2, 0))  # Rearrange to (H, W, C)
    
    # Scale to 0-255 and convert to uint8
    output_image = (output_tensor * 255.0).clip(0, 255).astype(np.uint8)

    # If the output is grayscale, convert it to 3 channels for OpenCV
    if output_image.shape[2] == 1:
        output_image = cv2.cvtColor(output_image, cv2.COLOR_GRAY2BGR)
    else:
        output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

    if "annotation" in kwargs:
        annotation = kwargs["annotation"]
        annotated_image = cv2.drawContours(output_image, [annotation[1]], 0, (0, 0, 0), 1)
        #split tail and head
        path = f"{annotation[0]}/annotated_images/{model_name}/" + os.path.split(path)[1]
        cv2.imwrite(f"{path}", annotated_image)

    return output_image


def adaptive_threshold(path="", **kwargs):
    image = kwargs["image"]
    block_size = kwargs.get("block_size", 11)
    c_value = kwargs.get("c_value", 2)

    # Ensure block_size is odd and greater than 1
    if block_size <= 1:
        print(f"Warning: block_size ({block_size}) must be > 1. Setting to 3.")
        block_size = 3
    elif block_size % 2 == 0:
        block_size += 1
        print(f"Warning: block_size must be odd. Adjusting to {block_size}.")

    # Convert image to grayscale for thresholding
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    # cv2.ADAPTIVE_THRESH_MEAN_C: threshold is mean of neighborhood area - C
    # cv2.ADAPTIVE_THRESH_GAUSSIAN_C: threshold is weighted sum (gaussian) - C
    thresh_image = cv2.adaptiveThreshold(
        gray,
        255, # Max value to assign
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, # Gaussian weighting is often preferred
        cv2.THRESH_BINARY, # Standard thresholding type
        block_size, # Neighborhood size
        c_value # Constant to subtract
    )

    # Convert back to BGR for consistency and color drawing
    thresh_image_bgr = cv2.cvtColor(thresh_image, cv2.COLOR_GRAY2BGR)

    if path:
        # Ensure directory exists if saving
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, thresh_image_bgr)

    if "annotation" in kwargs:
        annotation = kwargs["annotation"]
        # Draw on a copy
        annotated_image = cv2.drawContours(thresh_image_bgr.copy(), [annotation[1]], 0, (255, 255, 255), 1) # White contour
        # Construct annotated path
        annotated_dir = f"{annotation[0]}/annotated_images/adaptive_threshold/" # Specific directory
        os.makedirs(annotated_dir, exist_ok=True)
        # Use original path's basename for the annotated file name
        annotated_path = os.path.join(annotated_dir, os.path.basename(path)) if path else os.path.join(annotated_dir, "annotated_adaptive_thresh.png") # Default name if no path
        cv2.imwrite(annotated_path, annotated_image)

    return thresh_image_bgr

system_path = "/home/reddy"
prototxt_path = f'{system_path}/Bachelor_Thesis/HED_Files/deploy.prototxt'
caffemodel_path = f'{system_path}/Bachelor_Thesis/HED_Files/hed_pretrained_bsds.caffemodel'
net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
print("HED model loaded successfully")
