# from ultralytics import YOLO
# import cv2

# model = YOLO('/home/reddy/Bachelor_Thesis/trains/train_combined/weights/best.pt')

# img = cv2.imread('/data/reddy/Bachelor_Thesis/img5.jpg')

# results = model.predict(img, save = True, project = '/home/reddy/Bachelor_Thesis/predictions', conf = 0.8)



from ultralytics import YOLO
import cv2
import os
import numpy as np

# Load the trained model
model = YOLO('/home/reddy/Bachelor_Thesis/trains/transferred_train/weights/best.pt')

# Directory containing the images to test
image_dir = '/data/reddy/Bachelor_Thesis/tool_b'
output_dir = '/home/reddy/Bachelor_Thesis/predictions'
os.makedirs(output_dir, exist_ok=True)

# Set the grid size (e.g., 3 for 3x3, 4 for 4x4, etc.)
grid_size = 4  # Change to 3, 4, 5, or 6 based on your needs

# List to hold all prediction results for stitching
image_results = []

# Load images in their original size and predict
for filename in sorted(os.listdir(image_dir)):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Load image in its original size
        img_path = os.path.join(image_dir, filename)
        img = cv2.imread(img_path)

        # Run prediction on the original-sized image, saving results in a single folder
        results = model.predict(img, save=True, save_dir=output_dir, conf=0.8)
        
        # Save the visualized result and add it to the list for stitching
        result_image = results[0].plot()  # Visualize prediction
        image_results.append(result_image)

# Calculate the number of images required for the grid
required_images = grid_size * grid_size

# Add blank images if we have fewer than required images
if len(image_results) < required_images:
    # Get the shape of the first image as a reference for creating blank images
    ref_height, ref_width = image_results[0].shape[:2]
    blank_image = np.zeros((ref_height, ref_width, 3), dtype=np.uint8)  # Black image placeholder
    image_results.extend([blank_image] * (required_images - len(image_results)))

# Create the grid with filled blank images if necessary
rows = [np.hstack(image_results[i*grid_size:(i+1)*grid_size]) for i in range(grid_size)]
stitched_image = np.vstack(rows)  # Stack the rows vertically to form the grid

# Save the stitched image
stitched_image_path = os.path.join(output_dir, f'stitched_results_{image_dir.split("/")[-1]}_{grid_size}x{grid_size}.jpg')
cv2.imwrite(stitched_image_path, stitched_image)

print(f"Stitched {grid_size}x{grid_size} image saved at: {stitched_image_path}")




