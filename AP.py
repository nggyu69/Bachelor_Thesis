import os
import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import rotate, translate
from ultralytics import YOLO
import math

def yolo_obb_to_polygon(box):
    """
    Convert an oriented bounding box in YOLO format [x_center, y_center, width, height, angle] 
    into a Shapely Polygon.
    """
    cx, cy, w, h, angle = box
    theta = np.deg2rad(angle)

    polygon = Polygon([
        (-w / 2, -h / 2),
        ( w / 2, -h / 2),
        ( w / 2,  h / 2),
        (-w / 2,  h / 2)])

    polygon = rotate(polygon, theta, use_radians=True)
    polygon = translate(polygon, cx, cy)
    return polygon

def corners_to_xywha(corners_xyxyxyxy):
    """
    corners_xyxyxyxy: (x1, y1, x2, y2, x3, y3, x4, y4)
                      representing a single rotated rectangle in order.
    Returns (xc, yc, w, h, angle_radians).
    """
    # 1. Group corners
    pts = [
        (corners_xyxyxyxy[0], corners_xyxyxyxy[1]),
        (corners_xyxyxyxy[2], corners_xyxyxyxy[3]),
        (corners_xyxyxyxy[4], corners_xyxyxyxy[5]),
        (corners_xyxyxyxy[6], corners_xyxyxyxy[7])
    ]
    
    # 2. Center
    x_c = sum(x for x, _ in pts) / 4.0
    y_c = sum(y for _, y in pts) / 4.0
    
    # 3. Edge vectors
    dx1 = pts[1][0] - pts[0][0]
    dy1 = pts[1][1] - pts[0][1]
    dx2 = pts[2][0] - pts[1][0]
    dy2 = pts[2][1] - pts[1][1]
    
    # 4. Compute side lengths
    w = math.sqrt(dx1**2 + dy1**2)
    h = math.sqrt(dx2**2 + dy2**2)
    
    # If you want to ensure w >= h (or the opposite), you can reorder them here
    # but you'll also need to adjust the angle accordingly.
    
    # 5. Angle from edge 1
    angle = math.atan2(dy1, dx1)  # Radians, range: -pi to pi
    
    return (x_c, y_c, w, h, angle)


def obb_iou(pred_box, gt_box):
    """
    Compute the Intersection over Union (IoU) between a predicted YOLO-OBB box and a ground truth box,
    both in [x_center, y_center, width, height, angle] format.
    """
    poly_pred = yolo_obb_to_polygon(pred_box)
    poly_gt = yolo_obb_to_polygon(gt_box)
    
    # Uncomment these lines for debugging if needed:
    # print("Pred:", pred_box, "GT:", gt_box)
    # print()
    
    if not poly_pred.is_valid or not poly_gt.is_valid:
        return 0.0
    
    inter_area = poly_pred.intersection(poly_gt).area
    union_area = poly_pred.area + poly_gt.area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def load_ground_truth_labels(label_path, image_shape):
    """
    Load ground truth labels from a file in the standard OBB format.
    Each line should contain 9 values:
      class_id x1 y1 x2 y2 x3 y3 x4 y4
    The (x, y) coordinates are normalized (between 0 and 1).
    They are converted to absolute pixel values using the provided image shape (height, width),
    then converted to [x_center, y_center, width, height, angle] via cv2.minAreaRect.
    """
    labels = []
    height, width = image_shape[0], image_shape[1]
    with open(label_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 9:
            continue  # Skip lines not matching expected format
        class_id = 2
        # Multiply normalized coordinates by image dimensions.
        corners = []
        for i, val in enumerate(parts[1:]):
            num = float(val)
            # Even index: x coordinate, odd index: y coordinate.
            if i % 2 == 0:
                corners.append(num * width)
            else:
                corners.append(num * height)
        xywha = corners_to_xywha(corners)
        labels.append({"class_id": class_id, "bbox": xywha})
    return labels

def evaluate_detections(all_predictions, ground_truths, iou_threshold=0.5):
    """
    Evaluate detections over a set of images.
    
    all_predictions: list of dicts with keys:
        - "image_id": image filename or identifier
        - "class_id": predicted class id
        - "bbox": predicted bbox in [x_center, y_center, width, height, angle] format
        - "confidence": detection confidence score
    ground_truths: dict mapping image_id to a list of ground truth dicts,
                   each with keys "class_id" and "bbox" (in [x, y, w, h, angle] format)
                   
    Returns overall precision, recall, AP, and the per-threshold precision and recall arrays.
    """
    # Mark all ground truths as not yet detected
    for image_id in ground_truths:
        for gt in ground_truths[image_id]:
            gt["detected"] = False

    # Sort predictions by descending confidence
    all_predictions.sort(key=lambda x: x["confidence"], reverse=True)
    
    TP = np.zeros(len(all_predictions))
    FP = np.zeros(len(all_predictions))
    
    num_ground_truths = sum(len(ground_truths[img]) for img in ground_truths)
    
    # Match each prediction with the best available ground truth.
    for i, pred in enumerate(all_predictions):
        image_id = pred["image_id"]
        pred_bbox = pred["bbox"]
        pred_class = pred["class_id"]
        best_iou = 0.0
        best_gt = None
        
        for gt in ground_truths.get(image_id, []):
            if gt["class_id"] != pred_class:
                continue
            iou_val = obb_iou(pred_bbox, gt["bbox"])
            if iou_val > best_iou:
                best_iou = iou_val
                best_gt = gt
        
        if best_iou >= iou_threshold and best_gt is not None and not best_gt.get("detected", False):
            TP[i] = 1
            best_gt["detected"] = True
        else:
            FP[i] = 1
    
    # Compute cumulative sums for precision-recall curve
    cum_TP = np.cumsum(TP)
    cum_FP = np.cumsum(FP)
    
    # Compute precision and recall for each prediction
    precisions = cum_TP / (cum_TP + cum_FP + 1e-10)
    recalls = cum_TP / (num_ground_truths + 1e-10)
    
    # Compute AP using Pascal VOC 11-point interpolation
    recall_levels = np.linspace(0, 1, 11)
    precisions_interpolated = []
    for r_level in recall_levels:
        precisions_at_recall = precisions[recalls >= r_level]
        if precisions_at_recall.size > 0:
            precisions_interpolated.append(np.max(precisions_at_recall))
        else:
            precisions_interpolated.append(0.0)
    AP = np.mean(precisions_interpolated)
    
    final_precision = cum_TP[-1] / (cum_TP[-1] + cum_FP[-1]) if (cum_TP[-1] + cum_FP[-1]) > 0 else 0.0
    final_recall = cum_TP[-1] / num_ground_truths if num_ground_truths > 0 else 0.0
    
    return final_precision, final_recall, AP, precisions, recalls

def draw_obb(image, box, color=(0, 255, 0), thickness=2):
    """
    Draw an oriented bounding box on the image.
    box: [x_center, y_center, width, height, angle]
    """
    poly = yolo_obb_to_polygon(box)
    pts = np.array(list(poly.exterior.coords), np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(image, [pts], isClosed=True, color=color, thickness=thickness)

def draw_boxes(image, boxes, color=(0, 255, 0), thickness=2):
    """
    Draw multiple oriented bounding boxes on the image.
    boxes: list of boxes in [x_center, y_center, width, height, angle] format.
    """
    for box in boxes:
        draw_obb(image, box, color=color, thickness=thickness)

def main():
    # Set your directories and model path here.
    image_dir = "/home/reddy/Bachelor_Thesis/test_files/tool_holder_white_labeled/images"    # Directory containing your images
    label_dir = "/home/reddy/Bachelor_Thesis/test_files/tool_holder_white_labeled/labels"      # Directory containing your ground truth label files
    model_path = "/home/reddy/Bachelor_Thesis/trains/8object/8object_x/8object_x_control/weights/best.pt"  # Path to your trained YOLO-OBB model    
    
    # Create directories for saving visualizations if they don't exist.
    pred_save_dir = "/home/reddy/Bachelor_Thesis/evaluation/predicted"
    gt_save_dir = "/home/reddy/Bachelor_Thesis/evaluation/ground_truth"
    os.makedirs(pred_save_dir, exist_ok=True)
    os.makedirs(gt_save_dir, exist_ok=True)
    
    # List image paths (assumes images are .jpg or .png)
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
                   if f.lower().endswith((".jpg", ".png"))]
    
    # Load the YOLO-OBB model
    model = YOLO(model_path)
    
    all_predictions = []  # Collect predictions across all images.
    ground_truths = {}    # Dictionary mapping image_id to its ground truth labels.
    predicted_boxes_by_image = {}  # For drawing predicted boxes later.
    
    for image_path in image_paths:
        image_id = os.path.basename(image_path)
        # Read image to obtain its dimensions.
        img = cv2.imread(image_path)
        if img is None:
            continue  # Skip if the image cannot be loaded.
        height, width = img.shape[:2]

        # Run inference on the image.
        result = model(image_path, conf=0.9)[0]
        result_img = result.plot()  # Visualize prediction
        cv2.imwrite(f"/home/reddy/Bachelor_Thesis/evaluation/predicted_obb/{image_id}", result_img)
        # Extract predictions. Adjust attribute names if needed.
        # We assume each detected box has: .cls, .xywha (or .xywhr), and .conf.
        preds = []
        for box in result.obb:
            pred_dict = {
                "class_id": int(box.cls),
                "bbox": box.xyxyxyxy.tolist()[0],
                "confidence": float(box.conf),
                "image_id": image_id
            }
            preds.append(pred_dict)
        all_predictions.extend(preds)
        predicted_boxes_by_image[image_id] = [p["bbox"] for p in preds]
        
        # Load corresponding ground truth label file (expects the file has the same basename with .txt)
        label_filename = os.path.splitext(image_id)[0] + ".txt"
        label_path = os.path.join(label_dir, label_filename)
        if os.path.exists(label_path):
            gt_labels = load_ground_truth_labels(label_path, (height, width))
        else:
            gt_labels = []
        ground_truths[image_id] = gt_labels
        
        # For visualization, draw the predicted boxes and ground truth boxes and save images.
        img_pred = img.copy()
        img_gt = img.copy()
        
        # Draw predicted boxes in blue.
        draw_boxes(img_pred, predicted_boxes_by_image.get(image_id, []), color=(255, 0, 0), thickness=2)
        # Draw ground truth boxes in red.
        gt_boxes = [gt["bbox"] for gt in gt_labels]
        draw_boxes(img_gt, gt_boxes, color=(0, 0, 255), thickness=2)
        
        # Save images
        cv2.imwrite(os.path.join(pred_save_dir, image_id), img_pred)
        cv2.imwrite(os.path.join(gt_save_dir, image_id), img_gt)
    
    # Evaluate detections and compute metrics.
    precision, recall, AP, precisions, recalls = evaluate_detections(all_predictions, ground_truths, iou_threshold=0.5)
    
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"AP:        {AP:.3f}")

if __name__ == "__main__":
    main()
