import cv2
import numpy as np
import os
import edge_detections
from ultralytics import YOLO
import sys
from shapely.geometry import Polygon  # Need to install: pip install shapely
from collections import defaultdict
import yaml

def calculate_polygon_iou(box1, box2):
    """Calculate IoU between two polygons defined by 8 coordinates (xyxyxyxy)"""
    # Convert boxes to Shapely Polygon objects
    points1 = [(box1[i], box1[i+1]) for i in range(0, len(box1), 2)]
    points2 = [(box2[i], box2[i+1]) for i in range(0, len(box2), 2)]
    
    polygon1 = Polygon(points1)
    polygon2 = Polygon(points2)
    
    # Handle invalid polygons
    if not polygon1.is_valid or not polygon2.is_valid:
        return 0.0
    
    # Calculate intersection and union areas
    try:
        intersection_area = polygon1.intersection(polygon2).area
        union_area = polygon1.area + polygon2.area - intersection_area
        
        # Return IoU
        if union_area <= 0:
            return 0.0
        return intersection_area / union_area
    except Exception as e:
        print(f"Error calculating IoU: {e}")
        return 0.0

def calculate_metrics(gt_boxes, pred_boxes, iou_threshold=0.5):
    """Calculate precision, recall, F1, and mAP based on IoU matches"""
    # Match predictions to ground truth
    matched_gt_indices = set()
    true_positives = 0
    
    # For per-class metrics
    class_metrics = defaultdict(lambda: {"tp": 0, "fp": 0, "gt_count": 0})
    
    # Count ground truths per class
    for gt_class, _ in gt_boxes:
        class_metrics[gt_class]["gt_count"] += 1
    
    # For each prediction, find the best matching ground truth
    for pred_class, pred_box in pred_boxes:
        best_iou = 0
        best_gt_idx = -1
        
        # Compare with each ground truth
        for idx, (gt_class, gt_box) in enumerate(gt_boxes):
            # Only consider matching classes
            if pred_class == gt_class:
                iou = calculate_polygon_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
        
        # If IoU is above threshold, count as true positive
        if best_iou >= iou_threshold and best_gt_idx not in matched_gt_indices:
            true_positives += 1
            matched_gt_indices.add(best_gt_idx)
            class_metrics[pred_class]["tp"] += 1
        else:
            class_metrics[pred_class]["fp"] += 1
    
    # Calculate metrics
    false_positives = len(pred_boxes) - true_positives
    false_negatives = len(gt_boxes) - len(matched_gt_indices)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    # Calculate F1 score
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate AP for each class and then mAP
    ap_values = []
    for cls, metrics in class_metrics.items():
        if metrics["gt_count"] > 0:
            cls_precision = metrics["tp"] / (metrics["tp"] + metrics["fp"]) if (metrics["tp"] + metrics["fp"]) > 0 else 0
            cls_recall = metrics["tp"] / metrics["gt_count"] if metrics["gt_count"] > 0 else 0
            # Simplified AP (single point approximation)
            ap_values.append(cls_precision)
    
    # mAP is the mean of AP values across all classes
    mAP = np.mean(ap_values) if ap_values else 0
    
    return precision, recall, f1_score, mAP, true_positives, false_positives, false_negatives

def resize_pad_image(image, mask=False, new_shape=(640, 640)):
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

def read_label_file(label_path):
    """Read the label file and return class and bounding box coordinates for all objects."""
    gt_boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 9:  # class_id + 8 coordinates (xyxyxyxy)
                class_id = int(parts[0])
                coordinates = [float(x) for x in parts[1:9]]  # Extract 8 coordinates
                gt_boxes.append((class_id, coordinates))
    return gt_boxes

def draw_box(img, box, object_class, color, thickness=2):
    """Draw a polygon box on the image."""
    try:
        pts = np.array([[box[i], box[i+1]] for i in range(0, len(box), 2)], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True, color, thickness)
        if object_class is not None:
            # Draw class label
            label = f"{object_class}"
            cv2.putText(img, label, (int(box[0]), int(box[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    except Exception as e:
        print(f"Error drawing box: {e}")
    return img

def adjust_coordinates(box, scale, top, left):
    """Adjust coordinates based on scale and padding offset"""
    adjusted_box = box.copy()
    for i in range(0, len(box), 2):
        # Apply scale to x and y coordinates then add padding offset
        adjusted_box[i] = box[i] * scale + left
        adjusted_box[i+1] = box[i+1] * scale + top
    return adjusted_box

def process_image(image_path, label_path, model_name, save_image=True):
    """Process a single image and its labels."""
    # Load the original image
    model = models[model_name]["model"]
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Error: Could not load image from {image_path}")
        return None
    
    # Get original image dimensions
    orig_height, orig_width = original_image.shape[:2]

    # Resize and pad the image
    resized_image, scale, top, left = resize_pad_image(original_image, new_shape=(640, 640))

    if models[model_name]["args"] is not None:
        print(f"Preprocessing image for model {model_name}")
        models[model_name]["args"]["image"] = resized_image
        resized_image = preprocess(model_name, models[model_name])

    
    
    # Read ground truth labels
    try:
        gt_boxes = read_label_file(label_path)
        
        # If ground truth coordinates are normalized (0-1), convert to pixel coordinates
        # and adjust for resized image
        gt_boxes_adjusted = []
        for gt_class, gt_box in gt_boxes:
            # gt_class = class_map[gt_class]
            if max(gt_box) <= 1.0:
                # Denormalize to original image coords
                denormalized_box = gt_box.copy()
                for i in range(0, len(gt_box), 2):
                    denormalized_box[i] *= orig_width
                    denormalized_box[i+1] *= orig_height
                
                # Then adjust for resized/padded image
                adjusted_box = adjust_coordinates(denormalized_box, scale, top, left)
                gt_boxes_adjusted.append((gt_class, adjusted_box))
            else:
                # Already in pixel coordinates, just adjust
                adjusted_box = adjust_coordinates(gt_box, scale, top, left)
                gt_boxes_adjusted.append((gt_class, adjusted_box))
        
    except Exception as e:
        print(f"Error reading label file {label_path}: {e}")
        return None
    
    
    
    # Run model to get predictions
    results = model(resized_image, conf=0.9)
    
    # Collect predictions
    pred_boxes = []
    # Draw prediction boxes
    for result in results:
        if hasattr(result, 'obb') and result.obb is not None:
            for box in result.obb:                
                # Convert the box tensor to a list for drawing
                coords = box.xyxyxyxy
                pred_class = int(box.cls.item())
                if pred_class not in class_map:
                    continue
                pred_class = class_map[pred_class]
                # conf = box.conf
                pred_box = coords.cpu().numpy().flatten().tolist()               
                # Add to predictions list for metrics calculation
                pred_boxes.append((pred_class, pred_box))
    
    # Calculate metrics
    precision, recall, f1_score, mAP, tp, fp, fn = calculate_metrics(gt_boxes_adjusted, pred_boxes, iou_threshold=0.5)
    
    model_key = '_'.join(model_name.split('_')[2:])  # Extract the technique name
    
    all_metrics[model_key]['precision'].append(precision)
    all_metrics[model_key]['recall'].append(recall)
    all_metrics[model_key]['f1_score'].append(f1_score)
    all_metrics[model_key]['mAP'].append(mAP)

    if save_image:
        # Create output image
        output_image = resized_image.copy()
        
        # Draw ground truth boxes
        for gt_class, gt_box in gt_boxes_adjusted:
            output_image = draw_box(output_image, gt_box, gt_class, (0, 255, 0), 2)  # GT in green
        # Draw prediction boxes
        for pred_class, pred_box in pred_boxes:
            output_image = draw_box(output_image, pred_box, pred_class, (0, 0, 255), 2)  # Predictions in red
        # Add metrics to the image
        metrics_text1 = f"Precision: {precision:.2f}, Recall: {recall:.2f}"
        metrics_text2 = f"F1 Score: {f1_score:.2f}, mAP@0.5: {mAP:.2f}"
        details_text = f"TP: {tp}, FP: {fp}, FN: {fn}"
        
        # Draw text with background for better visibility
        font_scale = 0.5
        thickness = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 0, 0)  # Blue
        
        # Create background rectangles
        (text_w1, text_h1), _ = cv2.getTextSize(metrics_text1, font, font_scale, thickness)
        (text_w2, text_h2), _ = cv2.getTextSize(metrics_text2, font, font_scale, thickness)
        (text_w3, text_h3), _ = cv2.getTextSize(details_text, font, font_scale, thickness)
        
        # Draw semi-transparent background for text
        overlay = output_image.copy()
        cv2.rectangle(overlay, (5, 5), (max(text_w1, text_w2, text_w3) + 15, 95), (255, 255, 255), -1)
        # cv2.addWeighted(overlay, 0.7, output_image, 0.3, 0, output_image)
        
        # Draw text
        cv2.putText(output_image, metrics_text1, (10, 30), font, font_scale, color, thickness)
        cv2.putText(output_image, metrics_text2, (10, 60), font, font_scale, color, thickness)
        cv2.putText(output_image, details_text, (10, 90), font, font_scale, color, thickness)
    
        return output_image


def preprocess(model_name, model):
    if model["preprocess"]:
        img = model["preprocess"](**model["args"])
        if isinstance(img, list):
            img = img[int(model_name.split("HED")[-1])-1]
    return img

def save_latex(save_path="metrics_table.tex"):
    """
    Generate LaTeX table with metrics and save it to a file.
    
    Args:
        save_path (str): Path where to save the LaTeX file
    """
    # Create the LaTeX content as a string
    latex_content = []
    
    latex_content.append("\\begin{table}[H]")
    latex_content.append("    \\scriptsize")
    latex_content.append(f"    \\caption{{Evaluation Metrics for {test_set}}}")
    latex_content.append("    \\begin{tabular}{lcccccc}")
    latex_content.append("        \\toprule")
    latex_content.append("        \\textbf{Method} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1-Score} & \\textbf{mAP} \\\\")
    latex_content.append("        \\midrule")
    latex_content.append("        \\textbf{Baseline} & & & & \\\\")
    
    # Default color method
    if "control" in all_metrics:
        p_mean = np.mean(all_metrics["control"]['precision'])
        p_std = np.std(all_metrics["control"]['precision'])
        r_mean = np.mean(all_metrics["control"]['recall'])
        r_std = np.std(all_metrics["control"]['recall'])
        f1_mean = np.mean(all_metrics["control"]['f1_score'])
        f1_std = np.std(all_metrics["control"]['f1_score'])
        map_mean = np.mean(all_metrics["control"]['mAP'])
        map_std = np.std(all_metrics["control"]['mAP'])
        latex_content.append(f"        Render & {p_mean:.3f} ± {p_std:.3f} & {r_mean:.3f} ± {r_std:.3f} & {f1_mean:.3f} ± {f1_std:.3f} & {map_mean:.3f} ± {map_std:.3f} \\\\")
    else:
        latex_content.append("        Render & X ± Y & X ± Y & X ± Y & X ± Y \\\\")
    
    latex_content.append("        \\midrule")
    latex_content.append("        \\textbf{Edge Detection} & & & & \\\\")
    
    edge_methods = ["canny", "active_canny", "HED1", "HED2", "adaptive_threshold"]
    method_display_names = {
        "canny": "Canny Edge",
        "active_canny": "Active Canny Edge",
        "HED1": "HED1",
        "HED2": "HED2",
        "adaptive_threshold": "Adaptive Threshold"
    }
    
    # Edge detections
    for method in edge_methods:
        if method in all_metrics:
            p_mean = np.mean(all_metrics[method]['precision'])
            p_std = np.std(all_metrics[method]['precision'])
            r_mean = np.mean(all_metrics[method]['recall'])
            r_std = np.std(all_metrics[method]['recall'])
            f1_mean = np.mean(all_metrics[method]['f1_score'])
            f1_std = np.std(all_metrics[method]['f1_score'])
            map_mean = np.mean(all_metrics[method]['mAP'])
            map_std = np.std(all_metrics[method]['mAP'])
            display_name = method_display_names.get(method, method)
            latex_content.append(f"        {display_name} & {p_mean:.3f} ± {p_std:.3f} & {r_mean:.3f} ± {r_std:.3f} & {f1_mean:.3f} ± {f1_std:.3f} & {map_mean:.3f} ± {map_std:.3f} \\\\")
        else:
            display_name = method_display_names.get(method, method)
            latex_content.append(f"        {display_name} & X ± Y & X ± Y & X ± Y & X ± Y \\\\")
    
    latex_content.append("        \\midrule")
    latex_content.append("        \\textbf{Style Transfer} & & & & \\\\")
    
    # Style transfer methods
    style_methods = ["anime_style", "contour_style", "opensketch_style"]
    style_display_names = {
        "anime_style": "Anime Style",
        "contour_style": "Contour Style",
        "opensketch_style": "OpenSketch Style"
    }
    
    for method in style_methods:
        if method in all_metrics:
            p_mean = np.mean(all_metrics[method]['precision'])
            p_std = np.std(all_metrics[method]['precision'])
            r_mean = np.mean(all_metrics[method]['recall'])
            r_std = np.std(all_metrics[method]['recall'])
            f1_mean = np.mean(all_metrics[method]['f1_score'])
            f1_std = np.std(all_metrics[method]['f1_score'])
            map_mean = np.mean(all_metrics[method]['mAP'])
            map_std = np.std(all_metrics[method]['mAP'])
            display_name = style_display_names.get(method, method)
            latex_content.append(f"        {display_name} & {p_mean:.3f} ± {p_std:.3f} & {r_mean:.3f} ± {r_std:.3f} & {f1_mean:.3f} ± {f1_std:.3f} & {map_mean:.3f} ± {map_std:.3f} \\\\")
        else:
            display_name = style_display_names.get(method, method)
            latex_content.append(f"        {display_name} & X ± Y & X ± Y & X ± Y & X ± Y \\\\")
    
    latex_content.append("        \\bottomrule")
    latex_content.append("    \\end{tabular}")
    latex_content.append("\\end{table}")
    
    # Join all lines with newlines
    full_content = "\n".join(latex_content)
    
    # Save to file
    with open(save_path, 'w') as file:
        file.write(full_content)
    
    print(f"LaTeX table saved to {save_path}")

    # Also print to console for convenience
    print("\nTable content:")
    print(full_content)

def run_metrics(model_name, save_path, save_image):

    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for filename in os.listdir(images_dir):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            image_files.append(filename)
    
    if not image_files:
        print(f"No image files found in {images_dir}")
        return

    image_files.sort()
    
    print(f"Found {len(image_files)} images to process")
    
    processed_count = 0
    
    # Track overall metrics for summary
    total_precision = []
    total_recall = []
    total_f1 = []
    total_mAP = []
    
    # Process each image
    for img_file in image_files:
        img_path = os.path.join(images_dir, img_file)
        
        # Get the base filename without extension
        base_name = os.path.splitext(img_file)[0]
        
        # Find corresponding label file
        label_path = None
        for ext in ['.txt']:
            potential_path = os.path.join(labels_dir, base_name + ext)
            if os.path.exists(potential_path):
                label_path = potential_path
                break
                
        if label_path is None:
            print(f"No label file found for {img_file}")
            continue
        
        print(f"Processing {img_file} with label {os.path.basename(label_path)}")
        
        # Process the image
        result_img= process_image(img_path, label_path, model_name, save_image=save_image)
        
        if result_img is not None:
            # Save the result
            output_path = os.path.join(save_path, f"{base_name}.jpg")
            cv2.imwrite(output_path, result_img)
            print(f"Saved result to {output_path}")
            
            processed_count += 1

    print(f"\nProcessed {processed_count} images")
    

if __name__ == "__main__":
    name_map = {"bit_holder": 5,
                "bottle_holder": 4,
                "cup_holder": 1,
                "cutter_holder": 6,
                "scissor_holder": 0,
                "tool_holder": 2,
                }
    root_dir = "/home/reddy/Bachelor_Thesis/test_files/labeled_data/"
    test_sets = os.listdir(root_dir)

    for test_set in test_sets:
        print(f"Processing test set : {test_set}")
        images_dir = f"{root_dir}/{test_set}/train/images"
        labels_dir = f"{root_dir}/{test_set}/train/labels"
        with open(f"{root_dir}/{test_set}/data.yaml", 'r') as f:
            yaml_data = yaml.safe_load(f)
        
        class_map = {}
        for class_no, class_name in yaml_data["names"].items():
            class_map[name_map[class_name]] = class_no

        print("Class map:", class_map)

        for size in ["n", "s", "m", "l", "x"]:
            dataset = "8object"
            # size = "x"
            models = {
                    f"{dataset}_{size}_control" : {"model" : "",  "preprocess" : None, "args" : None}, 
                    f"{dataset}_{size}_canny" : {"model" : "", "preprocess" : edge_detections.canny_edge, "args" : {"image" : ""}}, 
                    f"{dataset}_{size}_active_canny" : {"model" : "", "preprocess" : edge_detections.active_canny, "args" : {"image" : ""}},
                    # f"{dataset}_{size}_HED1" : {"model" : "", "preprocess" : edge_detections.hed_edge, "args" : {"image" : "", "layer" : 1}},
                    # f"{dataset}_{size}_HED2" : {"model" : "", "preprocess" : edge_detections.hed_edge, "args" : {"image" : "", "layer" : 2}},
                    f"{dataset}_{size}_anime_style" : {"model" : "", "preprocess" : edge_detections.info_drawing, "args" : {"image" : "", "model_name" : "anime_style"}},
                    f"{dataset}_{size}_contour_style" : {"model" : "", "preprocess" : edge_detections.info_drawing, "args" : {"image" : "", "model_name" : "contour_style"}},
                    f"{dataset}_{size}_opensketch_style" : {"model" : "", "preprocess" : edge_detections.info_drawing, "args" : {"image" : "", "model_name" : "opensketch_style"}},
                    f"{dataset}_{size}_adaptive_threshold" : {"model" : "", "preprocess" : edge_detections.adaptive_threshold, "args" : {"image" : ""}},
                    }

            for model_name in models:
                #Uncomment to use normal model
                models[model_name]["model"] = YOLO(f'/home/reddy/Bachelor_Thesis/trains/{dataset}/{dataset}_{size}/{model_name}/weights/best.pt')
                #Uncomment to use openvino model
                # models[model_name]["model"] = YOLO(f'/home/reddy/Bachelor_Thesis/trains/{dataset}/{dataset}_{size}/{model_name}/weights/best_openvino_model')
                
                print(f"Loaded model: {model_name}")
            #for table
            all_metrics = {}
            for model_name in models:
                model_key = '_'.join(model_name.split('_')[2:])
                all_metrics[model_key] = {
                'precision': [], 'recall': [], 'f1_score': [], 'mAP': []
                }

            for model_name in models:
                output_dir = f"/home/reddy/Bachelor_Thesis/benchmarks/metrics/{test_set}/{dataset}_{size}"

                os.makedirs(f"{output_dir}/images/{'_'.join(model_name.split('_')[2:])}", exist_ok=True)

                run_metrics(model_name, save_path=f"{output_dir}/images/{'_'.join(model_name.split('_')[2:])}", save_image=False)
            
            save_latex(f"{output_dir}/{dataset}_{size}_latex.tex")