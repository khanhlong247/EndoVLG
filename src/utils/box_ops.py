import cv2
import numpy as np
from sklearn.cluster import KMeans

def get_boxes_from_mask(mask, target_color, max_boxes=3, threshold=0.60):
    """
    Extract bounding boxes from a mask using K-Means.
    Input: mask (H,W,3) or (H,W), target_color [B,G,R]
    Output: list [x1, y1, x2, y2]
    """
    if len(mask.shape) == 3:
        lower = np.maximum(np.array(target_color) - 10, 0)
        upper = np.minimum(np.array(target_color) + 10, 255)
        binary_mask = cv2.inRange(mask, lower, upper)
    else:
        binary_mask = (mask > 0).astype(np.uint8) * 255

    points = np.column_stack(np.where(binary_mask > 0)) # (y, x)
    
    if len(points) == 0:
        return []

    def get_bbox(pts):
        y_min, x_min = pts.min(axis=0)
        y_max, x_max = pts.max(axis=0)
        return [int(x_min), int(y_min), int(x_max), int(y_max)]

    full_box = get_bbox(points)
    x1, y1, x2, y2 = full_box
    box_area = (x2 - x1) * (y2 - y1)
    mask_area = len(points)

    if box_area == 0: return [full_box]
    
    coverage = mask_area / box_area

    if coverage >= threshold:
        return [full_box]

    best_boxes = [full_box]
    best_avg_coverage = coverage

    for k in range(2, max_boxes + 1):
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(points)
            
            current_boxes = []
            total_box_area = 0
            
            for i in range(k):
                cluster_points = points[labels == i]
                if len(cluster_points) < 10: continue
                c_box = get_bbox(cluster_points)
                w = c_box[2] - c_box[0]
                h = c_box[3] - c_box[1]
                total_box_area += (w * h)
                current_boxes.append(c_box)

            if total_box_area > 0:
                new_coverage = mask_area / total_box_area
                if new_coverage > best_avg_coverage + 0.05:
                    best_avg_coverage = new_coverage
                    best_boxes = current_boxes
                else:
                    break
        except Exception:
            continue

    return best_boxes