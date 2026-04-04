import cv2
import numpy as np
import os
from ultralytics import YOLO
from skimage.morphology import skeletonize
from django.conf import settings

# ---------------- HELPERS ----------------

def instance_masks_from_result(result, shape):
    if result.masks is None:
        return []
    masks = result.masks.data.cpu().numpy()
    h, w = shape
    out = []
    for m in masks:
        m = (m > 0.5).astype(np.uint8) * 255
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        out.append((m > 127).astype(np.uint8))
    return out

def compute_width(mask):
    if mask.sum() == 0:
        return None, None, None
    sk = skeletonize(mask > 0)
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    ys, xs = np.where(sk > 0)
    if len(xs) == 0:
        return None, None, None
    widths = dist[ys, xs] * 2
    idx = np.argmax(widths)
    # Ensure coordinates are standard Python ints to prevent OpenCV crashes
    return float(widths[idx]), (int(xs[idx]), int(ys[idx])), widths

# ---------------- MAIN DETECTOR CLASS ----------------

class CrackDetector:
    def __init__(self):
        self.coin_model_path = os.path.join(settings.BASE_DIR, 'detector', 'static', 'detector', 'models', 'coin_model.pt')
        self.crack_model_path = os.path.join(settings.BASE_DIR, 'detector', 'static', 'detector', 'models', 'crack_model.pt')
        self.coin_model = YOLO(self.coin_model_path)
        self.crack_model = YOLO(self.crack_model_path)

    def process_image(self, input_path, output_path, conf=0.25, iou=0.5, coin_diameter=18.5, mode='crack_only'):
        img = cv2.imread(input_path)
        if img is None: 
            return None, "Error: Could not read image."
        
        h_orig, w_orig = img.shape[:2]
        vis = img.copy()

        # ==========================================
        # MODE 1: CRACK DETECTION ONLY
        # ==========================================
        if mode == 'crack_only':
            crack_results = self.crack_model(img, conf=conf, iou=iou)[0]
            masks = instance_masks_from_result(crack_results, (h_orig, w_orig))
            
            crack_detected = len(masks) > 0
            
            # Extract actual model confidence
            model_conf = 0.0
            if crack_detected and crack_results.boxes is not None and len(crack_results.boxes.conf) > 0:
                model_conf = round(float(crack_results.boxes.conf.max().item()), 3)
            
            if crack_detected:
                merged_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
                for m in masks:
                    merged_mask = np.logical_or(merged_mask, m)
                merged_mask = merged_mask.astype(np.uint8)
                
                # Draw only the red mask overlay
                crack_overlay = np.zeros_like(vis)
                crack_overlay[merged_mask > 0] = [0, 0, 255] 
                vis = cv2.addWeighted(vis, 1.0, crack_overlay, 0.4, 0)
            
            # Resize for web display
            h, w = vis.shape[:2]
            max_width = 720
            if w > max_width:
                scale_ratio = max_width / w
                new_h = int(h * scale_ratio)
                vis = cv2.resize(vis, (max_width, new_h), interpolation=cv2.INTER_AREA)

            cv2.imwrite(output_path, vis)
            
            return {
                'mode': 'crack_only',
                'crack_detected': crack_detected,
                'model_conf': model_conf
            }, None

        # ==========================================
        # MODE 2: DETECTION + SEVERITY (Needs Coin)
        # ==========================================
        else:
            # 1. Coin Detection & Calibration 
            coin_results = self.coin_model(img, conf=conf, iou=iou)[0]
            if coin_results.boxes is None or len(coin_results.boxes) == 0:
                return None, "Reference coin not found. Ensure coin is visible or use 'Crack Detection Only' mode."
            
            boxes = coin_results.boxes.xyxy.cpu().numpy().tolist()
            box = boxes[0] 
            x1, y1, x2, y2 = box
            
            px = max(x2 - x1, y2 - y1)
            if px <= 0:
                return None, "Invalid coin dimensions."
                
            mm_per_pixel = float(coin_diameter / px)

            # 2. Crack Segmentation 
            crack_results = self.crack_model(img, conf=conf, iou=iou)[0]
            masks = instance_masks_from_result(crack_results, (h_orig, w_orig))

            if len(masks) == 0:
                return None, "No cracks detected."

            # Extract actual model confidence
            model_conf = 0.0
            if crack_results.boxes is not None and len(crack_results.boxes.conf) > 0:
                model_conf = round(float(crack_results.boxes.conf.max().item()), 3)

            merged_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
            for m in masks:
                merged_mask = np.logical_or(merged_mask, m)
            merged_mask = merged_mask.astype(np.uint8)

            # 3. Measurement
            width_px, point, all_widths = compute_width(merged_mask)
            if width_px is None:
                return None, "Measurement failed."

            # Explicitly cast to standard Python floats to prevent JSON serialization errors
            max_w_mm = float(width_px * mm_per_pixel)
            avg_w_mm = float(np.mean(all_widths) * mm_per_pixel)
            max_x, max_y = point 

            # 4. Updated Severity Classification Logic
            if max_w_mm < 0.10:
                severity_label = "Hairline crack"
                is_safe = True
            elif max_w_mm <= 0.30:
                severity_label = "Fine / Minor"
                is_safe = True
            elif max_w_mm <= 0.50:
                severity_label = "Moderate"
                is_safe = False
            else:
                severity_label = "Severe"
                is_safe = False

            # 5. Visualization (Mask + Box + Green Dot)
            crack_overlay = np.zeros_like(vis)
            crack_overlay[merged_mask > 0] = [0, 0, 255] 
            vis = cv2.addWeighted(vis, 1.0, crack_overlay, 0.4, 0)
            
            cv2.circle(vis, (max_x, max_y), 10, (0, 255, 0), -1) 
            cv2.circle(vis, (max_x, max_y), 15, (255, 255, 255), 2) 
            
            cv2.putText(vis, f"MAX: {max_w_mm:.2f}mm", (max_x + 20, max_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)

            h, w = vis.shape[:2]
            max_width_display = 720
            if w > max_width_display:
                scale_ratio = max_width_display / w
                new_h = int(h * scale_ratio)
                vis = cv2.resize(vis, (max_width_display, new_h), interpolation=cv2.INTER_AREA)

            cv2.imwrite(output_path, vis)
            
            return {
                'mode': 'crack_severity',
                'crack_detected': True,
                'max_width': round(max_w_mm, 2), 
                'avg_width': round(avg_w_mm, 2), 
                'scale': round(mm_per_pixel, 4),
                'severity': severity_label,
                'is_safe': is_safe,
                'model_conf': model_conf
            }, None