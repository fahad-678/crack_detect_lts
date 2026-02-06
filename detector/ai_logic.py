import cv2
import numpy as np
import os
from ultralytics import YOLO
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
from django.conf import settings

class CrackDetector:
    def __init__(self):
        # Load both models
        self.coin_model_path = os.path.join(settings.BASE_DIR, 'detector', 'static', 'detector', 'models', 'coin_model.pt')
        self.crack_model_path = os.path.join(settings.BASE_DIR, 'detector', 'static', 'detector', 'models', 'crack_model.pt')
        
        self.coin_model = YOLO(self.coin_model_path)
        self.crack_model = YOLO(self.crack_model_path)
        self.COIN_DIAMETER_MM = 18.51 # Standard measurement

    def process_image(self, input_path, output_path):
        image = cv2.imread(input_path)
        if image is None: return None, "Error: Could not read image."
        
        h_orig, w_orig = image.shape[:2]
        vis = image.copy()

        # 1. Coin Detection & Calibration
        coin_results = self.coin_model(image)[0]
        if len(coin_results.boxes) == 0:
            return None, "Reference coin not found."
        
        box = coin_results.boxes.xyxy[0].cpu().numpy().astype(int)
        mm_per_pixel = self.COIN_DIAMETER_MM / (((box[2]-box[0]) + (box[3]-box[1])) / 2)

        # 2. Crack Segmentation
        crack_results = self.crack_model(image)[0]
        if crack_results.masks is None:
            return None, "No cracks detected."

        mask_raw = crack_results.masks.data[0].cpu().numpy()
        mask = cv2.resize(mask_raw, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
        mask = (mask > 0.5).astype(np.uint8)

        # 3. Measurement & Finding Max Point
        skeleton = skeletonize(mask)
        distance = distance_transform_edt(mask)
        
        # Get all skeleton coordinates
        coords = np.column_stack(np.where(skeleton)) 
        widths_px = 2 * distance[skeleton]
        
        if len(widths_px) == 0:
            return None, "Measurement failed."

        # Find index of the maximum width
        max_idx = np.argmax(widths_px)
        max_w_mm = widths_px[max_idx] * mm_per_pixel
        avg_w_mm = np.mean(widths_px) * mm_per_pixel
        
        # Target coordinate (y, x)
        max_y, max_x = coords[max_idx]

        # 4. Visualization
        # Overlay the crack in Red
        crack_overlay = np.zeros_like(vis)
        crack_overlay[mask > 0] = [0, 0, 255] 
        vis = cv2.addWeighted(vis, 1.0, crack_overlay, 0.4, 0)
        
        # Highlight the MAX WIDTH point in Bright Green
        # We draw a circle and a label exactly where the max width was found
        cv2.circle(vis, (max_x, max_y), 10, (0, 255, 0), -1) # Solid green dot
        cv2.circle(vis, (max_x, max_y), 15, (255, 255, 255), 2) # White outer ring
        
        cv2.putText(vis, f"MAX: {max_w_mm:.2f}mm", (max_x + 20, max_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        # Draw the coin reference box in Yellow
        cv2.rectangle(vis, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 2)

        cv2.imwrite(output_path, vis)
        
        return {
            'max_width': round(max_w_mm, 2), 
            'avg_width': round(avg_w_mm, 2), 
            'scale': round(mm_per_pixel, 4)
        }, None