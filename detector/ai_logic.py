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
        vis = image.copy()

        # 1. Coin Detection for Calibration
        coin_results = self.coin_model(image)[0]
        if len(coin_results.boxes) == 0:
            return None, "Reference coin not found. Calibration failed."
        
        # Calculate mm per pixel
        box = coin_results.boxes.xyxy[0].cpu().numpy().astype(int)
        w, h = box[2] - box[0], box[3] - box[1]
        mm_per_pixel = self.COIN_DIAMETER_MM / ((w + h) / 2)

        # 2. Crack Segmentation
        crack_results = self.crack_model(image)[0]
        if crack_results.masks is None:
            return None, "No cracks detected in the image."

        mask = crack_results.masks.data[0].cpu().numpy()
        mask = cv2.resize((mask > 0.5).astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # 3. Measurement (Skeletonization)
        skeleton = skeletonize(mask)
        distance = distance_transform_edt(mask)
        width_mm = (2 * distance[skeleton]) * mm_per_pixel # Convert to mm

        # 4. Visualization
        crack_overlay = np.zeros_like(vis)
        crack_overlay[:, :, 2] = mask * 255 # Red Crack
        vis = cv2.addWeighted(vis, 1.0, crack_overlay, 0.5, 0)
        cv2.rectangle(vis, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 3) # Yellow Coin Box

        avg_w = np.mean(width_mm)
        max_w = np.max(width_mm)

        # Draw UI text
        cv2.putText(vis, f"Max: {max_w:.2f}mm", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.imwrite(output_path, vis)
        
        return {'max_width': round(max_w, 2), 'avg_width': round(avg_w, 2), 'scale': round(mm_per_pixel, 4)}, None