import cv2
import numpy as np
import os
from ultralytics import YOLO
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
from django.conf import settings

class CrackDetector:
    def __init__(self):
        # Path to your model inside the static folder
        self.model_path = os.path.join(settings.BASE_DIR, 'detector', 'static', 'detector', 'models', 'crack_model.pt')
        self.model = YOLO(self.model_path)
        
    def process_image(self, input_path, output_path):
        """
        Processes the image, detects cracks, measures width, 
        and saves the visual result.
        """
        image = cv2.imread(input_path)
        if image is None:
            return None, "Error: Could not read image."

        vis = image.copy()
        
        # 1. Run YOLOv8-segmentation
        results = self.model(image)[0]

        if results.masks is None:
            return None, "No cracks detected."

        # 2. Extract Mask
        mask = results.masks.data[0].cpu().numpy()
        mask = (mask > 0.5).astype(np.uint8)

        # Resize mask to original image size
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # 3. Measurement Logic (Skeletonization)
        skeleton = skeletonize(mask)
        distance = distance_transform_edt(mask)

        # width_px = 2 * distance at skeleton points
        width_px = 2 * distance[skeleton]
        
        if len(width_px) == 0:
            return None, "Width could not be calculated."

        # For this version, we are using pixel width. 
        # (We can add coin-calibration logic in the next step!)
        avg_width = np.mean(width_px)
        max_width = np.max(width_px)

        # 4. Create Visualization Overlay
        crack_overlay = np.zeros_like(vis)
        crack_overlay[:, :, 2] = mask * 255  # Red color for crack
        vis = cv2.addWeighted(vis, 1.0, crack_overlay, 0.5, 0)

        # Draw results on image
        cv2.putText(vis, f"Max Width: {max_width:.2f} px", (40, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(vis, f"Avg Width: {avg_width:.2f} px", (40, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 5. Save and Return
        cv2.imwrite(output_path, vis)
        
        return {
            'max_width': round(max_width, 2),
            'avg_width': round(avg_width, 2),
        }, None