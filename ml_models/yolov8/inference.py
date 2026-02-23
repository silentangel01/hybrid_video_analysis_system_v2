"""Run inference using loaded YOLOv8 models.""" 
# ml_models/yolov8/inference.py

from typing import List, Tuple, Optional
import cv2
import numpy as np

class YOLOInference:
    """
    Wrapper for running inference on YOLOv8 models.
    Returns standardized results for downstream processing.
    """

    @staticmethod
    def run_detection(
        model,
        image: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
        """
        Run object detection on single image frame.
        Args:
            model: Loaded YOLO model instance
            image: OpenCV image (numpy array)
            conf_threshold: Confidence threshold for filtering
            iou_threshold: IOU threshold for NMS
        Returns:
            List of detections: [(class_name, confidence, (x1,y1,x2,y2)), ...]
        """
        if model is None:
            return []

        # Run inference
        results = model.predict(
            source=image,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False  # Suppress console output
        )

        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                class_name = result.names[cls_id]
                detections.append((class_name, conf, (x1, y1, x2, y2)))

        return detections
