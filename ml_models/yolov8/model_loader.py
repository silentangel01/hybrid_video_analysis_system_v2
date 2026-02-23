# ml_models/yolov8/model_loader.py
"""Load YOLOv8 models from weights directory."""

from ultralytics import YOLO
import os
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class YOLOModelLoader:
    """
    Load and manage multiple YOLOv8 models for different detection tasks.
    Supports: vehicle detection, smoke/flame detection, etc.
    Models are loaded once and reused for efficiency.
    👉 Added robust class name mapping support.
    """

    def __init__(self, weights_dir: Optional[str] = None):
        """
        In order to deal with Yolo not Found.

        Args:
            weights_dir: Path to weights directory. If None, auto-detects project root.
        """
        if weights_dir is None:
            # 自动推断项目根目录（基于 model_loader.py 的位置）
            current_file = os.path.abspath(__file__)
            ml_models_dir = os.path.dirname(os.path.dirname(current_file))  # .../ml_models/
            project_root = os.path.dirname(ml_models_dir)  # .../Project Code/
            weights_dir = os.path.join(project_root, "ml_models", "yolov8", "weights")

        self.weights_dir = weights_dir
        self.models = {}
        self.model_names = {}

    def load_model(self, model_name: str, weight_file: str) -> bool:
        """
        Load a YOLOv8 model from .pt file.
        Args:
            model_name: Logical name (e.g., 'vehicle', 'smoke')
            weight_file: Filename under weights_dir (e.g., 'yolov8n.pt')
        Returns:
            True if loaded successfully, False otherwise.
        """
        weight_path = os.path.join(self.weights_dir, weight_file)

        if not os.path.exists(weight_path):
            logger.error(f"[ERROR] Weight file not found: {weight_path}")
            return False

        try:
            model = YOLO(weight_path)
            self.models[model_name] = model

            # 🧠 自动提取类别名映射
            if hasattr(model, 'names'):
                if isinstance(model.names, dict):
                    self.model_names[model_name] = model.names
                elif isinstance(model.names, (list, tuple)):
                    self.model_names[model_name] = {i: str(name) for i, name in enumerate(model.names)}
                else:
                    logger.warning(f"Unexpected type for model.names: {type(model.names)}")
                    self.model_names[model_name] = {i: f"class_{i}" for i in range(80)}
            else:
                logger.warning("Model has no 'names' attribute. Using default COCO 80-class mapping.")
                self.model_names[model_name] = {i: f"class_{i}" for i in range(80)}

            logger.info(f"[INFO] Model '{model_name}' loaded from {weight_path}")
            sample_classes = list(self.model_names[model_name].items())[:3]
            logger.info(f"    ├─ Total classes: {len(self.model_names[model_name])}")
            logger.info(f"    └─ Sample: {sample_classes}...")

            return True

        except Exception as e:
            logger.error(f"[ERROR] Failed to load model {model_name}: {str(e)}", exc_info=True)
            return False

    def get_model(self, model_name: str) -> Optional['YOLO']:
        """Get loaded model by name."""
        return self.models.get(model_name)

    def get_class_name(self, model_name: str, class_id: int) -> str:
        """
        Safely get human-readable class name from class ID.
        Never raises KeyError.
        """
        names = self.model_names.get(model_name, {})
        return names.get(class_id, f"unknown_class_{class_id}")

    def list_loaded_models(self) -> list:
        """Return list of currently loaded model names."""
        return list(self.models.keys())
