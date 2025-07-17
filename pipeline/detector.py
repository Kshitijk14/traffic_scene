import os
from pathlib import Path
from utils.config import CONFIG
from ultralytics import YOLO
import supervision as sv


# configurations
MODEL_WEIGHTS = Path(CONFIG["MODELS_WEIGHTS_8N"])
MODEL_CONFIG = Path(CONFIG["MODELS_CONFIG_8N"])


class VehicleDetector:
    def __init__(self, model_path=MODEL_WEIGHTS):
        self.model = YOLO(model_path)
        self.class_names = self.model.model.names
        
        self.box_annotator = sv.BoxAnnotator(
            color=(0, 255, 255),
            thickness=3,
        )

    def predict(self, frame):
        results = self.model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        return detections
