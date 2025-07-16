import os
from pathlib import Path
from utils.config import CONFIG
from utils.logger import setup_logger
from ultralytics import YOLO
import supervision as sv

# configurations
LOG_PATH = Path(CONFIG["LOG_DIR"])
MODEL_WEIGHTS = Path(CONFIG["MODELS_WEIGHTS_8N"])
MODEL_CONFIG = Path(CONFIG["MODELS_CONFIG_8N"])

# setup logging
LOG_DIR = os.path.join(os.getcwd(), LOG_PATH)
os.makedirs(LOG_DIR, exist_ok=True)  # Create the logs directory if it doesn't exist
LOG_FILE = os.path.join(LOG_DIR, "detector.log")


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
