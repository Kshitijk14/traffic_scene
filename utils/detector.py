import cv2, traceback
from pathlib import Path
import supervision as sv
from utils.config import CONFIG
from ultralytics import YOLO


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


def draw_box_and_label(frame, box, label, color, logger):
    try:
        logger.debug(f"[] Drawing box: {box}, label: {label}, color: {color}")
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(
            frame, label, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )
    except Exception as e:
        logger.error(f"[] Error drawing box and label: {e}")
        logger.debug(traceback.format_exc())
