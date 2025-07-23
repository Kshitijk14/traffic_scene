import cv2, traceback
from pathlib import Path
import numpy as np
import supervision as sv
from utils.config import CONFIG
from ultralytics import YOLO
from sahi.auto_model import AutoDetectionModel
from sahi.predict import get_sliced_prediction


# configurations
MODEL_WEIGHTS = Path(CONFIG["MODELS_WEIGHTS_8N"])


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


class SAHIVehicleDetector:
    def __init__(self, model_path=MODEL_WEIGHTS, conf_thres=0.25, device="cpu"):
        self.model = AutoDetectionModel.from_pretrained(
            model_type='ultralytics',
            model_path=str(model_path),
            confidence_threshold=conf_thres,
            device=device
        )

        self.box_annotator = sv.BoxAnnotator(
            color=(255, 0, 0),
            thickness=3,
        )

    def predict(self, frame):
        result = get_sliced_prediction(
            image=frame,
            detection_model=self.model,
            slice_height=512,
            slice_width=512,
            overlap_height_ratio=0.3,
            overlap_width_ratio=0.3,
        )

        boxes = []
        confidences = []
        class_ids = []

        for pred in result.object_prediction_list:
            boxes.append(pred.bbox.to_xyxy())
            confidences.append(pred.score.value)
            class_ids.append(pred.category.id)

        if len(boxes) == 0:
            return sv.Detections.empty()

        return sv.Detections(
            xyxy=np.array(boxes),
            confidence=np.array(confidences),
            class_id=np.array(class_ids)
        )


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
