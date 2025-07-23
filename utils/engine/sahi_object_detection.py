import numpy as np
import torch
from ultralytics import YOLO
from sahi.auto_model import AutoDetectionModel
from sahi.predict import get_sliced_prediction


class ObjectDetection:
    def __init__(self, model_path, conf_thres=0.25, iou_thres=0.45, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO(str(model_path)).to(self.device)
        self.model.conf = conf_thres
        self.model.iou = iou_thres
        self.class_names = self.model.names
        self.colors = self._generate_colors()

    def _generate_colors(self):
        # Assigns a unique color to each class
        np.random.seed(42)
        return {i: tuple(np.random.randint(0, 255, 3).tolist()) for i in range(len(self.model.names))}

    def detect(self, image, imgsz=1280):
        # Run YOLOv8 detection
        results = self.model.predict(image, imgsz=imgsz, device=self.device, verbose=False)
        boxes, class_ids, scores = [], [], []

        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    xyxy = box.xyxy[0].int().cpu().numpy()
                    conf = box.conf.item()
                    cls = int(box.cls.item())

                    boxes.append(tuple(xyxy))  # (x1, y1, x2, y2)
                    class_ids.append(cls)
                    scores.append(conf)

        return boxes, class_ids, scores


class SAHIObjectDetection:
    def __init__(self, model_path, conf_thres=0.25, device=None):
        self.device = device or "cuda"  # or "cpu"
        self.model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=str(model_path),
            confidence_threshold=conf_thres,
            device=self.device,
        )
        self.classes = self.model.model.model.names
        self.colors = self._generate_colors()

    def _generate_colors(self):
        np.random.seed(42)
        return {
            i: tuple(np.random.randint(0, 255, 3).tolist()) 
            for i in range(len(self.classes))
        }

    def detect(self, image, slice_height=512, slice_width=512, overlap_height_ratio=0.3, overlap_width_ratio=0.3):
        result = get_sliced_prediction(
            image,
            self.model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
        )
        return result.object_prediction_list
