import os, cv2
from pathlib import Path
import numpy as np
from utils.config import CONFIG
from utils.logger import setup_logger
from utils.engine.sahi_object_detection import ObjectDetection
from utils.engine.sahi_object_detection import SAHIObjectDetection


# configurations
LOG_PATH = Path(CONFIG["LOG_DIR"])
SOURCE_VIDEO_PATH = Path(CONFIG["SOURCE_VIDEO_PATH"])
TARGET_VIDEO_PATH = Path(CONFIG["TARGET_VIDEO_PATH"])
MODEL_YOLO_8M = Path(CONFIG["MODELS_WEIGHTS_8M"])

# setup logging
LOG_DIR = os.path.join(os.getcwd(), LOG_PATH)
os.makedirs(LOG_DIR, exist_ok=True)  # Create the logs directory if it doesn't exist
LOG_FILE = os.path.join(LOG_DIR, "detect_wo_sahi.log")


def logic_wo_sahi(frame):
    # initialize obj. detection
    od = ObjectDetection(model_path=MODEL_YOLO_8M)
    
    bboxes, class_ids, scores = od.detect(frame, imgsz=1280)
    for bbox, class_id, score in zip(bboxes, class_ids, scores):
        x1, x2, y1, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        class_name = od.class_names[class_id]
        cv2.putText(frame, f"{class_name}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

def logic_w_sahi(frame):
    # initialize sahi obj. detection
    od = SAHIObjectDetection(model_path=MODEL_YOLO_8M)
    
    predictions = od.detect(frame, slice_height=512, slice_width=512, overlap_height_ratio=0.3, overlap_width_ratio=0.3)
    for pred in predictions:
        bbox = pred.bbox
        class_id = pred.category.id
        score = pred.score.value
        
        x1, y1, x2, y2 = np.array([bbox.minx, bbox.miny, bbox.maxx, bbox.maxy], dtype=np.int32)
        cv2.rectangle(frame, (x1, y1), (x2, y2), od.colors[class_id], 2)
        
        # get class name
        class_name = od.classes[class_id]
        cv2.putText(frame, class_name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, od.colors[class_id], 2)


def main():
    # setup logger
    logger = setup_logger("detect_logger", LOG_FILE)
    logger.info(" ")
    logger.info("Starting object detection without SAHI...")

    # read video
    cap = cv2.VideoCapture(str(SOURCE_VIDEO_PATH))
    if not cap.isOpened():
        logger.error(f"Error opening video file: {SOURCE_VIDEO_PATH}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        logic_w_sahi(frame)

        # frame = cv2.resize(frame, (1280, 720), fx=0.5, fy=0.5)

        cv2.imshow("Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.info("Object detection completed.")


if __name__ == "__main__":
    main()