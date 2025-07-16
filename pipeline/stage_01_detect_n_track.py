import cv2, os
from pathlib import Path
from utils.config import CONFIG
from utils.logger import setup_logger
from utils.common import ensure_directory_exists
from pipeline.detector import VehicleDetector
from pipeline.tracker import Tracker
from tqdm import tqdm
import supervision as sv


# configurations
LOG_PATH = Path(CONFIG["LOG_DIR"])
SOURCE_VIDEO_PATH = Path(CONFIG["SOURCE_VIDEO_PATH"])
TARGET_VIDEO_PATH = Path(CONFIG["TARGET_VIDEO_PATH"])

# setup logging
LOG_DIR = os.path.join(os.getcwd(), LOG_PATH)
os.makedirs(LOG_DIR, exist_ok=True)  # Create the logs directory if it doesn't exist
LOG_FILE = os.path.join(LOG_DIR, "stage_01_detect_n_track.log")


CLASS_COLOR_MAP = {
    0: (0, 255, 255),   # person -> Yellow
    2: (0, 255, 0),     # car -> Green
    3: (255, 0, 0),     # motorcycle -> Blue
    7: (0, 0, 255),     # truck -> Red
}


def run_stage_01(video_path=SOURCE_VIDEO_PATH, output_path=TARGET_VIDEO_PATH):
    logger = setup_logger("stage_01_detect_n_track_logger", LOG_FILE)
    logger.info(" ")
    logger.info("Starting Stage 01: Detect and Track Vehicles")
    
    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    
    ensure_directory_exists(output_path)

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    detector = VehicleDetector()
    tracker = Tracker()

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0
    
    with tqdm(total=frame_count, desc="Processing Video", unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            detections = detector.predict(frame)
            tracked_detections = tracker.update(detections)

            detections = detector.predict(frame)
            tracked_detections = tracker.update(detections)
            
            annotated_frame = frame.copy()
            
            for i in range(len(tracked_detections.xyxy)):
                x1, y1, x2, y2 = map(int, tracked_detections.xyxy[i])
                class_id = int(tracked_detections.class_id[i])
                track_id = int(tracked_detections.tracker_id[i])

                color = CLASS_COLOR_MAP.get(class_id, (255, 255, 255))  # fallback: white
                label = f"#{track_id}"

                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)

                # Draw label
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )


            out.write(annotated_frame)
            frame_idx += 1
            pbar.update(1)

    cap.release()
    out.release()
    logger.info(f"Processed {frame_idx} frames.")
    logger.info("Stage 01 completed successfully.")


if __name__ == "__main__":
    run_stage_01()
