import cv2, os
from pathlib import Path
from utils.config import CONFIG
from utils.logger import setup_logger
from utils.common import ensure_directory_exists
from utils.save_track_checkpoints import save_object_log_csv
from pipeline.detector import VehicleDetector
from pipeline.tracker import Tracker
from tqdm import tqdm
from datetime import datetime, timedelta
from collections import defaultdict, deque
import supervision as sv


# configurations
LOG_PATH = Path(CONFIG["LOG_DIR"])
SOURCE_VIDEO_PATH = Path(CONFIG["SOURCE_VIDEO_PATH"])
TARGET_VIDEO_PATH = Path(CONFIG["TARGET_VIDEO_PATH"])
TRACK_CHECKPOINTS_PATH = Path(CONFIG["TRACK_CHECKPOINTS_DIR"])
TRACK_CHECKPOINTS_CLASS = CONFIG["TRACK_CHECKPOINTS_CLASS"]

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

DIRECTION_COLOR_MAP = {
        "UP": (0, 0, 255),       # Red
        "DOWN": (255, 0, 0),     # Blue
        "LEFT": (0, 255, 255),   # Yellow
        "RIGHT": (0, 255, 0)     # Green
    }


def estimate_speed(trail, fps):
    y_start, y_end = trail[0][1], trail[-1][1]
    distance = abs(y_end - y_start)
    time_sec = len(trail) / fps
    return int((distance / time_sec) * 3.6)

def estimate_direction(dir_trail):
    dx = dir_trail[-1][0] - dir_trail[0][0]
    dy = dir_trail[-1][1] - dir_trail[0][1]
    if abs(dx) > abs(dy):
        return "RIGHT" if dx > 0 else "LEFT"
    else:
        return "DOWN" if dy > 0 else "UP"

def draw_box_and_label(frame, box, label, color):
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    cv2.putText(
        frame, label, (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
    )

def draw_trace_path(frame, trail, color):
    for j in range(1, len(trail)):
        cv2.line(frame, trail[j - 1], trail[j], color, 2)

def draw_direction_arrow(frame, dir_trail, direction):
    if len(dir_trail) >= 2:
        x1_arrow, y1_arrow = dir_trail[0]
        x2_arrow, y2_arrow = dir_trail[-1]
        arrow_color = DIRECTION_COLOR_MAP[direction]
        cv2.arrowedLine(
            frame,
            (x1_arrow, y1_arrow),
            (x2_arrow, y2_arrow),
            arrow_color,
            2,
            tipLength=0.4
        )

def annotate_detections(frame, frame_idx, detections, trail_coords, direction_coords, object_logs, fps):
    annotated_frame = frame.copy()

    for i in range(len(detections.xyxy)):
        x1, y1, x2, y2 = map(int, detections.xyxy[i])
        class_id = int(detections.class_id[i])
        track_id = int(detections.tracker_id[i])
        color = CLASS_COLOR_MAP.get(class_id, (255, 255, 255))

        trail = trail_coords[track_id]
        dir_trail = direction_coords[track_id]

        if len(trail) < 2:
            label = f"#{track_id}"
        else:
            speed_kmph = estimate_speed(trail, fps)
            direction = estimate_direction(dir_trail)
            label = f"#{track_id} {speed_kmph} km/h {direction}"
            draw_direction_arrow(annotated_frame, dir_trail, direction)
            
            # Track object for logging
            if track_id not in object_logs:
                object_logs[track_id] = {
                    "class_id": class_id,
                    "speeds": [],
                    "entry_dir": direction,
                    "exit_dir": direction,
                    "entry_time": frame_idx / fps,
                    "exit_time": frame_idx / fps
                }
            else:
                object_logs[track_id]["exit_time"] = frame_idx / fps
            
            object_logs[track_id]["speeds"].append(speed_kmph)
            object_logs[track_id]["exit_dir"] = direction

        draw_box_and_label(annotated_frame, (x1, y1, x2, y2), label, color)
        draw_trace_path(annotated_frame, trail, color)

    return annotated_frame


def run_stage_01(video_path=SOURCE_VIDEO_PATH, output_path=TARGET_VIDEO_PATH):
    logger = setup_logger("stage_01_detect_n_track_logger", LOG_FILE)
    logger.info(" ")
    logger.info("Starting Stage 01: Detect and Track Vehicles")

    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ensure_directory_exists(output_path)

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    detector = VehicleDetector()
    tracker = Tracker()
    # trail_coords = defaultdict(lambda: deque(maxlen=int(fps * 5)))
    trail_coords = defaultdict(deque)
    direction_coords = defaultdict(deque)

    CSV_OUT_FILE = os.path.join(TRACK_CHECKPOINTS_PATH, (f"object_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"))
    object_logs = dict()
    active_track_ids = set()

    frame_idx = 0
    with tqdm(total=frame_count, desc="Processing Video", unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            detections = detector.predict(frame)
            tracked_detections = tracker.update(detections)
            
            current_ids = set(map(int, tracked_detections.tracker_id))
            exited_ids = active_track_ids - current_ids
            for exited_id in exited_ids:
                if exited_id in object_logs:
                    log = object_logs[exited_id]
                    if log["speeds"]:
                        avg_speed = sum(log["speeds"]) / len(log["speeds"])
                        save_object_log_csv(
                            CSV_OUT_FILE,
                            TRACK_CHECKPOINTS_CLASS,
                            {
                                "obj_tracker_id": exited_id,
                                "class_id": log["class_id"],
                                "avg_speed_kmph": round(avg_speed, 2),
                                "entry_direction": log["entry_dir"],
                                "exit_direction": log["exit_dir"],
                                "entry_time": str(timedelta(seconds=log["entry_time"])),
                                "exit_time": str(timedelta(seconds=log["exit_time"]))
                            }
                        )
                    del object_logs[exited_id]
            active_track_ids = current_ids

            points = tracked_detections.get_anchors_coordinates(
                anchor=sv.Position.BOTTOM_CENTER
            )
            
            for track_id, (x, y) in zip(tracked_detections.tracker_id, points):
                trail_coords[int(track_id)].append((int(x), int(y)))
                direction_coords[int(track_id)].append((int(x), int(y)))

            annotated_frame = annotate_detections(
                frame, frame_idx, tracked_detections, trail_coords, direction_coords, object_logs, fps
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
