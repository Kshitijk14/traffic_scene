import cv2, os, math
from pathlib import Path
from utils.config import CONFIG
from utils.logger import setup_logger
from utils.common import ensure_directory_exists
from utils.save_track_checkpoints import save_object_log_csv
from utils.zones import define_zones, draw_zones
from utils.detector import VehicleDetector
from utils.tracker import Tracker
from tqdm import tqdm
import traceback
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
    "N": (255, 0, 0),       # Red
    "NE": (255, 165, 0),    # Orange
    "E": (0, 255, 0),       # Green
    "SE": (0, 255, 255),    # Cyan
    "S": (0, 0, 255),       # Blue
    "SW": (128, 0, 128),    # Purple
    "W": (255, 255, 0),     # Yellow
    "NW": (255, 0, 255),    # Magenta
    "STILL": (192, 192, 192)  # Gray
}


def estimate_speed(trail, fps, logger):
    try:
        logger.debug(f"[] Estimating speed for trail: {trail}")
        
        y_start, y_end = trail[0][1], trail[-1][1]
        distance = abs(y_end - y_start)
        time_sec = len(trail) / fps
        return int((distance / time_sec) * 3.6)
    except Exception as e:
        logger.error(f"[] Error estimating speed: {e}")
        logger.debug(traceback.format_exc())
        return 

def estimate_cardinal_direction(dir_trail, logger, threshold=10):
    try:
        if len(dir_trail) < 2:
            logger.debug("[] Direction estimation: Not enough points.")
            return "UNKNOWN"

        start_point = dir_trail[0]
        end_point = dir_trail[-1]        
        dx = end_point[0] - start_point[0]
        dy = end_point[1] - start_point[1]
        
        logger.debug(f"[] Direction estimation: dx={dx}, dy={dy}")

        if abs(dx) < threshold and abs(dy) < threshold:
            return "STILL"

        angle_rad = math.atan2(-dy, dx)  # y-axis is inverted in image space
        angle_deg = math.degrees(angle_rad)
        angle_deg = (angle_deg + 360) % 360  # Normalize to [0, 360)
        
        logger.debug(f"[] Direction estimation: angle = {angle_deg:.2f}°")

        if 22.5 <= angle_deg < 67.5:
            return "NE"
        elif 67.5 <= angle_deg < 112.5:
            return "N"
        elif 112.5 <= angle_deg < 157.5:
            return "NW"
        elif 157.5 <= angle_deg < 202.5:
            return "W"
        elif 202.5 <= angle_deg < 247.5:
            return "SW"
        elif 247.5 <= angle_deg < 292.5:
            return "S"
        elif 292.5 <= angle_deg < 337.5:
            return "SE"
        else:
            return "E"
    except Exception as e:
        logger.error(f"[] Error estimating cardinal direction: {e}")
        logger.debug(traceback.format_exc())
        return "UNKNOWN"

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

def draw_trace_path(frame, trail, color, logger):
    try:
        logger.debug(f"[] Drawing trace path with color: {color}, trail length: {len(trail)}")
        for j in range(1, len(trail)):
            cv2.line(frame, trail[j - 1], trail[j], color, 2)
    except Exception as e:
        logger.error(f"[] Error drawing trace path: {e}")
        logger.debug(traceback.format_exc())

def draw_direction_arrow(frame, dir_trail, direction, logger):
    try:
        logger.debug(f"[] Drawing direction arrow for direction: {direction}, trail length: {len(dir_trail)}")
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
    except Exception as e:
        logger.error(f"[] Error drawing direction arrow: {e}")
        logger.debug(traceback.format_exc())

def annotate_detections(frame, frame_idx, detections, trail_coords, direction_coords, object_logs, fps, logger):
    try:
        annotated_frame = frame.copy()
        logger.debug(f"[] Annotating frame {frame_idx} with {len(detections.xyxy)} detections.")

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
                speed_kmph = estimate_speed(trail, fps, logger)
                direction = estimate_cardinal_direction(dir_trail, logger)
                label = f"#{track_id} {speed_kmph} km/h {direction}"
                draw_direction_arrow(annotated_frame, dir_trail, direction, logger)
                logger.debug(f"[] Track ID: {track_id}, Speed: {speed_kmph} km/h, Direction: {direction}")
                
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

            draw_box_and_label(annotated_frame, (x1, y1, x2, y2), label, color, logger)
            draw_trace_path(annotated_frame, trail, color, logger)
            logger.debug(f"[] Annotated frame {frame_idx} with track ID {track_id}, class ID {class_id}, label: {label}")

        return annotated_frame
    except Exception as e:
        logger.error(f"[] Error annotating detections: {e}")
        logger.debug(traceback.format_exc())
        return frame


def run_stage_01(video_path=SOURCE_VIDEO_PATH, output_path=TARGET_VIDEO_PATH):
    logger = setup_logger("stage_01_detect_n_track_logger", LOG_FILE)
    logger.info(" ")
    logger.info("Starting Stage 01: Detect and Track Vehicles")

    try:
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
        logger.info(f"Video opened: {video_path}, FPS: {fps}, Frame Count: {frame_count}")

        detector = VehicleDetector()
        tracker = Tracker()
        
        # trail_coords = defaultdict(lambda: deque(maxlen=int(fps * 5)))
        trail_coords = defaultdict(deque)
        direction_coords = defaultdict(deque)

        CSV_OUT_FILE = os.path.join(TRACK_CHECKPOINTS_PATH, (f"05-object_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"))
        object_logs = dict()
        active_track_ids = set()

        frame_idx = 0
        logger.info(f"Processing video: {video_path}, Output: {output_path}")
        logger.info(f"Video properties - Width: {width}, Height: {height}, FPS: {fps}, Frame Count: {frame_count}")
        
        zones = define_zones(width, height)
        
        with tqdm(total=frame_count, desc="Processing Video", unit="frame") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                detections = detector.predict(frame)
                tracked_detections = tracker.update(detections)
                
                current_ids = set(map(int, tracked_detections.tracker_id))
                exited_ids = active_track_ids - current_ids
                
                logger.debug(f"Frame {frame_idx}: Active IDs: {current_ids}, Exited IDs: {exited_ids}")
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
                logger.debug(f"Frame {frame_idx}: Active Track IDs: {active_track_ids}")

                points = tracked_detections.get_anchors_coordinates(
                    anchor=sv.Position.BOTTOM_CENTER
                )
                
                logger.debug(f"Frame {frame_idx}: Detected {len(points)} points.")
                for track_id, (x, y) in zip(tracked_detections.tracker_id, points):
                    trail_coords[int(track_id)].append((int(x), int(y)))
                    direction_coords[int(track_id)].append((int(x), int(y)))

                annotated_frame = annotate_detections(
                    frame, frame_idx, tracked_detections, trail_coords, direction_coords, object_logs, fps, logger
                )
                logger.debug(f"Frame {frame_idx}: Annotated frame with {len(tracked_detections.xyxy)} detections.")
                
                annotated_frame = draw_zones(annotated_frame, zones)

                out.write(annotated_frame)
                frame_idx += 1
                pbar.update(1)

        cap.release()
        out.release()
        logger.info(f"Processed {frame_idx} frames.")
        logger.info("Stage 01 completed successfully.")
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        logger.debug(traceback.format_exc())


if __name__ == "__main__":
    run_stage_01()
