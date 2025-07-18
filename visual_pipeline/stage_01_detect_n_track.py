import cv2, os, traceback
from pathlib import Path
from tqdm import tqdm
from utils.config import CONFIG
from utils.logger import setup_logger
import supervision as sv
from datetime import datetime, timedelta
from collections import defaultdict, deque
from utils.visual.detector import VehicleDetector, draw_box_and_label
from utils.visual.tracker import Tracker, draw_trace_path
from utils.visual.estimator import estimate_speed, estimate_cardinal_direction, draw_direction_arrow
from utils.visual.zones import define_zones, draw_zones
# from utils.visual.zones_utils import get_zone_for_point
from utils.common import ensure_directory_exists
from utils.visual.save_track_checkpoints import save_object_log_csv


# configurations
LOG_PATH = Path(CONFIG["LOG_DIR"])
SOURCE_VIDEO_PATH = Path(CONFIG["SOURCE_VIDEO_PATH"])
TARGET_VIDEO_PATH = Path(CONFIG["TARGET_VIDEO_PATH"])
TRACK_CHECKPOINTS_PATH = Path(CONFIG["TRACK_CHECKPOINTS_DIR"])
TRACK_CHECKPOINTS_CLASS = CONFIG["TRACK_CHECKPOINTS_CLASS"]
CONF_THRESHOLD = CONFIG["CONF_THRESHOLD"]

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


def annotate_detections(frame, frame_idx, detections, trail_coords, direction_coords, object_logs, fps, zones, logger):
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
                confidence = (float(detections.confidence[i]) * 100)
                
                label = (f"#{track_id}, {speed_kmph} km/h, {direction}, {confidence:.1f}%")
                draw_direction_arrow(annotated_frame, dir_trail, direction, logger)
                logger.debug(f"[] Track ID: {track_id}, Speed: {speed_kmph} km/h, Direction: {direction}, Confidence: {confidence:.1f}%")
                
                # Track object for logging
                if track_id not in object_logs:
                    object_logs[track_id] = {
                        "class_id": class_id,
                        "speeds": [],
                        "confidences": [],
                        "entry_dir": direction,
                        "exit_dir": direction,
                        "entry_time": frame_idx / fps,
                        "exit_time": frame_idx / fps
                    }
                else:
                    object_logs[track_id]["exit_time"] = frame_idx / fps
                
                object_logs[track_id]["speeds"].append(speed_kmph)
                object_logs[track_id]["confidences"].append(confidence)
                object_logs[track_id]["exit_dir"] = direction
                object_logs[track_id]["last_crop"] = frame[y1:y2, x1:x2]

            draw_box_and_label(annotated_frame, (x1, y1, x2, y2), label, color, logger)
            draw_trace_path(annotated_frame, trail, color, logger)
            logger.debug(f"[] Annotated frame {frame_idx} with track ID {track_id}, class ID {class_id}, label: {label}")

        return annotated_frame
    except Exception as e:
        logger.error(f"[] Error annotating detections: {e.__class__.__name__} - {e}")
        logger.debug(traceback.format_exc())
        return frame


def run_stage_01(video_path=SOURCE_VIDEO_PATH, output_path=TARGET_VIDEO_PATH):
    logger = setup_logger("stage_01_detect_n_track_logger", LOG_FILE)
    logger.info(" ")
    logger.info("Starting Stage 01: Detect and Track Vehicles")

    try:
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
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

        CSV_OUT_FILE = os.path.join(TRACK_CHECKPOINTS_PATH, (f"09-object_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"))
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
                            
                            avg_speed = (
                                round(sum(log["speeds"]) / len(log["speeds"]), 2)
                                if log["speeds"] else 0.0
                            )
                            avg_confidence = (
                                round(sum(log["confidences"]) / len(log["confidences"]), 2)
                                if log["confidences"] else 0.0
                            )
                            
                            save_object_log_csv(
                                CSV_OUT_FILE,
                                TRACK_CHECKPOINTS_CLASS,
                                {
                                    "obj_tracker_id": exited_id,
                                    "class_id": log["class_id"],
                                    "avg_speed_kmph": avg_speed,
                                    "avg_confidence": avg_confidence,
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
                    frame, frame_idx, tracked_detections, trail_coords, direction_coords, object_logs, fps, zones, logger
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
