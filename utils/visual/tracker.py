import cv2, traceback
import supervision as sv


class Tracker:
    def __init__(self):
        self.tracker = sv.ByteTrack()

    def update(self, detections: sv.Detections):
        return self.tracker.update_with_detections(detections)


def draw_trace_path(frame, trail, color, logger):
    try:
        logger.debug(f"[] Drawing trace path with color: {color}, trail length: {len(trail)}")
        for j in range(1, len(trail)):
            cv2.line(frame, trail[j - 1], trail[j], color, 2)
    except Exception as e:
        logger.error(f"[] Error drawing trace path: {e}")
        logger.debug(traceback.format_exc())
