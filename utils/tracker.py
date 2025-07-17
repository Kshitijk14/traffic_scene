import supervision as sv

class Tracker:
    def __init__(self):
        self.tracker = sv.ByteTrack()

    def update(self, detections: sv.Detections):
        return self.tracker.update_with_detections(detections)
