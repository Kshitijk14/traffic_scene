import cv2
import numpy as np


ZONE_COLORS = {
    "A": (255, 0, 0),    # Blue
    "B": (0, 255, 0),    # Green
    "C": (0, 0, 255),    # Red
    "D": (0, 255, 255),   # Yellow
    'E': (175, 140, 200)    # Purplish-Gray
}


def define_zones(frame_width, frame_height, pad_percent=0.05, b_pad_percent=0.10, t_pad_percent=0.20):
    left_w = int(pad_percent * frame_width)
    right_w = int(pad_percent * frame_width)
    bottom_h = int(b_pad_percent * frame_height)
    top_h = int(t_pad_percent * frame_height)
    
    zones = {
        "A": [ # Top
            (0, 0), (left_w, 0), 
            (left_w, frame_height), (0, frame_height)
        ],
        "B": [ # Right
            (frame_width - right_w, 0), (frame_width, 0), 
            (frame_width, frame_height), (frame_width - right_w, frame_height)
        ],
        "C": [ # Bottom
            (0, frame_height - bottom_h), (frame_width, frame_height - bottom_h), 
            (frame_width, frame_height), (0, frame_height)
        ],
        "D": [ # Left
            (0, 0), (frame_width, 0), 
            (frame_width, top_h), (0, top_h)
        ],
        "E": [ # Center
            (left_w, top_h), 
            (frame_width - right_w, top_h),
            (frame_width - right_w, frame_height - bottom_h), 
            (left_w, frame_height - bottom_h)
        ]
    }

    return zones

def draw_zones(frame, zones, alpha=0.3):
    overlay = frame.copy()
    
    for zone_name, points in zones.items():
        pts = np.array(points, dtype=np.int32)
        zone_color = ZONE_COLORS.get(zone_name, (255, 255, 255))
        
        cv2.fillPoly(overlay, [pts], color=zone_color)
        cv2.polylines(overlay, [pts], isClosed=True, color=(0, 0, 0), thickness=2)

        # Label the zone name in the center
        cx = int(sum([p[0] for p in points]) / len(points))
        cy = int(sum([p[1] for p in points]) / len(points))
        cv2.putText(
            overlay, zone_name, (cx - 10, cy),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2
        )

    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
