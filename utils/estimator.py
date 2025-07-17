import traceback, math, cv2


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