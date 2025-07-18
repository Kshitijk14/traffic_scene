DIRECTION_MAP = {
    ("N", "S"): "↓",
    ("S", "N"): "↑",
    ("W", "E"): "→",
    ("E", "W"): "←",
    ("N", "E"): "→",
    ("N", "W"): "←",
    ("S", "E"): "→",
    ("S", "W"): "←",
    ("W", "N"): "↑",
    ("W", "S"): "↓",
    ("E", "N"): "↑",
    ("E", "S"): "↓"
}


def get_final_direction(entry_zone, exit_zone):
    # Skip center zone, treat as None
    if entry_zone == "CENTER":
        entry_zone = None
    if exit_zone == "CENTER":
        exit_zone = None

    if not entry_zone or not exit_zone:
        return "UNKNOWN"
    
    if entry_zone == exit_zone:
        return "STILL"

    return DIRECTION_MAP.get((entry_zone, exit_zone), "UNKNOWN")