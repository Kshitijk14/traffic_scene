from shapely.geometry import Point, Polygon


ZONE_NAME_MAP = {
    "A": "WEST",
    "B": "EAST",
    "C": "SOUTH",
    "D": "NORTH",
    "E": "CENTER"
}


def get_zone_for_point(x, y, zones):
    point = Point(x, y)
    for name, pts in zones.items():
        polygon = Polygon(pts)
        if polygon.contains(point):
            return name  # Returns "A", "B", etc.
    return None