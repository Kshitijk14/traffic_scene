from shapely.geometry import Point, Polygon


ZONE_NAME_MAP = {
    "A": "W",
    "B": "E",
    "C": "S",
    "D": "N",
    "E": "CENTER"
}


def get_zone_for_point(x, y, zones):
    point = Point(x, y)
    for name, pts in zones.items():
        polygon = Polygon(pts)
        if polygon.contains(point):
            return ZONE_NAME_MAP.get(name)
    return None