from shapely.geometry import Point, Polygon


def get_zone_for_point(x, y, zones):
    point = Point(x, y)
    for name, pts in zones.items():
        polygon = Polygon(pts)
        if polygon.contains(point):
            return name
    return None