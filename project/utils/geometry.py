from matplotlib.path import Path

def is_inside_polygon(polygon_points, point):
    """
    다각형 내부에 점이 포함되는지 확인

    Args:
        polygon_points (list of (x, y)): 다각형 꼭짓점 리스트
        point ((x, y)): 검사할 점의 좌표

    Returns:
        bool: 점이 다각형 내부에 있으면 True
    """
    if not polygon_points or len(polygon_points) < 3:
        return False
    path = Path(polygon_points)
    return path.contains_point(point)
