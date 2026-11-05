"""Small geometry helpers. Boxes are (x1, y1, x2, y2); polygons are lists of (x, y)."""

from __future__ import annotations

from typing import Sequence

Point = tuple[float, float]
Box = tuple[float, float, float, float]


def point_in_polygon(x: float, y: float, poly: Sequence[Point]) -> bool:
    """Ray-casting test. Points exactly on an edge may go either way."""
    inside = False
    n = len(poly)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y):
            x_cross = xi + (y - yi) * (xj - xi) / (yj - yi)
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def in_any_polygon(x: float, y: float, polys: Sequence[Sequence[Point]]) -> bool:
    return any(point_in_polygon(x, y, p) for p in polys)


def expand_box(box: Box, margin: float) -> Box:
    x1, y1, x2, y2 = box
    return (x1 - margin, y1 - margin, x2 + margin, y2 + margin)


def box_contains(box: Box, x: float, y: float) -> bool:
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def boxes_overlap(a: Box, b: Box) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def distance(a: Point, b: Point) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def denormalize_polygons(polys_norm, width: int, height: int) -> list[list[Point]]:
    """Zones are saved as 0-1 fractions of the frame so they survive resizing."""
    return [[(x * width, y * height) for x, y in poly] for poly in polys_norm]
