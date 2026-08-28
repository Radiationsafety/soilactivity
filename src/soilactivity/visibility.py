"""Visibility matrix for barrier geometry in the Fredholm equation.

The visibility matrix Vis describes which raster cells are visible from
each other, accounting for buildings and structures that absorb
ionising radiation.  Vis[i,j] = 1 if the line segment from cell i to
cell j does not intersect any building, 0 otherwise.

Buildings are represented as axis-aligned rectangular polygons
(rectangles) with infinite height (conservative assumption for gamma
radiation shielding).

References
----------
1. Chizhov et al (2023) J. Radiol. Prot. 43 041505, Section 2.2.
2. Chizhov & Kryuchkov (2024) Nuclear Science and Technology.
"""
from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional

__all__ = [
    "compute_visibility_matrix",
    "line_intersects_rect",
    "rectangles_from_polygons",
    "visibility_radius_mask",
]


def line_intersects_rect(
    x1: float, y1: float,
    x2: float, y2: float,
    xmin: float, ymin: float,
    xmax: float, ymax: float,
) -> bool:
    """Check if line segment (x1,y1)-(x2,y2) intersects an axis-aligned rectangle.

    Uses the Liang-Barsky algorithm for efficient clipping test.

    Parameters
    ----------
    x1, y1, x2, y2 : float
        Line segment endpoints.
    xmin, ymin, xmax, ymax : float
        Rectangle bounds.

    Returns
    -------
    bool
        True if the segment intersects the rectangle.
    """
    dx = x2 - x1
    dy = y2 - y1

    p = [-dx, dx, -dy, dy]
    q = [x1 - xmin, xmax - x1, y1 - ymin, ymax - y1]

    u1 = 0.0
    u2 = 1.0

    for pk, qk in zip(p, q):
        if abs(pk) < 1e-15:
            # Parallel to this edge
            if qk < 0:
                return True  # Line is outside
        else:
            t = qk / pk
            if pk < 0:
                u1 = max(u1, t)
            else:
                u2 = min(u2, t)
            if u1 > u2:
                return False

    # Check if the intersection interval [u1, u2] is within [0, 1]
    if u2 < 0 or u1 > 1:
        return False

    return True


def rectangles_from_polygons(
    buildings: List[dict],
) -> List[Tuple[float, float, float, float]]:
    """Convert building dicts to (xmin, ymin, xmax, ymax) tuples.

    Each building dict must have keys 'x', 'y', 'width', 'height'
    (or 'x', 'y', 'w', 'h') defining an axis-aligned rectangle.

    Parameters
    ----------
    buildings : list of dict
        Building definitions.

    Returns
    -------
    list of (xmin, ymin, xmax, ymax)
    """
    rects = []
    for b in buildings:
        bx = b.get('x', b.get('left', 0.0))
        by = b.get('y', b.get('bottom', 0.0))
        bw = b.get('width', b.get('w', 0.0))
        bh = b.get('height', b.get('h', 0.0))
        rects.append((bx, by, bx + bw, by + bh))
    return rects


def compute_visibility_matrix(
    cx: np.ndarray,
    cy: np.ndarray,
    buildings: Optional[List[dict]] = None,
    visibility_radius: Optional[float] = None,
) -> np.ndarray:
    """Compute the visibility matrix Vis for a 2D raster.

    Vis[i, j] = 1 if cell j is visible from cell i (no building blocks
    the line of sight), 0 otherwise.  Vis[i, i] = 1 always.

    For performance, cells beyond visibility_radius are assumed
    not visible (Vis = 0) since their contribution is negligible.

    Parameters
    ----------
    cx : np.ndarray (N,)
        X-coordinates of cell centres (raster_to_vector order).
    cy : np.ndarray (N,)
        Y-coordinates of cell centres.
    buildings : list of dict or None
        Building definitions. Each dict: {'x':, 'y':, 'width':, 'height':}.
        If None or empty, all cells are mutually visible.
    visibility_radius : float or None
        Maximum line-of-sight distance [m]. If None, all pairs are checked.

    Returns
    -------
    np.ndarray, shape (N, N), dtype float64
        Visibility matrix with values 0.0 or 1.0.
    """
    N = len(cx)
    Vis = np.ones((N, N), dtype=np.float64)

    if buildings is None or len(buildings) == 0:
        # Barrier-free: optionally apply radius mask only
        if visibility_radius is not None:
            for i in range(N):
                dx = cx[i] - cx
                dy = cy[i] - cy
                dist = np.sqrt(dx**2 + dy**2)
                Vis[i, dist > visibility_radius] = 0.0
        return Vis

    rects = rectangles_from_polygons(buildings)

    for i in range(N):
        for j in range(i + 1, N):
            # Skip if beyond visibility radius
            if visibility_radius is not None:
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                if dx * dx + dy * dy > visibility_radius ** 2:
                    Vis[i, j] = 0.0
                    Vis[j, i] = 0.0
                    continue

            # Check line of sight
            blocked = False
            for (rxmin, rymin, rxmax, rymax) in rects:
                if line_intersects_rect(
                    cx[i], cy[i], cx[j], cy[j],
                    rxmin, rymin, rxmax, rymax
                ):
                    blocked = True
                    break

            if blocked:
                Vis[i, j] = 0.0
                Vis[j, i] = 0.0

    return Vis


def visibility_radius_mask(
    cx: np.ndarray,
    cy: np.ndarray,
    radius_m: float,
) -> np.ndarray:
    """Create a visibility mask based on distance only (no buildings).

    Vis[i,j] = 1 if horizontal distance(i,j) <= radius_m, else 0.
    Vis[i,i] = 1 always.

    This approximates a detector with limited angular aperture.

    Parameters
    ----------
    cx, cy : np.ndarray (N,)
        Cell centre coordinates.
    radius_m : float
        Maximum visibility distance [m].

    Returns
    -------
    np.ndarray (N, N)
        Binary mask.
    """
    N = len(cx)
    Vis = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        dx = cx[i] - cx
        dy = cy[i] - cy
        dist_sq = dx ** 2 + dy ** 2
        Vis[i, dist_sq <= radius_m ** 2] = 1.0
    return Vis
