from __future__ import annotations
import cv2
import numpy as np




def compute_delaunay(size_hw: tuple[int, int], points: np.ndarray) -> list[tuple[int, int, int]]:
    """Compute Delaunay triangles over target points.
    Returns list of index triplets into `points`.
    """
    h, w = size_hw
    rect = (0, 0, w, h)
    subdiv = cv2.Subdiv2D(rect)
    # Map point->index for fast lookup (use tuples of ints)
    pt2idx = {tuple(map(int, p)): i for i, p in enumerate(points)}
    for p in points:
        subdiv.insert((int(p[0]), int(p[1])))


    triangles = []
    for t in subdiv.getTriangleList():
        p0 = (int(t[0]), int(t[1]))
        p1 = (int(t[2]), int(t[3]))
        p2 = (int(t[4]), int(t[5]))
        if (0 <= p0[0] < w and 0 <= p0[1] < h and
            0 <= p1[0] < w and 0 <= p1[1] < h and
            0 <= p2[0] < w and 0 <= p2[1] < h):
            try:
                i0, i1, i2 = pt2idx[p0], pt2idx[p1], pt2idx[p2]
                triangles.append((i0, i1, i2))
            except KeyError:
            # Rarely, Subdiv may report verts not exactly in our set; skip them.
                continue
    return triangles