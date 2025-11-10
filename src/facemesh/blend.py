from __future__ import annotations
import cv2
import numpy as np




def face_mask_from_landmarks(size_hw: tuple[int, int], pts_tgt, feather: int = 15):
    """Create a convex hull mask from target landmarks; feather edges for smoothness."""
    h, w = size_hw
    hull_idx = cv2.convexHull(pts_tgt, returnPoints=False)
    hull = pts_tgt[hull_idx.squeeze()]


    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    if feather > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), feather)
    return mask




def seamless_clone(src_warped, tgt_img, mask_gray):
    """OpenCV Poisson blending (NORMAL_CLONE) centered on mask's centroid."""
    ys, xs = (mask_gray > 0).nonzero()
    if len(xs) == 0:
        return src_warped
    center = (int(xs.mean()), int(ys.mean()))
    return cv2.seamlessClone(src_warped, tgt_img, mask_gray, center, cv2.NORMAL_CLONE)