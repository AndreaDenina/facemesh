from __future__ import annotations
import cv2
import numpy as np




def _warp_triangle(src, dst, t_src, t_dst):
    r1 = cv2.boundingRect(np.float32([t_src]))
    r2 = cv2.boundingRect(np.float32([t_dst]))


    t1_rect = [(t_src[i][0] - r1[0], t_src[i][1] - r1[1]) for i in range(3)]
    t2_rect = [(t_dst[i][0] - r2[0], t_dst[i][1] - r2[1]) for i in range(3)]


    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_rect), (1.0, 1.0, 1.0), lineType=cv2.LINE_AA)


    img1_rect = src[r1[1]: r1[1]+r1[3], r1[0]: r1[0]+r1[2]]
    M = cv2.getAffineTransform(np.float32(t1_rect), np.float32(t2_rect))
    warped = cv2.warpAffine(img1_rect, M, (r2[2], r2[3]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


    dst_roi = dst[r2[1]: r2[1]+r2[3], r2[0]: r2[0]+r2[2]]
    dst_roi[:] = dst_roi * (1 - mask) + warped * mask




def warp_piecewise(src_img, tgt_img, pts_src, pts_tgt, triangles):
    """Piecewise affine warp from src onto tgt geometry.
    Returns a new image with warped src blended onto tgt.
    """
    out = tgt_img.copy()
    for i, j, k in triangles:
        t_src = [pts_src[i], pts_src[j], pts_src[k]]
        t_tgt = [pts_tgt[i], pts_tgt[j], pts_tgt[k]]
        _warp_triangle(src_img, out, t_src, t_tgt)
    return out