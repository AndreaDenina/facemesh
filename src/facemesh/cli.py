from __future__ import annotations
import argparse
import os
import cv2
import numpy as np


from .utils import imread, ensure_dir
from .landmarks import detect_landmarks
from .triangulation import compute_delaunay
from .warp2d import warp_piecewise
from .blend import face_mask_from_landmarks, seamless_clone

def main():
    p = argparse.ArgumentParser(description="2D face retarget using landmarks + Delaunay warp")
    p.add_argument("--src", required=True, help="path to source portrait image")
    p.add_argument("--tgt", required=True, help="path to target/reference face image")
    p.add_argument("--out", default="outputs/result.jpg", help="output image path")
    p.add_argument("--max-faces", type=int, default=1, help="max number of faces to detect")
    p.add_argument("--poisson", action="store_true", help="use Poisson seamless clone")
    args = p.parse_args()


    src = imread(args.src)
    tgt = imread(args.tgt)

    print(src)
    print(tgt)


    # Detect landmarks
    pts_src = detect_landmarks(src, max_faces=args.max_faces)
    pts_tgt = detect_landmarks(tgt, max_faces=1)
    if pts_src is None:
        raise SystemExit("No face found in source image")
    if pts_tgt is None:
        raise SystemExit("No face found in target image")


    # Triangulate on target space
    h, w = tgt.shape[:2]
    triangles = compute_delaunay((h, w), pts_tgt)
    if not triangles:
        raise SystemExit("Delaunay triangulation failed")


    # Warp source -> target geometry
    warped = warp_piecewise(src, tgt, pts_src, pts_tgt, triangles)


    # Optional masking / Poisson blending
    mask = face_mask_from_landmarks((h, w), pts_tgt, feather=15)
    mask3 = cv2.merge([mask, mask, mask])


    if args.poisson:
        out = seamless_clone(warped, tgt, mask)
    else:
    # Simple alpha composite
        out = cv2.convertScaleAbs(tgt * (1 - (mask3 / 255.0)) + warped * (mask3 / 255.0))


    ensure_dir(os.path.dirname(args.out) or ".")
    cv2.imwrite(args.out, out)
    print(f"Saved -> {args.out}")




if __name__ == "__main__":
    main()