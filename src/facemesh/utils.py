from __future__ import annotations
import os
import cv2




def imread(path: str):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img




def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)