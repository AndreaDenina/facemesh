from __future__ import annotations
import cv2
import numpy as np
import mediapipe as mp


_mp_face_mesh = mp.solutions.face_mesh


def detect_landmarks(img_bgr, max_faces: int = 1) -> np.ndarray | None:
    """Return (N,2) int32 array of 468 MediaPipe landmarks for the first face, or None."""
    h, w = img_bgr.shape[:2]
    with _mp_face_mesh.FaceMesh(
        static_image_mode=True,
        refine_landmarks=True,
        max_num_faces=max_faces,
    ) as fm:
        res = fm.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return None
    pts = [(int(lm.x * w), int(lm.y * h))
           for lm in res.multi_face_landmarks[0].landmark]
    return np.asarray(pts, dtype=np.int32)
