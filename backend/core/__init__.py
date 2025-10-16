"""
Backend core package for HomeFaceAI (moved under backend/)
This module re-exports the public API from the core implementation.
"""
from .api import (
    load_image_file,
    face_locations,
    batch_face_locations,
    face_landmarks,
    face_encodings,
    compare_faces,
    face_distance,
)

from .face_detector import FaceDetector

__all__ = [
    "load_image_file",
    "face_locations",
    "batch_face_locations",
    "face_landmarks",
    "face_encodings",
    "compare_faces",
    "face_distance",
    "FaceDetector",
]

__version__ = "1.0.0"
