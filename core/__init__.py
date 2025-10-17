# Khởi tạo package core
"""
Core packet for HomeFaceAI
Chứa các thành phần cốt lõi của hệ thống
"""

# import các module chính. trong core
from .api import load_image_file, face_locations, batch_face_locations, face_landmarks, face_encodings, compare_faces, face_distance


# Định nghĩa phiên bản của version (tuỳ chỉnh)
__version__ = "1.0.0"

# Định nghĩa __all__ để kiểm soát các export khi sử dụng from core import *
__all__ = ["FaceDetector", "FaceRecognizer", "PersonManager"]