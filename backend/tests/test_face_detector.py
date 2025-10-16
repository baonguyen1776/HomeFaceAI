#!/usr/bin/env python3
"""
Backend tests for FaceDetector (adjusted to backend.core)
"""
import os
import sys
import unittest
import numpy as np
from PIL import Image
import tempfile

# Ensure backend package is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from backend.core.face_detector import FaceDetector
except ImportError as e:
    print(f"ERROR importing backend.core.face_detector: {e}")
    raise


class TestFaceDetectorBackend(unittest.TestCase):
    def setUp(self):
        self.detector = FaceDetector()
        self.test_dir = tempfile.mkdtemp()
        self.sample_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        self.sample_image_path = os.path.join(self.test_dir, 'sample.jpg')
        Image.fromarray(self.sample_image).save(self.sample_image_path)

    def tearDown(self):
        for file in os.listdir(self.test_dir):
            os.remove(os.path.join(self.test_dir, file))
        os.rmdir(self.test_dir)

    def test_detect_faces_from_path(self):
        res = self.detector.detect_faces(self.sample_image_path)
        self.assertIsInstance(res, list)


if __name__ == '__main__':
    unittest.main()
