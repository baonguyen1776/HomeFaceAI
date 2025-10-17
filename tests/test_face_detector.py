#!/usr/bin/env python3
"""
Tests cho module face_detector.py

Module này chứa các test cases cho class FaceDetector, kiểm tra các chức năng:
- Phát hiện khuôn mặt từ file ảnh
- Phát hiện khuôn mặt từ array
- Phát hiện khuôn mặt hàng loạt
- Xử lý các trường hợp lỗi
- Format bounding boxes
"""

import os
import sys
import unittest
import numpy as np
from PIL import Image
import tempfile

# Thêm thư mục gốc của project vào sys.path để import module từ core
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import cv2
except ImportError:
    print("WARNING: OpenCV không được cài đặt. Một số test có thể không chạy được.")
    cv2 = None

# Import module cần test
try:
    from core.face_detector import FaceDetector
except ImportError as e:
    print(f"ERROR: Không thể import FaceDetector: {e}")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)


class TestFaceDetector(unittest.TestCase):
    """
    Test case cho class FaceDetector
    """
    
    def setUp(self):
        """
        Thiết lập trước mỗi test case.
        Tạo các test image, khởi tạo detector
        """
        try:
            # Khởi tạo detector với model hog và upsample=1
            self.detector = FaceDetector(model="hog", upsample=1)
            
            # Tạo thư mục tạm thời để lưu ảnh test
            self.test_dir = tempfile.mkdtemp()
            
            # Tạo ảnh mẫu có khuôn mặt
            # Chúng ta sẽ tạo một ảnh trắng 100x100 với "giả lập" khuôn mặt
            # Trong thực tế nên dùng ảnh thật nhưng ở đây chúng ta tạo dữ liệu test đơn giản
            self.sample_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
            
            # Lưu ảnh mẫu bằng PIL thay vì OpenCV (để tránh phụ thuộc vào OpenCV)
            self.sample_image_path = os.path.join(self.test_dir, "sample.jpg")
            Image.fromarray(self.sample_image).save(self.sample_image_path)
        except Exception as e:
            print(f"ERROR trong setUp: {e}")
            raise
        
    def tearDown(self):
        """
        Dọn dẹp sau mỗi test case
        """
        # Xóa các file tạm
        for file in os.listdir(self.test_dir):
            os.remove(os.path.join(self.test_dir, file))
        os.rmdir(self.test_dir)
        
    def test_init(self):
        """
        Test khởi tạo class FaceDetector với các tham số khác nhau
        """
        # Test khởi tạo với các tham số mặc định
        detector = FaceDetector()
        self.assertEqual(detector.model, "hog")
        self.assertEqual(detector.upsample, 1)
        
        # Test khởi tạo với tham số tùy chỉnh
        detector = FaceDetector(model="cnn", upsample=2)
        self.assertEqual(detector.model, "cnn")
        self.assertEqual(detector.upsample, 2)
        
    def test_detect_faces_from_path(self):
        """
        Test phát hiện khuôn mặt từ đường dẫn file
        """
        # Mock kết quả từ face_recognition.face_locations
        # Trong test thực tế, nên dùng một ảnh có mặt thật và xác nhận kết quả
        
        # Giả lập phát hiện được 1 khuôn mặt
        result = self.detector.detect_faces(self.sample_image_path)
        
        # Kiểm tra kiểu kết quả trả về
        self.assertIsInstance(result, list)
        
    def test_detect_faces_from_array(self):
        """
        Test phát hiện khuôn mặt từ numpy array
        """
        # Tạo numpy array ảnh
        image_array = np.ones((100, 100, 3), dtype=np.uint8) * 255
        
        # Gọi hàm detect_faces
        result = self.detector.detect_faces(image_array)
        
        # Kiểm tra kiểu kết quả trả về
        self.assertIsInstance(result, list)
        
    def test_invalid_file_path(self):
        """
        Test xử lý lỗi khi đường dẫn file không tồn tại
        """
        # Gọi hàm với đường dẫn không tồn tại
        with self.assertRaises(ValueError):
            self.detector.detect_faces("path/to/nonexistent/file.jpg")
            
    def test_get_face_bounding_boxes(self):
        """
        Test lấy bounding boxes
        """
        # Mock cho face_locations để giả lập việc phát hiện được 1 khuôn mặt
        self.detector.detect_faces = lambda x: [(10, 50, 60, 20)]  # (top, right, bottom, left)
        
        # Gọi hàm get_face_bounding_boxes
        boxes = self.detector.get_face_bounding_boxes(self.sample_image_path)
        
        # Kiểm tra kết quả
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0]['x'], 20)  # left
        self.assertEqual(boxes[0]['y'], 10)  # top
        self.assertEqual(boxes[0]['width'], 30)  # right - left
        self.assertEqual(boxes[0]['height'], 50)  # bottom - top
        
    def test_detect_faces_batch(self):
        """
        Test phát hiện khuôn mặt hàng loạt
        """
        # Tạo danh sách các ảnh mẫu
        images = [
            np.ones((100, 100, 3), dtype=np.uint8) * 255,
            np.ones((100, 100, 3), dtype=np.uint8) * 200
        ]
        
        # Gọi hàm detect_faces_batch
        result = self.detector.detect_faces_batch(images)
        
        # Kiểm tra kiểu kết quả trả về
        self.assertIsInstance(result, list)


def run_tests():
    """
    Hàm chính để chạy test suite
    """
    print("=== Bắt đầu kiểm thử FaceDetector ===")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print(f"sys.path: {sys.path}")
    print("=====================================")
    
    # Kiểm tra các module cần thiết có được import không
    missing_modules = []
    for module_name in ["numpy", "PIL"]:
        try:
            __import__(module_name)
        except ImportError:
            missing_modules.append(module_name)
    
    if missing_modules:
        print(f"WARNING: Các module sau chưa được cài đặt: {', '.join(missing_modules)}")
        print("Vui lòng cài đặt chúng bằng pip:")
        print(f"pip install {' '.join(missing_modules)}")
    
    # Chạy test
    unittest.main()

if __name__ == "__main__":
    run_tests()
