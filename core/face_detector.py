# Phát hiện khuôn mặt
import os 
import numpy as np
import face_recognition.api as face_recognition

class FaceDetector:
    """
    Class phát hiện khuôn mặt trong hình ảnh sử dụng thư viện face_recognition.
    """
    def __init__(self, model="hog", upsample=1):
        """
        Khởi tạo FaceDetector.
        
        Args:
            model (str): Model phát hiện mặt, "hog" (nhanh) hoặc "cnn" (chính xác hơn)
            upsample (int): Số lần upsample ảnh để tìm mặt nhỏ hơn
        """
        self.model = model
        self.upsample = upsample
    
    def detect_faces(self, image_path_or_array):
        """
        Phát hiện khuôn mặt trong ảnh.
        
        Args:
            image_path_or_array: Đường dẫn đến file ảnh hoặc numpy array của ảnh
            
        Returns:
            list: Danh sách các khuôn mặt phát hiện được dưới dạng (top, right, bottom, left)
        """
        # Nếu là đường dẫn, load ảnh, nếu là array thì dùng trực tiếp
        if isinstance(image_path_or_array, str):
            if not os.path.isfile(image_path_or_array):
                raise ValueError(f"File không tồn tại: {image_path_or_array}")
            image = face_recognition.load_image_file(image_path_or_array)
        else:
            image = image_path_or_array
            
        # Sử dụng face_locations từ thư viện face_recognition
        face_locations = face_recognition.face_locations(
            image, 
            number_of_times_to_upsample=self.upsample,
            model = self.model
        )
        
        return face_locations

    def detect_faces_batch(self, images, batch_size = 128):
        """
        Phát hiện khuôn mặt trong một batch ảnh (xử lý hàng loạt).
        
        Args:
            images (list): Danh sách các numpy array ảnh
            batch_size (int): Kích thước batch để xử lý cùng lúc
            
        Returns:
            list: Danh sách kết quả, mỗi phần tử là list các khuôn mặt trong một ảnh
        """
        # Sử dụng batch_face_locations từ thư viện face_recognition
        return face_recognition.batch_face_locations(
            images,
            number_of_times_to_upsample = self.upsample,
            batch_size = batch_size
        )
    
    def get_face_bounding_boxes(self, image_path_or_array):
        """
        Lấy bounding boxes của các khuôn mặt trong ảnh.
        
        Args:
            image_path_or_array: Đường dẫn đến file ảnh hoặc numpy array
            
        Returns:
            list: Danh sách các bounding box dưới dạng dict (x, y, width, height)
        """
        face_locs = self.detect_faces(image_path_or_array)
        
        # Chuyển đổi (top, right, bottom, left) thành (x, y, width, height)
        bounding_boxes = []
        for top, right, bottom, left in face_locs:
            bounding_boxes.append({
                'x': left,
                'y': top,
                'width': right - left,
                'height': bottom - top
            })
            
        return bounding_boxes

