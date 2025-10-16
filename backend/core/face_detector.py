"""
Module phát hiện khuôn mặt cho HomeFaceAI backend.

Cung cấp class FaceDetector để phát hiện và trích xuất thông tin khuôn mặt
từ các file ảnh hoặc numpy array sử dụng thư viện face_recognition.

Classes:
    FaceDetector: Class chính để phát hiện khuôn mặt
"""
import os
import numpy as np
import face_recognition.api as face_recognition


class FaceDetector:
    """
    Class phát hiện khuôn mặt trong hình ảnh sử dụng thư viện face_recognition.
    
    Hỗ trợ 2 model:
    - HOG (Histogram of Oriented Gradients): Nhanh, phù hợp với CPU
    - CNN (Convolutional Neural Network): Chính xác hơn, cần GPU để tối ưu
    
    Attributes:
        model (str): Tên model sử dụng ('hog' hoặc 'cnn')
        upsample (int): Số lần phóng to ảnh để tìm mặt nhỏ hơn (1 = không phóng, 2 = phóng 2x, v.v.)
    
    Example:
        >>> detector = FaceDetector(model="hog", upsample=1)
        >>> faces = detector.detect_faces("path/to/image.jpg")
        >>> boxes = detector.get_face_bounding_boxes("path/to/image.jpg")
    """
    
    def __init__(self, model="hog", upsample=1):
        """
        Khởi tạo FaceDetector với tham số cấu hình.
        
        Args:
            model (str, optional): Model phát hiện khuôn mặt. 
                - 'hog' (mặc định): Nhanh, phù hợp CPU
                - 'cnn': Chính xác hơn, phù hợp GPU
            upsample (int, optional): Số lần phóng to ảnh (mặc định: 1).
                Giá trị cao hơn = tìm được mặt nhỏ hơn nhưng chậm hơn.
                Phạm vi khuyến nghị: 1-2
        
        Raises:
            ValueError: Nếu model không phải 'hog' hoặc 'cnn'
        
        Examples:
            >>> detector_fast = FaceDetector()  # Mặc định: hog, upsample=1
            >>> detector_accurate = FaceDetector(model="cnn", upsample=2)
        """
        if model not in ("hog", "cnn"):
            raise ValueError(f"Model phải là 'hog' hoặc 'cnn', nhận được: {model}")
        
        self.model = model
        self.upsample = upsample

    def detect_faces(self, image_path_or_array):
        """
        Phát hiện khuôn mặt trong ảnh từ đường dẫn file hoặc numpy array.
        
        Hàm này chấp nhận cả đường dẫn file ảnh (str) hoặc numpy array,
        sau đó sử dụng model đã cấu hình để phát hiện vị trí các khuôn mặt.
        Kết quả trả về dưới dạng bounding boxes CSS (top, right, bottom, left).
        
        Args:
            image_path_or_array (str or np.ndarray): 
                - Nếu str: Đường dẫn đến file ảnh (jpg, png, bmp, v.v.)
                - Nếu np.ndarray: Array ảnh RGB hoặc BGR (height, width, 3)
        
        Returns:
            list: Danh sách tuple (top, right, bottom, left) cho mỗi khuôn mặt phát hiện được.
                - top (int): Pixel y từ trên cùng
                - right (int): Pixel x từ phải cùng
                - bottom (int): Pixel y từ dưới cùng
                - left (int): Pixel x từ trái cùng
                
                Nếu không phát hiện được mặt, trả về danh sách rỗng []
        
        Raises:
            ValueError: Nếu file không tồn tại (khi image_path_or_array là str)
            TypeError: Nếu ảnh không phải file hoặc array hợp lệ
        
        Examples:
            >>> detector = FaceDetector()
            >>> # Từ file
            >>> faces = detector.detect_faces("person.jpg")
            >>> print(faces)  # [(50, 150, 200, 100), (300, 400, 450, 350)]
            
            >>> # Từ numpy array
            >>> import cv2
            >>> img = cv2.imread("person.jpg")
            >>> img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            >>> faces = detector.detect_faces(img_rgb)
        """
        # Kiểm tra input và load ảnh nếu cần
        if isinstance(image_path_or_array, str):
            # Input là đường dẫn file
            if not os.path.isfile(image_path_or_array):
                raise ValueError(f"File không tồn tại: {image_path_or_array}")
            # Load ảnh từ file
            image = face_recognition.load_image_file(image_path_or_array)
        else:
            # Input là numpy array, dùng trực tiếp
            image = image_path_or_array

        # Gọi face_recognition.face_locations để phát hiện khuôn mặt
        face_locations = face_recognition.face_locations(
            image,
            number_of_times_to_upsample=self.upsample,
            model=self.model,
        )

        return face_locations

    def detect_faces_batch(self, images, batch_size=128):
        """
        Phát hiện khuôn mặt trong một batch (hàng loạt) ảnh.
        
        Hàm này tối ưu hóa việc xử lý nhiều ảnh cùng lúc bằng cách
        gom nhóm chúng thành batch, giúp tăng tốc độ xử lý so với
        gọi detect_faces() từng ảnh một lần.
        
        Chỉ hoạt động với model CNN (model='cnn'), HOG không hỗ trợ batch.
        
        Args:
            images (list): Danh sách các numpy array ảnh (mỗi ảnh là shape (height, width, 3))
            batch_size (int, optional): Kích thước batch để xử lý cùng lúc (mặc định: 128).
                Giá trị cao hơn = nhanh hơn nhưng dùng nhiều memory hơn.
                Phạm vi khuyến nghị: 32-256 tùy vào GPU memory
        
        Returns:
            list: Danh sách danh sách tuple, mỗi phần tử là kết quả detect_faces cho một ảnh.
                Ví dụ: [[(50, 150, 200, 100)], [(300, 400, 450, 350)]]
        
        Raises:
            ValueError: Nếu images là danh sách rỗng
            TypeError: Nếu images không phải list hoặc array không hợp lệ
        
        Notes:
            - Chỉ sử dụng khi xử lý nhiều ảnh (> 5 ảnh)
            - Với ít ảnh, detect_faces() có thể nhanh hơn do overhead batch
            - Model CNN yêu cầu CUDA/GPU để chạy nhanh
        
        Examples:
            >>> import cv2
            >>> detector = FaceDetector(model="cnn")
            >>> images = [
            ...     cv2.imread("img1.jpg"),
            ...     cv2.imread("img2.jpg"),
            ...     cv2.imread("img3.jpg")
            ... ]
            >>> all_faces = detector.detect_faces_batch(images)
            >>> print(all_faces)  # [[(50, 150, 200, 100)], [(300, 400, 450, 350)], []]
        """
        return face_recognition.batch_face_locations(
            images,
            number_of_times_to_upsample=self.upsample,
            batch_size=batch_size,
        )

    def get_face_bounding_boxes(self, image_path_or_array):
        """
        Lấy bounding boxes của khuôn mặt dưới dạng dict với tọa độ x, y, width, height.
        
        Hàm này gọi detect_faces() để lấy vị trí khuôn mặt, sau đó chuyển đổi
        từ định dạng CSS (top, right, bottom, left) sang định dạng x, y, width, height
        phổ biến trong các ứng dụng đồ họa và frontend.
        
        Args:
            image_path_or_array (str or np.ndarray): 
                File ảnh hoặc numpy array (giống như detect_faces)
        
        Returns:
            list: Danh sách dict, mỗi dict có các key:
                - 'x' (int): Tọa độ x từ trái (pixel)
                - 'y' (int): Tọa độ y từ trên (pixel)
                - 'width' (int): Chiều rộng bounding box (pixel)
                - 'height' (int): Chiều cao bounding box (pixel)
        
        Example:
            >>> detector = FaceDetector()
            >>> boxes = detector.get_face_bounding_boxes("person.jpg")
            >>> print(boxes)
            # Output:
            # [
            #     {'x': 100, 'y': 50, 'width': 80, 'height': 100},
            #     {'x': 350, 'y': 300, 'width': 75, 'height': 95}
            # ]
            
            >>> # Sử dụng trong backend API response
            >>> import json
            >>> response = {"faces": boxes}
            >>> print(json.dumps(response))
        
        Notes:
            - Định dạng x, y, width, height dễ sử dụng cho frontend (Flutter, React, v.v.)
            - Tọa độ tính theo pixel, bắt đầu từ góc trên-trái (0, 0)
            - Sẽ trả về danh sách rỗng nếu không phát hiện được mặt
        """
        # Phát hiện khuôn mặt sử dụng detect_faces()
        face_locs = self.detect_faces(image_path_or_array)
        
        # Chuyển đổi từ (top, right, bottom, left) sang dict x, y, width, height
        bounding_boxes = []
        for top, right, bottom, left in face_locs:
            bounding_boxes.append({
                'x': left,          # Tọa độ x từ trái
                'y': top,           # Tọa độ y từ trên
                'width': right - left,      # Chiều rộng = right - left
                'height': bottom - top      # Chiều cao = bottom - top
            })
        
        return bounding_boxes
