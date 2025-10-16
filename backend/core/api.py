# Quản lý dữ liệu người dùng và API cho nhận diện khuôn mặt (moved to backend/core)
"""
Module API cấp thấp cho nhận diện khuôn mặt sử dụng dlib và face_recognition.

Cung cấp các hàm API chính cho việc nhận diện khuôn mặt:
- Tải ảnh từ file
- Phát hiện vị trí khuôn mặt (HOG hoặc CNN)
- Trích xuất landmarks (68 điểm hoặc 5 điểm)
- Tính toán face encodings (128-D vectors)
- So sánh các khuôn mặt
- Tính khoảng cách giữa các face encodings

Các model được load từ face_recognition_models package:
- pose_predictor_68_point: Mô hình dự đoán 68 landmarks chi tiết
- pose_predictor_5_point: Mô hình dự đoán 5 landmarks đơn giản
- cnn_face_detector: Mô hình phát hiện mặt sử dụng CNN (chính xác)
- face_encoder: Mô hình mã hóa khuôn mặt thành vector 128D

Warning:
    Các model rất lớn (~500MB). Lần đầu import sẽ tự động download.
"""

import PIL.Image
import numpy as np
import dlib
from PIL import ImageFile

# Cho phép tải ảnh bị cắt ngắn (bị lỗi header nhưng vẫn có dữ liệu ảnh)
ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    import face_recognition_models
except Exception as e:
    print("[ERROR] Không thể import thư viện face_recognition_models. Vui lòng kiểm tra lại cài đặt.")
    raise ImportError("face_recognition_models không được cài đặt. Chạy: pip install face-recognition") from e

# ============================================================================
# KHỞI TẠO CÁC MÔ HÌNH DLIB (thực hiện lần đầu import, tốn thời gian)
# ============================================================================

# Tải model dự đoán pose 68 điểm cho landmarks khuôn mặt (chi tiết)
predictor_68_point_model = face_recognition_models.pose_predictor_model_location()
pose_predictor_68_point = dlib.shape_predictor(predictor_68_point_model)

# Tải model dự đoán pose 5 điểm (đơn giản hơn, nhanh hơn)
predictor_5_point_model = face_recognition_models.pose_predictor_five_point_model_location()
pose_predictor_5_point = dlib.shape_predictor(predictor_5_point_model)

# Tải model phát hiện khuôn mặt sử dụng CNN (chính xác hơn HOG)
cnn_face_detector_model = face_recognition_models.cnn_face_detector_model_location()
cnn_face_detector = dlib.cnn_face_detection_model_v1(cnn_face_detector_model)

# Tải model nhận diện khuôn mặt (mã hóa khuôn mặt thành vector 128D)
face_recognition_model_location = face_recognition_models.face_recognition_model_location()
face_encoder = dlib.face_recognition_model_v1(face_recognition_model_location)

# ============================================================================
# HELPER FUNCTIONS - Các hàm tiện ích để chuyển đổi định dạng
# ============================================================================

def _rect_to_css(rect):
    """
    Chuyển đổi từ định dạng dlib rectangle sang CSS format (top, right, bottom, left).
    
    Args:
        rect (dlib.rectangle): Đối tượng rectangle từ dlib
    
    Returns:
        tuple: (top, right, bottom, left) - pixel coordinates
    """
    return rect.top(), rect.right(), rect.bottom(), rect.left()


def _css_to_rect(css):
    """
    Chuyển đổi từ CSS format (top, right, bottom, left) sang dlib rectangle.
    
    Args:
        css (tuple): (top, right, bottom, left)
    
    Returns:
        dlib.rectangle: Đối tượng rectangle của dlib
    """
    return dlib.rectangle(css[3], css[0], css[1], css[2])


def _trim_css_to_bounds(css, image_shape):
    """
    Cắt/giới hạn CSS bounds để nằm trong kích thước ảnh (tránh vượt ra ngoài).
    
    Khi bounding box vượt quá biên của ảnh, hàm này sẽ cắt nó lại
    để nằm trong phạm vi (0, 0) đến (width, height).
    
    Args:
        css (tuple): (top, right, bottom, left) - bounding box
        image_shape (tuple): (height, width, channels) - kích thước ảnh
    
    Returns:
        tuple: (top, right, bottom, left) - bounding box đã được cắt
    """
    return (
        max(css[0], 0),                     # top: tối thiểu 0
        min(css[1], image_shape[1]),        # right: tối đa width
        min(css[2], image_shape[0]),        # bottom: tối đa height
        max(css[3], 0),                     # left: tối thiểu 0
    )

# ============================================================================
# PUBLIC API FUNCTIONS - Các hàm API công khai
# ============================================================================

def face_distance(face_encodings, face_to_compare):
    """
    Tính khoảng cách Euclidean giữa một tập hợp các face encodings và một face encoding.
    
    Khoảng cách càng nhỏ = hai khuôn mặt càng giống nhau.
    
    Args:
        face_encodings (list or np.ndarray): Danh sách các face encodings để so sánh
            Shape: (N, 128) hoặc list of (128,) arrays
        face_to_compare (np.ndarray): Face encoding để so sánh
            Shape: (128,)
    
    Returns:
        np.ndarray: Danh sách khoảng cách Euclidean
            Shape: (N,) - một khoảng cách cho mỗi encoding
            
            Thông thường:
            - distance < 0.6: Cùng một người
            - 0.6 <= distance <= 0.7: Có thể là cùng người
            - distance > 0.7: Khác người
    
    Examples:
        >>> import numpy as np
        >>> face1 = np.random.rand(128)
        >>> face2 = np.random.rand(128)
        >>> face3 = np.random.rand(128)
        >>> distances = face_distance([face2, face3], face1)
        >>> print(distances)  # [0.45, 0.82]
    """
    if len(face_encodings) == 0:
        return np.empty((0))

    return np.linalg.norm(face_encodings - face_to_compare, axis=1)


def load_image_file(file, mode="RGB"):
    """
    Tải file ảnh và chuyển đổi thành numpy array.
    
    Hỗ trợ các định dạng ảnh phổ biến: jpg, png, bmp, gif, v.v.
    
    Args:
        file (str or file-like): 
            - str: Đường dẫn đến file ảnh
            - file-like: File object (từ open() hoặc request.FILES)
        mode (str, optional): Chế độ màu để mở ảnh (mặc định: "RGB")
            - "RGB": 3 kênh màu (Red, Green, Blue)
            - "L": Grayscale (xám)
            - "RGBA": 4 kênh (RGB + Alpha/transparency)
    
    Returns:
        np.ndarray: Ảnh dưới dạng numpy array
            Shape: (height, width, 3) cho RGB
            Shape: (height, width, 4) cho RGBA
            Shape: (height, width) cho Grayscale
            dtype: uint8 (0-255)
    
    Examples:
        >>> # Từ file
        >>> image = load_image_file("person.jpg")
        >>> print(image.shape)  # (480, 640, 3)
        
        >>> # Từ file object (Flask request)
        >>> from flask import request
        >>> image = load_image_file(request.files['image'])
        
        >>> # Grayscale
        >>> gray_image = load_image_file("person.jpg", mode="L")
        >>> print(gray_image.shape)  # (480, 640)
    """
    im = PIL.Image.open(file)
    if mode:
        im = im.convert(mode)
    return np.array(im)


def _raw_face_locations(image, number_of_times_to_upsample=1, model="cnn"):
    """
    Phát hiện vị trí khuôn mặt trong ảnh sử dụng dlib (low-level).
    
    Hàm nội bộ, không sử dụng trực tiếp. Dùng face_locations() thay thế.
    
    Args:
        image (np.ndarray): Ảnh input (RGB)
        number_of_times_to_upsample (int): Số lần phóng to ảnh
        model (str): 'cnn' hoặc 'hog'
    
    Returns:
        list: Danh sách dlib.rectangle objects
    """
    if model == "cnn":
        return cnn_face_detector(image, number_of_times_to_upsample)
    else:
        detector = dlib.get_frontal_face_detector()
        return detector(image, number_of_times_to_upsample)


def face_locations(image, number_of_times_to_upsample=1, model="hog"):
    """
    Phát hiện vị trí khuôn mặt trong ảnh và trả về dưới dạng CSS format.
    
    Hàm chính để phát hiện khuôn mặt. Trả về vị trí dưới dạng (top, right, bottom, left).
    
    Args:
        image (np.ndarray): Ảnh input (RGB hoặc BGR, will be converted)
        number_of_times_to_upsample (int, optional): Số lần phóng to ảnh (mặc định: 1)
            - 0: Không phóng (nhanh nhất, chỉ tìm mặt lớn)
            - 1: Không phóng (tốc độ cân bằng)
            - 2: Phóng 2x (tìm được mặt nhỏ hơn nhưng chậm 4x)
            - 3: Phóng 3x (rất chậm, chỉ dùng khi cần thiết)
        model (str, optional): Model phát hiện (mặc định: "hog")
            - "hog": Nhanh, phù hợp CPU, độ chính xác 95%
            - "cnn": Chính xác hơn, phù hợp GPU, độ chính xác 99%
    
    Returns:
        list: Danh sách tuple (top, right, bottom, left) cho mỗi khuôn mặt
            Nếu không tìm được, trả về danh sách rỗng []
            
    Examples:
        >>> import cv2
        >>> # Load ảnh
        >>> image = cv2.imread("person.jpg")
        >>> image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV dùng BGR
        >>> 
        >>> # Phát hiện mặt - nhanh, chính xác vừa phải
        >>> faces = face_locations(image_rgb, model="hog")
        >>> print(faces)  # [(50, 150, 200, 100)]
        >>> 
        >>> # Phát hiện mặt - chính xác hơn nhưng chậm hơn
        >>> faces = face_locations(image_rgb, model="cnn")
        >>> 
        >>> # Phát hiện mặt nhỏ - phóng ảnh 2x
        >>> faces = face_locations(image_rgb, number_of_times_to_upsample=2)
    """
    if model == "cnn":
        return [
            _trim_css_to_bounds(_rect_to_css(face.rect), image.shape)
            for face in cnn_face_detector(image, number_of_times_to_upsample)
        ]
    else:
        return [
            _trim_css_to_bounds(_rect_to_css(face), image.shape)
            for face in _raw_face_locations(image, number_of_times_to_upsample, model)
        ]


def _raw_face_location_batched(images, number_of_times_to_upsample=1, batch_size=128):
    """
    Phát hiện khuôn mặt trong batch ảnh (low-level, chỉ dùng CNN).
    
    Hàm nội bộ, không sử dụng trực tiếp. Dùng batch_face_locations() thay thế.
    """
    return [cnn_face_detector(image, number_of_times_to_upsample) for image in images]


def batch_face_locations(images, number_of_times_to_upsample=1, batch_size=128):
    """
    Phát hiện khuôn mặt trong nhiều ảnh cùng lúc (xử lý batch).
    
    Hàm này tối ưu hóa việc xử lý nhiều ảnh bằng cách gom thành batch,
    giúp tăng tốc độ so với gọi face_locations từng ảnh một.
    
    Args:
        images (list): Danh sách numpy array ảnh (mỗi ảnh là (height, width, 3))
        number_of_times_to_upsample (int): Số lần phóng to ảnh
        batch_size (int): Kích thước batch xử lý cùng lúc
            - Giá trị cao hơn = nhanh hơn nhưng dùng nhiều RAM hơn
    
    Returns:
        list: Danh sách danh sách tuple, mỗi phần tử tương ứng một ảnh
            Ví dụ: [[(50, 150, 200, 100)], [(300, 400, 450, 350)], []]
    
    Examples:
        >>> import cv2
        >>> # Load nhiều ảnh
        >>> images = [
        ...     cv2.cvtColor(cv2.imread(f"img{i}.jpg"), cv2.COLOR_BGR2RGB)
        ...     for i in range(1, 4)
        ... ]
        >>> 
        >>> # Phát hiện mặt trong tất cả
        >>> all_faces = batch_face_locations(images)
        >>> print(all_faces)
        # [
        #     [(50, 150, 200, 100)],  # img1.jpg có 1 mặt
        #     [(300, 400, 450, 350)],  # img2.jpg có 1 mặt
        #     []                        # img3.jpg không có mặt
        # ]
    """
    def convert_cnn_detection_to_css(detections, image_shape):
        return [
            _trim_css_to_bounds(_rect_to_css(face.rect), image_shape)
            for face in detections
        ]

    raw_detections_batched = _raw_face_location_batched(images, number_of_times_to_upsample, batch_size)
    return [convert_cnn_detection_to_css(detections, images[idx].shape) for idx, detections in enumerate(raw_detections_batched)]




def _raw_face_landmarks(face_image, face_locations=None, model="large"):
    """
    Trích xuất landmarks từ ảnh sử dụng dlib (low-level).
    
    Hàm nội bộ, không sử dụng trực tiếp. Dùng face_landmarks() thay thế.
    """
    if face_locations is None:
        face_locations = _raw_face_locations(face_image)
    else:
        face_locations = [_css_to_rect(face_location) for face_location in face_locations]

    pose_predictor = pose_predictor_68_point
    if model == "small":
        pose_predictor = pose_predictor_5_point
    return [
        pose_predictor(face_image, face_location)
        for face_location in face_locations
    ]


def face_landmarks(face_image, face_locations=None, model="large"):
    """
    Trích xuất các điểm đặc trưng trên khuôn mặt (landmarks).
    
    Landmarks là các điểm quan trọng trên khuôn mặt: mắt, mũi, miệng, v.v.
    Hữu ích để định vị chính xác các bộ phận khuôn mặt.
    
    Args:
        face_image (np.ndarray): Ảnh input (RGB)
        face_locations (list, optional): Danh sách vị trí khuôn mặt từ face_locations()
            Nếu None, sẽ tự động phát hiện
        model (str, optional): Model landmarks (mặc định: "large")
            - "large": 68 điểm chi tiết (mắt, lông mày, mũi, cằm, miệng, v.v.)
            - "small": 5 điểm đơn giản (2 mắt, 2 mũi, 1 miệng) - nhanh hơn
    
    Returns:
        list: Danh sách dict, mỗi dict chứa tên và tọa độ của landmarks
            
        Cho model="large" (68 điểm):
            {
                "chin": [(x1, y1), ..., (x17, y17)],
                "left_eyebrow": [(x1, y1), ..., (x5, y5)],
                "right_eyebrow": [...],
                "nose_bridge": [...],
                "nose_tip": [...],
                "left_eye": [...],
                "right_eye": [...],
                "top_lip": [...],
                "bottom_lip": [...]
            }
        
        Cho model="small" (5 điểm):
            {
                "nose_tip": [(x, y)],
                "left_eye": [(x1, y1), (x2, y2)],
                "right_eye": [(x1, y1), (x2, y2)]
            }
    
    Examples:
        >>> image = load_image_file("person.jpg")
        >>> landmarks = face_landmarks(image, model="large")
        >>> print(landmarks)
        # [
        #     {
        #         "chin": [(150, 50), (151, 51), ...],
        #         "left_eye": [(100, 100), (120, 95)],
        #         ...
        #     }
        # ]
        >>> 
        >>> # Lấy tọa độ mắt trái
        >>> left_eye = landmarks[0]["left_eye"]
        >>> print(left_eye)  # [(100, 100), (120, 95), ...]
    """
    landmarks = _raw_face_landmarks(face_image, face_locations, model)
    landmarks_as_tuples = [[(p.x, p.y) for p in landmark.parts()] for landmark in landmarks]

    if model == 'large':
        return [{
            "chin": points[0:17],
            "left_eyebrow": points[17:22],
            "right_eyebrow": points[22:27],
            "nose_bridge": points[27:31],
            "nose_tip": points[31:36],
            "left_eye": points[36:42],
            "right_eye": points[42:48],
            "top_lip": points[48:55] + [points[64]] + [points[63]] + [points[62]] + [points[61]] + [points[60]],
            "bottom_lip": points[54:60] + [points[48]] + [points[60]] + [points[67]] + [points[66]] + [points[65]] + [points[64]]
        } for points in landmarks_as_tuples]
    elif model == 'small':
        return [{
            "nose_tip": [points[4]],
            "left_eye": points[2:4],
            "right_eye": points[0:2],
        } for points in landmarks_as_tuples]
    else:
        raise ValueError("Invalid landmarks model type. Supported models are ['small', 'large'].")


def face_encodings(face_image, known_face_locations=None, num_jitters=1, model="small"):
    """
    Trích xuất face encodings (vector 128D) từ ảnh.
    
    Face encoding là một vector 128 chiều đại diện cho đặc trưng khuôn mặt.
    Hai khuôn mặt của cùng một người sẽ có encodings gần giống nhau.
    
    Đây là hàm quan trọng nhất trong face recognition - dùng để so sánh khuôn mặt.
    
    Args:
        face_image (np.ndarray): Ảnh input (RGB)
        known_face_locations (list, optional): Danh sách vị trí khuôn mặt
            Nếu None, sẽ tự động phát hiện. Cấp nhanh hơn nếu cung cấp.
        num_jitters (int, optional): Số lần jittering để tăng độ chính xác (mặc định: 1)
            - 1: Nhanh, chính xác 95%
            - 5: Chậm 5x nhưng chính xác 99%
        model (str, optional): Model landmarks để căn chỉnh khuôn mặt
            - "small": 5 landmarks, nhanh
            - "large": 68 landmarks, chính xác hơn
    
    Returns:
        list: Danh sách numpy arrays, mỗi array là face encoding
            Shape: list of (128,) arrays
            dtype: float64
    
    Examples:
        >>> image = load_image_file("person.jpg")
        >>> encodings = face_encodings(image)
        >>> print(len(encodings))  # 1 (nếu có 1 mặt)
        >>> print(encodings[0].shape)  # (128,)
        >>> 
        >>> # Với jittering để tăng độ chính xác
        >>> encodings = face_encodings(image, num_jitters=5)
        >>> 
        >>> # Khi đã biết vị trí mặt
        >>> faces = face_locations(image)
        >>> encodings = face_encodings(image, known_face_locations=faces)
    
    Notes:
        - Jittering cao hơn = chính xác hơn nhưng chậm hơn
        - Encoding là deterministic nếu jitter=1 (cùng ảnh → cùng encoding)
        - Encoding không phải ảnh, không thể reverse để lấy lại ảnh (one-way)
    """
    raw_landmarks = _raw_face_landmarks(face_image, known_face_locations, model)

    return [
        np.array(face_encoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters))
        for raw_landmark_set in raw_landmarks
    ]


def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.7):
    """
    So sánh một face encoding với danh sách các face encodings đã biết.
    
    Hàm này tính khoảng cách Euclidean và so sánh với ngưỡng tolerance.
    
    Args:
        known_face_encodings (list): Danh sách face encodings đã biết
            - Có thể từ face_encodings()
            - Shape: list of (128,) arrays
        face_encoding_to_check (np.ndarray): Face encoding để so sánh
            Shape: (128,)
        tolerance (float, optional): Ngưỡng khoảng cách để coi là khớp (mặc định: 0.7)
            - 0.6: Chặt chẽ, ít false positive nhưng có thể miss
            - 0.7: Cân bằng (khuyến nghị)
            - 0.8: Lỏng lẻo, ít miss nhưng có thể false positive
    
    Returns:
        list: Danh sách boolean, True = khớp, False = không khớp
            Shape: (N,) - một boolean cho mỗi encoding trong known_face_encodings
    
    Examples:
        >>> # Tải ảnh 2 người
        >>> image1 = load_image_file("person1.jpg")
        >>> image2 = load_image_file("person2.jpg")
        >>> image3 = load_image_file("person1_again.jpg")
        >>> 
        >>> # Lấy encodings
        >>> encoding1 = face_encodings(image1)[0]
        >>> encoding2 = face_encodings(image2)[0]
        >>> encoding3 = face_encodings(image3)[0]
        >>> 
        >>> # So sánh
        >>> results = compare_faces([encoding1, encoding2], encoding3)
        >>> print(results)  # [True, False]
        >>> # encoding3 khớp với encoding1 (cùng người) nhưng không khớp encoding2
    
    Notes:
        - Khoảng cách < tolerance → True (khớp)
        - Khoảng cách >= tolerance → False (không khớp)
        - Tolerance nhỏ hơn = chặt chẽ hơn
        - Độ chính xác phụ thuộc vào chất lượng ảnh và alignment
    """
    distances = face_distance(known_face_encodings, face_encoding_to_check)
    return list(distances <= tolerance)