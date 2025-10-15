# Quản lý dữ liệu người dùng và API cho nhận diện khuôn mặt
"""
Module này cung cấp các hàm API chính cho việc nhận diện khuôn mặt,
bao gồm tải ảnh, phát hiện khuôn mặt, landmarks, encodings, so sánh và khoảng cách.
"""

import PIL.Image
import numpy as np
import dlib
from PIL import ImageFile

# Cho phép tải ảnh bị cắt ngắn
ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    import face_recognition_models
except Exception:
    print("[ERROR] Không thể import thư viện face_recognition_models. Vui lòng kiểm tra lại cài đặt.")
    quit()

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Tải model dự đoán pose 68 điểm cho landmarks khuôn mặt
predictor_68_point_model = face_recognition_models.pose_predictor_model_location()
pose_predictor_68_point = dlib.shape_predictor(predictor_68_point_model)

# Tải model dự đoán pose 5 điểm (đơn giản hơn)
predictor_5_point_model = face_recognition_models.pose_predictor_five_point_model_location()
pose_predictor_5_point = dlib.shape_predictor(predictor_5_point_model)

# Tải model phát hiện khuôn mặt sử dụng CNN
cnn_face_detector_model = face_recognition_models.cnn_face_detector_model_location()
cnn_face_detector = dlib.cnn_face_detection_model_v1(cnn_face_detector_model)

# Tải model nhận diện khuôn mặt (face encoding)
face_recognition_model_location = face_recognition_models.face_recognition_model_location()
face_encoder = dlib.face_recognition_model_v1(face_recognition_model_location)

def _rect_to_css(rect):
    """
    Chuyển đổi dlib 'rect' thành định dạng CSS (top, right, bottom, left)
    :param rect: the dlib 'rect' object
    :return: tuple (top, right, bottom, left)
    """
    return rect.top(), rect.right(), rect.bottom(), rect.left()

def _css_to_rect(css):
    """
    Chuyển đổi định dạng CSS (top, right, bottom, left) thành dlib 'rect'
    :param css: tuple (top, right, bottom, left)
    :return: dlib 'rect' object
    """
    return dlib.rectangle(css[3], css[0], css[1], css[2])

def _trim_css_to_bounds(css, image_shape):
    """
    Cắt định dạng CSS (top, right, bottom, left) để nằm trong kích thước ảnh
    :param css: tuple (top, right, bottom, left)
    :param image_shape: shape của ảnh (height, width, channels)
    :return: tuple (top, right, bottom, left) đã được cắt
    """
    return (
        max(css[0], 0),
        min(css[1], image_shape[1]),
        min(css[2], image_shape[0]),
        max(css[3], 0),
    )

def face_distance(face_encodings, face_to_compare):
    """
    Tính khoảng cách Euclidean giữa một tập hợp các face encodings và một face encoding để so sánh.
    :param face_encodings: danh sách các face encodings để so sánh
    :param face_to_compare: face encoding để so sánh
    :return: danh sách khoảng cách Euclidean
    """
    if len(face_encodings) == 0:
        return np.empty((0))

    return np.linalg.norm(face_encodings - face_to_compare, axis = 1)

def load_image_file(file, mode = "RGB"):
    """
    Tải một file ảnh vào numpy array.
    :param file: đường dẫn đến file ảnh hoặc file-like object
    :param mode: chế độ màu để mở ảnh (mặc định là "RGB")
    :return: ảnh dưới dạng numpy array
    """
    im = PIL.Image.open(file)
    if mode:
        im = im.convert(mode)
    return np.array(im)

def _raw_face_locations(image, number_of_times_to_upsample = 1, model = "cnn"):
    """
    Phát hiện vị trí khuôn mặt trong ảnh sử dụng dlib.
    :param img: ảnh đầu vào (numpy array)
    :param number_of_times_to_upsample: số lần upsample ảnh để phát hiện khuôn mặt nhỏ hơn
    :param model: mô hình phát hiện khuôn mặt ("hog" hoặc "cnn")
    :return: danh sách các dlib 'rect' objects
    """
    if model == "cnn":
        return cnn_face_detector(image, number_of_times_to_upsample)
    else:
        detector = dlib.get_frontal_face_detector()
        return detector(image, number_of_times_to_upsample)

def face_locations(image, number_of_times_to_upsample = 1, model = "hog"):
    """
    Phát hiện vị trí khuôn mặt trong ảnh và trả về định dạng CSS (top, right, bottom, left).
    :param img: ảnh đầu vào (numpy array)
    :param number_of_times_to_upsample: số lần upsample ảnh để phát hiện khuôn mặt nhỏ hơn
    :param model: mô hình phát hiện khuôn mặt ("hog" hoặc "cnn")
    :return: danh sách các bounding boxes dưới dạng (top, right, bottom, left)
    """
    if model == "cnn":
        return [
            _trim_css_to_bounds(_rect_to_css(face.rect), image.shape) 
            for face in cnn_face_detector(image, number_of_times_to_upsample)
        ]
    else:
        return [
            _trim_css_to_bounds(_rect_to_css(face, image.shape)) 
            for face in _raw_face_locations(image, number_of_times_to_upsample, model)
        ]

def _raw_face_location_batched(image, number_of_times_to_upsample = 1, batch_size = 128):
    """
    Trả về một mảng 2 chiều các đối tượng 'rect' của dlib đại diện cho khuôn mặt người trong ảnh, sử dụng bộ phát hiện khuôn mặt CNN.

    :param images: Một danh sách các ảnh (mỗi ảnh là một mảng numpy)
    :param number_of_times_to_upsample: Số lần phóng to ảnh để tìm khuôn mặt. Giá trị càng cao thì càng dễ phát hiện khuôn mặt nhỏ.
    :return: Một danh sách các đối tượng 'rect' của dlib đại diện cho vị trí khuôn mặt được tìm thấy
    """
    return cnn_face_detector(image, number_of_times_to_upsample)

def batch_face_locations(images, number_of_times_to_upsample = 1, batch_size = 128):
    """
    Trả về 1 mảng 2D các bounding boxes của khuôn mặt trong mỗi ảnh, sử dụng bộ phát hiện khuôn mặt CNN.
    Ở đây, mỗi bounding box được biểu diễn dưới dạng (top, right, bottom, left).
    :param imgs: danh sách ảnh đầu vào (mỗi ảnh là numpy array)
    :param number_of_times_to_upsample: số lần upsample ảnh để phát hiện khuôn mặt nhỏ hơn
    :param batch_size: kích thước batch để xử lý ảnh
    :return: danh sách các bounding boxes dưới dạng (top, right, bottom, left) cho mỗi ảnh
    """
    def convert_cnn_detection_to_css(detections):
        return [
            _trim_css_to_bounds(_rect_to_css(face.rect), images[0].shape)
            for face in detections
        ]

    raw_detections_batched = _raw_face_location_batched(images, number_of_times_to_upsample, batch_size)
    return [list(map(convert_cnn_detection_to_css, raw_detections_batched))]

def _raw_face_landmarks(face_image, face_locations = None, model = "large"):
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
    Cho vị trí các đặc điểm khuôn mặt (landmarks) như mắt, mũi, miệng.
    :param face_image: ảnh đầu vào (numpy array)
    :param face_locations: Tùy chọn cung cấp danh sách các vị trí khuôn mặt để kiểm tra.
    :param model: Tùy chọn - mô hình nào để sử dụng. "large" (mặc định) hoặc "small" chỉ trả về 5 điểm nhưng nhanh hơn.
    :return: Danh sách các dict chứa vị trí các đặc điểm khuôn mặt (mắt, mũi, v.v.)
    """
    landmarks = _raw_face_landmarks(face_image, face_locations, model)
    landmarks_as_tuples = [[(p.x, p.y) for p in landmark.parts()] for landmark in landmarks]

    # For a definition of each point index, see https://cdn-images-1.medium.com/max/1600/1*AbEg31EgkbXSQehuNJBlWg.png
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

def face_encodings(face_image, known_face_locations = None, num_jitters = 1, model = "small"):
    """
    Trích xuất face encodings (128-dimension face descriptors) từ ảnh.
    :param face_image: ảnh đầu vào (numpy array)
    :param known_face_locations: Danh sách các vị trí khuôn mặt để trích xuất encodings. Nếu không cung cấp, sẽ tự động phát hiện.
    :param num_jitters: Số lần jittering để tăng độ chính xác (mặc định là 1). Giá trị cao hơn sẽ chậm hơn.
    :param model: Tùy chọn - mô hình landmarks nào để sử dụng. "small" (mặc định) hoặc "large".
    :return: Danh sách các face encodings (mỗi encoding là một numpy array 128 chiều)
    """
    raw_landmarks = _raw_face_landmarks(face_image, known_face_locations, model)

    return [
        np.array(face_encoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters))
        for raw_landmark_set in raw_landmarks
    ]

def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.7):
    """
    So sánh một face encoding với một tập hợp các face encodings đã biết để xem có khớp không.
    :param known_face_encodings: danh sách các face encodings đã biết
    :param face_encoding_to_check: face encoding để so sánh
    :param tolerance: ngưỡng khoảng cách để coi là khớp (mặc định là 0.6). Giá trị thấp hơn sẽ chặt chẽ hơn.
    :return: danh sách boolean cho biết mỗi face encoding đã biết có khớp với face encoding để kiểm tra hay không
    """
    distances = face_distance(known_face_encodings, face_encoding_to_check)
    return list(distances <= tolerance)