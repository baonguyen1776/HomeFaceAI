# Phát hiện khuôn mặt (copied to backend/core)
import os
import numpy as np
import face_recognition.api as face_recognition


class FaceDetector:
    """
    Class phát hiện khuôn mặt trong hình ảnh sử dụng thư viện face_recognition.
    """
    def __init__(self, model="hog", upsample=1):
        self.model = model
        self.upsample = upsample

    def detect_faces(self, image_path_or_array):
        if isinstance(image_path_or_array, str):
            if not os.path.isfile(image_path_or_array):
                raise ValueError(f"File không tồn tại: {image_path_or_array}")
            image = face_recognition.load_image_file(image_path_or_array)
        else:
            image = image_path_or_array

        face_locations = face_recognition.face_locations(
            image,
            number_of_times_to_upsample=self.upsample,
            model=self.model,
        )

        return face_locations

    def detect_faces_batch(self, images, batch_size=128):
        return face_recognition.batch_face_locations(
            images,
            number_of_times_to_upsample=self.upsample,
            batch_size=batch_size,
        )

    def get_face_bounding_boxes(self, image_path_or_array):
        face_locs = self.detect_faces(image_path_or_array)
        bounding_boxes = []
        for top, right, bottom, left in face_locs:
            bounding_boxes.append({
                'x': left,
                'y': top,
                'width': right - left,
                'height': bottom - top
            })
        return bounding_boxes
