# YÊU CẦU HỌC VÀ CÔNG CỤ CHO DỰ ÁN HOMEFACEAI

## 1. Kiến thức cần học

### Python OOP

- Class, inheritance, interface, abstract class
- Quản lý package/module

### Xử lý ảnh & video

- OpenCV: Xử lý hình ảnh, video, camera
- PIL/Pillow: Xử lý ảnh cơ bản
- Numpy: Xử lý dữ liệu số

### Nhận diện khuôn mặt

- Face Recognition (thư viện dlib, face_recognition)
- Deep Learning cơ bản: CNN, embedding
- Cách huấn luyện và sử dụng model nhận diện

### Quản lý dữ liệu

- SQLite hoặc MySQL: Lưu trữ thông tin người dùng, sự kiện
- ORM: SQLAlchemy hoặc peewee

### Thông báo di động

- Firebase Cloud Messaging (FCM): Gửi thông báo push
- Xử lý notification trên Android/iOS (rung, chuông, hình ảnh)
- REST API: Giao tiếp với dịch vụ bên ngoài

### Xây dựng giao diện

- Tkinter/PyQt: Giao diện desktop
- Flask/FastAPI: Giao diện web đơn giản

### Kiểm thử & bảo trì

- Unit test: pytest, unittest
- Logging: logging module
- Quản lý môi trường: venv, pip

## 2. Thư viện cần cài đặt

```bash
face_recognition
opencv-python
numpy
Pillow
requests
sqlalchemy
peewee
firebase-admin
flask
pytest
logging
```

## 3. Tools hỗ trợ

- VS Code hoặc PyCharm
- Postman (test API)
- Git (quản lý mã nguồn)
- DBeaver/DB Browser (quản lý database)
- Android Studio/Xcode (test notification trên mobile)

## 4. APIs cần học và sử dụng

- OpenCV API: VideoCapture, image processing
- face_recognition API: face_locations, face_encodings, compare_faces
- Firebase Cloud Messaging API: gửi push notification
- RESTful API: requests, Flask/FastAPI
- SQLite/MySQL API: CRUD, ORM

## 5. Tài liệu tham khảo

- <https://docs.opencv.org/>
- <https://github.com/ageitgey/face_recognition>
- <https://firebase.google.com/docs/cloud-messaging>
- <https://docs.python.org/3/tutorial/classes.html>
- <https://docs.sqlalchemy.org/>
- <https://flask.palletsprojects.com/>
- <https://docs.pytest.org/>

## 6. Định hướng mở rộng

- Thêm nhận diện nhiều người cùng lúc
- Thêm các loại thông báo (email, SMS)
- Tích hợp AI nâng cao (YOLO, DeepFace)
- Xây dựng dashboard web quản lý
- Tích hợp IoT (mở cửa, camera IP)

---
**Ghi chú:**

- Nên học từng phần, thực hành với ví dụ nhỏ trước khi tích hợp vào dự án lớn.
- Luôn kiểm thử từng module độc lập.
- Quản lý mã nguồn và tài liệu rõ ràng.
