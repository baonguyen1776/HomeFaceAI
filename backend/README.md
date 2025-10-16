# HomeFaceAI Backend Structure

## Cấu trúc Backend

```
backend/
├── app.py                    # FastAPI server chính
├── core/                     # Core nhận diện khuôn mặt
│   ├── __init__.py
│   ├── api.py               # API wrapper cho face_recognition
│   ├── face_detector.py     # Class FaceDetector
│   └── face_recognition.py  # Face recognition utilities
├── models/                  # Database models
│   ├── __init__.py
│   ├── person.py            # Person model
│   └── detection_event.py   # Detection event model
├── services/                # Business logic services
│   ├── __init__.py
│   ├── camera_service.py    # Camera handling
│   ├── database_service.py  # Database operations
│   └── notification/        # Notification services
│       ├── __init__.py
│       ├── notification_base.py
│       ├── email_notifier.py
│       ├── sms_notifier.py
│       └── push_notifier.py
├── utils/                   # Utility functions
│   ├── __init__.py
│   ├── image_utils.py       # Image processing utilities
│   └── logger.py            # Logging utilities
├── tests/                   # Unit tests
│   ├── __init__.py
│   └── test_face_detector.py
├── requirements.txt         # Python dependencies
└── config.py                # Configuration file (optional)
```

## Chạy Backend

### Cài đặt dependencies
```bash
pip install -r backend/requirements.txt
```

### Chạy API server
```bash
cd backend
python app.py
```

### Chạy tests
```bash
cd backend
python -m unittest tests/test_face_detector.py
```

## API Endpoints

- **POST** `/detect-faces` - Phát hiện khuôn mặt từ ảnh upload
  - Request: multipart form-data with `file` parameter
  - Response: JSON với danh sách khuôn mặt phát hiện được
