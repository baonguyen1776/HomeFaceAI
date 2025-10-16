# HomeFaceAI - Face Recognition System

HomeFaceAI là một hệ thống nhận diện khuôn mặt hiện đại, kết hợp **Python backend** (nhận diện khuôn mặt) với **Flutter frontend** (giao diện di động).

## Cấu trúc dự án

```
HomeFaceAI/
├── backend/                 # Backend Python (FastAPI + Face Recognition)
│   ├── core/               # Core nhận diện khuôn mặt
│   ├── models/             # Database models
│   ├── services/           # Business logic
│   ├── utils/              # Utilities
│   ├── tests/              # Unit tests
│   ├── app.py              # FastAPI server
│   ├── requirements.txt    # Dependencies
│   └── README.md           # Backend documentation
│
├── frontend/               # Frontend Flutter (tạo sau)
│   ├── android/
│   ├── ios/
│   ├── lib/
│   └── pubspec.yaml
│
├── data/                   # Data folder (for images, models, etc.)
├── ui/                     # UI prototypes/designs (optional)
├── requirements.md         # Project requirements
└── README.md               # This file
```

## Yêu cầu hệ thống

- **Python 3.8+**
- **Flutter SDK** (cho frontend)
- **pip** (package manager Python)

## Cài đặt Backend

### 1. Clone repository
```bash
git clone https://github.com/baonguyen1776/HomeFaceAI.git
cd HomeFaceAI
```

### 2. Tạo virtual environment (khuyến nghị)
```bash
python3 -m venv face_env
source face_env/bin/activate  # macOS/Linux
# hoặc
face_env\Scripts\activate  # Windows
```

### 3. Cài dependencies
```bash
pip install -r backend/requirements.txt
```

### 4. Chạy API server
```bash
cd backend
python app.py
```

Server sẽ chạy tại `http://localhost:8000`

### 5. Chạy tests
```bash
cd backend
python -m unittest tests/test_face_detector.py
```

## API Documentation

API server chạy tại `http://localhost:8000` với các endpoints:

### POST /detect-faces
Phát hiện khuôn mặt trong ảnh upload

**Request:**
```
POST /detect-faces HTTP/1.1
Host: localhost:8000
Content-Type: multipart/form-data

file: <image-file>
```

**Response:**
```json
{
  "faces": [
    {
      "x": 100,
      "y": 150,
      "width": 80,
      "height": 100
    }
  ]
}
```

## Flutter Frontend (Coming Soon)

Frontend sẽ gọi backend API để thực hiện nhận diện khuôn mặt real-time từ camera điện thoại.

## Công nghệ sử dụng

### Backend
- **FastAPI** - Web framework
- **face_recognition** - Nhận diện khuôn mặt
- **dlib** - Deep learning library
- **OpenCV** - Image processing
- **NumPy** - Numerical computing

### Frontend
- **Flutter** - Cross-platform mobile framework
- **Dart** - Programming language

## Tác giả

- Nguyễn Phương Gia Bảo (baonguyen1776)

## License

MIT License
