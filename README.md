# HomeFaceAI - Face Recognition System

HomeFaceAI là một hệ thống nhận diện khuôn mặt hiện đại, kết hợp **Python backend** (nhận diện khuôn mặt) với **Flutter frontend** (giao diện di động).

## Cấu trúc dự án sau tái cấu trúc

```
HomeFaceAI/
├── backend/                    # Backend Python (Face Recognition + API)
│   ├── app.py                 # FastAPI server
│   ├── core/                  # Core face recognition modules
│   │   ├── __init__.py
│   │   ├── api.py            # Face recognition API wrapper
│   │   ├── face_detector.py  # FaceDetector class
│   │   └── face_recognition.py
│   ├── models/               # Database models
│   ├── services/             # Business logic services
│   ├── utils/                # Utility functions
│   ├── tests/                # Unit tests
│   ├── requirements.txt      # Python dependencies
│   └── README.md             # Backend documentation
│
├── frontend/                  # Frontend Flutter (tạo sau)
├── ui/                        # UI prototypes (from old project)
├── data/                      # Data folder
├── README.md                  # Project overview (This file)
└── .gitignore                # Git ignore rules
```

## Yêu cầu hệ thống

- **Python 3.8+**
- **Flutter SDK** (cho frontend)
- **macOS**: Xcode Command Line Tools
- **pip** (package manager Python)

## Cài đặt và chạy Backend

### 1. Activate virtual environment
```bash
source /path/to/face_env/bin/activate
```

### 2. Cài dependencies
```bash
pip install -r backend/requirements.txt
```

### 3. Chạy API server
```bash
cd backend
python app.py
```

Server sẽ chạy tại: **http://localhost:8000**

### 4. Chạy tests
```bash
cd backend
python -m unittest tests/test_face_detector.py -v
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

## Cải thiện từ cấu trúc cũ

### Những thay đổi chính

1. **Backend consolidation**: Tất cả code Python giờ ở `backend/`
2. **Tách biệt frontend/backend**: Dễ dàng phát triển song song
3. **Clean structure**: Xóa những file cũ không cần thiết ở root
4. **API server**: Thêm `backend/app.py` để expose API cho Flutter

### Các thư mục đã xóa ở root

- ❌ `core/` → ✅ `backend/core/`
- ❌ `tests/` → ✅ `backend/tests/`
- ❌ `app/` → ✅ Thêm `backend/app.py` thay thế
- ❌ `models/` → ✅ `backend/models/`
- ❌ `services/` → ✅ `backend/services/`
- ❌ `utils/` → ✅ `backend/utils/`

## Bước tiếp theo

1. ✅ Backend restructured
2. ⏳ Create Flutter frontend in `frontend/` directory
3. ⏳ Integrate frontend with backend API
4. ⏳ Deploy to production

## Troubleshooting

Nếu gặp vấn đề:

1. Kiểm tra `backend/requirements.txt` đã install đầy đủ không
2. Chạy `python -m unittest tests/test_face_detector.py -v` để test backend
3. Xem `SETUP.md` để hướng dẫn chi tiết

---

**Created**: October 16, 2025
**Last Updated**: October 16, 2025

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
