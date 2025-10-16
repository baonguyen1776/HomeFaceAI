# HomeFaceAI - Project Setup Guide

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
├── README.md                  # Project overview
└── .gitignore                # Git ignore rules
```

## Chạy Backend

### 1. Cài đặt dependencies
```bash
# Activate virtual environment
source /path/to/face_env/bin/activate

# Cài dependencies
pip install -r backend/requirements.txt
```

### 2. Chạy API server
```bash
cd backend
python app.py
```

Server sẽ chạy tại: **http://localhost:8000**

### 3. Chạy tests
```bash
cd backend
python -m unittest tests/test_face_detector.py -v
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

### Các thư mục giữ lại ở root

- ✅ `ui/` - UI prototypes/designs
- ✅ `data/` - Data folder
- ✅ `.git/` - Git repository
- ✅ `README.md` - Project overview
- ✅ `.gitignore` - Git ignore rules
- ✅ `requirements.md` - Project requirements

## Commit Git

Tất cả thay đổi đã được commit với message:

```
refactor: restructure project - move code to backend/ directory
```

## Bước tiếp theo

1. ✅ Backend restructured
2. ⏳ Create Flutter frontend
3. ⏳ Integrate frontend with backend API
4. ⏳ Deploy to production

## Support

Nếu gặp vấn đề:

1. Kiểm tra `backend/requirements.txt` đã install đầy đủ không
2. Chạy `python -m unittest tests/test_face_detector.py` để test backend
3. Xem `backend/README.md` để hiểu thêm API

---

**Created**: October 16, 2025
**Last Updated**: October 16, 2025
