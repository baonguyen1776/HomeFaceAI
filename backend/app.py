"""
Minimal FastAPI app exposing a face detection endpoint for the Flutter frontend.
"""
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import cv2
from backend.core.face_detector import FaceDetector

app = FastAPI(title="HomeFaceAI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize detector
detector = FaceDetector()

@app.post("/detect-faces")
async def detect_faces(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # Convert BGR to RGB for face_recognition
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = detector.get_face_bounding_boxes(image)
    return {"faces": faces}

if __name__ == "__main__":
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)
