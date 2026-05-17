"""
Emotion Recognition System — FastAPI Backend
Serves predictions from the trained CNN model.
"""

import io
import json
import base64
import os
import sqlite3
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect, Form
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf

# ─── Configuration ────────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(BASE_DIR, "models", "emotion_model.h5"))
DB_PATH = os.getenv("DB_PATH", os.path.join(BASE_DIR, "emotion.db"))
IMG_SIZE = (48, 48)
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.55"))

# Load Haarcascade classifier once
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# ─── App Setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Emotion Recognition API",
    description="Predicts facial emotion from an uploaded image.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Load Model Once at Startup ──────────────────────────────────────────────
model = None


@app.on_event("startup")
def startup():
    global model
    # Load model
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise RuntimeError(f"Could not load model: {e}")

    # Initialise database
    init_db()
    print(f"Database ready at {DB_PATH}")


# ─── Database ────────────────────────────────────────────────────────────────
def get_db():
    """Return a new connection (one per request)."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create the emotion_logs table if it doesn't exist."""
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS emotion_logs (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_name  TEXT    NOT NULL DEFAULT 'guest',
            emotion    TEXT    NOT NULL,
            confidence REAL    NOT NULL,
            timestamp  TEXT    NOT NULL
        )
    """)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(emotion_logs)").fetchall()}
    if "user_name" not in columns:
        conn.execute("ALTER TABLE emotion_logs ADD COLUMN user_name TEXT NOT NULL DEFAULT 'guest'")
    conn.commit()
    conn.close()


def save_prediction(emotion: str, confidence: float, user_name: str = "guest"):
    """Insert a prediction record into the database."""
    conn = get_db()
    conn.execute(
        "INSERT INTO emotion_logs (user_name, emotion, confidence, timestamp) VALUES (?, ?, ?, ?)",
        (user_name, emotion, confidence, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


# ─── Helper ──────────────────────────────────────────────────────────────────
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Convert raw bytes → model-ready (1, 48, 48, 3) RGB array.
    
    Accepts both grayscale and RGB images. Converts grayscale to RGB
    by repeating the channel 3 times (required by VGG16).
    """
    # Try to detect input format
    img = Image.open(io.BytesIO(image_bytes))
    
    # Convert to RGB (handles grayscale, RGBA, etc.)
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr.reshape(1, 48, 48, 3)  # VGG16 expects (1, 48, 48, 3)
    return arr


def detect_faces(image_bytes: bytes):
    """Detect ALL faces using Haarcascade and convert to RGB for VGG16.
    Returns list of (face_crop_rgb_array, bbox_normalised) tuples.
    bbox_normalised = {x, y, w, h} as fractions of image size.
    """
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return []
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ih, iw = gray.shape[:2]
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        return []
    
    results = []
    # Process all detected faces
    for x, y, w, h in faces:
        face_roi = img[y:y+h, x:x+w]  # Get color image region
        face_resized = cv2.resize(face_roi, IMG_SIZE)
        
        # Convert BGR to RGB for consistency
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        face_arr = face_rgb.astype(np.float32) / 255.0
        face_arr = face_arr.reshape(1, 48, 48, 3)  # VGG16 format: (1, 48, 48, 3)
        
        bbox = {"x": float(x / iw), "y": float(y / ih), "w": float(w / iw), "h": float(h / ih)}
        results.append((face_arr, bbox))
    
    return results


def detect_face(image_bytes: bytes):
    """Detect face using Haarcascade and convert to RGB for VGG16.
    Returns (face_crop_rgb_array, bbox_normalised) or (None, None).
    bbox_normalised = {x, y, w, h} as fractions of image size.
    
    This is the legacy single-face version. Use detect_faces for multi-face.
    """
    faces = detect_faces(image_bytes)
    if faces:
        # Return the largest face for backward compatibility
        return faces[0]
    return None, None


def run_prediction(image_bytes: bytes, multi_face: bool = False) -> dict:
    """Detect faces → predict emotion.
    
    Args:
        image_bytes: Raw image bytes
        multi_face: If True, return predictions for all detected faces
                   If False, return only the largest face (default for backward compatibility)
    
    Returns:
        Single prediction dict or list of prediction dicts depending on multi_face
    """
    if multi_face:
        #  Multi-face mode: detect and predict all faces
        face_results = detect_faces(image_bytes)
        if not face_results:
            # No faces, fall back to whole image
            processed = preprocess_image(image_bytes)
            predictions = model.predict(processed, verbose=0)[0]
            idx = int(np.argmax(predictions))
            confidence = round(float(predictions[idx]), 4)
            return {
                "faces": [
                    {
                        "emotion": EMOTION_LABELS[idx],
                        "confidence": confidence,
                        "face_detected": False,
                        "bbox": None,
                    }
                ]
            }
        
        # Predict for each face
        results = []
        for face_arr, bbox in face_results:
            predictions = model.predict(face_arr, verbose=0)[0]
            idx = int(np.argmax(predictions))
            confidence = round(float(predictions[idx]), 4)
            emotion = EMOTION_LABELS[idx]
            is_uncertain = confidence < CONFIDENCE_THRESHOLD
            if is_uncertain:
                emotion = "Uncertain"
            results.append({
                "emotion": emotion,
                "confidence": confidence,
                "is_uncertain": is_uncertain,
                "face_detected": True,
                "bbox": bbox,
            })
        
        return {"faces": results, "face_count": len(results)}
    
    else:
        # Single-face mode (backward compatible): detect largest face
        face_arr, bbox = detect_face(image_bytes)
        if face_arr is not None:
            processed = face_arr
        else:
            processed = preprocess_image(image_bytes)
            bbox = None

        predictions = model.predict(processed, verbose=0)[0]
        idx = int(np.argmax(predictions))
        confidence = round(float(predictions[idx]), 4)
        emotion = EMOTION_LABELS[idx]
        is_uncertain = confidence < CONFIDENCE_THRESHOLD
        if is_uncertain:
            emotion = "Uncertain"

        result = {
            "emotion": emotion,
            "confidence": confidence,
            "face_detected": bbox is not None,
            "is_uncertain": is_uncertain,
        }
        if bbox:
            result["bbox"] = bbox
        return result


# ─── Endpoints ───────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "running", "service": "Emotion Recognition API"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "database": DB_PATH,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...), user_name: str = Form("guest")):
    # Validate content type
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")

    try:
        image_bytes = await file.read()
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the image.")

    try:
        result = run_prediction(image_bytes, multi_face=False)
    except Exception:
        raise HTTPException(status_code=500, detail="Prediction failed.")

    # Save to database
    try:
        save_prediction(result["emotion"], result["confidence"], user_name=user_name)
    except Exception:
        pass  # logging failure should not block the response

    return result


@app.post("/predict/multi-face")
async def predict_multi_face(file: UploadFile = File(...), user_name: str = Form("guest")):
    """Detect and predict emotions for ALL faces in an image."""
    # Validate content type
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")

    try:
        image_bytes = await file.read()
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the image.")

    try:
        result = run_prediction(image_bytes, multi_face=True)
    except Exception:
        raise HTTPException(status_code=500, detail="Prediction failed.")

    # Save all predictions to database
    try:
        if "faces" in result:
            for face_pred in result["faces"]:
                save_prediction(face_pred["emotion"], face_pred["confidence"], user_name=user_name)
    except Exception:
        pass  # logging failure should not block the response

    return result


@app.get("/history")
def history(limit: int = 50, user_name: str | None = None):
    """Return the most recent prediction logs."""
    conn = get_db()
    if user_name:
        rows = conn.execute(
            "SELECT id, user_name, emotion, confidence, timestamp FROM emotion_logs "
            "WHERE user_name = ? ORDER BY id DESC LIMIT ?",
            (user_name, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, user_name, emotion, confidence, timestamp FROM emotion_logs "
            "ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    conn.close()
    return [{"id": r["id"], "user_name": r["user_name"], "emotion": r["emotion"],
             "confidence": r["confidence"], "timestamp": r["timestamp"]} for r in rows]


# ─── WebSocket for Real-Time Webcam ──────────────────────────────────────────
@app.websocket("/ws/predict")
async def ws_predict(websocket: WebSocket):
    """Accept base64 JPEG frames, return emotion + confidence as JSON."""
    await websocket.accept()
    user_name = websocket.query_params.get("user_name", "guest")
    try:
        while True:
            data = await websocket.receive_text()
            try:
                # strip optional data-url prefix
                if "," in data:
                    data = data.split(",", 1)[1]
                image_bytes = base64.b64decode(data)
                result = run_prediction(image_bytes)

                # save to db (non-blocking best-effort)
                try:
                    save_prediction(result["emotion"], result["confidence"], user_name=user_name)
                except Exception:
                    pass

                await websocket.send_text(json.dumps(result))
            except Exception as e:
                await websocket.send_text(
                    json.dumps({"error": str(e)})
                )
    except WebSocketDisconnect:
        pass
