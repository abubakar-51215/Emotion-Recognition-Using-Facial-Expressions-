# Emotion Recognition Using Facial Expressions

This project is a full-stack emotion recognition system that detects facial expressions from images or webcam input and predicts the most likely emotion class.

It includes:

- A Python backend for inference and prediction APIs
- A Next.js frontend for the dashboard, webcam flow, model comparison, and evaluation views
- Trained emotion classification models stored in the `models/` folder
- A facial expression dataset organized by emotion class under `dataset/`

The project is designed to support experimentation with different training scripts and model variants, including advanced, EfficientNet, ResNet50, and transfer-learning approaches.

## Run locally

- Backend: `python -m uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8000`
- Frontend: `cd frontend && npm run dev`

## Project Structure

- `backend/` - API server and prediction logic
- `frontend/` - user interface and dashboards
- `models/` - saved model files
- `dataset/` - training and test image sets
- `train*.py` - training and evaluation scripts
