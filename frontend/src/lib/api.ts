import axios from "axios";

const API_BASE = "http://127.0.0.1:8000";
const WS_BASE = "ws://127.0.0.1:8000";

const API = axios.create({
  baseURL: API_BASE,
});

export interface BBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface PredictionResult {
  emotion: string;
  confidence: number;
  face_detected?: boolean;
  bbox?: BBox;
  is_uncertain?: boolean;
  raw_emotion?: string;
  threshold?: number;
}

export interface MultiFaceResult {
  faces: PredictionResult[];
  face_count: number;
}

export interface HistoryRecord {
  id: number;
  user_name?: string;
  emotion: string;
  confidence: number;
  timestamp: string;
}

export async function predictEmotion(file: File, userName = "guest"): Promise<PredictionResult> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("user_name", userName);
  const { data } = await API.post<PredictionResult>("/predict", formData);
  return data;
}

/** Send a Blob (webcam frame) via the REST /predict endpoint. */
export async function predictFromBlob(blob: Blob, userName = "guest"): Promise<PredictionResult> {
  const formData = new FormData();
  formData.append("file", blob, "frame.jpg");
  formData.append("user_name", userName);
  const { data } = await API.post<PredictionResult>("/predict", formData);
  return data;
}

/** Predict emotions for ALL faces in an image (multi-face mode). */
export async function predictMultiFaceFromBlob(blob: Blob, userName = "guest"): Promise<MultiFaceResult> {
  const formData = new FormData();
  formData.append("file", blob, "frame.jpg");
  formData.append("user_name", userName);
  const { data } = await API.post<MultiFaceResult>("/predict/multi-face", formData);
  return data;
}

export async function fetchHistory(limit = 50, userName?: string): Promise<HistoryRecord[]> {
  const { data } = await API.get<HistoryRecord[]>('/history', {
    params: { limit, user_name: userName },
  });
  return data;
}

/** Create a WebSocket connection to the real-time prediction endpoint. */
export function createPredictionSocket(userName = "guest"): WebSocket {
  return new WebSocket(`${WS_BASE}/ws/predict?user_name=${encodeURIComponent(userName)}`);
}
