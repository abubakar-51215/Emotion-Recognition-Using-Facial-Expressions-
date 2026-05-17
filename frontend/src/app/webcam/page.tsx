"use client";

import React, { useRef, useState, useEffect, useCallback } from "react";
import Link from "next/link";
import {
  predictFromBlob,
  predictMultiFaceFromBlob,
  createPredictionSocket,
  PredictionResult,
  MultiFaceResult,
} from "../../lib/api";

/* ── Emotion visuals ─────────────────────────────────────────────────────── */
const COLORS: Record<string, string> = {
  Angry: "#ef4444",
  Disgust: "#a855f7",
  Fear: "#f97316",
  Happy: "#22c55e",
  Neutral: "#3b82f6",
  Sad: "#6366f1",
  Surprise: "#eab308",
};

const SOFT: Record<string, string> = {
  Angry: "#fee2e2",
  Disgust: "#f3e8ff",
  Fear: "#ffedd5",
  Happy: "#dcfce7",
  Neutral: "#dbeafe",
  Sad: "#e0e7ff",
  Surprise: "#fef9c3",
};

const EMOJI: Record<string, string> = {
  Angry: "😠",
  Disgust: "🤢",
  Fear: "😨",
  Happy: "😊",
  Neutral: "😐",
  Sad: "😢",
  Surprise: "😲",
};

type Mode = "rest" | "websocket";
type FaceMode = "single" | "multi";

/* ── Generate a short alert beep using Web Audio API ─────────────────────── */
function playAlertBeep() {
  try {
    const ctx = new AudioContext();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = "square";
    osc.frequency.value = 660;
    gain.gain.value = 0.18;
    osc.connect(gain).connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + 0.25);
    // second beep
    const osc2 = ctx.createOscillator();
    osc2.type = "square";
    osc2.frequency.value = 880;
    osc2.connect(gain);
    osc2.start(ctx.currentTime + 0.3);
    osc2.stop(ctx.currentTime + 0.55);
    setTimeout(() => ctx.close(), 1000);
  } catch {
    /* Web Audio not available */
  }
}

/* ======================================================================== */
export default function WebcamPage() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const captureCanvasRef = useRef<HTMLCanvasElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const [streaming, setStreaming] = useState(false);
  const [mode, setMode] = useState<Mode>("rest");
  const [faceMode, setFaceMode] = useState<FaceMode>("single");
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [multiFaceResults, setMultiFaceResults] = useState<PredictionResult[]>([]);
  const [fps, setFps] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [alertEnabled, setAlertEnabled] = useState(true);
  const [userName, setUserName] = useState("guest");

  const fpsCounter = useRef({ count: 0, last: Date.now() });
  const lastAlertRef = useRef(0); // throttle alerts

  useEffect(() => {
    const storedUser = window.localStorage.getItem("emotion-recognition-user");
    if (storedUser) {
      setUserName(storedUser);
    }
  }, []);

  useEffect(() => {
    window.localStorage.setItem("emotion-recognition-user", userName || "guest");
  }, [userName]);

  /* ── Start camera ──────────────────────────────────────────────────── */
  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: 640, height: 480 },
        audio: false,
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setError(null);
    } catch {
      setError("Camera access denied. Please allow camera permissions.");
    }
  }, []);

  const stopCamera = useCallback(() => {
    const stream = videoRef.current?.srcObject as MediaStream | null;
    stream?.getTracks().forEach((t) => t.stop());
    if (videoRef.current) videoRef.current.srcObject = null;
  }, []);

  /* ── Draw bounding box on overlay canvas ───────────────────────────── */
  const drawBBox = useCallback((res: PredictionResult) => {
    const canvas = overlayCanvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    const parent = canvas.parentElement;
    if (!parent) return;
    canvas.width = parent.clientWidth;
    canvas.height = parent.clientHeight;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (res.bbox && res.face_detected) {
      const { x, y, w, h } = res.bbox;
      // Mirror x because video is scaleX(-1)
      const bx = canvas.width - (x + w) * canvas.width;
      const by = y * canvas.height;
      const bw = w * canvas.width;
      const bh = h * canvas.height;

      const color = COLORS[res.emotion] ?? "#3b82f6";

      // Box
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.shadowColor = color;
      ctx.shadowBlur = 8;
      ctx.strokeRect(bx, by, bw, bh);
      ctx.shadowBlur = 0;

      // Corner accents
      const accent = 14;
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.moveTo(bx, by + accent);
      ctx.lineTo(bx, by);
      ctx.lineTo(bx + accent, by);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(bx + bw - accent, by);
      ctx.lineTo(bx + bw, by);
      ctx.lineTo(bx + bw, by + accent);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(bx, by + bh - accent);
      ctx.lineTo(bx, by + bh);
      ctx.lineTo(bx + accent, by + bh);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(bx + bw - accent, by + bh);
      ctx.lineTo(bx + bw, by + bh);
      ctx.lineTo(bx + bw, by + bh - accent);
      ctx.stroke();

      // Label
      const label = `${EMOJI[res.emotion] ?? ""} ${res.emotion} ${(res.confidence * 100).toFixed(0)}%`;
      ctx.font = "bold 15px Arial";
      const tm = ctx.measureText(label);
      const pad = 6;
      const lx = bx;
      const ly = by - 8;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.roundRect(lx - pad, ly - 18 - pad, tm.width + pad * 2, 18 + pad * 2, 6);
      ctx.fill();
      ctx.fillStyle = "#fff";
      ctx.fillText(label, lx, ly);
    }
  }, []);

  /* ── Draw multiple bounding boxes for multi-face mode ──────────────── */
  const drawMultiFaces = useCallback((faces: PredictionResult[]) => {
    const canvas = overlayCanvasRef.current;
    if (!canvas) return;

    const parent = canvas.parentElement;
    if (!parent) return;
    canvas.width = parent.clientWidth;
    canvas.height = parent.clientHeight;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    faces.forEach((res, idx) => {
      if (!res.bbox || !res.face_detected) return;

      const { x, y, w, h } = res.bbox;
      const bx = canvas.width - (x + w) * canvas.width;
      const by = y * canvas.height;
      const bw = w * canvas.width;
      const bh = h * canvas.height;

      const color = COLORS[res.emotion] ?? "#3b82f6";

      // Box
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.5;
      ctx.shadowColor = color;
      ctx.shadowBlur = 6;
      ctx.strokeRect(bx, by, bw, bh);
      ctx.shadowBlur = 0;

      // Corner accents
      const accent = 10;
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(bx, by + accent);
      ctx.lineTo(bx, by);
      ctx.lineTo(bx + accent, by);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(bx + bw - accent, by);
      ctx.lineTo(bx + bw, by);
      ctx.lineTo(bx + bw, by + accent);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(bx, by + bh - accent);
      ctx.lineTo(bx, by + bh);
      ctx.lineTo(bx + accent, by + bh);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(bx + bw - accent, by + bh);
      ctx.lineTo(bx + bw, by + bh);
      ctx.lineTo(bx + bw, by + bh - accent);
      ctx.stroke();

      // Label with face number
      const label = `Face ${idx + 1}`;
      ctx.font = "bold 13px Arial";
      const tm = ctx.measureText(label);
      const pad = 4;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.roundRect(bx - pad, by - 20 - pad, tm.width + pad * 2, 16 + pad * 2, 4);
      ctx.fill();
      ctx.fillStyle = "#fff";
      ctx.fillText(label, bx, by - 6);
    });
  }, []);

  /* ── Handle new result ─────────────────────────────────────────────── */
  const handleResult = useCallback(
    (res: PredictionResult) => {
      setResult(res);
      drawBBox(res);
      tickFps();

      // Angry alert (throttled to once every 3 seconds)
      if (alertEnabled && res.emotion === "Angry" && res.confidence > 0.5) {
        const now = Date.now();
        if (now - lastAlertRef.current > 3000) {
          lastAlertRef.current = now;
          playAlertBeep();
        }
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [alertEnabled, drawBBox]
  );

  /* ── Handle multi-face results ────────────────────────────────────── */
  const handleMultiFaceResult = useCallback(
    (result: MultiFaceResult) => {
      setMultiFaceResults(result.faces);
      drawMultiFaces(result.faces);
      tickFps();

      // Angry alert on any face
      if (alertEnabled) {
        const angryFace = result.faces.find(
          (f) => f.emotion === "Angry" && f.confidence > 0.5
        );
        if (angryFace) {
          const now = Date.now();
          if (now - lastAlertRef.current > 3000) {
            lastAlertRef.current = now;
            playAlertBeep();
          }
        }
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [alertEnabled, drawMultiFaces]
  );

  /* ── Capture a JPEG frame from the video ───────────────────────────── */
  const captureFrame = useCallback((): { blob: Promise<Blob | null>; dataUrl: string } | null => {
    const video = videoRef.current;
    const canvas = captureCanvasRef.current;
    if (!video || !canvas || video.readyState < 2) return null;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d")!;
    ctx.drawImage(video, 0, 0);

    const dataUrl = canvas.toDataURL("image/jpeg", 0.7);
    const blob = new Promise<Blob | null>((res) =>
      canvas.toBlob((b) => res(b), "image/jpeg", 0.7)
    );
    return { blob, dataUrl };
  }, []);

  /* ── FPS tracker ───────────────────────────────────────────────────── */
  const tickFps = useCallback(() => {
    fpsCounter.current.count++;
    const now = Date.now();
    if (now - fpsCounter.current.last >= 1000) {
      setFps(fpsCounter.current.count);
      fpsCounter.current = { count: 0, last: now };
    }
  }, []);

  /* ── REST polling mode (1 frame/sec) ───────────────────────────────── */
  const startRest = useCallback(() => {
    intervalRef.current = setInterval(async () => {
      const frame = captureFrame();
      if (!frame) return;
      const blob = await frame.blob;
      if (!blob) return;
      try {
        if (faceMode === "multi") {
          const res = await predictMultiFaceFromBlob(blob, userName);
          handleMultiFaceResult(res);
        } else {
          const res = await predictFromBlob(blob, userName);
          handleResult(res);
        }
      } catch {
        /* skip frame */
      }
    }, 1000);
  }, [captureFrame, handleResult, handleMultiFaceResult, faceMode, userName]);

  const stopRest = useCallback(() => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    intervalRef.current = null;
  }, []);

  /* ── WebSocket streaming mode ──────────────────────────────────────── */
  const startWs = useCallback(() => {
    const ws = createPredictionSocket(userName);
    wsRef.current = ws;

    ws.onopen = () => {
      intervalRef.current = setInterval(() => {
        const frame = captureFrame();
        if (!frame) return;
        ws.send(frame.dataUrl);
      }, 333);
    };

    ws.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        if (data.emotion) {
          handleResult(data as PredictionResult);
        }
      } catch {
        /* ignore */
      }
    };

    ws.onerror = () => setError("WebSocket error — is the backend running?");
    ws.onclose = () => { stopRest(); };
  }, [captureFrame, handleResult, stopRest, userName]);

  const stopWs = useCallback(() => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    intervalRef.current = null;
    wsRef.current?.close();
    wsRef.current = null;
  }, []);

  /* ── Toggle streaming ──────────────────────────────────────────────── */
  const toggle = async () => {
    if (streaming) {
      mode === "rest" ? stopRest() : stopWs();
      stopCamera();
      setStreaming(false);
      setResult(null);
      setMultiFaceResults([]);
      setFps(0);
      // clear overlay
      const ctx = overlayCanvasRef.current?.getContext("2d");
      if (ctx) ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    } else {
      await startCamera();
      setStreaming(true);
      setTimeout(() => {
        mode === "rest" ? startRest() : startWs();
      }, 600);
    }
  };

  /* cleanup on unmount */
  useEffect(() => {
    return () => {
      stopRest();
      stopWs();
      stopCamera();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /* ==================================================================== */
  return (
    <main className="min-h-screen bg-linear-to-br from-slate-50 via-white to-blue-50 text-slate-800">
      {/* ── Navbar ────────────────────────────────────────────────────── */}
      <nav className="sticky top-0 z-50 backdrop-blur-md bg-white/70 border-b border-slate-200">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-linear-to-br from-violet-500 to-blue-500 flex items-center justify-center text-white text-lg font-bold shadow-md">
              E
            </div>
            <div>
              <h1 className="text-lg font-bold text-slate-800 leading-tight">
                Emotion Recognition
              </h1>
              <p className="text-xs text-slate-400">Real-Time Webcam Detection</p>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <Link
              href="/"
              className="text-sm font-medium text-slate-500 hover:text-violet-600 transition"
            >
              Dashboard
            </Link>
            <Link
              href="/models"
              className="text-sm font-medium text-slate-500 hover:text-violet-600 transition"
            >
              Models
            </Link>
            <Link
              href="/evaluation"
              className="text-sm font-medium text-slate-500 hover:text-violet-600 transition"
            >
              Evaluation
            </Link>
            <Link
              href="/webcam"
              className="text-sm font-medium text-violet-600 border-b-2 border-violet-500 pb-0.5"
            >
              Webcam
            </Link>
            <span className="text-xs px-3 py-1 rounded-full bg-emerald-50 text-emerald-600 font-medium border border-emerald-200">
              ● Online
            </span>
          </div>
        </div>
      </nav>

      <div className="max-w-6xl mx-auto px-6 py-8 space-y-8">
        {/* ── Title ──────────────────────────────────────────────────── */}
        <div className="text-center">
          <h2 className="text-3xl font-extrabold bg-linear-to-r from-violet-600 to-blue-500 bg-clip-text text-transparent">
            Live Webcam Emotion Detection
          </h2>
          <p className="text-slate-500 mt-2">
            Choose a mode and start your camera — face detection draws a bounding box automatically.
          </p>
          <p className="mt-3 text-sm font-medium text-slate-500">
            Active account: <span className="text-violet-600">{userName || "guest"}</span>
          </p>
        </div>

        {/* ── Controls row ───────────────────────────────────────────── */}
        <div className="flex flex-wrap justify-center items-center gap-4">
          {(["rest", "websocket"] as Mode[]).map((m) => (
            <button
              key={m}
              disabled={streaming}
              onClick={() => setMode(m)}
              className={`px-6 py-2.5 rounded-full text-sm font-semibold transition-all duration-200 border
                ${mode === m
                  ? "bg-linear-to-r from-violet-600 to-blue-500 text-white border-transparent shadow-lg shadow-violet-200"
                  : "bg-white text-slate-600 border-slate-200 hover:border-violet-300"
                }
                disabled:opacity-50 disabled:cursor-not-allowed`}
            >
              {m === "rest" ? "🔄 REST Polling (1 fps)" : "⚡ WebSocket (3 fps)"}
            </button>
          ))}

          {/* face mode toggle */}
          {(["single", "multi"] as FaceMode[]).map((fm) => (
            <button
              key={fm}
              disabled={streaming}
              onClick={() => setFaceMode(fm)}
              className={`px-6 py-2.5 rounded-full text-sm font-semibold transition-all duration-200 border
                ${faceMode === fm
                  ? "bg-linear-to-r from-emerald-600 to-teal-500 text-white border-transparent shadow-lg shadow-emerald-200"
                  : "bg-white text-slate-600 border-slate-200 hover:border-emerald-300"
                }
                disabled:opacity-50 disabled:cursor-not-allowed`}
            >
              {fm === "single" ? "👤 Single Face" : "👥 Multi-Face"}
            </button>
          ))}

          {/* angry alert toggle */}
          <button
            onClick={() => setAlertEnabled((v) => !v)}
            className={`px-5 py-2.5 rounded-full text-sm font-semibold border transition-all duration-200
              ${alertEnabled
                ? "bg-red-50 text-red-600 border-red-200"
                : "bg-white text-slate-400 border-slate-200"
              }`}
          >
            {alertEnabled ? "🔔 Angry Alert ON" : "🔕 Angry Alert OFF"}
          </button>
        </div>

        {/* ── Video + Overlay ────────────────────────────────────────── */}
        <div className="flex flex-col items-center gap-6">
          <div className="relative w-full max-w-2xl aspect-video rounded-2xl overflow-hidden bg-slate-900 shadow-xl border border-slate-200">
            <video
              ref={videoRef}
              className="w-full h-full object-cover"
              style={{ transform: "scaleX(-1)" }}
              muted
              playsInline
            />
            {/* hidden canvas for frame capture */}
            <canvas ref={captureCanvasRef} className="hidden" />
            {/* overlay canvas for bounding box */}
            <canvas
              ref={overlayCanvasRef}
              className="absolute inset-0 w-full h-full pointer-events-none"
              style={{ zIndex: 10 }}
            />

            {/* Emotion overlay badges on video */}
            {streaming && (faceMode === "single" ? result : multiFaceResults.length > 0) && (
              <div className="absolute inset-0 pointer-events-none flex flex-col justify-between p-4" style={{ zIndex: 20 }}>
                {/* top badges */}
                {faceMode === "single" && result ? (
                  <div className="flex items-center gap-2">
                    <div
                      className="px-4 py-2 rounded-xl font-bold text-lg backdrop-blur-sm shadow-lg"
                      style={{
                        backgroundColor: COLORS[result.emotion] + "dd",
                        color: "#fff",
                      }}
                    >
                      {EMOJI[result.emotion]} {result.emotion}
                    </div>
                    <div className="px-3 py-2 rounded-xl backdrop-blur-sm bg-black/40 text-white text-sm font-semibold">
                      {(result.confidence * 100).toFixed(1)}%
                    </div>
                    {result.face_detected && (
                      <div className="px-3 py-2 rounded-xl backdrop-blur-sm bg-emerald-500/70 text-white text-xs font-semibold">
                        Face Detected
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="flex items-center gap-2 max-w-full overflow-x-auto pb-2">
                    <span className="px-3 py-2 rounded-xl backdrop-blur-sm bg-black/40 text-white text-sm font-semibold whitespace-nowrap">
                      {multiFaceResults.length} Faces Detected
                    </span>
                  </div>
                )}

                {/* bottom bar */}
                <div className="flex items-center justify-between">
                  <span className="px-3 py-1.5 rounded-lg backdrop-blur-sm bg-black/40 text-white text-xs font-medium">
                    {mode === "rest" ? "REST" : "WebSocket"} · {fps} fps · {faceMode === "multi" ? "Multi" : "Single"}
                  </span>
                  <span className="px-3 py-1.5 rounded-lg backdrop-blur-sm bg-red-500/80 text-white text-xs font-medium animate-pulse">
                    ● LIVE
                  </span>
                </div>
              </div>
            )}

            {/* placeholder when camera off */}
            {!streaming && (
              <div className="absolute inset-0 flex flex-col items-center justify-center bg-linear-to-br from-slate-100 to-slate-200 text-slate-400 gap-3">
                <svg className="w-16 h-16" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 002.25-2.25v-9A2.25 2.25 0 0013.5 5.25h-9A2.25 2.25 0 002.25 7.5v9A2.25 2.25 0 004.5 18.75z" />
                </svg>
                <p className="text-sm font-medium">Camera is off</p>
              </div>
            )}
          </div>

          {/* start / stop button */}
          <button
            onClick={toggle}
            className={`px-10 py-3.5 rounded-full font-semibold text-white shadow-lg transition-all duration-300 hover:scale-[1.03]
              ${streaming
                ? "bg-linear-to-r from-red-500 to-rose-500 shadow-red-200 hover:shadow-red-300"
                : "bg-linear-to-r from-violet-600 to-blue-500 shadow-violet-200 hover:shadow-violet-300"
              }`}
          >
            {streaming ? "⏹ Stop Camera" : "📷 Start Camera"}
          </button>

          {error && (
            <p className="text-red-500 text-sm bg-red-50 px-4 py-2 rounded-lg border border-red-200">
              {error}
            </p>
          )}
        </div>

        {/* ── Live Result Card ───────────────────────────────────────── */}
        {result && faceMode === "single" && (
          <div className="flex justify-center">
            <div
              className="w-full max-w-xl rounded-2xl p-5 shadow-lg border transition-all duration-300"
              style={{
                backgroundColor: SOFT[result.emotion] ?? "#f8fafc",
                borderColor: COLORS[result.emotion] + "40",
              }}
            >
              <div className="flex items-center gap-5">
                <div
                  className="w-16 h-16 rounded-xl flex items-center justify-center text-4xl shadow-inner"
                  style={{ backgroundColor: COLORS[result.emotion] + "15" }}
                >
                  {EMOJI[result.emotion]}
                </div>
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <p className="text-xs font-medium text-slate-500 uppercase tracking-wider">
                      Current Emotion
                    </p>
                    {result.emotion === "Angry" && alertEnabled && (
                      <span className="text-xs px-2 py-0.5 rounded-full bg-red-100 text-red-600 font-semibold animate-pulse">
                        ⚠ ALERT
                      </span>
                    )}
                  </div>
                  <p className="text-2xl font-extrabold" style={{ color: COLORS[result.emotion] }}>
                    {result.emotion}
                  </p>
                  <div className="mt-2 flex items-center gap-3">
                    <div className="flex-1 h-2.5 rounded-full bg-white/80 overflow-hidden shadow-inner">
                      <div
                        className="h-full rounded-full transition-all duration-300"
                        style={{
                          width: `${result.confidence * 100}%`,
                          backgroundColor: COLORS[result.emotion],
                        }}
                      />
                    </div>
                    <span className="text-sm font-bold" style={{ color: COLORS[result.emotion] }}>
                      {(result.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ── Multi-Face Results Grid ────────────────────────────────── */}
        {multiFaceResults.length > 0 && faceMode === "multi" && (
          <div className="flex justify-center">
            <div className="w-full max-w-4xl">
              <div className="text-center mb-4">
                <p className="text-sm font-semibold text-slate-600">
                  👥 Detected {multiFaceResults.length} Face{multiFaceResults.length !== 1 ? "s" : ""}
                </p>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {multiFaceResults.map((face, idx) => (
                  <div
                    key={idx}
                    className="rounded-2xl p-4 shadow-lg border transition-all duration-300"
                    style={{
                      backgroundColor: SOFT[face.emotion] ?? "#f8fafc",
                      borderColor: COLORS[face.emotion] + "40",
                    }}
                  >
                    <div className="flex items-center gap-3">
                      <div
                        className="w-12 h-12 rounded-lg flex items-center justify-center text-3xl shadow-inner"
                        style={{ backgroundColor: COLORS[face.emotion] + "15" }}
                      >
                        {EMOJI[face.emotion]}
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="text-xs font-medium text-slate-500 uppercase tracking-wider">
                          Face {idx + 1}
                        </p>
                        <p
                          className="text-lg font-extrabold truncate"
                          style={{ color: COLORS[face.emotion] }}
                        >
                          {face.emotion}
                        </p>
                        <p className="text-sm font-semibold text-slate-600">
                          {(face.confidence * 100).toFixed(1)}%
                        </p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* ── Info cards ─────────────────────────────────────────────── */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 max-w-3xl mx-auto">
          <div className="bg-white rounded-2xl p-5 shadow-sm border border-slate-100">
            <h3 className="font-bold text-slate-700 mb-2">🔄 REST Polling</h3>
            <p className="text-sm text-slate-500 leading-relaxed">
              1 frame/sec via HTTP POST. Simple and reliable.
            </p>
          </div>
          <div className="bg-white rounded-2xl p-5 shadow-sm border border-slate-100">
            <h3 className="font-bold text-slate-700 mb-2">⚡ WebSocket</h3>
            <p className="text-sm text-slate-500 leading-relaxed">
              ~3 fps streaming for near real-time detection.
            </p>
          </div>
          <div className="bg-white rounded-2xl p-5 shadow-sm border border-slate-100">
            <h3 className="font-bold text-slate-700 mb-2">📦 Face Detection</h3>
            <p className="text-sm text-slate-500 leading-relaxed">
              OpenCV Haarcascade detects faces with bounding box overlay.
            </p>
          </div>
        </div>
      </div>

      {/* ── Footer ───────────────────────────────────────────────────── */}
      <footer className="text-center py-6 text-xs text-slate-400 border-t border-slate-200 mt-8">
        Emotion Recognition System &middot; Built with TensorFlow, FastAPI &amp; Next.js
      </footer>
    </main>
  );
}
