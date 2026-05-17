"use client";

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { predictEmotion, fetchHistory, PredictionResult, HistoryRecord } from "../lib/api";

type DateWindow = "7" | "30" | "90" | "all";

const COLORS: Record<string, string> = {
  Angry: "#ef4444",
  Disgust: "#a855f7",
  Fear: "#f97316",
  Happy: "#22c55e",
  Neutral: "#3b82f6",
  Sad: "#6366f1",
  Surprise: "#eab308",
  Uncertain: "#64748b",
};

const SOFT_COLORS: Record<string, string> = {
  Angry: "#fee2e2",
  Disgust: "#f3e8ff",
  Fear: "#ffedd5",
  Happy: "#dcfce7",
  Neutral: "#dbeafe",
  Sad: "#e0e7ff",
  Surprise: "#fef9c3",
  Uncertain: "#e2e8f0",
};

const EMOJI: Record<string, string> = {
  Angry: "😠",
  Disgust: "🤢",
  Fear: "😨",
  Happy: "😊",
  Neutral: "😐",
  Sad: "😢",
  Surprise: "😲",
  Uncertain: "◌",
};

const EMOTION_ORDER = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise", "Uncertain"];

function formatConfidence(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

function getDateCutoff(window: DateWindow) {
  if (window === "all") return null;
  const days = Number(window);
  return Date.now() - days * 24 * 60 * 60 * 1000;
}

function buildCsv(history: HistoryRecord[]) {
  const rows = [
    ["Emotion Recognition Report"],
    ["Timestamp", "Emotion", "Confidence", "Status"],
    ...history.map((entry) => [
      entry.timestamp,
      entry.emotion,
      entry.confidence.toFixed(4),
      entry.emotion === "Uncertain" ? "Review" : "Normal",
    ]),
  ];

  return rows.map((row) => row.join(",")).join("\n");
}

function hexToRgb(hex: string) {
  const normalized = hex.replace("#", "");
  return {
    r: Number.parseInt(normalized.slice(0, 2), 16),
    g: Number.parseInt(normalized.slice(2, 4), 16),
    b: Number.parseInt(normalized.slice(4, 6), 16),
  };
}

export default function Dashboard() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [history, setHistory] = useState<HistoryRecord[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [darkMode, setDarkMode] = useState(false);
  const [dateWindow, setDateWindow] = useState<DateWindow>("30");
  const [emotionFilter, setEmotionFilter] = useState<string>("All");
  const [userName, setUserName] = useState("guest");
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const storedUser = window.localStorage.getItem("emotion-recognition-user");
    if (storedUser) {
      setUserName(storedUser);
    }
  }, []);

  useEffect(() => {
    window.localStorage.setItem("emotion-recognition-user", userName || "guest");
  }, [userName]);

  const loadHistory = useCallback(async (activeUser: string) => {
    try {
      const data = await fetchHistory(300, activeUser);
      setHistory(data);
    } catch {
      setHistory([]);
    }
  }, []);

  useEffect(() => {
    loadHistory(userName);
  }, [loadHistory, userName]);

  useEffect(() => {
    return () => {
      if (preview) {
        URL.revokeObjectURL(preview);
      }
    };
  }, [preview]);

  const filteredHistory = useMemo(() => {
    const cutoff = getDateCutoff(dateWindow);
    return history.filter((entry) => {
      if (cutoff !== null && new Date(entry.timestamp).getTime() < cutoff) {
        return false;
      }
      if (emotionFilter !== "All" && entry.emotion !== emotionFilter) {
        return false;
      }
      return true;
    });
  }, [history, dateWindow, emotionFilter]);

  const analytics = useMemo(() => {
    const emotionCounts: Record<string, number> = {};
    let confidenceSum = 0;
    let uncertainCount = 0;

    filteredHistory.forEach((entry) => {
      emotionCounts[entry.emotion] = (emotionCounts[entry.emotion] || 0) + 1;
      confidenceSum += entry.confidence;
      if (entry.emotion === "Uncertain") {
        uncertainCount += 1;
      }
    });

    const topEmotion =
      Object.entries(emotionCounts).sort((a, b) => b[1] - a[1])[0]?.[0] || "N/A";

    return {
      total: filteredHistory.length,
      averageConfidence: filteredHistory.length ? confidenceSum / filteredHistory.length : 0,
      topEmotion,
      uncertainCount,
      uncertainRate: filteredHistory.length ? uncertainCount / filteredHistory.length : 0,
      emotionCounts,
    };
  }, [filteredHistory]);

  const distributionData = useMemo(
    () =>
      EMOTION_ORDER.map((emotion) => ({
        emotion,
        count: analytics.emotionCounts[emotion] || 0,
      })),
    [analytics.emotionCounts]
  );

  const timelineData = useMemo(
    () =>
      filteredHistory
        .slice()
        .reverse()
        .slice(0, 30)
        .map((entry, index) => ({
          time: index + 1,
          confidence: Number(entry.confidence.toFixed(2)),
          emotion: entry.emotion,
        })),
    [filteredHistory]
  );

  const resultEmotion = result?.emotion ?? "Neutral";
  const resultIsUncertain = result?.emotion === "Uncertain" || result?.is_uncertain;

  const handleFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFile(e.target.files?.[0] ?? null);
    setResult(null);
    setError(null);
    const nextFile = e.target.files?.[0] ?? null;
    if (preview) {
      URL.revokeObjectURL(preview);
    }
    setPreview(nextFile ? URL.createObjectURL(nextFile) : null);
  };

  const selectFile = (nextFile: File | null) => {
    setFile(nextFile);
    setResult(null);
    setError(null);
    if (preview) {
      URL.revokeObjectURL(preview);
    }
    setPreview(nextFile ? URL.createObjectURL(nextFile) : null);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const droppedFile = e.dataTransfer.files?.[0] ?? null;
    if (droppedFile?.type.startsWith("image/")) {
      selectFile(droppedFile);
    }
  };

  const handlePredict = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const nextResult = await predictEmotion(file, userName);
      setResult(nextResult);
      await loadHistory(userName);
    } catch {
      setError("Prediction failed. Make sure the backend is running.");
    } finally {
      setLoading(false);
    }
  };

  const exportCSV = () => {
    if (filteredHistory.length === 0) {
      alert("No filtered history to export.");
      return;
    }

    const csv = buildCsv(filteredHistory);
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `emotion_report_${new Date().toISOString().split("T")[0]}.csv`;
    anchor.click();
    URL.revokeObjectURL(url);
  };

  const exportPDF = async () => {
    if (filteredHistory.length === 0) {
      alert("No filtered history to export.");
      return;
    }

    const { jsPDF } = await import("jspdf");
    const doc = new jsPDF({ unit: "pt", format: "a4" });
    const pageWidth = doc.internal.pageSize.getWidth();
    const pageHeight = doc.internal.pageSize.getHeight();
    const margin = 44;
    const accent = hexToRgb("#7c3aed");
    const primary = hexToRgb("#0f172a");
    const subtle = hexToRgb("#64748b");

    doc.setFillColor(accent.r, accent.g, accent.b);
    doc.rect(0, 0, pageWidth, 78, "F");
    doc.setTextColor(255, 255, 255);
    doc.setFont("helvetica", "bold");
    doc.setFontSize(20);
    doc.text("Emotion Intelligence Report", margin, 34);
    doc.setFontSize(10);
    doc.setFont("helvetica", "normal");
    doc.text("Branded export generated from the dashboard analytics view", margin, 52);

    doc.setTextColor(primary.r, primary.g, primary.b);
    doc.setFontSize(14);
    doc.setFont("helvetica", "bold");
    doc.text("Summary", margin, 112);

    const summaryLines = [
      `Generated: ${new Date().toLocaleString()}`,
      `Date filter: ${dateWindow === "all" ? "All time" : `Last ${dateWindow} days`}`,
      `Emotion filter: ${emotionFilter}`,
      `Total filtered records: ${filteredHistory.length}`,
      `Top emotion: ${analytics.topEmotion}`,
      `Average confidence: ${formatConfidence(analytics.averageConfidence)}`,
      `Uncertain cases: ${analytics.uncertainCount}`,
      result ? `Latest prediction: ${resultEmotion} (${formatConfidence(result.confidence)})` : "Latest prediction: None",
    ];

    doc.setFontSize(11);
    doc.setFont("helvetica", "normal");
    doc.setTextColor(subtle.r, subtle.g, subtle.b);
    doc.text(summaryLines, margin, 132, { maxWidth: pageWidth - margin * 2, lineHeightFactor: 1.45 });

    doc.setTextColor(primary.r, primary.g, primary.b);
    doc.setFont("helvetica", "bold");
    doc.setFontSize(14);
    doc.text("Recent predictions", margin, 260);

    const rows = filteredHistory.slice(0, 12);
    let cursorY = 282;
    doc.setFont("helvetica", "normal");
    doc.setFontSize(10);

    rows.forEach((entry, index) => {
      if (cursorY > pageHeight - 60) {
        doc.addPage();
        cursorY = 54;
        doc.setTextColor(primary.r, primary.g, primary.b);
        doc.setFont("helvetica", "bold");
        doc.setFontSize(14);
        doc.text("Recent predictions (continued)", margin, cursorY);
        cursorY += 22;
        doc.setFont("helvetica", "normal");
        doc.setFontSize(10);
      }

      const line = `${index + 1}. ${new Date(entry.timestamp).toLocaleString()} | ${entry.emotion} | ${formatConfidence(entry.confidence)}`;
      doc.setTextColor(subtle.r, subtle.g, subtle.b);
      doc.text(line, margin, cursorY, { maxWidth: pageWidth - margin * 2 });
      cursorY += 18;
    });

    doc.setTextColor(primary.r, primary.g, primary.b);
    doc.setFont("helvetica", "italic");
    doc.setFontSize(9);
    doc.text("Emotion Recognition System · FastAPI + Next.js · PDF summary", margin, pageHeight - 24);
    doc.save(`emotion_report_${new Date().toISOString().split("T")[0]}.pdf`);
  };

  return (
    <main
      className={`min-h-screen transition-colors duration-300 ${
        darkMode
          ? "bg-linear-to-br from-slate-950 via-slate-900 to-blue-950 text-slate-100"
          : "bg-linear-to-br from-slate-50 via-white to-blue-50 text-slate-800"
      }`}
    >
      <nav
        className={`sticky top-0 z-50 border-b backdrop-blur-xl transition-colors ${
          darkMode ? "border-slate-800 bg-slate-950/75" : "border-slate-200 bg-white/75"
        }`}
      >
        <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-4 sm:px-6">
          <div className="flex items-center gap-3">
            <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-linear-to-br from-violet-500 to-blue-500 font-bold text-white shadow-lg shadow-violet-200/60">
              E
            </div>
            <div>
              <h1 className={`text-lg font-semibold leading-tight ${darkMode ? "text-white" : "text-slate-900"}`}>
                Emotion Recognition
              </h1>
              <p className={`text-xs ${darkMode ? "text-slate-400" : "text-slate-500"}`}>
                AI dashboard for emotion analysis and review
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3 sm:gap-4">
            <Link href="/" className="text-sm font-medium text-violet-600">
              Dashboard
            </Link>
            <Link
              href="/models"
              className={`text-sm font-medium transition ${darkMode ? "text-slate-400 hover:text-violet-300" : "text-slate-500 hover:text-violet-600"}`}
            >
              Models
            </Link>
            <Link
              href="/evaluation"
              className={`text-sm font-medium transition ${darkMode ? "text-slate-400 hover:text-violet-300" : "text-slate-500 hover:text-violet-600"}`}
            >
              Evaluation
            </Link>
            <Link
              href="/webcam"
              className={`text-sm font-medium transition ${darkMode ? "text-slate-400 hover:text-violet-300" : "text-slate-500 hover:text-violet-600"}`}
            >
              Webcam
            </Link>
            <button
              onClick={() => setDarkMode((value) => !value)}
              className={`rounded-xl px-3 py-2 text-sm transition ${
                darkMode ? "bg-slate-800 text-slate-200 hover:bg-slate-700" : "bg-slate-100 text-slate-700 hover:bg-slate-200"
              }`}
            >
              {darkMode ? "☀️ Light" : "🌙 Dark"}
            </button>
          </div>
        </div>
      </nav>

      <div className="mx-auto max-w-7xl space-y-8 px-4 py-8 sm:px-6">
        <section className="grid gap-8 lg:grid-cols-[1.08fr_0.92fr] lg:items-start">
          <div
            className={`rounded-[36px] border p-10 shadow-[0_24px_100px_rgba(15,23,42,0.08)] ${
              darkMode ? "border-slate-800 bg-slate-900/75" : "border-white/80 bg-white/90"
            }`}
          >
            <div className="flex flex-wrap gap-2">
              <span className="rounded-full bg-violet-100 px-3 py-1 text-xs font-semibold text-violet-700">Date filters</span>
              <span className="rounded-full bg-blue-100 px-3 py-1 text-xs font-semibold text-blue-700">Confidence trends</span>
              <span className="rounded-full bg-amber-100 px-3 py-1 text-xs font-semibold text-amber-700">Uncertain state</span>
            </div>
            <h2 className={`mt-6 max-w-xl text-4xl font-black tracking-tight sm:text-5xl lg:text-6xl ${darkMode ? "text-white" : "text-slate-950"}`}>
              A cleaner, more professional emotion analytics dashboard.
            </h2>
            <p className={`mt-5 max-w-xl text-base leading-8 ${darkMode ? "text-slate-300" : "text-slate-600"}`}>
              Upload a face image, inspect the prediction result, and review the recent history with filters,
              trend charts, and a low-confidence review state.
            </p>
            <div className="mt-4 flex flex-wrap items-center gap-3">
              <span className={`rounded-full border px-3 py-1 text-xs font-semibold ${darkMode ? "border-emerald-800 bg-emerald-950/40 text-emerald-300" : "border-emerald-200 bg-emerald-50 text-emerald-600"}`}>
                Live system
              </span>
              <label className={`flex items-center gap-2 rounded-full border px-3 py-2 text-xs font-medium ${darkMode ? "border-slate-800 bg-slate-950/40 text-slate-300" : "border-slate-200 bg-white text-slate-600"}`}>
                <span>Account</span>
                <input
                  value={userName}
                  onChange={(e) => setUserName(e.target.value)}
                  className="w-28 bg-transparent outline-none"
                  placeholder="guest"
                />
              </label>
            </div>
            <div className="mt-8 grid gap-4 sm:grid-cols-3">
              {[
                { label: "Filtered results", value: analytics.total.toString() },
                { label: "Top emotion", value: analytics.topEmotion },
                { label: "Review rate", value: formatConfidence(analytics.uncertainRate) },
              ].map((item) => (
                <div
                  key={item.label}
                  className={`rounded-3xl border px-5 py-4 ${darkMode ? "border-slate-800 bg-slate-950/55" : "border-slate-100 bg-slate-50/90"}`}
                >
                  <p className={`text-xs uppercase tracking-[0.2em] ${darkMode ? "text-slate-500" : "text-slate-400"}`}>
                    {item.label}
                  </p>
                  <p className={`mt-2 text-xl font-semibold ${darkMode ? "text-slate-100" : "text-slate-900"}`}>{item.value}</p>
                </div>
              ))}
            </div>
          </div>

          <div
            className={`rounded-[36px] border p-7 shadow-[0_24px_100px_rgba(15,23,42,0.08)] ${
              darkMode ? "border-slate-800 bg-slate-900/75" : "border-white/80 bg-white/90"
            }`}
          >
            <div
              onClick={() => inputRef.current?.click()}
              onDragOver={(e) => {
                e.preventDefault();
                setDragOver(true);
              }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleDrop}
              className={`flex min-h-128 cursor-pointer flex-col items-center justify-center rounded-[30px] border-2 border-dashed px-5 py-8 text-center transition-all duration-300 ${
                dragOver
                  ? "border-violet-400 bg-violet-50/80"
                  : preview
                    ? darkMode
                      ? "border-slate-700 bg-slate-950/40"
                      : "border-slate-200 bg-slate-50"
                    : darkMode
                      ? "border-slate-700 bg-slate-950/20 hover:border-violet-400 hover:bg-slate-950/40"
                      : "border-slate-200 bg-white hover:border-violet-400 hover:bg-violet-50/50"
              }`}
            >
              {preview ? (
                <img src={preview} alt="preview" className="max-h-80 w-full rounded-3xl object-contain" />
              ) : (
                <>
                  <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-linear-to-br from-violet-100 to-blue-100 text-2xl shadow-inner">
                    📤
                  </div>
                  <p className={`mt-5 text-xl font-semibold ${darkMode ? "text-slate-100" : "text-slate-800"}`}>
                    Drop an image here
                  </p>
                  <p className={`mt-2 max-w-sm text-sm leading-6 ${darkMode ? "text-slate-400" : "text-slate-500"}`}>
                    PNG or JPG up to 5 MB. The model will flag weak predictions as uncertain.
                  </p>
                </>
              )}
            </div>

            <input ref={inputRef} type="file" accept="image/*" className="hidden" onChange={handleFile} />

            <button
              onClick={handlePredict}
              disabled={!file || loading}
              className="mt-6 w-full rounded-full bg-linear-to-r from-violet-600 to-blue-500 px-6 py-4 text-sm font-semibold text-white shadow-lg shadow-violet-200/60 transition hover:scale-[1.01] hover:shadow-xl disabled:cursor-not-allowed disabled:opacity-40"
            >
              {loading ? "Analyzing..." : "Predict emotion"}
            </button>
          </div>
        </section>

        {error && (
          <div className="rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-600">
            {error}
          </div>
        )}

        {result && (
          <section className="rounded-[36px] border p-7 shadow-[0_24px_100px_rgba(15,23,42,0.08)] sm:p-8" style={{ backgroundColor: SOFT_COLORS[resultEmotion] ?? "#f8fafc", borderColor: `${COLORS[resultEmotion] ?? "#94a3b8"}33` }}>
            <div className="flex flex-col gap-6 md:flex-row md:items-center">
              <div className="flex h-24 w-24 items-center justify-center rounded-[30px] text-5xl shadow-inner" style={{ backgroundColor: `${COLORS[resultEmotion] ?? "#94a3b8"}18` }}>
                {EMOJI[resultEmotion] ?? "◌"}
              </div>
              <div className="flex-1">
                <div className="flex flex-wrap items-center gap-2">
                  <p className="text-sm font-semibold uppercase tracking-[0.18em] text-slate-500">Latest prediction</p>
                  {resultIsUncertain && (
                    <span className="rounded-full bg-amber-100 px-3 py-1 text-xs font-semibold text-amber-700">
                      Low confidence, review recommended
                    </span>
                  )}
                </div>
                <h3 className="mt-2 text-4xl font-black sm:text-5xl" style={{ color: COLORS[resultEmotion] ?? "#64748b" }}>
                  {resultEmotion}
                </h3>
                <div className="mt-5 flex items-center gap-3">
                  <div className="h-3 flex-1 overflow-hidden rounded-full bg-white/70 shadow-inner">
                    <div className="h-full rounded-full transition-all duration-700" style={{ width: `${result.confidence * 100}%`, backgroundColor: COLORS[resultEmotion] ?? "#64748b" }} />
                  </div>
                  <span className="text-sm font-bold" style={{ color: COLORS[resultEmotion] ?? "#64748b" }}>
                    {formatConfidence(result.confidence)}
                  </span>
                </div>
                <p className="mt-3 text-sm text-slate-600">
                  {resultIsUncertain
                    ? "The backend tagged this result as uncertain because the model confidence is below the threshold."
                    : "This result is strong enough to be treated as a direct emotion label."}
                </p>
              </div>
            </div>
          </section>
        )}

        <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          {[
            { label: "Total records", value: filteredHistory.length, icon: "📊" },
            { label: "Top emotion", value: analytics.topEmotion, icon: "🏆" },
            { label: "Average confidence", value: filteredHistory.length ? formatConfidence(analytics.averageConfidence) : "—", icon: "🎯" },
            { label: "Uncertain cases", value: analytics.uncertainCount, icon: "⚠️" },
          ].map((card) => (
            <div key={card.label} className={`rounded-3xl border p-5 shadow-sm transition ${darkMode ? "border-slate-800 bg-slate-900/75" : "border-white/80 bg-white/90"}`}>
              <p className="text-2xl">{card.icon}</p>
              <p className={`mt-3 text-2xl font-black ${darkMode ? "text-white" : "text-slate-900"}`}>{card.value}</p>
              <p className={`mt-1 text-xs uppercase tracking-[0.18em] ${darkMode ? "text-slate-500" : "text-slate-400"}`}>{card.label}</p>
            </div>
          ))}
        </section>

        <section className={`rounded-[36px] border p-7 shadow-[0_24px_100px_rgba(15,23,42,0.08)] ${darkMode ? "border-slate-800 bg-slate-900/75" : "border-white/80 bg-white/90"}`}>
          <div className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
            <div>
              <h3 className={`text-2xl font-black ${darkMode ? "text-white" : "text-slate-900"}`}>Analytics</h3>
              <p className={`mt-1 text-sm ${darkMode ? "text-slate-400" : "text-slate-500"}`}>
                Filter the history by date range or emotion to inspect trends.
              </p>
            </div>
            <div className="grid gap-3 sm:grid-cols-2">
              <label className={`flex flex-col gap-2 rounded-2xl border px-4 py-3 text-sm ${darkMode ? "border-slate-800 bg-slate-950/40" : "border-slate-100 bg-slate-50"}`}>
                <span className={darkMode ? "text-slate-400" : "text-slate-500"}>Date range</span>
                <select value={dateWindow} onChange={(e) => setDateWindow(e.target.value as DateWindow)} className={`rounded-xl border px-3 py-2 text-sm outline-none ${darkMode ? "border-slate-700 bg-slate-900 text-slate-100" : "border-slate-200 bg-white text-slate-800"}`}>
                  <option value="7">Last 7 days</option>
                  <option value="30">Last 30 days</option>
                  <option value="90">Last 90 days</option>
                  <option value="all">All time</option>
                </select>
              </label>
              <label className={`flex flex-col gap-2 rounded-2xl border px-4 py-3 text-sm ${darkMode ? "border-slate-800 bg-slate-950/40" : "border-slate-100 bg-slate-50"}`}>
                <span className={darkMode ? "text-slate-400" : "text-slate-500"}>Emotion</span>
                <select value={emotionFilter} onChange={(e) => setEmotionFilter(e.target.value)} className={`rounded-xl border px-3 py-2 text-sm outline-none ${darkMode ? "border-slate-700 bg-slate-900 text-slate-100" : "border-slate-200 bg-white text-slate-800"}`}>
                  <option value="All">All emotions</option>
                  {EMOTION_ORDER.map((emotion) => (
                    <option key={emotion} value={emotion}>
                      {emotion}
                    </option>
                  ))}
                </select>
              </label>
            </div>
          </div>

          <div className="mt-6 grid gap-6 lg:grid-cols-2">
            <div className={`rounded-3xl border p-5 ${darkMode ? "border-slate-800 bg-slate-950/40" : "border-slate-100 bg-slate-50"}`}>
              <h4 className={`text-base font-bold ${darkMode ? "text-slate-100" : "text-slate-800"}`}>Emotion distribution</h4>
              <p className={`mt-1 text-xs ${darkMode ? "text-slate-400" : "text-slate-500"}`}>Counts from the currently filtered history.</p>
              {filteredHistory.length === 0 ? (
                <p className={`mt-6 text-sm ${darkMode ? "text-slate-400" : "text-slate-500"}`}>No data for the selected filters.</p>
              ) : (
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={distributionData}>
                    <CartesianGrid strokeDasharray="3 3" stroke={darkMode ? "#334155" : "#e2e8f0"} />
                    <XAxis dataKey="emotion" tick={{ fill: darkMode ? "#94a3b8" : "#64748b", fontSize: 11 }} />
                    <YAxis allowDecimals={false} tick={{ fill: darkMode ? "#94a3b8" : "#64748b", fontSize: 11 }} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: darkMode ? "#0f172a" : "#fff",
                        border: `1px solid ${darkMode ? "#334155" : "#e2e8f0"}`,
                        borderRadius: 12,
                      }}
                    />
                    <Bar dataKey="count" radius={[10, 10, 0, 0]}>
                      {distributionData.map((entry) => (
                        <Cell key={entry.emotion} fill={COLORS[entry.emotion] ?? "#94a3b8"} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              )}
            </div>

            <div className={`rounded-3xl border p-5 ${darkMode ? "border-slate-800 bg-slate-950/40" : "border-slate-100 bg-slate-50"}`}>
              <h4 className={`text-base font-bold ${darkMode ? "text-slate-100" : "text-slate-800"}`}>Confidence trend</h4>
              <p className={`mt-1 text-xs ${darkMode ? "text-slate-400" : "text-slate-500"}`}>Recent predictions in chronological order.</p>
              {timelineData.length === 0 ? (
                <p className={`mt-6 text-sm ${darkMode ? "text-slate-400" : "text-slate-500"}`}>No data for the selected filters.</p>
              ) : (
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={timelineData}>
                    <CartesianGrid strokeDasharray="3 3" stroke={darkMode ? "#334155" : "#e2e8f0"} />
                    <XAxis dataKey="time" tick={{ fill: darkMode ? "#94a3b8" : "#64748b", fontSize: 11 }} />
                    <YAxis domain={[0, 1]} tick={{ fill: darkMode ? "#94a3b8" : "#64748b", fontSize: 11 }} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: darkMode ? "#0f172a" : "#fff",
                        border: `1px solid ${darkMode ? "#334155" : "#e2e8f0"}`,
                        borderRadius: 12,
                      }}
                      formatter={(value) => (typeof value === "number" ? value.toFixed(2) : value)}
                    />
                    <Line type="monotone" dataKey="confidence" stroke="#8b5cf6" strokeWidth={3} dot={false} isAnimationActive={false} />
                  </LineChart>
                </ResponsiveContainer>
              )}
            </div>
          </div>
        </section>

        <section className={`rounded-[36px] border p-7 shadow-[0_24px_100px_rgba(15,23,42,0.08)] ${darkMode ? "border-slate-800 bg-slate-900/75" : "border-white/80 bg-white/90"}`}>
          <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <h3 className={`text-2xl font-black ${darkMode ? "text-white" : "text-slate-900"}`}>Prediction history</h3>
              <p className={`mt-1 text-sm ${darkMode ? "text-slate-400" : "text-slate-500"}`}>
                Showing {filteredHistory.length} of {history.length} records.
              </p>
            </div>
            <button
              onClick={exportPDF}
              disabled={filteredHistory.length === 0}
              className="rounded-xl border border-violet-200 bg-white px-4 py-2 text-sm font-semibold text-violet-700 transition hover:border-violet-300 hover:bg-violet-50 disabled:cursor-not-allowed disabled:opacity-40"
            >
              Export PDF
            </button>
            <button
              onClick={exportCSV}
              disabled={filteredHistory.length === 0}
              className="rounded-xl bg-linear-to-r from-violet-600 to-blue-500 px-4 py-2 text-sm font-semibold text-white shadow-lg shadow-violet-200/60 transition hover:shadow-xl disabled:cursor-not-allowed disabled:opacity-40"
            >
              Export CSV
            </button>
          </div>

          {filteredHistory.length === 0 ? (
            <p className={`mt-6 text-sm ${darkMode ? "text-slate-400" : "text-slate-500"}`}>No predictions found for the current filters.</p>
          ) : (
            <div className="mt-5 overflow-hidden rounded-2xl border border-slate-100">
              <div className={`max-h-75 overflow-y-auto ${darkMode ? "bg-slate-950/40" : "bg-white"}`}>
                <table className="w-full text-sm">
                  <thead className={`sticky top-0 ${darkMode ? "bg-slate-900" : "bg-white"}`}>
                    <tr className={`border-b text-left text-xs uppercase tracking-[0.2em] ${darkMode ? "border-slate-800 text-slate-500" : "border-slate-100 text-slate-400"}`}>
                      <th className="px-4 py-3">#</th>
                      <th className="px-4 py-3">Emotion</th>
                      <th className="px-4 py-3">Confidence</th>
                      <th className="px-4 py-3">Status</th>
                      <th className="px-4 py-3">Timestamp</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredHistory.map((row, index) => {
                      const isReview = row.emotion === "Uncertain";
                      return (
                        <tr key={row.id} className={`border-b transition ${darkMode ? "border-slate-800 hover:bg-slate-900/70" : "border-slate-100 hover:bg-slate-50"}`}>
                          <td className={`px-4 py-3 text-xs ${darkMode ? "text-slate-500" : "text-slate-400"}`}>{index + 1}</td>
                          <td className="px-4 py-3 font-semibold" style={{ color: COLORS[row.emotion] ?? "#64748b" }}>
                            <span className="mr-2">{EMOJI[row.emotion] ?? "◌"}</span>
                            {row.emotion}
                          </td>
                          <td className="px-4 py-3">
                            <span className="inline-flex rounded-full px-3 py-1 text-xs font-semibold" style={{ backgroundColor: `${SOFT_COLORS[row.emotion] ?? "#e2e8f0"}`, color: COLORS[row.emotion] ?? "#64748b" }}>
                              {formatConfidence(row.confidence)}
                            </span>
                          </td>
                          <td className="px-4 py-3">
                            <span className={`inline-flex rounded-full px-3 py-1 text-xs font-semibold ${isReview ? "bg-amber-100 text-amber-700" : darkMode ? "bg-emerald-950 text-emerald-300" : "bg-emerald-50 text-emerald-700"}`}>
                              {isReview ? "Review" : "Confirmed"}
                            </span>
                          </td>
                          <td className={`px-4 py-3 text-xs ${darkMode ? "text-slate-500" : "text-slate-400"}`}>
                            {new Date(row.timestamp).toLocaleString()}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </section>
      </div>

      <footer className={`border-t py-6 text-center text-xs ${darkMode ? "border-slate-800 text-slate-500" : "border-slate-200 text-slate-400"}`}>
        Emotion Recognition System · Built with TensorFlow, FastAPI &amp; Next.js
      </footer>
    </main>
  );
}
