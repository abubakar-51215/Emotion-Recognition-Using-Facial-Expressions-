import Link from "next/link";
import evaluation from "../../data/evaluation-summary.json";

const classes = evaluation.classes as string[];
const confusionMatrix = evaluation.confusion_matrix as number[][];
const perClass = evaluation.per_class as Array<{
  emotion: string;
  precision: number;
  recall: number;
  f1: number;
  support: number;
}>;
const misclassifications = evaluation.top_misclassifications as Array<{
  true: string;
  predicted: string;
  count: number;
}>;

function percent(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

function maxMatrixValue(matrix: number[][]) {
  return Math.max(...matrix.flat(), 1);
}

export default function EvaluationPage() {
  const maxValue = maxMatrixValue(confusionMatrix);
  const totalSamples = perClass.reduce((sum, item) => sum + item.support, 0);

  return (
    <main className="min-h-screen bg-linear-to-br from-slate-50 via-white to-blue-50 text-slate-800">
      <nav className="sticky top-0 z-50 border-b border-slate-200 bg-white/75 backdrop-blur-xl">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-4 sm:px-6">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.3em] text-violet-600">Evaluation</p>
            <h1 className="text-lg font-semibold text-slate-900">Model Diagnostics</h1>
          </div>
          <div className="flex items-center gap-4">
            <Link href="/" className="text-sm font-medium text-slate-500 transition hover:text-violet-600">
              Dashboard
            </Link>
            <Link href="/models" className="text-sm font-medium text-slate-500 transition hover:text-violet-600">
              Models
            </Link>
            <Link href="/evaluation" className="text-sm font-medium text-violet-600 border-b-2 border-violet-500 pb-0.5">
              Evaluation
            </Link>
            <Link href="/webcam" className="text-sm font-medium text-slate-500 transition hover:text-violet-600">
              Webcam
            </Link>
          </div>
        </div>
      </nav>

      <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:py-12 space-y-8">
        <section className="rounded-4xl border border-white bg-white/90 p-8 shadow-[0_20px_80px_rgba(15,23,42,0.08)]">
          <p className="text-xs font-semibold uppercase tracking-[0.3em] text-violet-600">Real evaluation data</p>
          <h2 className="mt-4 text-4xl font-black tracking-tight text-slate-950">
            Confusion matrix, class metrics, and misclassification review.
          </h2>
          <p className="mt-4 max-w-3xl text-base leading-7 text-slate-600">
            These metrics were computed from the trained model against the test split, so you can present concrete evidence
            of performance rather than a theoretical slide.
          </p>

          <div className="mt-8 grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            {[
              { label: "Accuracy", value: percent(evaluation.accuracy), accent: "text-violet-700" },
              { label: "Macro Precision", value: percent(evaluation.macro_precision), accent: "text-blue-700" },
              { label: "Macro Recall", value: percent(evaluation.macro_recall), accent: "text-emerald-700" },
              { label: "Macro F1", value: percent(evaluation.macro_f1), accent: "text-amber-700" },
            ].map((card) => (
              <div key={card.label} className="rounded-3xl border border-slate-100 bg-slate-50 p-5">
                <p className="text-xs uppercase tracking-[0.22em] text-slate-400">{card.label}</p>
                <p className={`mt-3 text-3xl font-black ${card.accent}`}>{card.value}</p>
              </div>
            ))}
          </div>
        </section>

        <section className="grid gap-6 xl:grid-cols-[1.1fr_0.9fr]">
          <div className="rounded-4xl border border-white bg-white/90 p-6 shadow-[0_20px_80px_rgba(15,23,42,0.08)]">
            <div className="flex items-center justify-between gap-4">
              <div>
                <h3 className="text-2xl font-black text-slate-900">Confusion Matrix</h3>
                <p className="mt-1 text-sm text-slate-500">Rows are true labels, columns are predictions.</p>
              </div>
              <span className="rounded-full bg-slate-100 px-3 py-1 text-xs font-semibold text-slate-600">
                {totalSamples.toLocaleString()} samples
              </span>
            </div>

            <div className="mt-6 overflow-x-auto">
              <table className="min-w-full border-separate border-spacing-2">
                <thead>
                  <tr>
                    <th className="w-24"></th>
                    {classes.map((label) => (
                      <th key={label} className="rounded-2xl bg-slate-50 px-3 py-2 text-xs font-semibold text-slate-500">
                        {label}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {classes.map((rowLabel, rowIndex) => (
                    <tr key={rowLabel}>
                      <th className="rounded-2xl bg-slate-50 px-3 py-2 text-left text-xs font-semibold text-slate-500">
                        {rowLabel}
                      </th>
                      {confusionMatrix[rowIndex].map((cell, colIndex) => {
                        const intensity = cell / maxValue;
                        const isDiagonal = rowIndex === colIndex;
                        return (
                          <td key={`${rowLabel}-${classes[colIndex]}`}>
                            <div
                              className={`flex h-14 w-full items-center justify-center rounded-2xl px-2 text-sm font-semibold ${
                                isDiagonal ? "text-slate-950" : "text-slate-700"
                              }`}
                              style={{
                                backgroundColor: isDiagonal
                                  ? `rgba(37, 99, 235, ${0.15 + intensity * 0.55})`
                                  : `rgba(148, 163, 184, ${0.12 + intensity * 0.35})`,
                              }}
                            >
                              {cell}
                            </div>
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="space-y-6">
            <section className="rounded-4xl border border-white bg-white/90 p-6 shadow-[0_20px_80px_rgba(15,23,42,0.08)]">
              <h3 className="text-2xl font-black text-slate-900">Top Misclassifications</h3>
              <p className="mt-1 text-sm text-slate-500">Most common mistakes are useful talking points during evaluation.</p>
              <div className="mt-5 space-y-3">
                {misclassifications.map((item, index) => (
                  <div key={`${item.true}-${item.predicted}`} className="flex items-center justify-between rounded-2xl border border-slate-100 bg-slate-50 px-4 py-3">
                    <div>
                      <p className="text-sm font-semibold text-slate-900">
                        {index + 1}. {item.true} → {item.predicted}
                      </p>
                      <p className="text-xs text-slate-500">Frequently confused classes</p>
                    </div>
                    <span className="rounded-full bg-violet-100 px-3 py-1 text-xs font-semibold text-violet-700">
                      {item.count}
                    </span>
                  </div>
                ))}
              </div>
            </section>

            <section className="rounded-4xl border border-white bg-slate-950 p-6 text-slate-100 shadow-[0_20px_80px_rgba(15,23,42,0.12)]">
              <h3 className="text-2xl font-black">Interpretation</h3>
              <p className="mt-3 text-sm leading-6 text-slate-300">
                The strongest classes can be highlighted as stable predictions, while the confused pairs help explain
                where the model still needs improvement.
              </p>
              <ul className="mt-5 space-y-3 text-sm text-slate-300">
                <li>• Use the diagonal cells to show where the model is correct.</li>
                <li>• Mention the top misclassification pairs as honest limitations.</li>
                <li>• Compare per-class precision and recall when answering evaluator questions.</li>
              </ul>
            </section>
          </div>
        </section>

        <section className="rounded-4xl border border-white bg-white/90 p-6 shadow-[0_20px_80px_rgba(15,23,42,0.08)]">
          <div className="flex items-center justify-between gap-4">
            <div>
              <h3 className="text-2xl font-black text-slate-900">Per-Class Metrics</h3>
              <p className="mt-1 text-sm text-slate-500">Precision, recall, F1, and support for each emotion label.</p>
            </div>
          </div>

          <div className="mt-6 grid gap-4 xl:grid-cols-2">
            {perClass.map((item) => (
              <div key={item.emotion} className="rounded-3xl border border-slate-100 bg-slate-50 p-5">
                <div className="flex items-center justify-between gap-4">
                  <h4 className="text-lg font-bold text-slate-900">{item.emotion}</h4>
                  <span className="rounded-full bg-white px-3 py-1 text-xs font-semibold text-slate-500">
                    Support: {item.support}
                  </span>
                </div>
                <div className="mt-4 grid gap-3 sm:grid-cols-3">
                  {[
                    { label: "Precision", value: item.precision },
                    { label: "Recall", value: item.recall },
                    { label: "F1", value: item.f1 },
                  ].map((metric) => (
                    <div key={metric.label} className="rounded-2xl border border-slate-100 bg-white px-4 py-3">
                      <p className="text-[11px] uppercase tracking-[0.22em] text-slate-400">{metric.label}</p>
                      <p className="mt-2 text-xl font-black text-slate-900">{percent(metric.value)}</p>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </section>
      </div>
    </main>
  );
}
