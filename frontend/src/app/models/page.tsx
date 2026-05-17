import Link from "next/link";

const MODELS = [
  {
    name: "VGG16 Advanced",
    script: "train_advanced.py",
    accuracy: "~65%",
    size: "56.77 MB",
    speed: "Moderate",
    status: "Recommended",
    useCase: "Best balance for a demo and course evaluation.",
    tone: "from-violet-500 to-blue-500",
  },
  {
    name: "ResNet50",
    script: "train_resnet50.py",
    accuracy: "~63%",
    size: "102 MB",
    speed: "Slower",
    status: "Alternative",
    useCase: "Useful if you want to show a deeper architecture.",
    tone: "from-rose-500 to-orange-500",
  },
  {
    name: "EfficientNetB0",
    script: "train_efficientnet.py",
    accuracy: "~61%",
    size: "24 MB",
    speed: "Fast",
    status: "Mobile-ready",
    useCase: "Good story for lightweight deployment.",
    tone: "from-emerald-500 to-teal-500",
  },
  {
    name: "Baseline VGG16",
    script: "train_transfer.py",
    accuracy: "~58%",
    size: "56.77 MB",
    speed: "Fast",
    status: "Baseline",
    useCase: "Handy as a comparison point against the improved model.",
    tone: "from-slate-500 to-slate-700",
  },
];

const HIGHLIGHTS = [
  "The advanced VGG16 variant is the best default for the current version 1 demo.",
  "EfficientNet is the strongest option if you want to talk about compact deployment.",
  "ResNet50 is valuable as a deeper comparison model, even if it is heavier.",
];

export default function ModelsPage() {
  return (
    <main className="min-h-screen bg-linear-to-br from-slate-50 via-white to-blue-50 text-slate-800">
      <nav className="sticky top-0 z-50 border-b border-slate-200 bg-white/75 backdrop-blur-xl">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-4 sm:px-6">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.3em] text-violet-600">Model comparison</p>
            <h1 className="text-lg font-semibold text-slate-900">Emotion Recognition Models</h1>
          </div>
          <div className="flex items-center gap-4">
            <Link href="/" className="text-sm font-medium text-slate-500 transition hover:text-violet-600">
              Dashboard
            </Link>
            <Link href="/evaluation" className="text-sm font-medium text-slate-500 transition hover:text-violet-600">
              Evaluation
            </Link>
            <Link href="/models" className="text-sm font-medium text-violet-600 border-b-2 border-violet-500 pb-0.5">
              Models
            </Link>
            <Link href="/webcam" className="text-sm font-medium text-slate-500 transition hover:text-violet-600">
              Webcam
            </Link>
          </div>
        </div>
      </nav>

      <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:py-12">
        <section className="rounded-4xl border border-white bg-white/85 p-8 shadow-[0_20px_80px_rgba(15,23,42,0.08)]">
          <p className="text-xs font-semibold uppercase tracking-[0.3em] text-violet-600">Professional feature</p>
          <h2 className="mt-4 text-4xl font-black tracking-tight text-slate-950">
            Compare the trained models before choosing the demo version.
          </h2>
          <p className="mt-4 max-w-3xl text-base leading-7 text-slate-600">
            This page is built from the project’s training notes and lets your team explain why one model is used for the
            current release and how the alternatives compare on size, speed, and expected accuracy.
          </p>

          <div className="mt-8 grid gap-4 md:grid-cols-3">
            {HIGHLIGHTS.map((item) => (
              <div key={item} className="rounded-3xl border border-slate-100 bg-slate-50 p-5 text-sm leading-6 text-slate-600">
                {item}
              </div>
            ))}
          </div>
        </section>

        <section className="mt-8 grid gap-6 lg:grid-cols-2">
          {MODELS.map((model, index) => (
            <article key={model.name} className="rounded-[28px] border border-white bg-white/90 p-6 shadow-[0_20px_80px_rgba(15,23,42,0.08)]">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <p className="text-xs font-semibold uppercase tracking-[0.25em] text-slate-400">{model.script}</p>
                  <h3 className="mt-2 text-2xl font-black text-slate-900">
                    {index + 1}. {model.name}
                  </h3>
                </div>
                <span className={`rounded-full bg-linear-to-r ${model.tone} px-3 py-1 text-xs font-semibold text-white`}>
                  {model.status}
                </span>
              </div>

              <p className="mt-4 text-sm leading-6 text-slate-600">{model.useCase}</p>

              <div className="mt-6 grid gap-3 sm:grid-cols-3">
                {[
                  { label: "Accuracy", value: model.accuracy },
                  { label: "Size", value: model.size },
                  { label: "Speed", value: model.speed },
                ].map((metric) => (
                  <div key={metric.label} className="rounded-2xl border border-slate-100 bg-slate-50 px-4 py-3">
                    <p className="text-[11px] uppercase tracking-[0.22em] text-slate-400">{metric.label}</p>
                    <p className="mt-2 text-base font-semibold text-slate-900">{metric.value}</p>
                  </div>
                ))}
              </div>
            </article>
          ))}
        </section>

        <section className="mt-8 rounded-[28px] border border-slate-100 bg-slate-950 p-8 text-slate-100 shadow-[0_20px_80px_rgba(15,23,42,0.12)]">
          <p className="text-xs font-semibold uppercase tracking-[0.3em] text-violet-300">Recommended talking point</p>
          <h3 className="mt-3 text-2xl font-black">Use VGG16 Advanced as the demo model, then show the comparison page to justify the choice.</h3>
          <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-300">
            This gives your evaluator a clear story: a stronger demo model for version 1, plus a transparent comparison of
            alternative architectures for future versions.
          </p>
        </section>
      </div>
    </main>
  );
}
