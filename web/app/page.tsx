"use client";

import { useEffect, useRef, useState } from "react";

const API_URL =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") || "http://localhost:8000";

type Detection = { bbox: number[]; confidence: number; label: string };
type ModelResult = {
  model: string;
  latency_ms: number;
  detections: Detection[];
  annotated_image_b64?: string | null;
  error?: string | null;
};

const STATIC_MODELS = [
  "RT-DETR-L",
  "YOLO 11n",
  "ViT-Artery",
  "Faster-RCNN-MobileNet",
  "MobileDet-SSDLite",
  "NanoDet",
  "RF-DETR",
  "DenseNet",
  "ConvNeXt-tiny",
  "SqueezeNet",
  "Xception",
];

export default function Page() {
  const [models, setModels] = useState<string[]>(STATIC_MODELS);
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set([STATIC_MODELS[0]]));
  const [threshold, setThreshold] = useState(0.5);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [results, setResults] = useState<ModelResult[]>([]);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const r = await fetch(`${API_URL}/models`, {
          cache: "no-store",
          headers: { "ngrok-skip-browser-warning": "true" },
        });
        if (!r.ok) throw new Error(`status ${r.status}`);
        const data = await r.json();
        if (cancelled) return;
        if (Array.isArray(data.models) && data.models.length) setModels(data.models);
        setBackendOnline(true);
      } catch {
        if (!cancelled) setBackendOnline(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  function onFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0] || null;
    setFile(f);
    setPreview(f ? URL.createObjectURL(f) : null);
    setResults([]);
  }

  function toggle(name: string) {
    setSelected((prev) => {
      const next = new Set(prev);
      next.has(name) ? next.delete(name) : next.add(name);
      return next;
    });
  }

  async function run() {
    if (!file) {
      setError("Upload an image first.");
      return;
    }
    if (selected.size === 0) {
      setError("Select at least one model.");
      return;
    }
    setError(null);
    setRunning(true);
    setResults([]);
    try {
      const fd = new FormData();
      fd.append("image", file);
      fd.append("models", Array.from(selected).join(","));
      fd.append("threshold", String(threshold));
      const r = await fetch(`${API_URL}/infer`, {
        method: "POST",
        body: fd,
        headers: { "ngrok-skip-browser-warning": "true" },
      });
      if (!r.ok) {
        const text = await r.text();
        throw new Error(`HTTP ${r.status}: ${text}`);
      }
      const data = await r.json();
      setResults(data.results || []);
    } catch (e: any) {
      setError(e.message || String(e));
    } finally {
      setRunning(false);
    }
  }

  return (
    <main className="mx-auto max-w-7xl px-6 py-10">
      <header className="mb-8 flex flex-wrap items-baseline justify-between gap-4">
        <div>
          <h1 className="text-3xl font-semibold tracking-tight">
            Vessel Vision Inference Hub
          </h1>
          <p className="mt-1 text-sm text-slate-400">
            11 vessel-detection models, one image, side-by-side comparison.
          </p>
        </div>
        <StatusPill online={backendOnline} apiUrl={API_URL} />
      </header>

      {backendOnline === false && <LandingBanner apiUrl={API_URL} />}

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[360px_1fr]">
        <section className="rounded-xl border border-slate-800 bg-slate-900/40 p-5">
          <h2 className="mb-3 text-sm font-medium uppercase tracking-wider text-slate-400">
            Input
          </h2>

          <label className="block">
            <span className="text-sm text-slate-300">Image</span>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={onFileChange}
              className="mt-2 block w-full text-sm text-slate-400 file:mr-3 file:rounded-md file:border-0 file:bg-indigo-600 file:px-3 file:py-2 file:text-sm file:font-medium file:text-white hover:file:bg-indigo-500"
            />
          </label>

          {preview && (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={preview}
              alt="preview"
              className="mt-3 max-h-64 w-full rounded-md object-contain border border-slate-800"
            />
          )}

          <div className="mt-5">
            <div className="flex items-center justify-between text-sm">
              <span className="text-slate-300">Threshold</span>
              <span className="font-mono text-slate-400">{threshold.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min={0.05}
              max={0.95}
              step={0.05}
              value={threshold}
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
              className="mt-1 w-full accent-indigo-500"
            />
          </div>

          <div className="mt-5">
            <div className="mb-2 flex items-center justify-between text-sm">
              <span className="text-slate-300">Models ({selected.size})</span>
              <button
                type="button"
                className="text-xs text-indigo-400 hover:text-indigo-300"
                onClick={() =>
                  setSelected(selected.size === models.length ? new Set() : new Set(models))
                }
              >
                {selected.size === models.length ? "Clear" : "All"}
              </button>
            </div>
            <div className="max-h-64 space-y-1 overflow-auto rounded-md border border-slate-800 bg-slate-950/40 p-2">
              {models.map((m) => (
                <label key={m} className="flex cursor-pointer items-center gap-2 rounded px-2 py-1 text-sm hover:bg-slate-800/50">
                  <input
                    type="checkbox"
                    className="accent-indigo-500"
                    checked={selected.has(m)}
                    onChange={() => toggle(m)}
                  />
                  <span>{m}</span>
                </label>
              ))}
            </div>
          </div>

          <button
            onClick={run}
            disabled={running || backendOnline === false}
            className="mt-5 w-full rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white shadow disabled:cursor-not-allowed disabled:bg-slate-700 hover:bg-indigo-500"
          >
            {running ? "Running…" : "Run inference"}
          </button>

          {error && (
            <p className="mt-3 rounded-md border border-red-900/60 bg-red-950/40 px-3 py-2 text-sm text-red-300">
              {error}
            </p>
          )}
        </section>

        <section className="rounded-xl border border-slate-800 bg-slate-900/40 p-5">
          <h2 className="mb-3 text-sm font-medium uppercase tracking-wider text-slate-400">
            Results
          </h2>
          {results.length === 0 && !running && (
            <p className="text-sm text-slate-500">
              Upload an image, pick models, hit run. Outputs appear here side-by-side.
            </p>
          )}
          {running && <p className="text-sm text-slate-400">Running inference…</p>}
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            {results.map((r) => (
              <div
                key={r.model}
                className="overflow-hidden rounded-lg border border-slate-800 bg-slate-950/40"
              >
                <div className="flex items-center justify-between border-b border-slate-800 px-3 py-2 text-xs">
                  <span className="font-medium text-slate-200">{r.model}</span>
                  <span className="font-mono text-slate-400">
                    {r.error ? "ERROR" : `${r.latency_ms.toFixed(1)} ms · ${r.detections.length} det`}
                  </span>
                </div>
                {r.error ? (
                  <p className="p-3 text-xs text-red-300">{r.error}</p>
                ) : r.annotated_image_b64 ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    src={r.annotated_image_b64}
                    alt={r.model}
                    className="w-full object-contain"
                  />
                ) : null}
              </div>
            ))}
          </div>
        </section>
      </div>

      <footer className="mt-12 border-t border-slate-800 pt-6 text-xs text-slate-500">
        Backend runs locally on your GPU via Docker — see project README.
      </footer>
    </main>
  );
}

function StatusPill({ online, apiUrl }: { online: boolean | null; apiUrl: string }) {
  const color =
    online === null ? "bg-slate-600" : online ? "bg-emerald-500" : "bg-red-500";
  const label = online === null ? "checking…" : online ? "backend online" : "backend offline";
  return (
    <div className="flex items-center gap-2 rounded-full border border-slate-800 bg-slate-900/60 px-3 py-1.5 text-xs">
      <span className={`h-2 w-2 rounded-full ${color}`} />
      <span className="text-slate-300">{label}</span>
      <span className="font-mono text-slate-500">{apiUrl}</span>
    </div>
  );
}

function LandingBanner({ apiUrl }: { apiUrl: string }) {
  return (
    <div className="mb-6 rounded-xl border border-amber-900/40 bg-amber-950/30 p-4 text-sm text-amber-100">
      <p className="font-medium">No local backend detected at {apiUrl}.</p>
      <p className="mt-1 text-amber-200/80">
        This UI runs inference on a local GPU. To enable the demo, clone the repo and
        start the backend:
      </p>
      <pre className="mt-2 overflow-x-auto rounded-md bg-black/40 p-3 font-mono text-xs text-amber-100">
{`git clone https://github.com/<you>/inference-hub
cd inference-hub
docker compose up`}
      </pre>
    </div>
  );
}
