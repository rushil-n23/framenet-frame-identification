import { useEffect, useMemo, useState } from "react";
import Papa from "papaparse";
import OverviewPage from "./pages/OverviewPage";
import ExperimentResultsPage from "./pages/ExperimentResultsPage";
import type { Row } from "./utils/metrics";
import {
  normalizeKey,
  getPrompt,
  getModel,
  toNumber,
  toPercentNumber,
  labelModel,
  labelPrompt,
} from "./utils/metrics";

type Page = "Overview" | "Experiment Results";

const CSV_PATH = "/results/experiment_history.csv";

async function loadCsv(): Promise<Row[]> {
  const response = await fetch(`${CSV_PATH}?t=${Date.now()}`);

  if (!response.ok) {
    throw new Error(`CSV not found at ${CSV_PATH}`);
  }

  const text = await response.text();

  return new Promise((resolve, reject) => {
    Papa.parse<Row>(text, {
      header: true,
      skipEmptyLines: true,
      complete: (result) => {
        const cleanedRows = result.data.map((row) => {
          const cleaned: Row = {};

          Object.entries(row).forEach(([key, value]) => {
            cleaned[normalizeKey(key)] = String(value ?? "").trim();
          });

          return cleaned;
        });

        resolve(
          cleanedRows.filter((row) =>
            Object.values(row).some((value) => value !== "")
          )
        );
      },
      error: reject,
    });
  });
}

export default function App() {
  const [rows, setRows] = useState<Row[]>([]);
  const [activePage, setActivePage] = useState<Page>("Overview");
  const [activePrompt, setActivePrompt] = useState("all");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState("Not loaded yet");

  async function refreshCsv() {
    try {
      setLoading(true);

      const data = await loadCsv();

      setRows(data);
      setError(null);
      setLastUpdated(new Date().toLocaleTimeString());
    } catch (err) {
      console.error(err);
      setRows([]);
      setError(
        "CSV could not be loaded. Check dashboard/public/results/experiment_history.csv"
      );
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    refreshCsv();

    const interval = setInterval(refreshCsv, 5000);
    return () => clearInterval(interval);
  }, []);

  const promptOptions = useMemo(() => {
    return Array.from(new Set(rows.map(getPrompt).filter(Boolean)));
  }, [rows]);

  const filteredOverviewRows =
    activePrompt === "all"
      ? rows
      : rows.filter((row) => getPrompt(row) === activePrompt);

  const best = useMemo(() => {
    if (rows.length === 0) return null;

    return [...rows].sort(
      (a, b) => toNumber(b.cluster_accuracy) - toNumber(a.cluster_accuracy)
    )[0];
  }, [rows]);

  const chartData = filteredOverviewRows.map((row) => ({
    name: `${labelModel(getModel(row))} / ${labelPrompt(getPrompt(row))}`,
    Accuracy: toPercentNumber(row.accuracy),
    "Macro F1": toPercentNumber(row.f1_macro),
    Cluster: toPercentNumber(row.cluster_accuracy),
    "Cluster Top-3": toPercentNumber(row.cluster_top3_accuracy),
    Unknown: toPercentNumber(row.unknown_rate),
  }));

  const trendData = rows.map((row) => ({
    name: `${labelModel(getModel(row))} / ${labelPrompt(getPrompt(row))}`,
    Accuracy: toPercentNumber(row.accuracy),
    Cluster: toPercentNumber(row.cluster_accuracy),
    "Top-3": toPercentNumber(row.cluster_top3_accuracy),
  }));

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex">
      <aside className="hidden lg:flex w-72 bg-slate-900 border-r border-slate-800 p-6 flex-col">
        <div>
          <h1 className="text-2xl font-bold">FrameNet Lab</h1>
          <p className="text-sm text-slate-400 mt-1">NLP Experiment Tracker</p>
        </div>

        <nav className="mt-10 space-y-2 text-sm">
          {(["Overview", "Experiment Results"] as Page[]).map((page) => (
            <button
              key={page}
              onClick={() => setActivePage(page)}
              className={`w-full text-left px-4 py-3 rounded-xl ${
                activePage === page
                  ? "bg-cyan-500/10 text-cyan-300 border border-cyan-500/20"
                  : "text-slate-400 hover:bg-slate-800"
              }`}
            >
              {page}
            </button>
          ))}
        </nav>

        <div className="mt-auto text-xs text-slate-500">
          Thesis dashboard · history CSV
        </div>
      </aside>

      <main className="flex-1 p-6 lg:p-8 overflow-x-hidden">
        <header className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 border-b border-slate-800 pb-6">
          <div>
            <p className="text-cyan-400 text-sm font-medium">
              FrameNet Semantic Frame Classification
            </p>
            <h2 className="text-3xl font-bold mt-1">{activePage}</h2>
            <p className="text-slate-400 text-sm mt-2">
              Data source: <code>{CSV_PATH}</code>
            </p>
            <p className="text-xs text-slate-500 mt-2">
              Last updated: {lastUpdated}
            </p>
          </div>

          <button
            onClick={refreshCsv}
            className="px-4 py-2 rounded-lg text-sm bg-cyan-500 text-slate-950 font-semibold hover:bg-cyan-400"
          >
            Refresh CSV
          </button>
        </header>

        {error && (
          <div className="mt-6 rounded-xl border border-rose-500/30 bg-rose-500/10 p-4 text-sm text-rose-300">
            {error}
          </div>
        )}

        {loading && rows.length === 0 && (
          <div className="mt-6 rounded-xl border border-slate-700 bg-slate-900 p-4 text-sm text-slate-300">
            Loading CSV...
          </div>
        )}

        {activePage === "Overview" ? (
          <OverviewPage
            rows={rows}
            best={best}
            chartData={chartData}
            trendData={trendData}
            promptOptions={promptOptions}
            activePrompt={activePrompt}
            setActivePrompt={setActivePrompt}
          />
        ) : (
          <ExperimentResultsPage rows={rows} />
        )}
      </main>
    </div>
  );
}