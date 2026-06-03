import { useMemo, useState } from "react";
import type { ReactNode } from "react";
import type { Row } from "../utils/metrics";
import {
  getModel,
  getPrompt,
  toNumber,
  formatMetric,
  labelModel,
  labelPrompt,
  inputClass,
} from "../utils/metrics";

type SortDirection = "asc" | "desc";

type RunOption = {
  run_id: string;
  timestamp: string;
  label: string;
};

export default function ExperimentResultsPage({ rows }: { rows: Row[] }) {
  const [search, setSearch] = useState("");
  const [modelFilter, setModelFilter] = useState("all");
  const [promptFilter, setPromptFilter] = useState("all");
  const [runFilter, setRunFilter] = useState("all");
  const [sortKey, setSortKey] = useState("timestamp");
  const [sortDirection, setSortDirection] = useState<SortDirection>("desc");
  const [showRunInfo, setShowRunInfo] = useState(true);

  const runOptions = useMemo<RunOption[]>(() => {
    const runMap = new Map<string, { run_id: string; timestamp: string }>();

    rows.forEach((row) => {
      if (!row.run_id) return;

      if (!runMap.has(row.run_id)) {
        runMap.set(row.run_id, {
          run_id: row.run_id,
          timestamp: row.timestamp || "",
        });
      }
    });

    return Array.from(runMap.values())
      .sort(
        (a, b) =>
          new Date(a.timestamp || "").getTime() -
          new Date(b.timestamp || "").getTime()
      )
      .map((run, index) => ({
        ...run,
        label: `Run ${index + 1}`,
      }));
  }, [rows]);

  const runLabels = useMemo(() => {
    const labels: Record<string, string> = {};

    runOptions.forEach((run) => {
      labels[run.run_id] = run.label;
    });

    return labels;
  }, [runOptions]);

  const modelOptions = useMemo(() => {
    return Array.from(new Set(rows.map(getModel).filter(Boolean)));
  }, [rows]);

  const promptOptions = useMemo(() => {
    return Array.from(new Set(rows.map(getPrompt).filter(Boolean)));
  }, [rows]);

  const visibleRows = useMemo(() => {
    return rows
      .filter((row) => {
        const fullText = Object.values(row).join(" ").toLowerCase();

        return (
          fullText.includes(search.toLowerCase()) &&
          (modelFilter === "all" || getModel(row) === modelFilter) &&
          (promptFilter === "all" || getPrompt(row) === promptFilter) &&
          (runFilter === "all" || row.run_id === runFilter)
        );
      })
      .sort((a, b) => {
        if (sortKey === "timestamp") {
          const aTime = new Date(a.timestamp || "").getTime();
          const bTime = new Date(b.timestamp || "").getTime();

          if (aTime !== bTime) {
            return sortDirection === "asc" ? aTime - bTime : bTime - aTime;
          }

          return getModel(a).localeCompare(getModel(b));
        }

        const aValue = toNumber(a[sortKey]);
        const bValue = toNumber(b[sortKey]);

        if (aValue !== bValue) {
          return sortDirection === "asc" ? aValue - bValue : bValue - aValue;
        }

        return getModel(a).localeCompare(getModel(b));
      });
  }, [
    rows,
    search,
    modelFilter,
    promptFilter,
    runFilter,
    sortKey,
    sortDirection,
  ]);

  function resetFilters() {
    setSearch("");
    setModelFilter("all");
    setPromptFilter("all");
    setRunFilter("all");
    setSortKey("timestamp");
    setSortDirection("desc");
    setShowRunInfo(true);
  }

  return (
    <section className="mt-6">
      <Panel title="Customizable Experiment Results Table">
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-6 gap-4 mb-6">
          <Field label="Search">
            <input
              aria-label="Search experiment results"
              title="Search experiment results"
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Search run, model, prompt..."
              className={inputClass}
            />
          </Field>

          <Field label="Run">
            <select
              aria-label="Filter by run"
              title="Filter by run"
              value={runFilter}
              onChange={(event) => setRunFilter(event.target.value)}
              className={inputClass}
            >
              <option value="all">All Runs</option>
              {runOptions.map((run) => (
                <option key={run.run_id} value={run.run_id}>
                  {run.label} — {run.timestamp}
                </option>
              ))}
            </select>
          </Field>

          <Field label="Model">
            <select
              aria-label="Filter by model"
              title="Filter by model"
              value={modelFilter}
              onChange={(event) => setModelFilter(event.target.value)}
              className={inputClass}
            >
              <option value="all">All Models</option>
              {modelOptions.map((model) => (
                <option key={model} value={model}>
                  {labelModel(model)}
                </option>
              ))}
            </select>
          </Field>

          <Field label="Prompt">
            <select
              aria-label="Filter by prompt"
              title="Filter by prompt"
              value={promptFilter}
              onChange={(event) => setPromptFilter(event.target.value)}
              className={inputClass}
            >
              <option value="all">All Prompts</option>
              {promptOptions.map((prompt) => (
                <option key={prompt} value={prompt}>
                  {labelPrompt(prompt)}
                </option>
              ))}
            </select>
          </Field>

          <Field label="Sort By">
            <select
              aria-label="Sort experiment results by metric"
              title="Sort experiment results by metric"
              value={sortKey}
              onChange={(event) => setSortKey(event.target.value)}
              className={inputClass}
            >
              <option value="timestamp">Timestamp</option>
              <option value="accuracy">Accuracy</option>
              <option value="cluster_accuracy">Cluster Accuracy</option>
              <option value="cluster_gain">Cluster Gain</option>
              <option value="cluster_top3_accuracy">Cluster Top-3 Accuracy</option>
              <option value="f1_macro">Macro F1</option>
              <option value="unknown_rate">UNKNOWN Rate</option>
            </select>
          </Field>

          <Field label="Direction">
            <select
              aria-label="Sort direction"
              title="Sort direction"
              value={sortDirection}
              onChange={(event) =>
                setSortDirection(event.target.value as SortDirection)
              }
              className={inputClass}
            >
              <option value="desc">Highest / Newest First</option>
              <option value="asc">Lowest / Oldest First</option>
            </select>
          </Field>
        </div>

        <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 mb-4">
          <p className="text-sm text-slate-400">
            Showing {visibleRows.length} of {rows.length} records ·{" "}
            {runOptions.length} stored runs
          </p>

          <div className="flex gap-2">
            <button
              onClick={() => setShowRunInfo((value) => !value)}
              className="px-4 py-2 rounded-lg text-sm bg-slate-800 text-slate-300 hover:bg-slate-700"
            >
              {showRunInfo ? "Hide Run Info" : "Show Run Info"}
            </button>

            <button
              onClick={resetFilters}
              className="px-4 py-2 rounded-lg text-sm bg-slate-800 text-slate-300 hover:bg-slate-700"
            >
              Reset Filters
            </button>
          </div>
        </div>

        <div className="overflow-x-auto rounded-xl border border-slate-800">
          <table className="w-full text-sm">
            <thead className="bg-slate-800/70">
              <tr className="text-left text-slate-300">
                {showRunInfo && (
                  <>
                    <th className="px-4 py-3">Run</th>
                    <th className="px-4 py-3">Timestamp</th>
                  </>
                )}
                <th className="px-4 py-3">Model</th>
                <th className="px-4 py-3">Prompt</th>
                <th className="px-4 py-3">Accuracy</th>
                <th className="px-4 py-3">Cluster Accuracy</th>
                <th className="px-4 py-3">Gain</th>
                <th className="px-4 py-3">Top-3</th>
                <th className="px-4 py-3">Macro F1</th>
                <th className="px-4 py-3">UNKNOWN</th>
                <th className="px-4 py-3">Examples</th>
              </tr>
            </thead>

            <tbody>
              {visibleRows.map((row, index) => (
                <tr
                  key={`${row.run_id || "run"}-${index}`}
                  className="border-t border-slate-800 hover:bg-slate-800/50"
                >
                  {showRunInfo && (
                    <>
                      <td className="px-4 py-3">
                        <RunBadge label={runLabels[row.run_id] || "-"} />
                      </td>

                      <td className="px-4 py-3 whitespace-nowrap text-slate-400">
                        {row.timestamp || "-"}
                      </td>
                    </>
                  )}

                  <td className="px-4 py-3">
                    <ModelBadge model={getModel(row)} />
                  </td>

                  <td className="px-4 py-3">
                    <PromptBadge prompt={getPrompt(row)} />
                  </td>

                  <td className="px-4 py-3 font-semibold">
                    {formatMetric(row.accuracy, "accuracy")}
                  </td>

                  <td className="px-4 py-3 font-semibold text-emerald-300">
                    {formatMetric(row.cluster_accuracy, "cluster_accuracy")}
                  </td>

                  <td className="px-4 py-3">
                    <GainBadge value={row.cluster_gain} />
                  </td>

                  <td className="px-4 py-3 text-violet-300 font-medium">
                    {formatMetric(
                      row.cluster_top3_accuracy,
                      "cluster_top3_accuracy"
                    )}
                  </td>

                  <td className="px-4 py-3">
                    {formatMetric(row.f1_macro, "f1_macro")}
                  </td>

                  <td className="px-4 py-3 text-rose-300">
                    {formatMetric(row.unknown_rate, "unknown_rate")}
                  </td>

                  <td className="px-4 py-3">
                    {formatMetric(row.n_examples, "n_examples")}
                  </td>
                </tr>
              ))}

              {visibleRows.length === 0 && (
                <tr>
                  <td
                    colSpan={showRunInfo ? 11 : 9}
                    className="px-4 py-8 text-center text-slate-400"
                  >
                    No rows match the selected filters.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </Panel>
    </section>
  );
}

function RunBadge({ label }: { label: string }) {
  return (
    <span className="inline-flex items-center rounded-full bg-slate-700 border border-slate-600 px-3 py-1 text-xs font-semibold text-slate-200">
      {label}
    </span>
  );
}

function ModelBadge({ model }: { model: string }) {
  let className =
    "inline-flex items-center rounded-full px-3 py-1 text-xs font-semibold border ";

  if (model.includes("large")) {
    className += "bg-emerald-500/10 text-emerald-300 border-emerald-500/30";
  } else if (model.includes("base")) {
    className += "bg-cyan-500/10 text-cyan-300 border-cyan-500/30";
  } else if (model.includes("small")) {
    className += "bg-amber-500/10 text-amber-300 border-amber-500/30";
  } else {
    className += "bg-slate-700 text-slate-300 border-slate-600";
  }

  return <span className={className}>{labelModel(model)}</span>;
}

function PromptBadge({ prompt }: { prompt: string }) {
  let className =
    "inline-flex items-center rounded-lg px-3 py-1 text-xs font-semibold border ";

  if (prompt === "few_shot") {
    className += "bg-violet-500/10 text-violet-300 border-violet-500/30";
  } else if (prompt === "guided") {
    className += "bg-blue-500/10 text-blue-300 border-blue-500/30";
  } else if (prompt === "zero_shot") {
    className += "bg-slate-500/10 text-slate-300 border-slate-500/30";
  } else {
    className += "bg-slate-700 text-slate-300 border-slate-600";
  }

  return <span className={className}>{labelPrompt(prompt)}</span>;
}

function GainBadge({ value }: { value: string | undefined }) {
  const gain = toNumber(value);
  const formatted = formatMetric(value, "cluster_gain");

  if (gain > 0) {
    return (
      <span className="inline-flex items-center gap-1 rounded-full bg-emerald-500/10 border border-emerald-500/30 px-3 py-1 text-xs font-semibold text-emerald-300">
        ↑ {formatted}
      </span>
    );
  }

  if (gain < 0) {
    return (
      <span className="inline-flex items-center gap-1 rounded-full bg-rose-500/10 border border-rose-500/30 px-3 py-1 text-xs font-semibold text-rose-300">
        ↓ {formatted}
      </span>
    );
  }

  return (
    <span className="inline-flex items-center gap-1 rounded-full bg-slate-700 border border-slate-600 px-3 py-1 text-xs font-semibold text-slate-300">
      {formatted}
    </span>
  );
}

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div>
      <label className="text-xs text-slate-400">{label}</label>
      <div className="mt-2">{children}</div>
    </div>
  );
}

function Panel({
  title,
  children,
  className = "",
}: {
  title: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section
      className={`rounded-2xl bg-slate-900 border border-slate-800 p-6 shadow-lg ${className}`}
    >
      <h3 className="font-semibold text-lg mb-4">{title}</h3>
      {children}
    </section>
  );
}