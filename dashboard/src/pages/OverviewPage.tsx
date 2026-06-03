import type { ReactNode } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  LineChart,
  Line,
  CartesianGrid,
  Legend,
} from "recharts";
import type { Row } from "../utils/metrics";
import {
  getModel,
  getPrompt,
  labelModel,
  labelPrompt,
  formatMetric,
} from "../utils/metrics";

export default function OverviewPage({
  rows,
  best,
  chartData,
  trendData,
  promptOptions,
  activePrompt,
  setActivePrompt,
}: {
  rows: Row[];
  best: Row | null;
  chartData: any[];
  trendData: any[];
  promptOptions: string[];
  activePrompt: string;
  setActivePrompt: (prompt: string) => void;
}) {
  return (
    <>
      <section className="flex flex-wrap gap-2 mt-6">
        <button
          onClick={() => setActivePrompt("all")}
          className={`px-4 py-2 rounded-lg text-sm ${
            activePrompt === "all"
              ? "bg-cyan-500 text-slate-950 font-semibold"
              : "bg-slate-800 text-slate-300"
          }`}
        >
          All Prompts
        </button>

        {promptOptions.map((prompt) => (
          <button
            key={prompt}
            onClick={() => setActivePrompt(prompt)}
            className={`px-4 py-2 rounded-lg text-sm ${
              activePrompt === prompt
                ? "bg-cyan-500 text-slate-950 font-semibold"
                : "bg-slate-800 text-slate-300"
            }`}
          >
            {labelPrompt(prompt)}
          </button>
        ))}
      </section>

      <section className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-5 mt-6">
        <Kpi
          title="Best Model"
          value={labelModel(getModel(best || {}))}
          note={labelPrompt(getPrompt(best || {}))}
        />
        <Kpi
          title="Standard Accuracy"
          value={formatMetric(best?.accuracy, "accuracy")}
          note="without cluster evaluation"
        />
        <Kpi
          title="Cluster Accuracy"
          value={formatMetric(best?.cluster_accuracy, "cluster_accuracy")}
          note="semantic cluster match"
        />
        <Kpi
          title="Rows Loaded"
          value={String(rows.length)}
          note="from experiment history"
        />
      </section>

      <section className="grid grid-cols-1 xl:grid-cols-3 gap-6 mt-6">
        <Panel title="Standard vs Cluster Performance" className="xl:col-span-2">
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="name" stroke="#94a3b8" />
              <YAxis stroke="#94a3b8" domain={[0, 100]} />
              <Tooltip
                formatter={(value) => `${Number(value).toFixed(1)}%`}
                contentStyle={{
                  background: "#0f172a",
                  border: "1px solid #334155",
                }}
              />
              <Legend />
              <Bar dataKey="Accuracy" fill="#22d3ee" radius={[6, 6, 0, 0]} />
              <Bar dataKey="Macro F1" fill="#38bdf8" radius={[6, 6, 0, 0]} />
              <Bar dataKey="Cluster" fill="#34d399" radius={[6, 6, 0, 0]} />
              <Bar dataKey="Cluster Top-3" fill="#a78bfa" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Panel>

        <Panel title="UNKNOWN Rate">
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="name" stroke="#94a3b8" />
              <YAxis stroke="#94a3b8" domain={[0, 100]} />
              <Tooltip
                formatter={(value) => `${Number(value).toFixed(1)}%`}
                contentStyle={{
                  background: "#0f172a",
                  border: "1px solid #334155",
                }}
              />
              <Bar dataKey="Unknown" fill="#fb7185" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Panel>
      </section>

      <Panel title="Performance Trend Across Stored Runs" className="mt-6">
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={trendData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis dataKey="name" stroke="#94a3b8" />
            <YAxis stroke="#94a3b8" domain={[0, 100]} />
            <Tooltip
              formatter={(value) => `${Number(value).toFixed(1)}%`}
              contentStyle={{
                background: "#0f172a",
                border: "1px solid #334155",
              }}
            />
            <Legend />
            <Line type="monotone" dataKey="Accuracy" stroke="#22d3ee" strokeWidth={3} />
            <Line type="monotone" dataKey="Cluster" stroke="#34d399" strokeWidth={3} />
            <Line type="monotone" dataKey="Top-3" stroke="#a78bfa" strokeWidth={3} />
          </LineChart>
        </ResponsiveContainer>
      </Panel>
    </>
  );
}

function Kpi({
  title,
  value,
  note,
}: {
  title: string;
  value: string;
  note: string;
}) {
  return (
    <div className="rounded-2xl bg-slate-900 border border-slate-800 p-5 shadow-lg">
      <p className="text-sm text-slate-400">{title}</p>
      <p className="text-2xl font-bold mt-2">{value}</p>
      <p className="text-xs text-cyan-300 mt-2">{note}</p>
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