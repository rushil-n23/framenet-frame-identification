export type Row = Record<string, string>;

export function normalizeKey(key: string) {
  return key.replace(/^\uFEFF/, "").trim();
}

export function getPrompt(row: Row) {
  return row.condition || row["Prompt Condition"] || row.prompt_condition || "";
}

export function getModel(row: Row) {
  return row.model || row.Model || "";
}

export function toNumber(value: string | undefined) {
  if (!value) return 0;
  return Number(String(value).replace("%", "").trim()) || 0;
}

export function toPercentNumber(value: string | undefined) {
  const num = toNumber(value);
  return Math.abs(num) <= 1 ? num * 100 : num;
}

export function formatMetric(value: string | undefined, key: string) {
  if (!value) return "-";

  const num = toNumber(value);

  if (key === "n_examples") return String(Math.round(num));

  if (Math.abs(num) <= 1) return `${(num * 100).toFixed(1)}%`;

  return `${num.toFixed(1)}%`;
}

export function labelModel(model?: string) {
  if (!model) return "-";

  return model
    .replaceAll("_", " ")
    .replace(/\bflan\b/i, "FLAN")
    .replace(/\bt5\b/i, "T5")
    .replace(/\bsmall\b/i, "Small")
    .replace(/\bbase\b/i, "Base")
    .replace(/\blarge\b/i, "Large");
}

export function labelPrompt(prompt?: string) {
  if (!prompt) return "-";

  return prompt
    .replace("zero_shot", "Zero-shot")
    .replace("few_shot", "Few-shot")
    .replace("guided", "Guided")
    .replaceAll("_", " ");
}

export function prettyColumnName(key: string) {
  const names: Record<string, string> = {
    accuracy: "Accuracy",
    precision_macro: "Precision",
    recall_macro: "Recall",
    f1_macro: "Macro F1",
    cluster_accuracy: "Cluster Accuracy",
    cluster_top3_accuracy: "Cluster Top-3 Accuracy",
    cluster_gain: "Cluster Gain",
    cluster_top3_gain: "Cluster Top-3 Gain",
    unknown_rate: "UNKNOWN Rate",
    n_examples: "Examples",
  };

  return names[key] || key;
}

export const inputClass =
  "w-full rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-sm outline-none focus:border-cyan-400";