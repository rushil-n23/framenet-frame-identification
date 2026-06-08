from pathlib import Path

import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score


PREDICTIONS_PATH = Path("data/outputs/all_model_predictions.csv")
DATA_PATH = Path("data/processed/framenet_processed.csv")

OUTPUT_SUMMARY_PATH = Path("results/frame_clustering_comparison.csv")
OUTPUT_DETAILS_PATH = Path("results/frame_clustering_assignments.csv")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CLUSTER_CONFIGS = [
    {
        "method": "agglomerative",
        "n_clusters": 8,
        "name": "agglomerative_k8_current",
    },
    {
        "method": "agglomerative",
        "n_clusters": 10,
        "name": "agglomerative_k10",
    },
    {
        "method": "agglomerative",
        "n_clusters": 12,
        "name": "agglomerative_k12",
    },
    {
        "method": "agglomerative",
        "n_clusters": 15,
        "name": "agglomerative_k15",
    },
    {
        "method": "kmeans",
        "n_clusters": 8,
        "name": "kmeans_k8",
    },
    {
        "method": "kmeans",
        "n_clusters": 10,
        "name": "kmeans_k10",
    },
    {
        "method": "kmeans",
        "n_clusters": 12,
        "name": "kmeans_k12",
    },
    {
        "method": "kmeans",
        "n_clusters": 15,
        "name": "kmeans_k15",
    },
]


def clean_frame(frame):
    return str(frame).strip()


def clean_text(text):
    return str(text).replace("_", " ").strip()


def load_frames(df):
    return sorted(df["gold_frame"].dropna().astype(str).unique().tolist())


def build_frame_texts(df, frames):
    frame_texts = []

    for frame in frames:
        rows = df[df["gold_frame"] == frame]

        lexical_units = (
            rows["lexical_unit"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        examples = (
            rows["sentence"]
            .dropna()
            .astype(str)
            .head(3)
            .tolist()
        )

        if "frame_definition" in df.columns:
            definitions = (
                rows["frame_definition"]
                .dropna()
                .astype(str)
                .unique()
            )
            definition = definitions[0] if len(definitions) > 0 else ""
        else:
            definition = ""

        frame_text = (
            f"Frame: {clean_text(frame)}. "
            f"Definition: {definition}. "
            f"Lexical units: {', '.join(lexical_units[:10])}. "
            f"Examples: {' '.join(examples[:3])}"
        )

        frame_texts.append(frame_text)

    return frame_texts


def cluster_embeddings(embeddings, method, n_clusters):
    if method == "agglomerative":
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="cosine",
            linkage="average",
        )
        return model.fit_predict(embeddings)

    if method == "kmeans":
        model = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=20,
        )
        return model.fit_predict(embeddings)

    raise ValueError(f"Unknown clustering method: {method}")


def compute_cluster_accuracy(pred_df, frame_to_cluster):
    correct = []

    for _, row in pred_df.iterrows():
        gold = clean_frame(row["gold_frame"])
        pred = clean_frame(row["prediction"])

        if pred == "UNKNOWN":
            correct.append(False)
            continue

        if gold not in frame_to_cluster or pred not in frame_to_cluster:
            correct.append(False)
            continue

        correct.append(frame_to_cluster[gold] == frame_to_cluster[pred])

    return float(np.mean(correct))


def compute_exact_accuracy(pred_df):
    return float(
        (pred_df["gold_frame"].astype(str).str.strip()
         == pred_df["prediction"].astype(str).str.strip()).mean()
    )


def summarize_cluster_balance(labels, total_frames):
    counts = pd.Series(labels).value_counts().sort_index()

    largest = int(counts.max())
    smallest = int(counts.min())
    mean_size = float(counts.mean())
    std_size = float(counts.std())

    largest_share = largest / total_frames
    imbalance_ratio = largest / smallest if smallest > 0 else np.inf

    return {
        "largest_cluster_size": largest,
        "smallest_cluster_size": smallest,
        "mean_cluster_size": mean_size,
        "std_cluster_size": std_size,
        "largest_cluster_share": largest_share,
        "imbalance_ratio": imbalance_ratio,
        "cluster_sizes": dict(counts),
    }


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {DATA_PATH}")

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(f"Missing predictions: {PREDICTIONS_PATH}")

    df = pd.read_csv(DATA_PATH)
    pred_df = pd.read_csv(PREDICTIONS_PATH)

    frames = load_frames(df)

    print(f"Loaded {len(frames)} unique frames.")
    print(f"Loaded {len(pred_df)} predictions.")

    print("Building frame text representations...")
    frame_texts = build_frame_texts(df, frames)

    print(f"Encoding frames with {EMBEDDING_MODEL}...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    embeddings = embedder.encode(
        frame_texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    exact_accuracy = compute_exact_accuracy(pred_df)

    summary_rows = []
    assignment_rows = []

    for config in CLUSTER_CONFIGS:
        name = config["name"]
        method = config["method"]
        n_clusters = config["n_clusters"]

        print(f"\nRunning {name}...")

        labels = cluster_embeddings(
            embeddings=embeddings,
            method=method,
            n_clusters=n_clusters,
        )

        frame_to_cluster = {
            frame: int(label)
            for frame, label in zip(frames, labels)
        }

        balance = summarize_cluster_balance(labels, len(frames))

        cluster_accuracy = compute_cluster_accuracy(
            pred_df=pred_df,
            frame_to_cluster=frame_to_cluster,
        )

        semantic_gain = cluster_accuracy - exact_accuracy

        if n_clusters > 1 and n_clusters < len(frames):
            try:
                sil = silhouette_score(
                    embeddings,
                    labels,
                    metric="cosine",
                )
            except Exception:
                sil = None
        else:
            sil = None

        summary_rows.append({
            "config": name,
            "method": method,
            "n_clusters": n_clusters,
            "n_frames": len(frames),
            "exact_accuracy": exact_accuracy,
            "cluster_accuracy": cluster_accuracy,
            "semantic_gain": semantic_gain,
            "largest_cluster_size": balance["largest_cluster_size"],
            "smallest_cluster_size": balance["smallest_cluster_size"],
            "mean_cluster_size": balance["mean_cluster_size"],
            "std_cluster_size": balance["std_cluster_size"],
            "largest_cluster_share": balance["largest_cluster_share"],
            "imbalance_ratio": balance["imbalance_ratio"],
            "silhouette_score": sil,
            "cluster_sizes": balance["cluster_sizes"],
        })

        for frame, label in zip(frames, labels):
            assignment_rows.append({
                "config": name,
                "method": method,
                "n_clusters": n_clusters,
                "frame": frame,
                "cluster": int(label),
            })

    summary_df = pd.DataFrame(summary_rows)
    assignment_df = pd.DataFrame(assignment_rows)

    OUTPUT_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(OUTPUT_SUMMARY_PATH, index=False)
    assignment_df.to_csv(OUTPUT_DETAILS_PATH, index=False)

    print("\n=== Clustering Comparison Summary ===")
    print(
        summary_df[
            [
                "config",
                "cluster_accuracy",
                "semantic_gain",
                "largest_cluster_size",
                "smallest_cluster_size",
                "largest_cluster_share",
                "imbalance_ratio",
                "silhouette_score",
            ]
        ].to_string(index=False)
    )

    print("\nSaved summary to:")
    print(OUTPUT_SUMMARY_PATH)

    print("\nSaved assignments to:")
    print(OUTPUT_DETAILS_PATH)


if __name__ == "__main__":
    main()