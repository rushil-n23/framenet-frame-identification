import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity


PRED_PATH = Path("data/outputs/all_model_predictions.csv")
HISTORY_PATH = Path("results/experiment_history.csv")
CLUSTER_OUTPUT_PATH = Path("results/frame_clusters.csv")
DETAILED_OUTPUT_PATH = Path("data/outputs/all_model_predictions_cluster_top3.csv")

N_CLUSTERS = 8


def clean_text(text):
    return str(text).replace("_", " ").strip()


def to_bool(value):
    if isinstance(value, bool):
        return value

    if pd.isna(value):
        return False

    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def build_frame_clusters(allowed_frames, frame_embeddings):
    clustering = AgglomerativeClustering(
        n_clusters=N_CLUSTERS,
        metric="cosine",
        linkage="average",
    )

    cluster_ids = clustering.fit_predict(frame_embeddings)

    frame_to_cluster = {
        frame: cluster_id
        for frame, cluster_id in zip(allowed_frames, cluster_ids)
    }

    cluster_df = pd.DataFrame({
        "frame": allowed_frames,
        "cluster": cluster_ids,
    }).sort_values(["cluster", "frame"])

    return frame_to_cluster, cluster_df


def get_prediction_cluster(
    prediction,
    prediction_cluster_cache,
    allowed_frames,
    frame_embeddings,
    embedder,
    frame_to_cluster,
):
    if pd.isna(prediction) or prediction == "UNKNOWN":
        return -1

    if prediction in prediction_cluster_cache:
        return prediction_cluster_cache[prediction]

    pred_embedding = embedder.encode(
        [clean_text(prediction)],
        normalize_embeddings=True,
    )

    similarities = cosine_similarity(pred_embedding, frame_embeddings)[0]
    best_index = np.argmax(similarities)
    nearest_frame = allowed_frames[best_index]
    cluster = frame_to_cluster[nearest_frame]

    prediction_cluster_cache[prediction] = cluster

    return cluster


def get_cluster_top3(
    prediction,
    prediction_top3_cache,
    allowed_frames,
    frame_embeddings,
    embedder,
    frame_to_cluster,
):
    if pd.isna(prediction) or prediction == "UNKNOWN":
        return ["UNKNOWN", "UNKNOWN", "UNKNOWN"]

    if prediction in prediction_top3_cache:
        return prediction_top3_cache[prediction]

    pred_embedding = embedder.encode(
        [clean_text(prediction)],
        normalize_embeddings=True,
    )

    similarities = cosine_similarity(pred_embedding, frame_embeddings)[0]

    predicted_cluster = get_prediction_cluster(
        prediction,
        {},
        allowed_frames,
        frame_embeddings,
        embedder,
        frame_to_cluster,
    )

    candidate_indices = [
        i
        for i, frame in enumerate(allowed_frames)
        if frame_to_cluster[frame] == predicted_cluster
    ]

    if len(candidate_indices) < 3:
        candidate_indices = list(range(len(allowed_frames)))

    ranked_candidates = sorted(
        candidate_indices,
        key=lambda i: similarities[i],
        reverse=True,
    )

    top_indices = ranked_candidates[:3]

    while len(top_indices) < 3:
        top_indices.append(top_indices[-1])

    top3 = [allowed_frames[i] for i in top_indices]
    prediction_top3_cache[prediction] = top3

    return top3


def hit_hard_negative(row):
    negatives = str(row.get("hard_negative_frames", "")).split("|")
    negatives = [n for n in negatives if n and n != "nan"]
    return row["prediction"] in negatives


def safe_mean(series):
    if len(series) == 0:
        return np.nan
    return series.mean()


def append_to_history(results):
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

    if HISTORY_PATH.exists():
        old_history = pd.read_csv(HISTORY_PATH)
        history = pd.concat([old_history, results], ignore_index=True)
    else:
        history = results

    history.to_csv(HISTORY_PATH, index=False)


def main():
    if not PRED_PATH.exists():
        raise FileNotFoundError(f"Missing predictions file: {PRED_PATH}")

    df = pd.read_csv(PRED_PATH)

    required_columns = {"model", "condition", "gold_frame", "prediction"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if "is_ambiguous_lu" not in df.columns:
        df["is_ambiguous_lu"] = False

    if "is_long_context" not in df.columns:
        df["is_long_context"] = False

    if "hard_negative_frames" not in df.columns:
        df["hard_negative_frames"] = ""

    df["is_ambiguous_lu"] = df["is_ambiguous_lu"].apply(to_bool)
    df["is_long_context"] = df["is_long_context"].apply(to_bool)

    allowed_frames = sorted(df["gold_frame"].dropna().unique().tolist())

    print(f"Loaded {len(df)} predictions.")
    print(f"Detected {len(allowed_frames)} frames.")
    print(f"Using {N_CLUSTERS} semantic clusters.")

    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    frame_texts = [clean_text(frame) for frame in allowed_frames]

    frame_embeddings = embedder.encode(
        frame_texts,
        normalize_embeddings=True,
    )

    frame_to_cluster, cluster_df = build_frame_clusters(
        allowed_frames,
        frame_embeddings,
    )

    prediction_cluster_cache = {}
    prediction_top3_cache = {}

    df["gold_cluster"] = df["gold_frame"].map(frame_to_cluster)

    df["predicted_cluster"] = df["prediction"].apply(
        lambda pred: get_prediction_cluster(
            pred,
            prediction_cluster_cache,
            allowed_frames,
            frame_embeddings,
            embedder,
            frame_to_cluster,
        )
    )

    df["cluster_top3_predictions"] = df["prediction"].apply(
        lambda pred: get_cluster_top3(
            pred,
            prediction_top3_cache,
            allowed_frames,
            frame_embeddings,
            embedder,
            frame_to_cluster,
        )
    )

    df["exact_match"] = df["gold_frame"] == df["prediction"]
    df["cluster_match"] = df["gold_cluster"] == df["predicted_cluster"]

    df["cluster_top3_match"] = df.apply(
        lambda row: row["gold_frame"] in row["cluster_top3_predictions"],
        axis=1,
    )

    df["hard_negative_error"] = df.apply(hit_hard_negative, axis=1)

    run_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rows = []

    for (model, condition), group in df.groupby(["model", "condition"]):
        y_true = group["gold_frame"]
        y_pred = group["prediction"]

        accuracy = accuracy_score(y_true, y_pred)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )

        cluster_accuracy = group["cluster_match"].mean()
        cluster_top3_accuracy = group["cluster_top3_match"].mean()
        unknown_rate = (group["prediction"] == "UNKNOWN").mean()

        ambiguous_group = group[group["is_ambiguous_lu"] == True]
        long_context_group = group[group["is_long_context"] == True]

        ambiguous_accuracy = safe_mean(ambiguous_group["exact_match"])
        long_context_accuracy = safe_mean(long_context_group["exact_match"])

        rows.append({
            "run_id": run_id,
            "timestamp": timestamp,
            "model": model,
            "condition": condition,
            "accuracy": accuracy,
            "precision_macro": precision,
            "recall_macro": recall,
            "f1_macro": f1,
            "cluster_accuracy": cluster_accuracy,
            "cluster_top3_accuracy": cluster_top3_accuracy,
            "cluster_gain": cluster_accuracy - accuracy,
            "cluster_top3_gain": cluster_top3_accuracy - accuracy,
            "unknown_rate": unknown_rate,
            "hard_negative_error_rate": group["hard_negative_error"].mean(),
            "ambiguous_accuracy": ambiguous_accuracy,
            "long_context_accuracy": long_context_accuracy,
            "ambiguous_examples": len(ambiguous_group),
            "long_context_examples": len(long_context_group),
            "n_examples": len(group),
        })

    results = pd.DataFrame(rows).sort_values(
        ["timestamp", "model", "condition"]
    )

    CLUSTER_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DETAILED_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    cluster_df.to_csv(CLUSTER_OUTPUT_PATH, index=False)
    df.to_csv(DETAILED_OUTPUT_PATH, index=False)

    append_to_history(results)

    print("\n=== Current Run Results ===")
    print(results.to_string(index=False))

    print(f"\nSaved experiment history to: {HISTORY_PATH}")
    print(f"Saved frame clusters to: {CLUSTER_OUTPUT_PATH}")
    print(f"Saved detailed predictions to: {DETAILED_OUTPUT_PATH}")


if __name__ == "__main__":
    main()