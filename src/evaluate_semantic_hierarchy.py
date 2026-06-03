from pathlib import Path
from collections import defaultdict, deque

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


PREDICTIONS_PATH = Path("data/outputs/all_model_predictions_cluster_top3.csv")
BASIC_PREDICTIONS_PATH = Path("data/outputs/all_model_predictions.csv")

FRAME_CLUSTERS_PATH = Path("results/frame_clusters.csv")
RELATIONS_PATH = Path("data/framenet/frame_relations.csv")

OUTPUT_PATH = Path("results/semantic_hierarchy_evaluation.csv")
DETAILED_OUTPUT_PATH = Path("data/outputs/all_model_predictions_semantic_hierarchy.csv")


RELATION_WEIGHTS = {
    "Inheritance": 0.85,
    "Perspective_on": 0.85,
    "Subframe": 0.75,
    "Using": 0.70,
    "Causative_of": 0.70,
    "Inchoative_of": 0.70,
    "Precedes": 0.55,
    "See_also": 0.50,
}


def clean_frame(value):
    return str(value).strip()


def load_predictions():
    if PREDICTIONS_PATH.exists():
        return pd.read_csv(PREDICTIONS_PATH)

    if BASIC_PREDICTIONS_PATH.exists():
        return pd.read_csv(BASIC_PREDICTIONS_PATH)

    raise FileNotFoundError(
        f"Could not find {PREDICTIONS_PATH} or {BASIC_PREDICTIONS_PATH}"
    )


def load_cluster_map():
    if not FRAME_CLUSTERS_PATH.exists():
        print("No frame_clusters.csv found. Cluster metrics will be disabled.")
        return {}

    clusters = pd.read_csv(FRAME_CLUSTERS_PATH)

    if "frame" not in clusters.columns or "cluster" not in clusters.columns:
        print("frame_clusters.csv must contain columns: frame, cluster")
        return {}

    return {
        clean_frame(row["frame"]): row["cluster"]
        for _, row in clusters.iterrows()
    }


def load_relation_graph():
    """
    Optional file:
        data/framenet/frame_relations.csv

    Required columns:
        source_frame,target_frame,relation_type

    Example:
        Commerce_buy,Commerce_goods-transfer,Perspective_on
        Commerce_sell,Commerce_goods-transfer,Perspective_on
    """
    graph = defaultdict(list)

    if not RELATIONS_PATH.exists():
        print("No frame relation CSV found.")
        print("Hierarchy metric will use cluster/domain only.")
        return graph

    relations = pd.read_csv(RELATIONS_PATH)

    required = {"source_frame", "target_frame", "relation_type"}
    missing = required - set(relations.columns)

    if missing:
        raise ValueError(f"Missing relation columns: {missing}")

    for _, row in relations.iterrows():
        source = clean_frame(row["source_frame"])
        target = clean_frame(row["target_frame"])
        relation = clean_frame(row["relation_type"])

        graph[source].append((target, relation))
        graph[target].append((source, relation))

    print(f"Loaded {len(relations)} frame relations.")
    return graph


def shortest_relation_distance(graph, source, target, max_depth=2):
    if source == target:
        return 0, "Exact"

    if source not in graph:
        return None, None

    queue = deque([(source, 0, None)])
    visited = {source}

    while queue:
        current, depth, first_relation = queue.popleft()

        if depth >= max_depth:
            continue

        for neighbor, relation in graph[current]:
            if neighbor in visited:
                continue

            next_relation = first_relation or relation

            if neighbor == target:
                return depth + 1, next_relation

            visited.add(neighbor)
            queue.append((neighbor, depth + 1, next_relation))

    return None, None


def same_cluster(gold, pred, cluster_map):
    if pred == "UNKNOWN":
        return False

    if gold not in cluster_map or pred not in cluster_map:
        return False

    return cluster_map[gold] == cluster_map[pred]


def same_domain(row):
    """
    Since each row only stores the gold domain, this metric is conservative.
    If you later create a frame_to_domain map, this can be improved.
    """
    pred = clean_frame(row["prediction"])

    if pred == "UNKNOWN":
        return False

    # If exact, same domain is definitely true.
    if clean_frame(row["gold_frame"]) == pred:
        return True

    # We cannot reliably infer predicted domain from current prediction file.
    return False


def semantic_score(row, cluster_map, relation_graph):
    gold = clean_frame(row["gold_frame"])
    pred = clean_frame(row["prediction"])

    if pred == "UNKNOWN":
        return 0.0, "unknown"

    if gold == pred:
        return 1.0, "exact"

    if same_cluster(gold, pred, cluster_map):
        return 0.80, "same_cluster"

    distance, relation = shortest_relation_distance(
        relation_graph,
        gold,
        pred,
        max_depth=2,
    )

    if distance == 1:
        return RELATION_WEIGHTS.get(relation, 0.70), f"direct_{relation}"

    if distance == 2:
        return 0.55, "relation_distance_2"

    return 0.0, "unrelated"


def main():
    pred_df = load_predictions()
    cluster_map = load_cluster_map()
    relation_graph = load_relation_graph()

    required = {"model", "condition", "gold_frame", "prediction"}
    missing = required - set(pred_df.columns)

    if missing:
        raise ValueError(f"Missing required prediction columns: {missing}")

    semantic_scores = []
    semantic_reasons = []
    cluster_correct = []
    hierarchy_correct = []

    for _, row in pred_df.iterrows():
        gold = clean_frame(row["gold_frame"])
        pred = clean_frame(row["prediction"])

        score, reason = semantic_score(row, cluster_map, relation_graph)

        semantic_scores.append(score)
        semantic_reasons.append(reason)
        cluster_correct.append(same_cluster(gold, pred, cluster_map))

        hierarchy_correct.append(
            score >= 0.55
        )

    pred_df["semantic_score"] = semantic_scores
    pred_df["semantic_reason"] = semantic_reasons
    pred_df["hierarchy_semantic_correct"] = hierarchy_correct
    pred_df["flat_cluster_correct"] = cluster_correct

    rows = []

    for (model, condition), group in pred_df.groupby(["model", "condition"]):
        exact_accuracy = accuracy_score(
            group["gold_frame"],
            group["prediction"],
        )

        unknown_rate = group["prediction"].eq("UNKNOWN").mean()

        cluster_accuracy = group["flat_cluster_correct"].mean()
        hierarchy_accuracy = group["hierarchy_semantic_correct"].mean()
        weighted_semantic_score = group["semantic_score"].mean()

        reason_counts = (
            group["semantic_reason"]
            .value_counts(normalize=True)
            .mul(100)
            .round(2)
            .to_dict()
        )

        rows.append({
            "model": model,
            "condition": condition,
            "exact_accuracy": exact_accuracy,
            "flat_cluster_accuracy": cluster_accuracy,
            "hierarchy_semantic_accuracy": hierarchy_accuracy,
            "weighted_semantic_score": weighted_semantic_score,
            "semantic_gain_over_exact": hierarchy_accuracy - exact_accuracy,
            "unknown_rate": unknown_rate,
            "n_examples": len(group),
            "semantic_reason_breakdown_percent": reason_counts,
        })

    results = pd.DataFrame(rows)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DETAILED_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    results.to_csv(OUTPUT_PATH, index=False)
    pred_df.to_csv(DETAILED_OUTPUT_PATH, index=False)

    print("\n=== Hierarchy-Aware Semantic Evaluation ===")
    print(results.to_string(index=False))

    print("\nSaved summary to:")
    print(OUTPUT_PATH)

    print("\nSaved detailed predictions to:")
    print(DETAILED_OUTPUT_PATH)


if __name__ == "__main__":
    main()