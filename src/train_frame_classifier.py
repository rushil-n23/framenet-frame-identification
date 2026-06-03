import json
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


DATA_PATH = Path("data/processed/framenet_processed.csv")

# This output path lets your existing evaluate.py work without changes.
OUTPUT_PATH = Path("data/outputs/all_model_predictions.csv")

# Extra files for analysis.
SUPERVISED_OUTPUT_PATH = Path("data/outputs/supervised_frame_classifier_predictions.csv")
SPLIT_INFO_PATH = Path("results/supervised_split_info.json")
METRICS_PATH = Path("results/supervised_classifier_metrics.csv")

TEST_SIZE = 0.20
RANDOM_STATE = 42

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# If True, this adds a very strong lexical-unit shortcut:
# if a lexical unit has only one frame in training, directly predict that frame.
USE_LU_SINGLETON_RULE = True

# If True, ambiguous LUs are constrained to frames seen with that LU in training.
USE_LU_CANDIDATE_CONSTRAINT = True

# Fallback retrieval candidates when lexical unit is unseen.
FALLBACK_TOP_K = 15


def normalize_text(value):
    return str(value).strip()


def normalize_lu(value):
    return str(value).strip().lower()


def clean_frame(frame):
    return str(frame).strip()


def make_input_text(row):
    sentence = normalize_text(row["sentence"])
    lexical_unit = normalize_text(row["lexical_unit"])

    return (
        f"Sentence: {sentence}\n"
        f"Target lexical unit: {lexical_unit}\n"
        f"Task: identify the FrameNet semantic frame."
    )


def safe_bool(value):
    if isinstance(value, bool):
        return value

    if pd.isna(value):
        return False

    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def can_stratify(labels):
    counts = pd.Series(labels).value_counts()
    return counts.min() >= 2


def split_dataset(df):
    labels = df["gold_frame"].astype(str)

    stratify_labels = labels if can_stratify(labels) else None

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=stratify_labels,
    )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, test_df


def build_lu_to_frames(train_df):
    lu_to_frames = {}

    for _, row in train_df.iterrows():
        lu = normalize_lu(row["lexical_unit"])
        frame = clean_frame(row["gold_frame"])

        if lu not in lu_to_frames:
            lu_to_frames[lu] = set()

        lu_to_frames[lu].add(frame)

    return {
        lu: sorted(list(frames))
        for lu, frames in lu_to_frames.items()
    }


def build_frame_texts(train_df, all_frames):
    frame_texts = []

    for frame in all_frames:
        rows = train_df[train_df["gold_frame"] == frame]

        examples = (
            rows["sentence"]
            .dropna()
            .astype(str)
            .head(3)
            .tolist()
        )

        lexical_units = (
            rows["lexical_unit"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        if "frame_definition" in rows.columns:
            definitions = (
                rows["frame_definition"]
                .dropna()
                .astype(str)
                .unique()
            )
            definition = definitions[0] if len(definitions) > 0 else ""
        else:
            definition = ""

        text = (
            f"Frame: {frame.replace('_', ' ')}. "
            f"Definition: {definition}. "
            f"Lexical units: {', '.join(lexical_units[:10])}. "
            f"Examples: {' '.join(examples)}"
        )

        frame_texts.append(text)

    return frame_texts


def train_classifier(train_df, embedder):
    print("Encoding training examples...")
    train_texts = [make_input_text(row) for _, row in train_df.iterrows()]

    X_train = embedder.encode(
        train_texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df["gold_frame"].astype(str))

    print("Training LogisticRegression classifier...")

    clf = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        solver="lbfgs",
        n_jobs=-1,
    )

    clf.fit(X_train, y_train)

    return clf, label_encoder


def predict_with_lu_constrained_classifier(
    row,
    clf,
    label_encoder,
    row_embedding,
    lu_to_frames,
    frame_embeddings,
    frame_names,
):
    lu = normalize_lu(row["lexical_unit"])
    lu_candidates = lu_to_frames.get(lu, [])

    class_labels = list(label_encoder.classes_)

    # Very strong rule:
    # If this lexical unit only appeared with one frame in the training data,
    # use that frame directly.
    if USE_LU_SINGLETON_RULE and len(lu_candidates) == 1:
        return lu_candidates[0], lu_candidates, "lu_singleton_rule"

    probabilities = clf.predict_proba(row_embedding.reshape(1, -1))[0]

    # If LU has multiple candidate frames, choose the highest-probability frame
    # among only those candidate frames.
    if USE_LU_CANDIDATE_CONSTRAINT and len(lu_candidates) > 1:
        candidate_indices = [
            i for i, frame in enumerate(class_labels)
            if frame in lu_candidates
        ]

        if candidate_indices:
            best_index = max(candidate_indices, key=lambda i: probabilities[i])
            return class_labels[best_index], lu_candidates, "lu_constrained_classifier"

    # If the lexical unit was unseen or unusable, fallback to global classifier.
    best_global_index = int(np.argmax(probabilities))
    global_prediction = class_labels[best_global_index]

    # Add an embedding fallback candidate list for analysis.
    sims = cosine_similarity(
        row_embedding.reshape(1, -1),
        frame_embeddings,
    )[0]

    top_indices = sims.argsort()[::-1][:FALLBACK_TOP_K]
    fallback_candidates = [frame_names[i] for i in top_indices]

    return global_prediction, fallback_candidates, "global_classifier"


def run_predictions(
    test_df,
    clf,
    label_encoder,
    embedder,
    lu_to_frames,
    frame_embeddings,
    frame_names,
):
    print("Encoding test examples...")
    test_texts = [make_input_text(row) for _, row in test_df.iterrows()]

    X_test = embedder.encode(
        test_texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    results = []

    for idx, row in tqdm(
        list(test_df.iterrows()),
        desc="Predicting test examples",
    ):
        row_embedding = X_test[idx]

        prediction, candidate_frames, decision_source = (
            predict_with_lu_constrained_classifier(
                row=row,
                clf=clf,
                label_encoder=label_encoder,
                row_embedding=row_embedding,
                lu_to_frames=lu_to_frames,
                frame_embeddings=frame_embeddings,
                frame_names=frame_names,
            )
        )

        gold_frame = clean_frame(row["gold_frame"])

        results.append({
            "model": "supervised_lu_embedding_classifier",
            "model_type": "supervised_embedding_classifier",
            "condition": "lu_constrained_logreg",

            "sentence": row["sentence"],
            "lexical_unit": row["lexical_unit"],
            "gold_frame": gold_frame,
            "prediction": prediction,

            "candidate_frames": "|".join(candidate_frames),
            "gold_in_candidates": gold_frame in candidate_frames,
            "decision_source": decision_source,

            "domain": row.get("domain", "unknown"),
            "sentence_length": row.get("sentence_length", None),
            "is_long_context": row.get("is_long_context", False),
            "is_ambiguous_lu": row.get("is_ambiguous_lu", False),
            "hard_negative_frames": row.get("hard_negative_frames", ""),
            "frame_definition": row.get("frame_definition", ""),
        })

    return pd.DataFrame(results)


def evaluate_quick(pred_df):
    y_true = pred_df["gold_frame"]
    y_pred = pred_df["prediction"]

    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    unknown_rate = (pred_df["prediction"] == "UNKNOWN").mean()

    metrics = pd.DataFrame([{
        "model": "supervised_lu_embedding_classifier",
        "condition": "lu_constrained_logreg",
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "unknown_rate": unknown_rate,
        "n_examples": len(pred_df),
        "gold_in_candidates": pred_df["gold_in_candidates"].mean(),
        "lu_singleton_rule_rate": (
            pred_df["decision_source"] == "lu_singleton_rule"
        ).mean(),
        "lu_constrained_classifier_rate": (
            pred_df["decision_source"] == "lu_constrained_classifier"
        ).mean(),
        "global_classifier_rate": (
            pred_df["decision_source"] == "global_classifier"
        ).mean(),
    }])

    return metrics


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    required_columns = {"sentence", "lexical_unit", "gold_frame"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=["sentence", "lexical_unit", "gold_frame"])
    df["gold_frame"] = df["gold_frame"].astype(str)

    if "is_ambiguous_lu" in df.columns:
        df["is_ambiguous_lu"] = df["is_ambiguous_lu"].apply(safe_bool)

    if "is_long_context" in df.columns:
        df["is_long_context"] = df["is_long_context"].apply(safe_bool)

    print(f"Loaded {len(df)} examples.")
    print(f"Detected {df['gold_frame'].nunique()} frames.")
    print(f"Detected {df['lexical_unit'].nunique()} lexical units.")

    train_df, test_df = split_dataset(df)

    print(f"Train examples: {len(train_df)}")
    print(f"Test examples: {len(test_df)}")
    print(f"Train frames: {train_df['gold_frame'].nunique()}")
    print(f"Test frames: {test_df['gold_frame'].nunique()}")

    print("\nLoading embedding model...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    clf, label_encoder = train_classifier(train_df, embedder)

    lu_to_frames = build_lu_to_frames(train_df)

    frame_names = sorted(train_df["gold_frame"].unique().tolist())
    frame_texts = build_frame_texts(train_df, frame_names)

    print("Encoding frame metadata for fallback retrieval...")
    frame_embeddings = embedder.encode(
        frame_texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    pred_df = run_predictions(
        test_df=test_df,
        clf=clf,
        label_encoder=label_encoder,
        embedder=embedder,
        lu_to_frames=lu_to_frames,
        frame_embeddings=frame_embeddings,
        frame_names=frame_names,
    )

    metrics_df = evaluate_quick(pred_df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUPERVISED_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_INFO_PATH.parent.mkdir(parents=True, exist_ok=True)

    pred_df.to_csv(OUTPUT_PATH, index=False)
    pred_df.to_csv(SUPERVISED_OUTPUT_PATH, index=False)
    metrics_df.to_csv(METRICS_PATH, index=False)

    split_info = {
        "data_path": str(DATA_PATH),
        "embedding_model": EMBEDDING_MODEL,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "train_examples": len(train_df),
        "test_examples": len(test_df),
        "train_frames": int(train_df["gold_frame"].nunique()),
        "test_frames": int(test_df["gold_frame"].nunique()),
        "use_lu_singleton_rule": USE_LU_SINGLETON_RULE,
        "use_lu_candidate_constraint": USE_LU_CANDIDATE_CONSTRAINT,
        "fallback_top_k": FALLBACK_TOP_K,
    }

    SPLIT_INFO_PATH.write_text(
        json.dumps(split_info, indent=2),
        encoding="utf-8",
    )

    print("\n=== Supervised Classifier Results ===")
    print(metrics_df.to_string(index=False))

    print("\nDecision source breakdown:")
    print(
        pred_df["decision_source"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
        .astype(str)
        + "%"
    )

    print("\nSaved predictions to:")
    print(OUTPUT_PATH)

    print("\nAlso saved:")
    print(SUPERVISED_OUTPUT_PATH)
    print(METRICS_PATH)
    print(SPLIT_INFO_PATH)


if __name__ == "__main__":
    main()