import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import pipeline


# =========================
# Paths
# =========================

DATA_PATH = Path("data/processed/framenet_processed.csv")
PROMPT_PATH = Path("prompts/classifier_reranked_qwen.txt")

# Existing evaluation scripts read this file.
OUTPUT_PATH = Path("data/outputs/all_model_predictions.csv")

# Backup copy for this specific experiment.
HYBRID_OUTPUT_PATH = Path(
    "data/outputs/hybrid_qwen_classifier_candidates_predictions.csv"
)

METRICS_PATH = Path(
    "results/hybrid_qwen_classifier_candidates_metrics.csv"
)

SPLIT_INFO_PATH = Path(
    "results/hybrid_qwen_classifier_candidates_split.json"
)


# =========================
# Config
# =========================

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

SLM_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
SLM_MODEL_LABEL = "qwen2_5_0_5b_classifier_candidates"

TEST_SIZE = 0.20
RANDOM_STATE = 42

FINAL_TOP_K = 5
EXAMPLES_PER_FRAME = 1

BATCH_SIZE = 1
MAX_NEW_TOKENS = 12


# =========================
# Helpers
# =========================

def normalize_lu(value):
    return str(value).strip().lower()


def clean_frame(value):
    return str(value).strip()


def safe_bool(value):
    if isinstance(value, bool):
        return value

    if pd.isna(value):
        return False

    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def make_classifier_input(row):
    return (
        f"Sentence: {row['sentence']}\n"
        f"Target lexical unit: {row['lexical_unit']}\n"
        f"Task: identify the FrameNet semantic frame."
    )


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

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def build_lu_to_frames(train_df):
    """
    Maps each lexical unit to frames seen with it in training.
    Example:
        buy.v -> Commerce_buy
        run.v -> Motion, Operating_a_system, etc.
    """
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


def train_candidate_classifier(train_df, embedder):
    """
    Trains a supervised embedding classifier.
    This classifier is NOT the final predictor.
    It only generates better candidate frames for Qwen.
    """
    train_texts = [
        make_classifier_input(row)
        for _, row in train_df.iterrows()
    ]

    print("Encoding classifier training examples...")

    X_train = embedder.encode(
        train_texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df["gold_frame"].astype(str))

    print("Training supervised candidate generator...")

    clf = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        solver="lbfgs",
        n_jobs=-1,
    )

    clf.fit(X_train, y_train)

    return clf, label_encoder


def get_classifier_candidates(
    row,
    row_embedding,
    clf,
    label_encoder,
    lu_to_frames,
):
    """
    Candidate generation logic:

    1. Include frames seen with the same lexical unit in training.
    2. Fill remaining slots using classifier probability ranking.
    3. Return top-k candidate frames for Qwen.

    Qwen still makes the final prediction.
    """
    class_labels = list(label_encoder.classes_)

    probabilities = clf.predict_proba(
        row_embedding.reshape(1, -1)
    )[0]

    ranked_indices = np.argsort(probabilities)[::-1]
    ranked_classifier_frames = [
        class_labels[i]
        for i in ranked_indices
    ]

    lu = normalize_lu(row["lexical_unit"])
    lu_candidates = lu_to_frames.get(lu, [])

    candidates = []
    candidate_sources = {}

    # First add LU-based candidates.
    for frame in lu_candidates:
        if frame not in candidates:
            candidates.append(frame)
            candidate_sources[frame] = "lexical_unit"

    # Then fill remaining slots using supervised classifier ranking.
    for frame in ranked_classifier_frames:
        if frame not in candidates:
            candidates.append(frame)
            candidate_sources[frame] = "classifier"

        if len(candidates) >= FINAL_TOP_K:
            break

    return candidates[:FINAL_TOP_K], candidate_sources


def build_example_texts(train_df):
    texts = []

    for _, row in train_df.iterrows():
        texts.append(
            f"Sentence: {row['sentence']} "
            f"Target lexical unit: {row['lexical_unit']} "
            f"Frame: {row['gold_frame']}"
        )

    return texts


def retrieve_examples_for_candidates(
    row,
    train_df,
    candidate_frames,
    example_embeddings,
    query_embedding,
):
    """
    Retrieves one example per candidate frame.
    These examples are given to Qwen as guidance.
    """
    similarities = cosine_similarity(
        query_embedding,
        example_embeddings,
    )[0]

    dynamic_examples = []

    for frame in candidate_frames:
        frame_indices = train_df.index[
            train_df["gold_frame"] == frame
        ].tolist()

        if not frame_indices:
            continue

        ranked = sorted(
            frame_indices,
            key=lambda idx: similarities[idx],
            reverse=True,
        )

        for idx in ranked[:EXAMPLES_PER_FRAME]:
            dynamic_examples.append(train_df.loc[idx])

    return dynamic_examples


def make_frame_list(candidate_frames):
    return "\n".join(
        [f"- {frame}" for frame in candidate_frames]
    )


def make_retrieved_examples_text(dynamic_examples):
    if not dynamic_examples:
        return "No retrieved examples available."

    blocks = []

    for ex in dynamic_examples:
        blocks.append(
            f"""Sentence: {ex['sentence']}
Target lexical unit: {ex['lexical_unit']}
Frame: {ex['gold_frame']}"""
        )

    return "\n\n".join(blocks)


def build_prompt(
    row,
    candidate_frames,
    dynamic_examples,
    prompt_template,
):
    return prompt_template.format(
        frame_list=make_frame_list(candidate_frames),
        retrieved_examples=make_retrieved_examples_text(dynamic_examples),
        sentence=row["sentence"],
        lexical_unit=row["lexical_unit"],
    )


def extract_generated_text(output):
    """
    Handles different Hugging Face pipeline output shapes.
    """
    if isinstance(output, list):
        if len(output) == 0:
            return ""

        if isinstance(output[0], dict):
            return output[0].get("generated_text", "")

        return str(output[0])

    if isinstance(output, dict):
        return output.get("generated_text", "")

    return str(output)


def clean_prediction(output, candidate_frames):
    """
    Forces the output to match one of the candidate frame names.
    If Qwen outputs something else, return UNKNOWN.
    """
    text = str(output).strip()
    text = text.replace("Frame:", "").strip()

    lower = text.lower()
    normalized = lower.replace(" ", "_")

    for frame in candidate_frames:
        if frame.lower() == lower:
            return frame

    for frame in candidate_frames:
        if frame.lower() == normalized:
            return frame

    for frame in candidate_frames:
        if frame.lower() in lower:
            return frame

    for frame in candidate_frames:
        if frame.lower().replace("_", " ") in lower:
            return frame

    return "UNKNOWN"


def load_qwen_generator():
    print(f"Loading SLM: {SLM_MODEL_NAME}")

    device = 0 if torch.cuda.is_available() else -1

    return pipeline(
        task="text-generation",
        model=SLM_MODEL_NAME,
        device=device,
        torch_dtype=torch.float32,
    )


def predict_with_qwen(generator, prompts, candidate_batches):
    generation_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "batch_size": BATCH_SIZE,
        "return_full_text": False,
    }

    with torch.inference_mode():
        outputs = generator(prompts, **generation_kwargs)

    predictions = []
    raw_outputs = []

    for output, candidates in zip(outputs, candidate_batches):
        generated = extract_generated_text(output)
        prediction = clean_prediction(generated, candidates)

        predictions.append(prediction)
        raw_outputs.append(generated)

    return predictions, raw_outputs


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

    unknown_rate = (
        pred_df["prediction"] == "UNKNOWN"
    ).mean()

    return pd.DataFrame([{
        "model": SLM_MODEL_LABEL,
        "condition": "classifier_candidates_dynamic_fewshot",
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "unknown_rate": unknown_rate,
        "n_examples": len(pred_df),
        "gold_in_candidates": pred_df["gold_in_candidates"].mean(),
    }])


# =========================
# Main
# =========================

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {DATA_PATH}")

    if not PROMPT_PATH.exists():
        raise FileNotFoundError(f"Missing prompt file: {PROMPT_PATH}")

    prompt_template = PROMPT_PATH.read_text(encoding="utf-8")

    df = pd.read_csv(DATA_PATH)
    df = df.dropna(
        subset=["sentence", "lexical_unit", "gold_frame"]
    ).copy()

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

    clf, label_encoder = train_candidate_classifier(train_df, embedder)
    lu_to_frames = build_lu_to_frames(train_df)

    print("\nEncoding training examples for dynamic retrieval...")

    example_texts = build_example_texts(train_df)
    example_embeddings = embedder.encode(
        example_texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    print("\nEncoding test examples...")

    test_texts = [
        make_classifier_input(row)
        for _, row in test_df.iterrows()
    ]

    X_test = embedder.encode(
        test_texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    print("\nBuilding Qwen prompts...")

    prompts = []
    candidate_batches = []
    row_records = []
    retrieved_example_frames_records = []
    candidate_source_records = []

    for idx, row in tqdm(
        list(test_df.iterrows()),
        desc="Preparing hybrid prompts",
    ):
        row_embedding = X_test[idx]

        candidate_frames, candidate_sources = get_classifier_candidates(
            row=row,
            row_embedding=row_embedding,
            clf=clf,
            label_encoder=label_encoder,
            lu_to_frames=lu_to_frames,
        )

        dynamic_examples = retrieve_examples_for_candidates(
            row=row,
            train_df=train_df,
            candidate_frames=candidate_frames,
            example_embeddings=example_embeddings,
            query_embedding=row_embedding.reshape(1, -1),
        )

        prompt = build_prompt(
            row=row,
            candidate_frames=candidate_frames,
            dynamic_examples=dynamic_examples,
            prompt_template=prompt_template,
        )

        prompts.append(prompt)
        candidate_batches.append(candidate_frames)
        row_records.append(row)

        retrieved_example_frames_records.append(
            "|".join(
                [ex["gold_frame"] for ex in dynamic_examples]
            )
        )

        candidate_source_records.append(
            json.dumps(candidate_sources)
        )

    candidate_recall = np.mean([
        clean_frame(row["gold_frame"]) in candidates
        for row, candidates in zip(row_records, candidate_batches)
    ])

    print(f"\nCandidate recall: {candidate_recall:.4f}")

    generator = load_qwen_generator()

    results = []

    print("\nRunning Qwen final frame prediction...")

    for start in tqdm(
        range(0, len(prompts), BATCH_SIZE),
        desc="Qwen prediction",
    ):
        batch_prompts = prompts[start:start + BATCH_SIZE]
        batch_candidates = candidate_batches[start:start + BATCH_SIZE]
        batch_rows = row_records[start:start + BATCH_SIZE]
        batch_retrieved_frames = retrieved_example_frames_records[
            start:start + BATCH_SIZE
        ]
        batch_candidate_sources = candidate_source_records[
            start:start + BATCH_SIZE
        ]

        predictions, raw_outputs = predict_with_qwen(
            generator=generator,
            prompts=batch_prompts,
            candidate_batches=batch_candidates,
        )

        for (
            row,
            prediction,
            raw_output,
            candidates,
            retrieved_frames,
            candidate_sources,
        ) in zip(
            batch_rows,
            predictions,
            raw_outputs,
            batch_candidates,
            batch_retrieved_frames,
            batch_candidate_sources,
        ):
            gold_frame = clean_frame(row["gold_frame"])

            results.append({
                "model": SLM_MODEL_LABEL,
                "model_type": "small_language_model_with_supervised_candidates",
                "condition": "classifier_candidates_dynamic_fewshot",

                "sentence": row["sentence"],
                "lexical_unit": row["lexical_unit"],
                "gold_frame": gold_frame,
                "prediction": prediction,
                "raw_output": raw_output,

                "candidate_frames": "|".join(candidates),
                "gold_in_candidates": gold_frame in candidates,
                "candidate_sources": candidate_sources,
                "retrieved_example_frames": retrieved_frames,

                "domain": row.get("domain", "unknown"),
                "sentence_length": row.get("sentence_length", None),
                "is_long_context": row.get("is_long_context", False),
                "is_ambiguous_lu": row.get("is_ambiguous_lu", False),
                "hard_negative_frames": row.get("hard_negative_frames", ""),
                "frame_definition": row.get("frame_definition", ""),
            })

        # Save partial progress after each batch.
        partial_df = pd.DataFrame(results)
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

        partial_df.to_csv(OUTPUT_PATH, index=False)
        partial_df.to_csv(HYBRID_OUTPUT_PATH, index=False)

    pred_df = pd.DataFrame(results)
    metrics_df = evaluate_quick(pred_df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    HYBRID_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_INFO_PATH.parent.mkdir(parents=True, exist_ok=True)

    pred_df.to_csv(OUTPUT_PATH, index=False)
    pred_df.to_csv(HYBRID_OUTPUT_PATH, index=False)
    metrics_df.to_csv(METRICS_PATH, index=False)

    split_info = {
        "data_path": str(DATA_PATH),
        "embedding_model": EMBEDDING_MODEL,
        "slm_model": SLM_MODEL_NAME,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "train_examples": len(train_df),
        "test_examples": len(test_df),
        "train_frames": int(train_df["gold_frame"].nunique()),
        "test_frames": int(test_df["gold_frame"].nunique()),
        "final_top_k": FINAL_TOP_K,
        "examples_per_frame": EXAMPLES_PER_FRAME,
        "candidate_recall": float(candidate_recall),
    }

    SPLIT_INFO_PATH.write_text(
        json.dumps(split_info, indent=2),
        encoding="utf-8",
    )

    print("\n=== Hybrid SLM Results ===")
    print(metrics_df.to_string(index=False))

    print("\nSaved active predictions to:")
    print(OUTPUT_PATH)

    print("\nAlso saved:")
    print(HYBRID_OUTPUT_PATH)
    print(METRICS_PATH)
    print(SPLIT_INFO_PATH)


if __name__ == "__main__":
    main()