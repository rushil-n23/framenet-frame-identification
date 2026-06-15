import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline,
)


# =========================
# Paths
# =========================

DATA_PATH = Path("data/processed/framenet_processed.csv")
OUTPUT_DIR = Path("data/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_PATH = OUTPUT_DIR / "all_model_predictions.csv"


# =========================
# Experiment settings
# =========================

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

TOP_K = 15
DYNAMIC_EXAMPLES = 3
FEW_SHOT_EXAMPLES = 3
ONE_SHOT_EXAMPLES = 1

MAX_NEW_TOKENS = 16
BATCH_SIZE = 1

DEVICE = 0 if torch.cuda.is_available() else -1
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32


# =========================
# Models
# =========================

MODEL_CONFIGS = {
    # FLAN / Google
    "flan-t5-large": {
        "hf_name": "google/flan-t5-large",
        "type": "seq2seq",
    },
    "flan-t5-xl": {
        "hf_name": "google/flan-t5-xl",
        "type": "seq2seq",
    },

    # Qwen
    "qwen-0.5b": {
        "hf_name": "Qwen/Qwen2.5-0.5B-Instruct",
        "type": "causal",
        "chat": True,
    },
    "qwen-1.5b": {
        "hf_name": "Qwen/Qwen2.5-1.5B-Instruct",
        "type": "causal",
        "chat": True,
    },
    "qwen-3b": {
        "hf_name": "Qwen/Qwen2.5-3B-Instruct",
        "type": "causal",
        "chat": True,
    },

    # Microsoft Phi
    "phi-3.5-mini": {
        "hf_name": "microsoft/Phi-3.5-mini-instruct",
        "type": "causal",
        "chat": True,
        "trust_remote_code": True,
    },

    # SmolLM
    "smollm2-1.7b": {
        "hf_name": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "type": "causal",
        "chat": True,
    },

    # GPT baselines
    "gpt2-large": {
        "hf_name": "gpt2-large",
        "type": "causal",
        "chat": False,
    },

    # TinyLlama baseline
    "tinyllama-1.1b": {
        "hf_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "type": "causal",
        "chat": True,
    },
}


PROMPTING_STRATEGIES = [
    "zero_shot",
    "one_shot",
    "few_shot",
    "guided",
    "dynamic_few_shot",
    "dynamic_lu_rerank",
]


# =========================
# Utilities
# =========================

def clean_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def normalize_frame_name(x):
    return clean_text(x).replace(" ", "_")


def get_lexical_unit(row):
    for col in ["lexical_unit", "target_word", "lu", "target"]:
        if col in row and clean_text(row[col]):
            return clean_text(row[col])
    return ""


def get_sentence(row):
    for col in ["sentence", "text", "example"]:
        if col in row and clean_text(row[col]):
            return clean_text(row[col])
    raise ValueError("No sentence/text column found.")


def get_gold_frame(row):
    for col in ["gold_frame", "frame", "intended_frame"]:
        if col in row and clean_text(row[col]):
            return normalize_frame_name(row[col])
    raise ValueError("No gold_frame/frame/intended_frame column found.")


def safe_lower(x):
    return str(x).lower().strip()


def exact_frame_match(text, valid_frames):
    """
    Extract the predicted frame from raw model output.
    Tries exact match, substring match, and simple cleanup.
    """
    if text is None:
        return "UNKNOWN"

    raw = str(text).strip()
    if not raw:
        return "UNKNOWN"

    # Remove common wrappers
    cleaned = raw
    cleaned = cleaned.replace("Frame:", "")
    cleaned = cleaned.replace("Answer:", "")
    cleaned = cleaned.replace("Prediction:", "")
    cleaned = cleaned.strip()

    # Take first line only
    cleaned = cleaned.split("\n")[0].strip()

    # Remove punctuation around label
    cleaned = re.sub(r"[^A-Za-z0-9_\- ]", "", cleaned).strip()
    cleaned_norm = cleaned.replace(" ", "_")

    valid_set = set(valid_frames)

    if cleaned_norm in valid_set:
        return cleaned_norm

    # Sometimes model writes a sentence containing the frame
    raw_norm = raw.replace(" ", "_")
    for frame in valid_frames:
        pattern = r"\b" + re.escape(frame) + r"\b"
        if re.search(pattern, raw_norm):
            return frame

    # Case-insensitive match
    lower_to_frame = {f.lower(): f for f in valid_frames}
    if cleaned_norm.lower() in lower_to_frame:
        return lower_to_frame[cleaned_norm.lower()]

    for frame in valid_frames:
        if frame.lower() in raw_norm.lower():
            return frame

    return "UNKNOWN"


# =========================
# Data loading
# =========================

def load_dataset(sample_size=None):
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    df["sentence"] = df.apply(get_sentence, axis=1)
    df["gold_frame"] = df.apply(get_gold_frame, axis=1)
    df["lexical_unit"] = df.apply(get_lexical_unit, axis=1)

    if sample_size is not None:
        df = df.head(sample_size).copy()

    return df


def build_frame_inventory(df):
    frames = sorted(df["gold_frame"].dropna().astype(str).unique().tolist())

    frame_texts = {}

    for frame in frames:
        rows = df[df["gold_frame"] == frame]

        examples = rows["sentence"].dropna().astype(str).head(5).tolist()
        lexical_units = rows["lexical_unit"].dropna().astype(str).unique().tolist()

        definition = ""
        if "frame_definition" in rows.columns:
            defs = rows["frame_definition"].dropna().astype(str).unique().tolist()
            if defs:
                definition = defs[0]

        frame_text = (
            f"Frame: {frame.replace('_', ' ')}. "
            f"Definition: {definition}. "
            f"Lexical units: {', '.join(lexical_units[:15])}. "
            f"Examples: {' '.join(examples[:5])}"
        )

        frame_texts[frame] = frame_text

    return frames, frame_texts


# =========================
# Retrieval
# =========================

def encode_inventory(embedder, frames, frame_texts):
    texts = [frame_texts[f] for f in frames]
    embeddings = embedder.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return embeddings


def retrieve_candidate_frames(sentence, lexical_unit, frames, frame_embeddings, embedder, top_k=TOP_K):
    query = f"Lexical unit: {lexical_unit}. Sentence: {sentence}"
    query_emb = embedder.encode([query], normalize_embeddings=True)

    sims = cosine_similarity(query_emb, frame_embeddings)[0]
    top_indices = np.argsort(sims)[::-1][:top_k]

    return [frames[i] for i in top_indices]


def retrieve_candidate_frames_lu_rerank(
    sentence,
    lexical_unit,
    frames,
    frame_texts,
    frame_embeddings,
    embedder,
    top_k=TOP_K,
):
    query = f"Lexical unit: {lexical_unit}. Sentence: {sentence}"
    query_emb = embedder.encode([query], normalize_embeddings=True)

    sims = cosine_similarity(query_emb, frame_embeddings)[0]

    lu = safe_lower(lexical_unit)
    boosted_scores = []

    for i, frame in enumerate(frames):
        score = sims[i]
        frame_text = safe_lower(frame_texts[frame])

        # small lexical-unit boost
        if lu and lu in frame_text:
            score += 0.08

        boosted_scores.append(score)

    top_indices = np.argsort(boosted_scores)[::-1][:top_k]
    return [frames[i] for i in top_indices]


def retrieve_dynamic_examples(row_idx, df, sentence_embeddings, top_n=DYNAMIC_EXAMPLES):
    sims = cosine_similarity(
        sentence_embeddings[row_idx].reshape(1, -1),
        sentence_embeddings,
    )[0]

    sims[row_idx] = -1

    top_indices = np.argsort(sims)[::-1][:top_n]

    examples = []
    for idx in top_indices:
        ex = df.iloc[idx]
        examples.append({
            "sentence": ex["sentence"],
            "lexical_unit": ex["lexical_unit"],
            "gold_frame": ex["gold_frame"],
        })

    return examples


def get_static_examples(df, n):
    examples = []

    used_frames = set()

    for _, row in df.iterrows():
        frame = row["gold_frame"]

        if frame in used_frames:
            continue

        examples.append({
            "sentence": row["sentence"],
            "lexical_unit": row["lexical_unit"],
            "gold_frame": row["gold_frame"],
        })

        used_frames.add(frame)

        if len(examples) >= n:
            break

    return examples


# =========================
# Prompt construction
# =========================

def format_examples(examples):
    if not examples:
        return ""

    lines = []

    for i, ex in enumerate(examples, start=1):
        lines.append(
            f"Example {i}:\n"
            f"Sentence: {ex['sentence']}\n"
            f"Lexical unit: {ex['lexical_unit']}\n"
            f"Frame: {ex['gold_frame']}\n"
        )

    return "\n".join(lines)


def format_candidates(candidates):
    return "\n".join([f"- {c}" for c in candidates])


def build_prompt(
    strategy,
    sentence,
    lexical_unit,
    valid_frames,
    candidates=None,
    examples=None,
):
    all_frames_text = ", ".join(valid_frames)

    base_instruction = (
        "You are performing FrameNet semantic frame identification.\n"
        "Return exactly one frame label and nothing else.\n"
        "Do not explain your answer.\n"
        "If unsure, choose the closest valid frame label.\n"
    )

    input_block = (
        f"Sentence: {sentence}\n"
        f"Lexical unit: {lexical_unit}\n"
    )

    if strategy == "zero_shot":
        return (
            f"{base_instruction}\n"
            f"Valid frame labels:\n{all_frames_text}\n\n"
            f"{input_block}\n"
            f"Frame:"
        )

    if strategy == "one_shot":
        return (
            f"{base_instruction}\n"
            f"Valid frame labels:\n{all_frames_text}\n\n"
            f"{format_examples(examples)}\n"
            f"Now identify the frame for this input.\n"
            f"{input_block}\n"
            f"Frame:"
        )

    if strategy == "few_shot":
        return (
            f"{base_instruction}\n"
            f"Valid frame labels:\n{all_frames_text}\n\n"
            f"{format_examples(examples)}\n"
            f"Now identify the frame for this input.\n"
            f"{input_block}\n"
            f"Frame:"
        )

    if strategy == "guided":
        return (
            f"{base_instruction}\n"
            f"Choose only from the candidate frames below.\n\n"
            f"Candidate frames:\n{format_candidates(candidates)}\n\n"
            f"{input_block}\n"
            f"Frame:"
        )

    if strategy == "dynamic_few_shot":
        return (
            f"{base_instruction}\n"
            f"Use the examples to understand the task, then choose only from the candidate frames.\n\n"
            f"{format_examples(examples)}\n"
            f"Candidate frames:\n{format_candidates(candidates)}\n\n"
            f"{input_block}\n"
            f"Frame:"
        )

    if strategy == "dynamic_lu_rerank":
        return (
            f"{base_instruction}\n"
            f"The candidate frames were retrieved using semantic similarity and lexical-unit reranking.\n"
            f"Use the examples to understand the task, then choose only from the candidate frames.\n\n"
            f"{format_examples(examples)}\n"
            f"Candidate frames:\n{format_candidates(candidates)}\n\n"
            f"{input_block}\n"
            f"Frame:"
        )

    raise ValueError(f"Unknown strategy: {strategy}")


# =========================
# Model loading/generation
# =========================

def load_generation_pipeline(model_key):
    config = MODEL_CONFIGS[model_key]
    hf_name = config["hf_name"]
    model_type = config["type"]
    trust_remote_code = config.get("trust_remote_code", False)

    print(f"\nLoading model: {model_key} -> {hf_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        hf_name,
        trust_remote_code=trust_remote_code,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if model_type == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(
            hf_name,
            torch_dtype=TORCH_DTYPE,
            trust_remote_code=trust_remote_code,
            device_map="auto" if torch.cuda.is_available() else None,
        )

        gen_pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            device=DEVICE if not torch.cuda.is_available() else None,
        )

    elif model_type == "causal":
        model = AutoModelForCausalLM.from_pretrained(
            hf_name,
            torch_dtype=TORCH_DTYPE,
            trust_remote_code=trust_remote_code,
            device_map="auto" if torch.cuda.is_available() else None,
        )

        gen_pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=DEVICE if not torch.cuda.is_available() else None,
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return gen_pipe, tokenizer, config


def apply_chat_template_if_needed(prompt, tokenizer, config):
    if not config.get("chat", False):
        return prompt

    messages = [
        {
            "role": "system",
            "content": "You are a precise FrameNet frame identification system.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return prompt


def generate_prediction(gen_pipe, tokenizer, config, prompt):
    final_prompt = apply_chat_template_if_needed(prompt, tokenizer, config)

    model_type = config["type"]

    if model_type == "seq2seq":
        output = gen_pipe(
            final_prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            num_beams=1,
        )
        return output[0]["generated_text"]

    output = gen_pipe(
        final_prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        return_full_text=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    return output[0]["generated_text"]


# =========================
# Running experiments
# =========================

def run_single_experiment(
    df,
    model_key,
    strategy,
    frames,
    frame_texts,
    frame_embeddings,
    sentence_embeddings,
    embedder,
    resume=True,
):
    existing = None

    if resume and PREDICTIONS_PATH.exists():
        existing = pd.read_csv(PREDICTIONS_PATH)
        existing = existing[
            (existing["model_key"] == model_key)
            & (existing["prompting_strategy"] == strategy)
        ]

    done_ids = set()
    if existing is not None and "row_id" in existing.columns:
        done_ids = set(existing["row_id"].astype(int).tolist())

    gen_pipe, tokenizer, config = load_generation_pipeline(model_key)

    static_one = get_static_examples(df, ONE_SHOT_EXAMPLES)
    static_few = get_static_examples(df, FEW_SHOT_EXAMPLES)

    rows = []

    total = len(df)
    start_time = time.time()

    print(f"\nRunning: model={model_key}, strategy={strategy}, total={total}")

    for row_idx, row in df.iterrows():
        if int(row_idx) in done_ids:
            continue

        sentence = row["sentence"]
        lexical_unit = row["lexical_unit"]
        gold_frame = row["gold_frame"]

        candidates = None
        examples = None

        if strategy in ["guided", "dynamic_few_shot"]:
            candidates = retrieve_candidate_frames(
                sentence=sentence,
                lexical_unit=lexical_unit,
                frames=frames,
                frame_embeddings=frame_embeddings,
                embedder=embedder,
                top_k=TOP_K,
            )

        elif strategy == "dynamic_lu_rerank":
            candidates = retrieve_candidate_frames_lu_rerank(
                sentence=sentence,
                lexical_unit=lexical_unit,
                frames=frames,
                frame_texts=frame_texts,
                frame_embeddings=frame_embeddings,
                embedder=embedder,
                top_k=TOP_K,
            )

        if strategy == "one_shot":
            examples = static_one

        elif strategy == "few_shot":
            examples = static_few

        elif strategy in ["dynamic_few_shot", "dynamic_lu_rerank"]:
            examples = retrieve_dynamic_examples(
                row_idx=row_idx,
                df=df,
                sentence_embeddings=sentence_embeddings,
                top_n=DYNAMIC_EXAMPLES,
            )

        prompt = build_prompt(
            strategy=strategy,
            sentence=sentence,
            lexical_unit=lexical_unit,
            valid_frames=frames,
            candidates=candidates,
            examples=examples,
        )

        try:
            raw_output = generate_prediction(
                gen_pipe=gen_pipe,
                tokenizer=tokenizer,
                config=config,
                prompt=prompt,
            )
        except Exception as e:
            print(f"Generation error at row {row_idx}: {e}")
            raw_output = ""

        prediction = exact_frame_match(raw_output, frames)

        gold_in_candidates = None
        if candidates is not None:
            gold_in_candidates = gold_frame in candidates

        result = {
            "row_id": int(row_idx),
            "model_key": model_key,
            "model_name": MODEL_CONFIGS[model_key]["hf_name"],
            "prompting_strategy": strategy,
            "sentence": sentence,
            "lexical_unit": lexical_unit,
            "gold_frame": gold_frame,
            "prediction": prediction,
            "raw_output": raw_output,
            "exact_match": prediction == gold_frame,
            "gold_in_candidates": gold_in_candidates,
            "top_k": TOP_K if candidates is not None else None,
            "candidate_frames": json.dumps(candidates) if candidates is not None else None,
        }

        # carry over useful metadata if present
        for col in [
            "domain",
            "is_ambiguous_lu",
            "is_ambiguous",
            "is_long_context",
            "hard_negative_frames",
            "sentence_length",
        ]:
            if col in df.columns:
                result[col] = row[col]

        rows.append(result)

        if len(rows) % 10 == 0:
            save_partial(rows)
            rows = []

        if (row_idx + 1) % 25 == 0:
            elapsed = time.time() - start_time
            print(
                f"[{model_key} | {strategy}] "
                f"{row_idx + 1}/{total} done | elapsed {elapsed / 60:.1f} min"
            )

    if rows:
        save_partial(rows)

    print(f"Finished model={model_key}, strategy={strategy}")


def save_partial(rows):
    out_df = pd.DataFrame(rows)

    if PREDICTIONS_PATH.exists():
        existing = pd.read_csv(PREDICTIONS_PATH)
        combined = pd.concat([existing, out_df], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["row_id", "model_key", "prompting_strategy"],
            keep="last",
        )
    else:
        combined = out_df

    combined.to_csv(PREDICTIONS_PATH, index=False)


# =========================
# Summary
# =========================

def summarize_results():
    if not PREDICTIONS_PATH.exists():
        print("No predictions found.")
        return

    df = pd.read_csv(PREDICTIONS_PATH)

    summary = (
        df.groupby(["model_key", "prompting_strategy"])
        .agg(
            n=("exact_match", "count"),
            exact_accuracy=("exact_match", "mean"),
            unknown_rate=("prediction", lambda x: (x == "UNKNOWN").mean()),
            hits_at_k=("gold_in_candidates", lambda x: x.dropna().mean() if x.notna().any() else np.nan),
        )
        .reset_index()
    )

    summary["exact_accuracy"] *= 100
    summary["unknown_rate"] *= 100
    summary["hits_at_k"] *= 100

    summary_path = OUTPUT_DIR / "prompting_strategy_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n=== Prompting Strategy Summary ===")
    print(summary.to_string(index=False))
    print(f"\nSaved summary to {summary_path}")


# =========================
# Main
# =========================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--models",
        nargs="+",
        default=["qwen-0.5b"],
        choices=list(MODEL_CONFIGS.keys()),
        help="Model keys to run.",
    )

    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["guided", "dynamic_few_shot"],
        choices=PROMPTING_STRATEGIES,
        help="Prompting strategies to run.",
    )

    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Use only first N examples for quick testing.",
    )

    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Do not skip already completed model/strategy/row combinations.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("Loading dataset...")
    df = load_dataset(sample_size=args.sample_size)

    print(f"Loaded {len(df)} examples.")

    print("Building frame inventory...")
    frames, frame_texts = build_frame_inventory(df)

    print(f"Loaded {len(frames)} unique frames.")

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("Encoding frame inventory...")
    frame_embeddings = encode_inventory(
        embedder=embedder,
        frames=frames,
        frame_texts=frame_texts,
    )

    print("Encoding sentences for dynamic few-shot retrieval...")
    sentence_embeddings = embedder.encode(
        df["sentence"].tolist(),
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    for model_key in args.models:
        for strategy in args.strategies:
            run_single_experiment(
                df=df,
                model_key=model_key,
                strategy=strategy,
                frames=frames,
                frame_texts=frame_texts,
                frame_embeddings=frame_embeddings,
                sentence_embeddings=sentence_embeddings,
                embedder=embedder,
                resume=not args.no_resume,
            )

    summarize_results()


if __name__ == "__main__":
    main()