import pandas as pd
from pathlib import Path
from tqdm import tqdm

import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


DATA_PATH = Path("data/processed/framenet_processed.csv")
OUTPUT_PATH = Path("data/outputs/all_model_predictions.csv")

# None = full dataset.
MAX_EXAMPLES = None

TOP_K = 15
DYNAMIC_EXAMPLES = 3
BATCH_SIZE = 1

MODELS = {
    "flan_t5_large": {
        "model_name": "google/flan-t5-large",
        "task": "text2text-generation",
        "type": "encoder_decoder_instruction_model",
        "conditions": ["guided"],
    },
    "qwen2_5_0_5b_instruct": {
        "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
        "task": "text-generation",
        "type": "compact_instruction_llm",
        "conditions": ["guided", "dynamic_few_shot"],
    },
}


def clean_text(text):
    return str(text).replace("_", " ").strip()


def get_allowed_frames(df):
    return sorted(df["gold_frame"].dropna().unique().tolist())


def build_frame_texts(df, allowed_frames):
    frame_texts = []

    for frame in allowed_frames:
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

        text = (
            f"Frame: {clean_text(frame)}. "
            f"Definition: {definition}. "
            f"Lexical units: {', '.join(lexical_units[:10])}. "
            f"Examples: {' '.join(examples[:3])}"
        )

        frame_texts.append(text)

    return frame_texts


def build_example_bank(df):
    example_texts = []

    for _, row in df.iterrows():
        example_texts.append(
            f"Sentence: {row['sentence']} "
            f"Target lexical unit: {row['lexical_unit']} "
            f"Frame: {row['gold_frame']}"
        )

    return example_texts


def retrieve_candidate_frames(row, allowed_frames, frame_embeddings, embedder):
    query = (
        f"Sentence: {row['sentence']} "
        f"Target lexical unit: {row['lexical_unit']}"
    )

    query_embedding = embedder.encode(
        [query],
        normalize_embeddings=True,
    )

    similarities = cosine_similarity(query_embedding, frame_embeddings)[0]
    top_indices = similarities.argsort()[::-1][:TOP_K]

    candidate_frames = [allowed_frames[i] for i in top_indices]

    return candidate_frames, query_embedding


def retrieve_dynamic_examples(row, df, example_embeddings, query_embedding):
    similarities = cosine_similarity(query_embedding, example_embeddings)[0]
    ranked_indices = similarities.argsort()[::-1]

    examples = []

    for idx in ranked_indices:
        candidate = df.iloc[idx]

        if str(candidate["sentence"]) == str(row["sentence"]):
            continue

        examples.append(candidate)

        if len(examples) >= DYNAMIC_EXAMPLES:
            break

    return examples


def make_candidate_frame_list(candidate_frames):
    return "\n".join([f"- {frame}" for frame in candidate_frames])


def make_dynamic_examples_text(dynamic_examples):
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


def build_guided_prompt(row, candidate_frames):
    frame_list = make_candidate_frame_list(candidate_frames)

    return f"""You are performing FrameNet semantic frame identification.

Choose exactly one frame from the candidate list.

Candidate frames:
{frame_list}

Sentence:
{row["sentence"]}

Target lexical unit:
{row["lexical_unit"]}

Rules:
- Use the target lexical unit in context.
- Choose only from the candidate frames.
- Return exactly one frame name.
- Do not explain.

Frame:"""


def build_dynamic_few_shot_prompt(row, candidate_frames, dynamic_examples):
    frame_list = make_candidate_frame_list(candidate_frames)
    examples_text = make_dynamic_examples_text(dynamic_examples)

    return f"""You are performing FrameNet semantic frame identification.

Choose exactly one frame from the candidate list.

Candidate frames:
{frame_list}

Relevant retrieved examples:
{examples_text}

Now classify this sentence.

Sentence:
{row["sentence"]}

Target lexical unit:
{row["lexical_unit"]}

Rules:
- Use the target lexical unit in context.
- Use the retrieved examples as guidance.
- Choose only from the candidate frames.
- Return exactly one frame name.
- Do not explain.

Frame:"""


def build_prompt(condition, row, candidate_frames, dynamic_examples):
    if condition == "guided":
        return build_guided_prompt(row, candidate_frames)

    if condition == "dynamic_few_shot":
        return build_dynamic_few_shot_prompt(
            row=row,
            candidate_frames=candidate_frames,
            dynamic_examples=dynamic_examples,
        )

    raise ValueError(f"Unknown condition: {condition}")


def extract_generated_text(output):
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
    output_clean = str(output).strip()
    output_clean = output_clean.replace("Frame:", "").strip()

    output_lower = output_clean.lower()
    output_normalized = output_lower.replace(" ", "_")

    for frame in candidate_frames:
        if frame.lower() == output_lower:
            return frame

    for frame in candidate_frames:
        if frame.lower() == output_normalized:
            return frame

    for frame in candidate_frames:
        if frame.lower() in output_lower:
            return frame

    for frame in candidate_frames:
        if frame.lower().replace("_", " ") in output_lower:
            return frame

    return "UNKNOWN"


def batch_predict(generator, model_info, prompts, candidate_frame_batches):
    generation_kwargs = {
        "max_new_tokens": 12,
        "do_sample": False,
        "batch_size": BATCH_SIZE,
    }

    if model_info["task"] == "text-generation":
        generation_kwargs["return_full_text"] = False

    with torch.inference_mode():
        outputs = generator(prompts, **generation_kwargs)

    predictions = []
    raw_outputs = []

    for output, candidate_frames in zip(outputs, candidate_frame_batches):
        generated_text = extract_generated_text(output)
        prediction = clean_prediction(generated_text, candidate_frames)

        predictions.append(prediction)
        raw_outputs.append(generated_text)

    return predictions, raw_outputs


def load_generator(model_info):
    print(f"Loading model: {model_info['model_name']}")

    return pipeline(
        task=model_info["task"],
        model=model_info["model_name"],
        device=0 if torch.cuda.is_available() else -1,
        torch_dtype=torch.float32,
    )


def save_results(results):
    out_df = pd.DataFrame(results)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_PATH, index=False)


def main():
    torch.set_num_threads(4)

    full_df = pd.read_csv(DATA_PATH)
    df = full_df.copy()

    if MAX_EXAMPLES is not None:
        df = df.head(MAX_EXAMPLES)

    allowed_frames = get_allowed_frames(full_df)

    print(f"Loaded {len(df)} evaluation examples.")
    print(f"Detected {len(allowed_frames)} total frames.")
    print(f"Top-k candidate frames: {TOP_K}")
    print(f"Dynamic examples: {DYNAMIC_EXAMPLES}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    print("\nLoading sentence embedding model...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    print("Encoding frame metadata for candidate retrieval...")
    frame_texts = build_frame_texts(full_df, allowed_frames)
    frame_embeddings = embedder.encode(
        frame_texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    print("Encoding example bank for dynamic few-shot retrieval...")
    example_texts = build_example_bank(full_df)
    example_embeddings = embedder.encode(
        example_texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    results = []

    for model_label, model_info in MODELS.items():
        print("\n" + "=" * 70)
        print(f"Running model: {model_label}")
        print(f"Model path: {model_info['model_name']}")
        print(f"Conditions: {model_info['conditions']}")
        print("=" * 70)

        generator = load_generator(model_info)

        for condition in model_info["conditions"]:
            print(f"\nCondition: {condition}")

            prompts = []
            row_records = []
            candidate_frame_batches = []
            retrieved_example_frames_records = []

            for _, row in tqdm(
                list(df.iterrows()),
                desc=f"Building prompts | {model_label} | {condition}",
            ):
                candidate_frames, query_embedding = retrieve_candidate_frames(
                    row=row,
                    allowed_frames=allowed_frames,
                    frame_embeddings=frame_embeddings,
                    embedder=embedder,
                )

                if condition == "dynamic_few_shot":
                    dynamic_examples = retrieve_dynamic_examples(
                        row=row,
                        df=full_df,
                        example_embeddings=example_embeddings,
                        query_embedding=query_embedding,
                    )
                else:
                    dynamic_examples = []

                prompt = build_prompt(
                    condition=condition,
                    row=row,
                    candidate_frames=candidate_frames,
                    dynamic_examples=dynamic_examples,
                )

                prompts.append(prompt)
                row_records.append(row)
                candidate_frame_batches.append(candidate_frames)
                retrieved_example_frames_records.append(
                    "|".join([ex["gold_frame"] for ex in dynamic_examples])
                )

            for start in tqdm(
                range(0, len(prompts), BATCH_SIZE),
                desc=f"Predicting | {model_label} | {condition}",
            ):
                batch_prompts = prompts[start:start + BATCH_SIZE]
                batch_rows = row_records[start:start + BATCH_SIZE]
                batch_candidates = candidate_frame_batches[
                    start:start + BATCH_SIZE
                ]
                batch_retrieved_frames = retrieved_example_frames_records[
                    start:start + BATCH_SIZE
                ]

                predictions, raw_outputs = batch_predict(
                    generator=generator,
                    model_info=model_info,
                    prompts=batch_prompts,
                    candidate_frame_batches=batch_candidates,
                )

                for row, prediction, raw_output, candidates, retrieved_frames in zip(
                    batch_rows,
                    predictions,
                    raw_outputs,
                    batch_candidates,
                    batch_retrieved_frames,
                ):
                    results.append({
                        "model": model_label,
                        "model_type": model_info["type"],
                        "condition": condition,

                        "top_k": TOP_K,
                        "dynamic_examples": (
                            DYNAMIC_EXAMPLES
                            if condition == "dynamic_few_shot"
                            else 0
                        ),

                        "sentence": row["sentence"],
                        "lexical_unit": row["lexical_unit"],
                        "gold_frame": row["gold_frame"],
                        "prediction": prediction,
                        "raw_output": raw_output,

                        "candidate_frames": "|".join(candidates),
                        "gold_in_candidates": row["gold_frame"] in candidates,
                        "retrieved_example_frames": retrieved_frames,

                        "domain": row.get("domain", "unknown"),
                        "sentence_length": row.get("sentence_length", None),
                        "is_long_context": row.get("is_long_context", False),
                        "is_ambiguous_lu": row.get("is_ambiguous_lu", False),
                        "hard_negative_frames": row.get("hard_negative_frames", ""),
                        "frame_definition": row.get("frame_definition", ""),
                    })

                save_results(results)

            save_results(results)

    final_df = pd.DataFrame(results)
    save_results(results)

    print("\nSaved predictions to:")
    print(OUTPUT_PATH)

    print("\nCandidate recall:")
    print(
        final_df.groupby(["model", "condition"])["gold_in_candidates"]
        .mean()
        .reset_index()
        .to_string(index=False)
    )

    print("\nUNKNOWN rate preview:")
    print(
        final_df.assign(is_unknown=final_df["prediction"].eq("UNKNOWN"))
        .groupby(["model", "condition"])["is_unknown"]
        .mean()
        .reset_index()
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()