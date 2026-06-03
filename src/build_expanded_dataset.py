import pandas as pd
from pathlib import Path
from collections import defaultdict
from nltk.corpus import framenet as fn

OUT_PATH = Path("data/processed/framenet_processed.csv")

MAX_FRAMES = 100
MAX_EXAMPLES_PER_FRAME = 10
MIN_SENTENCE_WORDS = 5

AMBIGUOUS_LUS = {
    "run", "break", "hold", "drive", "charge", "take", "make",
    "get", "go", "come", "keep", "set", "turn", "stand", "fall",
    "move", "carry", "draw", "cut", "pass", "open", "close",
}

DOMAIN_KEYWORDS = {
    "news_politics": ["government", "minister", "president", "policy", "council", "war", "court"],
    "science_medical": ["patient", "disease", "chemical", "cells", "research", "medical", "drug"],
    "business": ["company", "market", "bank", "finance", "trade", "industry", "employment"],
    "sports": ["game", "team", "player", "match", "race", "goal", "league"],
    "legal": ["court", "law", "legal", "case", "judge", "crime", "police"],
    "daily_life": ["home", "family", "friend", "car", "food", "room", "children"],
}

def clean_text(text):
    if text is None:
        return ""
    return " ".join(str(text).replace("\n", " ").split())

def infer_domain(sentence):
    s = sentence.lower()
    scores = {}

    for domain, words in DOMAIN_KEYWORDS.items():
        scores[domain] = sum(1 for w in words if w in s)

    best_domain = max(scores, key=scores.get)

    if scores[best_domain] == 0:
        return "general"

    return best_domain

def sentence_len(sentence):
    return len(str(sentence).split())

def get_lu_root(lu_name):
    return lu_name.split(".")[0].lower().strip()

def collect_lu_to_frames():
    lu_to_frames = defaultdict(set)

    for frame in fn.frames():
        for lu_name in frame.lexUnit.keys():
            root = get_lu_root(lu_name)
            lu_to_frames[root].add(frame.name)

    return lu_to_frames

def main():
    rows = []
    frame_counts = defaultdict(int)
    lu_to_frames = collect_lu_to_frames()

    frames = fn.frames()

    for frame in frames:
        frame_name = frame.name
        frame_definition = clean_text(frame.definition)

        if len(frame_counts) >= MAX_FRAMES:
            break

        for lu_name, lu in frame.lexUnit.items():
            lexical_unit = get_lu_root(lu_name)

            try:
                exemplars = lu.exemplars
            except Exception:
                continue

            for ex in exemplars:
                if frame_counts[frame_name] >= MAX_EXAMPLES_PER_FRAME:
                    break

                sentence = clean_text(ex.text)

                if not sentence:
                    continue

                if sentence_len(sentence) < MIN_SENTENCE_WORDS:
                    continue

                hard_negatives = sorted(
                    f for f in lu_to_frames.get(lexical_unit, set())
                    if f != frame_name
                )

                rows.append({
                    "sentence": sentence,
                    "lexical_unit": lexical_unit,
                    "gold_frame": frame_name,
                    "frame_definition": frame_definition,
                    "domain": infer_domain(sentence),
                    "sentence_length": sentence_len(sentence),
                    "is_long_context": sentence_len(sentence) >= 20,
                    "is_ambiguous_lu": lexical_unit in AMBIGUOUS_LUS or len(hard_negatives) > 1,
                    "hard_negative_frames": "|".join(hard_negatives[:5]),
                })

                frame_counts[frame_name] += 1

    df = pd.DataFrame(rows)

    df = df.drop_duplicates(subset=["sentence", "lexical_unit", "gold_frame"])
    df = df.dropna(subset=["sentence", "lexical_unit", "gold_frame"])

    df = df.sort_values(
        ["is_ambiguous_lu", "is_long_context", "gold_frame"],
        ascending=[False, False, True],
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"Saved dataset to: {OUT_PATH}")
    print(f"Rows: {len(df)}")
    print(f"Frames: {df['gold_frame'].nunique()}")
    print(f"Ambiguous examples: {df['is_ambiguous_lu'].sum()}")
    print(f"Long-context examples: {df['is_long_context'].sum()}")
    print("\nDomain distribution:")
    print(df["domain"].value_counts())

if __name__ == "__main__":
    main()