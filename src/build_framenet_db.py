import pandas as pd
from pathlib import Path
from nltk.corpus import framenet as fn
from collections import defaultdict

OUT_PATH = Path("data/raw/framenet_sample.csv")

MAX_FRAMES = 20
MAX_EXAMPLES_PER_FRAME = 10

def clean_text(text):
    if text is None:
        return ""
    return " ".join(str(text).split())

def main():
    rows = []
    frame_counts = defaultdict(int)

    frames = fn.frames()

    for frame in frames:
        frame_name = frame.name
        frame_definition = clean_text(frame.definition)

        lexical_units = frame.lexUnit

        for lu_name, lu in lexical_units.items():
            lexical_unit = lu_name.split(".")[0]

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

                rows.append({
                    "sentence": sentence,
                    "lexical_unit": lexical_unit,
                    "gold_frame": frame_name,
                    "frame_definition": frame_definition
                })

                frame_counts[frame_name] += 1

        if len(frame_counts) >= MAX_FRAMES:
            break

    df = pd.DataFrame(rows)

    df = df.drop_duplicates(subset=["sentence", "lexical_unit", "gold_frame"])
    df = df.dropna()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"Saved dataset to {OUT_PATH}")
    print(f"Rows: {len(df)}")
    print(f"Frames: {df['gold_frame'].nunique()}")
    print(df.head())

if __name__ == "__main__":
    main()