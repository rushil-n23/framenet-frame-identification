import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw/framenet_sample.csv")
OUT_PATH = Path("data/processed/framenet_processed.csv")

REQUIRED_COLUMNS = [
    "sentence",
    "lexical_unit",
    "gold_frame",
    "frame_definition"
]

def main():
    df = pd.read_csv(RAW_PATH)

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df.dropna(subset=REQUIRED_COLUMNS)
    df = df.drop_duplicates()

    df["sentence"] = df["sentence"].astype(str).str.strip()
    df["lexical_unit"] = df["lexical_unit"].astype(str).str.strip()
    df["gold_frame"] = df["gold_frame"].astype(str).str.strip()
    df["frame_definition"] = df["frame_definition"].astype(str).str.strip()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print("Preprocessing complete.")
    print(f"Rows: {len(df)}")
    print(f"Frames: {df['gold_frame'].nunique()}")
    print(f"Saved to: {OUT_PATH}")

if __name__ == "__main__":
    main()