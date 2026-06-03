import pandas as pd
from pathlib import Path

PRED_PATH = Path("data/outputs/all_model_predictions.csv")
ERROR_PATH = Path("results/error_analysis.csv")
FRAME_ERROR_PATH = Path("results/errors_by_frame.csv")

def main():
    df = pd.read_csv(PRED_PATH)

    df["correct"] = df["gold_frame"] == df["prediction"]

    errors = df[df["correct"] == False].copy()

    def classify_error(row):
        if row["prediction"] == "UNKNOWN":
            return "unknown_prediction"
        return "wrong_frame_prediction"

    errors["error_type"] = errors.apply(classify_error, axis=1)

    errors.to_csv(ERROR_PATH, index=False)

    frame_errors = (
        df.groupby(["model", "condition", "gold_frame"])
        .agg(
            total=("gold_frame", "count"),
            correct=("correct", "sum")
        )
        .reset_index()
    )

    frame_errors["accuracy"] = frame_errors["correct"] / frame_errors["total"]
    frame_errors.to_csv(FRAME_ERROR_PATH, index=False)

    print(f"Saved detailed errors to: {ERROR_PATH}")
    print(f"Saved frame-level errors to: {FRAME_ERROR_PATH}")
    print(f"Total errors: {len(errors)}")

if __name__ == "__main__":
    main()