import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


PRED_PATH = Path("data/outputs/all_model_predictions.csv")
RESULT_PATH = Path("results/all_model_evaluation.csv")
SLICE_RESULT_PATH = Path("results/error_slice_evaluation.csv")


def main():
    if not PRED_PATH.exists():
        raise FileNotFoundError(f"Missing predictions file: {PRED_PATH}")

    df = pd.read_csv(PRED_PATH)

    required_columns = {"model", "condition", "gold_frame", "prediction"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)

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

        unknown_rate = (group["prediction"] == "UNKNOWN").mean()

        rows.append({
            "model": model,
            "condition": condition,
            "accuracy": accuracy,
            "precision_macro": precision,
            "recall_macro": recall,
            "f1_macro": f1,
            "unknown_rate": unknown_rate,
            "n_examples": len(group),
        })

    results = pd.DataFrame(rows).sort_values(["model", "condition"])
    results.to_csv(RESULT_PATH, index=False)

    slice_rows = []

    slice_columns = [
        "domain",
        "is_long_context",
        "is_ambiguous_lu",
    ]

    for slice_col in slice_columns:
        if slice_col not in df.columns:
            continue

        for (model, condition, slice_value), group in df.groupby(
            ["model", "condition", slice_col]
        ):
            y_true = group["gold_frame"]
            y_pred = group["prediction"]

            accuracy = accuracy_score(y_true, y_pred)
            unknown_rate = (group["prediction"] == "UNKNOWN").mean()

            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )

            slice_rows.append({
                "analysis_group": slice_col,
                "group_value": slice_value,
                "model": model,
                "condition": condition,
                "accuracy": accuracy,
                "precision_macro": precision,
                "recall_macro": recall,
                "f1_macro": f1,
                "unknown_rate": unknown_rate,
                "n_examples": len(group),
            })

    if slice_rows:
        slice_results = pd.DataFrame(slice_rows).sort_values(
            ["analysis_group", "group_value", "model", "condition"]
        )
        slice_results.to_csv(SLICE_RESULT_PATH, index=False)
    else:
        slice_results = pd.DataFrame()
        slice_results.to_csv(SLICE_RESULT_PATH, index=False)

    print("\n=== Evaluation Results ===")
    print(results.to_string(index=False))

    print(f"\nSaved evaluation results to: {RESULT_PATH}")
    print(f"Saved slice evaluation results to: {SLICE_RESULT_PATH}")


if __name__ == "__main__":
    main()