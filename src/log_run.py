import pandas as pd
from pathlib import Path
from datetime import datetime

STANDARD_RESULTS = Path("results/all_model_evaluation.csv")
CLUSTER_RESULTS = Path("results/cluster_top3_evaluation.csv")
RUN_HISTORY = Path("results/run_history.csv")


def main():
    if not STANDARD_RESULTS.exists():
        print("Missing standard results.")
        return

    standard = pd.read_csv(STANDARD_RESULTS)

    if CLUSTER_RESULTS.exists():
        cluster = pd.read_csv(CLUSTER_RESULTS)
    else:
        cluster = None

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rows = []

    for _, row in standard.iterrows():
        record = {
            "run_time": timestamp,
            "model": row["model"],
            "condition": row["condition"],
            "accuracy": row["accuracy"],
            "precision_macro": row["precision_macro"],
            "recall_macro": row["recall_macro"],
            "f1_macro": row["f1_macro"],
            "unknown_rate": row["unknown_rate"],
            "n_examples": row["n_examples"]
        }

        if cluster is not None:
            match = cluster[
                (cluster["model"] == row["model"]) &
                (cluster["condition"] == row["condition"])
            ]

            if len(match) > 0:
                c = match.iloc[0]
                record["cluster_accuracy"] = c["cluster_accuracy"]
                record["cluster_top3_accuracy"] = c["cluster_top3_accuracy"]
                record["cluster_gain"] = c["cluster_gain"]

        rows.append(record)

    new_history = pd.DataFrame(rows)

    if RUN_HISTORY.exists():
        old_history = pd.read_csv(RUN_HISTORY)
        history = pd.concat([old_history, new_history], ignore_index=True)
    else:
        history = new_history

    RUN_HISTORY.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(RUN_HISTORY, index=False)

    print(f"Saved run history to {RUN_HISTORY}")


if __name__ == "__main__":
    main()