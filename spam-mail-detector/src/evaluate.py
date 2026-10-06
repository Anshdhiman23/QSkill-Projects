import json
from .config import METRICS_PATH


def main():
    if not METRICS_PATH.exists():
        print("No metrics found. Run: python -m src.train")
        return

    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    print(f"Dataset rows: {metrics['dataset_rows']}")
    print(f"Selected model: {metrics['selected_model']}")

    for name in ("naive_bayes", "logistic_regression"):
        result = metrics[name]
        print(f"\n{name.replace('_', ' ').title()}")
        for metric in ("accuracy", "precision", "recall", "f1"):
            print(f"{metric.title():10}: {result[metric]:.4f}")


if __name__ == "__main__":
    main()
