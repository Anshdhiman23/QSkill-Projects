import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from .config import (
    MODEL_DIR,
    MODEL_INFO_PATH,
    MODEL_PATH,
    METRICS_PATH,
    NB_CM_PATH,
    LR_CM_PATH,
    RANDOM_STATE,
    REPORT_DIR,
    TEST_SIZE,
    VECTORIZER_PATH,
)
from .data_loader import load_dataset
from .preprocessing import clean_text


def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "report": classification_report(
            y_test, pred, target_names=["ham", "spam"], zero_division=0,
            output_dict=True
        ),
        "predictions": pred,
    }


def save_confusion_matrix(y_test, predictions, path, title):
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Ham", "Spam"],
        yticklabels=["Ham", "Spam"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    df = load_dataset()
    df["clean_message"] = df["message"].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_message"],
        df["target"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["target"],
    )

    vectorizer = TfidfVectorizer(
        lowercase=False,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    nb = MultinomialNB(alpha=0.5)
    lr = LogisticRegression(max_iter=2000, class_weight="balanced")

    nb.fit(X_train_tfidf, y_train)
    lr.fit(X_train_tfidf, y_train)

    nb_results = evaluate(nb, X_test_tfidf, y_test)
    lr_results = evaluate(lr, X_test_tfidf, y_test)

    save_confusion_matrix(
        y_test, nb_results["predictions"], NB_CM_PATH,
        "Naive Bayes Confusion Matrix"
    )
    save_confusion_matrix(
        y_test, lr_results["predictions"], LR_CM_PATH,
        "Logistic Regression Confusion Matrix"
    )

    # Choose the model with the higher F1-score.
    if lr_results["f1"] >= nb_results["f1"]:
        best_name = "Logistic Regression"
        best_model = lr
        best_results = lr_results
    else:
        best_name = "Multinomial Naive Bayes"
        best_model = nb
        best_results = nb_results

    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(best_model, MODEL_PATH)

    metrics = {
        "dataset_rows": int(len(df)),
        "ham_count": int((df["target"] == 0).sum()),
        "spam_count": int((df["target"] == 1).sum()),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "naive_bayes": {
            k: v for k, v in nb_results.items() if k != "predictions"
        },
        "logistic_regression": {
            k: v for k, v in lr_results.items() if k != "predictions"
        },
        "selected_model": best_name,
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    model_info = {
        "model_name": best_name,
        "feature_count": int(X_train_tfidf.shape[1]),
    }
    with open(MODEL_INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=2)

    print("\n=== Spam Mail Detector Training ===")
    print(f"Dataset rows: {len(df)}")
    print(f"Ham: {metrics['ham_count']}")
    print(f"Spam: {metrics['spam_count']}")
    print(f"Features: {X_train_tfidf.shape[1]}")
    print("\nNaive Bayes:")
    print(classification_report(
        y_test, nb_results["predictions"],
        target_names=["ham", "spam"], zero_division=0
    ))
    print("Logistic Regression:")
    print(classification_report(
        y_test, lr_results["predictions"],
        target_names=["ham", "spam"], zero_division=0
    ))
    print(f"Selected model: {best_name}")
    print(f"Saved model: {MODEL_PATH}")


if __name__ == "__main__":
    main()
