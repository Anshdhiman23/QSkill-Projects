import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)



iris = load_iris()

X = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

y = pd.Series(iris.target)


print("Dataset shape:", X.shape)

print("\nFeatures:")
print(X.columns.tolist())

print("\nClasses:")
print(iris.target_names)



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))



models = {

    "KNN": Pipeline([
        (
            "scaler",
            StandardScaler()
        ),

        (
            "classifier",
            KNeighborsClassifier(
                n_neighbors=5
            )
        )
    ]),


    "Logistic Regression": Pipeline([
        (
            "scaler",
            StandardScaler()
        ),

        (
            "classifier",
            LogisticRegression(
                max_iter=1000
            )
        )
    ]),


    "Decision Tree": DecisionTreeClassifier(
        max_depth=4,
        random_state=42
    )
}



os.makedirs("models", exist_ok=True)

os.makedirs("outputs", exist_ok=True)



results = []

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 5)
)


for ax, (name, model) in zip(
    axes,
    models.items()
):

    print("\n")
    print("=" * 60)
    print(name)
    print("=" * 60)



    model.fit(
        X_train,
        y_train
    )



    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        average="weighted"
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted"
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted"
    )


    cv_scores = cross_val_score(
        model,
        X,
        y,
        cv=5
    )

    cv_mean = cv_scores.mean()


    # -----------------------------
    # Print results
    # -----------------------------

    print(
        f"Accuracy:  {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall:    {recall:.4f}"
    )

    print(
        f"F1 Score:  {f1:.4f}"
    )

    print(
        f"CV Mean:   {cv_mean:.4f}"
    )


    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=iris.target_names
        )
    )



    cm = confusion_matrix(
        y_test,
        predictions
    )

    print("Confusion Matrix:")

    print(cm)


    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=iris.target_names
    )

    disp.plot(ax=ax)

    ax.set_title(name)


    results.append({

        "Model": name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1 Score": f1,

        "CV Mean Accuracy": cv_mean
    })


plt.tight_layout()

plt.savefig(
    "outputs/confusion_matrices.png",
    dpi=150
)

plt.show()


results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    by="CV Mean Accuracy",
    ascending=False
)


print("\n")
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    results_df.to_string(
        index=False
    )
)


results_df.to_csv(
    "outputs/model_comparison.csv",
    index=False
)



best_model_name = results_df.iloc[0]["Model"]

best_model = models[
    best_model_name
]


print("\nBest model:")
print(best_model_name)


best_model.fit(
    X,
    y
)

model_data = {

    "model": best_model,

    "model_name": best_model_name,

    "target_names":
        iris.target_names.tolist(),

    "feature_names":
        iris.feature_names
}


joblib.dump(
    model_data,
    "models/best_iris_model.joblib"
)


print(
    "\nModel saved successfully:"
)

print(
    "models/best_iris_model.joblib"
)