import argparse
import json
import joblib

from .config import MODEL_INFO_PATH, MODEL_PATH, VECTORIZER_PATH
from .preprocessing import clean_text


def load_artifacts():
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        raise FileNotFoundError(
            "Model artifacts not found. Run `python -m src.train` first."
        )

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    model_name = "Trained classifier"
    if MODEL_INFO_PATH.exists():
        with open(MODEL_INFO_PATH, "r", encoding="utf-8") as f:
            model_name = json.load(f).get("model_name", model_name)

    return model, vectorizer, model_name


def predict_message(message: str):
    model, vectorizer, model_name = load_artifacts()
    cleaned = clean_text(message)
    features = vectorizer.transform([cleaned])

    prediction = int(model.predict(features)[0])

    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(features)[0][1])

    return {
        "label": "spam" if prediction == 1 else "ham",
        "spam_probability": probability,
        "model": model_name,
    }


def main():
    parser = argparse.ArgumentParser(description="Classify a message as spam or ham.")
    parser.add_argument("message", nargs="+", help="Message text to classify.")
    args = parser.parse_args()

    message = " ".join(args.message)
    result = predict_message(message)

    print(f"Prediction: {result['label'].upper()}")
    if result["spam_probability"] is not None:
        print(f"Spam probability: {result['spam_probability']:.2%}")
    print(f"Model: {result['model']}")


if __name__ == "__main__":
    main()
