from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports"

DATASET_PATH = DATA_DIR / "SMSSpamCollection"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"
MODEL_PATH = MODEL_DIR / "spam_classifier.joblib"
MODEL_INFO_PATH = MODEL_DIR / "model_info.json"

METRICS_PATH = REPORT_DIR / "metrics.json"
NB_CM_PATH = REPORT_DIR / "naive_bayes_confusion_matrix.png"
LR_CM_PATH = REPORT_DIR / "logistic_regression_confusion_matrix.png"

TEST_SIZE = 0.20
RANDOM_STATE = 42
