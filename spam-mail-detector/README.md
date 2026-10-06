# Spam Mail Detector

An end-to-end NLP project that classifies messages as **spam** or **ham** using TF-IDF and two classical ML models:

- Multinomial Naive Bayes
- Logistic Regression
- Accuracy, precision, recall, F1-score
- Confusion matrix
- Interactive Streamlit web app
- Automatic model/data artifact creation

## Project structure

```text
spam-mail-detector/
├── data/
│   └── README.md
├── models/
│   └── .gitkeep
├── reports/
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── app.py
├── requirements.txt
└── README.md
```

## 1. Install

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Then:

```bash
pip install -r requirements.txt
```

## 2. Get the dataset

Use the **UCI SMS Spam Collection** dataset.

Download `SMSSpamCollection` and place it here:

```text
data/SMSSpamCollection
```

The file should be tab-separated and contain:

```text
ham<TAB>message text
spam<TAB>message text
```

You can also use another dataset if it has the same two-column structure.

## 3. Train the models

From the project root:

```bash
python -m src.train
```

This will:

1. Load the dataset.
2. Clean the text.
3. Split it into training/testing data.
4. Build a TF-IDF representation.
5. Train Naive Bayes and Logistic Regression.
6. Select the better model using F1-score.
7. Save the model and vectorizer in `models/`.
8. Save evaluation metrics and confusion matrices in `reports/`.

## 4. Test from the command line

```bash
python -m src.predict "Congratulations! You won a free prize. Claim now!"
```

Or:

```bash
python -m src.predict "Can you send me the notes from today's class?"
```

## 5. Run the web application

First train the model, then:

```bash
streamlit run app.py
```

Open the URL Streamlit prints, paste a message, and click **Classify**.

## Notes

- The first training run creates all model artifacts.
- The vectorizer is fitted only on training data to avoid test-set leakage.
- Metrics are saved to `reports/metrics.json`.
- The app reports the model's estimated probability, not a guarantee that a message is spam.
