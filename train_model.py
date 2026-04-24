import os
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from text_utils import clean_text

DATA_DIR = "data"
RAW_FILE_PATHS = [
    os.path.join(DATA_DIR, "SMSSpamCollection"),
    "SMSSpamCollection",
]
CSV_FILE_PATHS = [
    os.path.join(DATA_DIR, "SMSSpamCollection.csv"),
    "SMSSpamCollection.csv",
]
MODEL_DIR = "models"
PIPELINE_FILE = os.path.join(MODEL_DIR, "spam_pipeline.pkl")
MODEL_FILE = os.path.join(MODEL_DIR, "spam_model.pkl")
VECTORIZER_FILE = os.path.join(MODEL_DIR, "vectorizer.pkl")


def find_local_file(paths):
    for path in paths:
        if os.path.exists(path):
            return path
    return None


def load_dataset() -> pd.DataFrame:
    csv_path = find_local_file(CSV_FILE_PATHS)
    if csv_path:
        print(f"📂 Loading CSV dataset from {csv_path}...")
        return pd.read_csv(csv_path)

    raw_path = find_local_file(RAW_FILE_PATHS)
    if raw_path:
        os.makedirs(DATA_DIR, exist_ok=True)
        csv_path = os.path.join(DATA_DIR, "SMSSpamCollection.csv")
        print(f"📂 Converting raw dataset from {raw_path} to CSV at {csv_path}...")
        df_raw = pd.read_csv(raw_path, sep='\t', header=None, names=['label', 'message'])
        df_raw.to_csv(csv_path, index=False)
        print(f"✅ Created CSV dataset: {csv_path}")
        return df_raw

    raise FileNotFoundError(
        "Local dataset not found. Place 'data/SMSSpamCollection' or 'SMSSpamCollection' and/or 'data/SMSSpamCollection.csv' in the project folder."
    )


def build_pipeline() -> Pipeline:
    return Pipeline([
        (
            'tfidf',
            TfidfVectorizer(
                preprocessor=clean_text,
                ngram_range=(1, 2),
                max_features=7000,
                stop_words=None,
            ),
        ),
        (
            'classifier',
            LogisticRegression(
                solver='liblinear',
                class_weight='balanced',
                max_iter=1000,
            ),
        ),
    ])


def train():
    df = load_dataset()
    print(f"Dataset shape: {df.shape}")

    df = df.dropna(subset=['message'])
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    label_counts = df['label'].value_counts(normalize=True).to_dict()
    print(f"Label distribution: {label_counts}")

    X = df['message']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print("[[TN, FP]")
    print(" [FN, TP]]")
    print(cm)

    # Calculate additional metrics from confusion matrix
    tn, fp, fn, tp = cm.ravel()
    print("\nDetailed Metrics:")
    print(f"True Negatives (TN): {tn} - Correctly identified ham messages")
    print(f"False Positives (FP): {fp} - Ham messages incorrectly classified as spam")
    print(f"False Negatives (FN): {fn} - Spam messages incorrectly classified as ham")
    print(f"True Positives (TP): {tp} - Correctly identified spam messages")

    # Calculate rates
    total = tn + fp + fn + tp
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    print("\nRates:")
    print(f"Accuracy: {accuracy:.4f} - Overall correct predictions")
    print(f"Precision: {precision:.4f} - Of predicted spam, how many were actually spam")
    print(f"Recall/Sensitivity: {recall:.4f} - Of actual spam, how many were detected")
    print(f"Specificity: {specificity:.4f} - Of actual ham, how many were correctly identified")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, PIPELINE_FILE)
    joblib.dump(pipeline.named_steps['classifier'], MODEL_FILE)
    joblib.dump(pipeline.named_steps['tfidf'], VECTORIZER_FILE)
    print(f"✅ Saved pipeline to {PIPELINE_FILE}")
    print(f"✅ Saved model to {MODEL_FILE}")
    print(f"✅ Saved vectorizer to {VECTORIZER_FILE}")


if __name__ == '__main__':
    train()