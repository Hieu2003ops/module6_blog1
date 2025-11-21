import os
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


def preprocess_dataset(
    cleaned_csv_path="/opt/airflow/data/clean/cleaned_reviews.csv",
    split_dir="/opt/airflow/data/split",
    vectorizer_dir="/opt/airflow/data/vectorizers"
):
    """
    Preprocess cleaned dataset:
    - Train/test split
    - TF-IDF (word-level)
    - TF-IDF (ngram 2,2)
    - Save splits + vectorizers
    """

    # Ensure folders exist
    os.makedirs(split_dir, exist_ok=True)
    os.makedirs(vectorizer_dir, exist_ok=True)

    if not os.path.exists(cleaned_csv_path):
        raise FileNotFoundError(f"Cleaned CSV not found: {cleaned_csv_path}")

    # Load cleaned dataset
    df = pd.read_csv(cleaned_csv_path)

    # Target + text
    X = df["full_review"]
    y = df["Sentiment_Label"] if "Sentiment_Label" in df.columns else None

    if y is None:
        raise ValueError("Column 'Sentiment_Label' not found. Ensure sentiment labeling is done before preprocess.")

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ----- TF-IDF Word-level -----
    tfidf_word = TfidfVectorizer()
    tfidf_word.fit(X_train)

    X_train_word = tfidf_word.transform(X_train)
    X_test_word  = tfidf_word.transform(X_test)

    # Save word vectorizer
    with open(os.path.join(vectorizer_dir, "tfidf_word.pkl"), "wb") as f:
        pickle.dump(tfidf_word, f)

    # ----- TF-IDF N-gram (2,2) -----
    tfidf_ngram = TfidfVectorizer(ngram_range=(2, 2))
    tfidf_ngram.fit(X_train)

    X_train_ngram = tfidf_ngram.transform(X_train)
    X_test_ngram  = tfidf_ngram.transform(X_test)

    # Save n-gram vectorizer
    with open(os.path.join(vectorizer_dir, "tfidf_ngram.pkl"), "wb") as f:
        pickle.dump(tfidf_ngram, f)

    # ----- Save train/test splits -----
    X_train.to_csv(os.path.join(split_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(split_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(split_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(split_dir, "y_test.csv"), index=False)

    print(f"[PREPROCESS] Splits saved to: {split_dir}")
    print(f"[PREPROCESS] Vectorizers saved to: {vectorizer_dir}")

    return {
        "X_train": os.path.join(split_dir, "X_train.csv"),
        "X_test": os.path.join(split_dir, "X_test.csv"),
        "y_train": os.path.join(split_dir, "y_train.csv"),
        "y_test": os.path.join(split_dir, "y_test.csv"),
        "tfidf_word": os.path.join(vectorizer_dir, "tfidf_word.pkl"),
        "tfidf_ngram": os.path.join(vectorizer_dir, "tfidf_ngram.pkl"),
    }
