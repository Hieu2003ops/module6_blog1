import os
import pickle
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer


def train_models(
    split_dir="/opt/airflow/data/split",
    vectorizer_dir="/opt/airflow/data/vectorizers",
    model_dir="/opt/airflow/data/models"
):
    """
    Train Logistic Regression models (word-level + ngram-level)
    using the split data and saved TF-IDF vectorizers.
    """

    # Ensure output folder exists
    os.makedirs(model_dir, exist_ok=True)

    # ====== LOAD SPLIT DATA ======
    X_train_path = os.path.join(split_dir, "X_train.csv")
    X_test_path  = os.path.join(split_dir, "X_test.csv")
    y_train_path = os.path.join(split_dir, "y_train.csv")
    y_test_path  = os.path.join(split_dir, "y_test.csv")

    if not all(os.path.exists(p) for p in [X_train_path, X_test_path, y_train_path, y_test_path]):
        raise FileNotFoundError("Missing split data. Please run preprocess.py first.")

    X_train = pd.read_csv(X_train_path)["full_review"]
    X_test  = pd.read_csv(X_test_path)["full_review"]
    y_train = pd.read_csv(y_train_path).iloc[:, 0]
    y_test  = pd.read_csv(y_test_path).iloc[:, 0]

    # ====== LOAD VECTORIZERS ======
    word_vec_path  = os.path.join(vectorizer_dir, "tfidf_word.pkl")
    ngram_vec_path = os.path.join(vectorizer_dir, "tfidf_ngram.pkl")

    if not os.path.exists(word_vec_path) or not os.path.exists(ngram_vec_path):
        raise FileNotFoundError("Vectorizer .pkl files missing. Run preprocess.py first.")

    with open(word_vec_path, "rb") as f:
        tfidf_word = pickle.load(f)

    with open(ngram_vec_path, "rb") as f:
        tfidf_ngram = pickle.load(f)

    # ====== TRANSFORM DATA ======
    X_train_word = tfidf_word.transform(X_train)
    X_test_word  = tfidf_word.transform(X_test)

    X_train_ngram = tfidf_ngram.transform(X_train)
    X_test_ngram  = tfidf_ngram.transform(X_test)

    # ====== MODEL 1: LOGISTIC REGRESSION (WORD LEVEL) ======
    log_word = LogisticRegression(max_iter=200)
    log_word.fit(X_train_word, y_train)

    y_pred_word = log_word.predict(X_test_word)
    acc_word = accuracy_score(y_test, y_pred_word)

    print("\n===== WORD LEVEL MODEL =====")
    print("Accuracy:", acc_word)
    print("Report:\n", classification_report(y_test, y_pred_word))

    # Save model
    word_model_path = os.path.join(model_dir, "log_word.pkl")
    with open(word_model_path, "wb") as f:
        pickle.dump(log_word, f)

    # ====== MODEL 2: LOGISTIC REGRESSION (N-GRAM LEVEL) ======
    log_ngram = LogisticRegression(max_iter=200)
    log_ngram.fit(X_train_ngram, y_train)

    y_pred_ngram = log_ngram.predict(X_test_ngram)
    acc_ngram = accuracy_score(y_test, y_pred_ngram)

    print("\n===== N-GRAM LEVEL MODEL =====")
    print("Accuracy:", acc_ngram)
    print("Report:\n", classification_report(y_test, y_pred_ngram))

    # Save model
    ngram_model_path = os.path.join(model_dir, "log_ngram.pkl")
    with open(ngram_model_path, "wb") as f:
        pickle.dump(log_ngram, f)

    print(f"\n[TRAIN] Models saved to: {model_dir}")

    return {
        "word_model": word_model_path,
        "ngram_model": ngram_model_path,
        "accuracy_word": acc_word,
        "accuracy_ngram": acc_ngram
    }
